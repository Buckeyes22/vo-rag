"""book_rag DAG orchestrator (VOR-7).

Tests the orchestration contract — step order, idempotency, canonical-id
derivation, failure handling, the advanced-understanding gate, and the
live-seam adapters — by stubbing the step coroutines. The real step functions
are covered by their own module tests; an end-to-end run against live
Qdrant/Postgres/Garage is the @pytest.mark.integration path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.models.artifact import ArtifactType, KnowledgeArtifact, LicenseClass, TrustLevel
from vo_rag.core.models.envelope import (
    EnvelopeStatus,
    IntakeEnvelope,
    IntakeSource,
    TargetPipeline,
)
from vo_rag.core.parsing.contract import ElementType, ExtractedImage, ParsedDocument, ParsedElement
from vo_rag.core.storage.figures import FigureStore
from vo_rag.pipelines.book_rag import workflow as workflow_mod
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.source_identity import canonical_source_id
from vo_rag.pipelines.book_rag.steps.chunk_embed import do_build_chunks, do_upsert_raw
from vo_rag.pipelines.book_rag.workflow import (
    RAW_COLLECTION,
    BookRagContext,
    BookRagPipeline,
    PipelineOutput,
)

_TS = datetime(2026, 1, 1, tzinfo=UTC)
_HASH = "a" * 64
_VERIFICATION = {"status": "verified", "sha256": _HASH, "size_bytes": 42}
SRC = "ab" * 32
IMG_A_REF = FigureStore.figure_ref(SRC, 1, 0, "png")


def _envelope(*, status: EnvelopeStatus = EnvelopeStatus.RECEIVED) -> IntakeEnvelope:
    return IntakeEnvelope(
        id="01KT6XE8M1MW42M0Y0JJKRZSVP",  # pragma: allowlist secret
        source_type=IntakeSource.FILE_DROP,
        raw_content={
            "garage_key": "drops/book.pdf",
            "file_hash": _HASH,
            "object_verification": _VERIFICATION,
        },
        status=status,
        created_at=_TS,
        updated_at=_TS,
    )


def _parsed_doc(
    *,
    source_id: str = "src-1",
    title: str = "Parsed Book Title",
) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        source_id=source_id,
        title=title,
        author="Parsed Author",
        total_pages=1,
        elements=[
            ParsedElement(
                element_id="e-1",
                element_type=ElementType.TITLE,
                text="Chapter 1",
                page_number=1,
            ),
            ParsedElement(
                element_id="e-2",
                element_type=ElementType.NARRATIVE_TEXT,
                text="Narrative text forming a chunk for source identity tests.",
                page_number=1,
            ),
        ],
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="test",
    )


def _counting_store(existing_points: int = 0) -> SimpleNamespace:
    """A qdrant_store whose `count_source` reports `existing_points` for any source.

    `ingest` consults this before doing any work (VOR-1403), so the default of 0
    keeps every other test on the "not a duplicate, run the steps" path.
    """
    calls: list[tuple[str, str]] = []

    async def _count_source(collection: str, source_id: str) -> int:
        calls.append((collection, source_id))
        return existing_points

    return SimpleNamespace(count_source=_count_source, count_source_calls=calls)


def _pipeline(**overrides: Any) -> BookRagPipeline:
    base: dict[str, Any] = {
        "garage": SimpleNamespace(),
        "qdrant_store": _counting_store(),
        "embedding_client": SimpleNamespace(),
        "llm_client": SimpleNamespace(),
        "retriever": SimpleNamespace(),
        "parser_router": SimpleNamespace(),
        "figures": SimpleNamespace(),
    }
    base.update(overrides)
    return BookRagPipeline(**base)


class _FakeAcquire:
    def __init__(self, conn: _FakePgConn) -> None:
        self.conn = conn

    async def __aenter__(self) -> _FakePgConn:
        return self.conn

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class _FakePgConn:
    def __init__(self, *, key: str, row: dict[str, Any] | None) -> None:
        self.key = key
        self.row = row
        self.calls: list[tuple[str, str]] = []

    async def fetchrow(self, sql: str, key: str) -> dict[str, Any] | None:
        self.calls.append((sql, key))
        if key == self.key:
            return self.row
        return None


class _FakePgPool:
    def __init__(self, conn: _FakePgConn) -> None:
        self.conn = conn
        self.acquire_count = 0

    def acquire(self) -> _FakeAcquire:
        self.acquire_count += 1
        return _FakeAcquire(self.conn)


class _FakeQdrantStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[Any]]] = []

    async def upsert_vectors(self, collection: str, points: list[Any]) -> None:
        self.upserts.append((collection, points))


def _stub_all_steps(pipeline: BookRagPipeline, order: list[str]) -> None:
    """Replace every step coroutine with a recorder that mutates the context."""

    def _rec(name: str) -> Any:
        async def _step(ctx: BookRagContext) -> None:
            order.append(name)
            if name == "register_source":
                ctx.source_id = "vo-rag-corpus-sha256:" + _HASH

        return _step

    pipeline._register_source = _rec("register_source")  # type: ignore[method-assign]
    pipeline._parse = _rec("parse")  # type: ignore[method-assign]
    pipeline._visual = _rec("visual")  # type: ignore[method-assign]
    pipeline._chunk_embed = _rec("chunk_embed")  # type: ignore[method-assign]
    pipeline._sterilize = _rec("sterilize")  # type: ignore[method-assign]
    pipeline._advanced_understanding = _rec("advanced_understanding")  # type: ignore[method-assign]
    pipeline._register_artifact = _rec("register_artifact")  # type: ignore[method-assign]
    pipeline._validate = _rec("validate")  # type: ignore[method-assign]
    pipeline._retrieval_smoke = _rec("retrieval_smoke")  # type: ignore[method-assign]


# ── book identity resolution ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_book_metadata_prefers_pg_source_identity_and_payload_gets_project_id() -> None:
    source_id = canonical_source_id(_HASH)
    conn = _FakePgConn(
        key=source_id,
        row={
            "id": source_id,
            "title": "PG Book",
            "author": "PG Author",
            "license_class": "licensed_internal",
            "project_id": "motorsport",
            "file_hash": _HASH,
        },
    )
    pipeline = _pipeline(pg_pool=_FakePgPool(conn))
    env = _envelope()
    env.metadata = {
        "title": "Envelope Book",
        "author": "Envelope Author",
        "license_class": "unlicensed",
        "tags": ["envelope-tag"],
    }
    ctx = BookRagContext(envelope=env, config=pipeline.config, source_id=source_id)
    ctx.parsed = _parsed_doc(source_id=source_id)

    metadata = await pipeline._book_metadata(ctx)
    chunks = do_build_chunks(ctx.parsed, book_metadata=metadata)
    fake_store = _FakeQdrantStore()
    await do_upsert_raw(
        chunks,
        {"dense": [[0.1] * 1024], "sparse": [{1: 0.5}], "colbert": [[[0.1] * 1024]]},
        fake_store,
        source_id,
    )

    assert metadata["source_book"] == "PG Book (PG Author)"
    assert metadata["title"] == "PG Book"
    assert metadata["author"] == "PG Author"
    assert metadata["license_class"] == "licensed_internal"
    assert metadata["project_id"] == "motorsport"
    assert metadata["canonical_source_id"] == source_id
    assert metadata["domain_tags"] == ["envelope-tag"]
    assert chunks[0].metadata["project_id"] == "motorsport"
    assert fake_store.upserts[0][1][0].payload["project_id"] == "motorsport"
    assert [key for _, key in conn.calls] == [source_id]


@pytest.mark.asyncio
async def test_book_metadata_without_pg_pool_keeps_envelope_identity() -> None:
    source_id = canonical_source_id(_HASH)
    pipeline = _pipeline()
    env = _envelope()
    env.metadata = {
        "title": "Envelope Book",
        "author": "Envelope Author",
        "license_class": "licensed_internal",
        "domain_tags": ["vehicle-dynamics"],
    }
    ctx = BookRagContext(envelope=env, config=pipeline.config, source_id=source_id)
    ctx.parsed = _parsed_doc(source_id=source_id, title="Parsed Fallback")

    metadata = await pipeline._book_metadata(ctx)

    assert metadata["source_book"] == "Envelope Book (Envelope Author)"
    assert metadata["title"] == "Envelope Book"
    assert metadata["author"] == "Envelope Author"
    assert metadata["domain_tags"] == ["vehicle-dynamics"]
    assert metadata["license_class"] == "licensed_internal"
    assert metadata["canonical_source_id"] == source_id
    assert metadata.get("project_id") is None


@pytest.mark.asyncio
async def test_book_metadata_pg_miss_falls_back_to_envelope_identity() -> None:
    source_id = canonical_source_id(_HASH)
    conn = _FakePgConn(key="different-source", row=None)
    pipeline = _pipeline(pg_pool=_FakePgPool(conn))
    env = _envelope()
    env.metadata = {
        "title": "Envelope Book",
        "author": "Envelope Author",
        "license_class": "licensed_internal",
        "tags": ["fallback-tag"],
    }
    ctx = BookRagContext(envelope=env, config=pipeline.config, source_id=source_id)
    ctx.parsed = _parsed_doc(source_id=source_id)

    metadata = await pipeline._book_metadata(ctx)

    assert metadata["source_book"] == "Envelope Book (Envelope Author)"
    assert metadata["title"] == "Envelope Book"
    assert metadata["author"] == "Envelope Author"
    assert metadata["domain_tags"] == ["fallback-tag"]
    assert metadata["license_class"] == "licensed_internal"
    assert metadata["canonical_source_id"] == source_id
    assert metadata.get("project_id") is None
    assert [key for _, key in conn.calls] == [
        source_id,
        f"vo-rag-corpus-sha256:{source_id}",
        source_id.upper(),
    ]


# ── DAG order + terminal status ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_runs_dag_in_correct_order() -> None:
    pipeline = _pipeline()
    order: list[str] = []
    _stub_all_steps(pipeline, order)
    env = _envelope()

    out = await pipeline.ingest(env)

    assert order == [
        "register_source",
        "parse",
        "visual",
        "chunk_embed",
        "sterilize",
        "advanced_understanding",
        "register_artifact",
        "validate",
        "retrieval_smoke",
    ]
    assert env.status is EnvelopeStatus.COMPLETED
    assert isinstance(out, PipelineOutput)
    assert out.idempotent_noop is False


# ── quality gate at completion (VOR-1434) ────────────────────────────


class _FakeQualityScrollClient:
    """`AsyncQdrantClient.scroll` stand-in for `health.collect_stats`: returns
    every payload on the first page (`offset=None`), then stops."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.scroll_calls: list[Any] = []
        self.collections: list[str] = []

    async def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: Any,
        with_payload: Any,
        with_vectors: bool,
        limit: int,
        offset: Any = None,
    ) -> tuple[list[Any], Any]:
        self.scroll_calls.append(scroll_filter)
        self.collections.append(collection_name)
        if offset is not None:
            return [], None
        points = [
            SimpleNamespace(id=f"p-{i}", payload=payload) for i, payload in enumerate(self.payloads)
        ]
        return points, None


class _BoomScrollClient:
    """A Qdrant client whose `scroll` always raises -- used to prove the
    quality gate is best-effort."""

    async def scroll(self, **kwargs: Any) -> tuple[list[Any], Any]:
        raise RuntimeError("boom")


class _FakeQualityPool:
    """A Postgres pool stand-in recording every `execute` call."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        return "UPDATE 1"

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        return []


async def _no_dup_count_source(collection: str, source_id: str) -> int:
    return 0


def _quality_writes(pool: _FakeQualityPool) -> list[tuple[str, tuple[Any, ...]]]:
    """Only the quality writes (the metrics merges share the pool)."""
    return [(query, args) for query, args in pool.executed if "{quality}" in query]


def _quality_store(payloads: list[dict[str, Any]]) -> SimpleNamespace:
    return SimpleNamespace(
        count_source=_no_dup_count_source,
        _client=_FakeQualityScrollClient(payloads),
    )


@pytest.mark.asyncio
async def test_ingest_records_quality_findings_to_envelope_metadata() -> None:
    """VOR-1434: the workflow's completion step runs `book_findings` on
    the book it just upserted and writes the result to
    `intake_envelopes.metadata->'quality'` -- the image-only EPUB shape
    (VOR-1433: an EPUB parsed to 1 text chunk + N uncaptioned figures)."""
    payloads = [
        {
            "content": "the only real text in this book",
            "chunk_type": "text",
            "page_number": 1,
            "parser_used": "epub",
        },
        *[
            {
                "content": "Section: Foo",
                "chunk_type": "figure_composite",
                "page_number": i + 1,
                "parser_used": "epub",
            }
            for i in range(9)
        ],
    ]
    pool = _FakeQualityPool()
    pipeline = _pipeline(qdrant_store=_quality_store(payloads), pg_pool=pool)
    order: list[str] = []
    _stub_all_steps(pipeline, order)
    env = _envelope()

    out = await pipeline.ingest(env)

    assert env.status is EnvelopeStatus.COMPLETED
    assert out.idempotent_noop is False
    [(query, args)] = _quality_writes(pool)
    assert "metadata" in query
    assert "quality" in query
    envelope_id, quality = args
    assert envelope_id == str(env.id)
    assert quality["findings"] == ["degenerate", "empty_figures", "image_only_suspect"]
    assert quality["stats"]["chunks"] == 10
    assert quality["stats"]["figure_chunks"] == 9


@pytest.mark.asyncio
async def test_ingest_writes_a_clean_quality_report_when_nothing_trips() -> None:
    long_text = "well-parsed prose, " * 40
    payloads = [
        {
            "content": long_text,
            "chunk_type": "text",
            "page_number": i + 1,
            "parser_used": "docling",
        }
        for i in range(5)
    ]
    pool = _FakeQualityPool()
    pipeline = _pipeline(qdrant_store=_quality_store(payloads), pg_pool=pool)
    _stub_all_steps(pipeline, [])

    await pipeline.ingest(_envelope())

    [(_, (_, quality))] = _quality_writes(pool)
    assert quality["findings"] == []


@pytest.mark.asyncio
async def test_ingest_completes_even_when_the_quality_check_raises() -> None:
    """The quality gate must never fail an otherwise-successful ingest."""
    store = SimpleNamespace(count_source=_no_dup_count_source, _client=_BoomScrollClient())
    pool = _FakeQualityPool()
    pipeline = _pipeline(qdrant_store=store, pg_pool=pool)
    _stub_all_steps(pipeline, [])
    env = _envelope()

    out = await pipeline.ingest(env)

    assert env.status is EnvelopeStatus.COMPLETED
    assert out.status is EnvelopeStatus.COMPLETED
    assert _quality_writes(pool) == []  # never partially written


@pytest.mark.asyncio
async def test_ingest_scrolls_the_pipelines_raw_collection_override() -> None:
    """`collect_stats`/`_collect_one`/
    `scroll_book_payloads` must be told the pipeline's own raw collection
    (`VORAG_RAW_COLLECTION`, e.g. `ops/rc_ingest_batch.py` bulk runs, plumbed
    through as `runner.py`'s `raw_collection` -> `BookRagPipeline.raw_collection`)
    -- scrolling the hardcoded canonical collection instead would silently
    miss every point a bulk run just upserted."""
    scroll_client = _FakeQualityScrollClient(
        [{"content": "x" * 400, "chunk_type": "text", "page_number": 1, "parser_used": "docling"}]
    )
    store = SimpleNamespace(count_source=_no_dup_count_source, _client=scroll_client)
    pool = _FakeQualityPool()
    pipeline = _pipeline(qdrant_store=store, pg_pool=pool, raw_collection="vo_rag_books_raw_bulk")
    _stub_all_steps(pipeline, [])

    await pipeline.ingest(_envelope())

    assert scroll_client.collections == ["vo_rag_books_raw_bulk"]
    [(_, (_, quality))] = _quality_writes(pool)
    assert quality["stats"]["chunks"] == 1


@pytest.mark.asyncio
async def test_ingest_records_no_chunks_finding_when_the_book_has_zero_points() -> None:
    """A completed book that comes back with zero
    Qdrant points (wrong collection scrolled, or genuinely nothing written)
    must never be recorded as a clean quality report."""
    pool = _FakeQualityPool()
    pipeline = _pipeline(qdrant_store=_quality_store([]), pg_pool=pool)
    _stub_all_steps(pipeline, [])

    await pipeline.ingest(_envelope())

    [(_, (_, quality))] = _quality_writes(pool)
    assert quality["findings"] == ["no_chunks"]
    assert quality["stats"]["chunks"] == 0


# ── idempotency ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_completed_envelope_is_noop() -> None:
    pipeline = _pipeline()
    order: list[str] = []
    _stub_all_steps(pipeline, order)
    env = _envelope(status=EnvelopeStatus.COMPLETED)

    out = await pipeline.ingest(env)

    assert order == []  # no step ran
    assert out.idempotent_noop is True


@pytest.mark.asyncio
async def test_already_ingested_file_hash_is_skipped_without_running_steps() -> None:
    """VOR-1403: the same bytes under a new envelope must not be re-embedded.

    Chunk point ids are deterministic in source_id, so a re-ingest re-upserts
    byte-identical points -- full parse+embed cost for zero information gained.
    """
    store = _counting_store(existing_points=412)
    pipeline = _pipeline(qdrant_store=store)
    order: list[str] = []
    _stub_all_steps(pipeline, order)
    env = _envelope()

    out = await pipeline.ingest(env)

    assert order == ["register_source"]  # identity is verified before the duplicate decision
    assert out.idempotent_noop is True
    assert out.source_id == canonical_source_id(_HASH)
    assert env.status is EnvelopeStatus.COMPLETED
    assert store.count_source_calls == [(RAW_COLLECTION, canonical_source_id(_HASH))]


@pytest.mark.asyncio
async def test_unseen_file_hash_runs_the_full_pipeline() -> None:
    store = _counting_store(existing_points=0)
    pipeline = _pipeline(qdrant_store=store)
    order: list[str] = []
    _stub_all_steps(pipeline, order)

    out = await pipeline.ingest(_envelope())

    assert order[0] == "register_source"
    assert out.idempotent_noop is False


@pytest.mark.asyncio
async def test_force_reingest_env_overrides_the_duplicate_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator full-overwrite re-ingests (VOR-1393, VOR-309) must still work."""
    monkeypatch.setenv("VORAG_BOOK_RAG_FORCE_REINGEST", "true")
    store = _counting_store(existing_points=412)
    pipeline = _pipeline(qdrant_store=store)
    order: list[str] = []
    _stub_all_steps(pipeline, order)

    out = await pipeline.ingest(_envelope())

    assert order[0] == "register_source"
    assert out.idempotent_noop is False
    assert store.count_source_calls == []  # not even consulted


@pytest.mark.asyncio
async def test_envelope_without_file_hash_is_rejected_before_duplicate_detection() -> None:
    store = _counting_store(existing_points=412)
    pipeline = _pipeline(qdrant_store=store)
    order: list[str] = []
    env = _envelope()
    env.raw_content = {"garage_key": "drops/book.pdf"}

    with pytest.raises(ValueError, match="verification and file_hash"):
        await pipeline.ingest(env)
    assert order == []


@pytest.mark.asyncio
async def test_duplicate_skip_honours_raw_collection_override() -> None:
    store = _counting_store(existing_points=1)
    pipeline = _pipeline(qdrant_store=store, raw_collection="vo_rag_books_ab_v1")
    _stub_all_steps(pipeline, [])

    out = await pipeline.ingest(_envelope())

    assert store.count_source_calls == [("vo_rag_books_ab_v1", canonical_source_id(_HASH))]
    assert out.source_id == canonical_source_id(_HASH)
    assert out.idempotent_noop is True


# ── artifact registration ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_artifact_inserts_book_rag_knowledge_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[Any, KnowledgeArtifact]] = []

    async def _fake_insert_artifact(pool: Any, artifact: KnowledgeArtifact) -> None:
        captured.append((pool, artifact))

    monkeypatch.setattr(workflow_mod, "insert_artifact", _fake_insert_artifact, raising=False)
    pg_pool = object()
    pipeline = _pipeline(pg_pool=pg_pool)
    env = _envelope()
    env.metadata = {
        "title": "Envelope Book",
        "author": "Envelope Author",
        "license_class": "cleared",
        "tags": ["vehicle-dynamics"],
    }
    ctx = BookRagContext(
        envelope=env,
        config=pipeline.config,
        source_id=canonical_source_id(_HASH),
        raw_point_ids=["raw-p1", "raw-p2"],
    )
    ctx.parsed = _parsed_doc(source_id=ctx.source_id, title="Parsed Fallback")

    await pipeline._register_artifact(ctx)

    assert len(captured) == 1
    assert captured[0][0] is pg_pool
    artifact = captured[0][1]
    assert artifact.pipeline is TargetPipeline.BOOK_RAG
    assert artifact.artifact_type is ArtifactType.EXTRACTION
    assert artifact.title == "Envelope Book"
    assert artifact.content == {"source_id": ctx.source_id}
    assert artifact.tags == ["vehicle-dynamics"]
    assert artifact.license_class is LicenseClass.CLEARED
    assert artifact.trust_level is TrustLevel.TRUSTED
    assert artifact.qdrant_collection == RAW_COLLECTION
    assert artifact.qdrant_point_ids == ["raw-p1", "raw-p2"]
    assert len(artifact.source_provenance) == 1
    assert artifact.source_provenance[0].source_id == ctx.source_id
    assert artifact.source_provenance[0].source_title == "Envelope Book"
    assert artifact.source_provenance[0].source_type == "file_drop"
    assert artifact.source_provenance[0].contribution == 1.0


@pytest.mark.asyncio
async def test_register_artifact_id_is_deterministic_from_source_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[KnowledgeArtifact] = []

    async def _fake_insert_artifact(pool: Any, artifact: KnowledgeArtifact) -> None:
        captured.append(artifact)

    monkeypatch.setattr(workflow_mod, "insert_artifact", _fake_insert_artifact, raising=False)
    pipeline = _pipeline(pg_pool=object())
    first = BookRagContext(
        envelope=_envelope(),
        config=pipeline.config,
        source_id=canonical_source_id(_HASH),
        raw_point_ids=["raw-p1"],
    )
    second = BookRagContext(
        envelope=_envelope(),
        config=pipeline.config,
        source_id=canonical_source_id(_HASH),
        raw_point_ids=["raw-p2"],
    )

    await pipeline._register_artifact(first)
    await pipeline._register_artifact(second)

    assert captured[0].id == captured[1].id
    assert len(captured[0].id) == 26


@pytest.mark.asyncio
async def test_register_artifact_skips_when_pg_pool_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def _fake_insert_artifact(pool: Any, artifact: KnowledgeArtifact) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(workflow_mod, "insert_artifact", _fake_insert_artifact, raising=False)
    pipeline = _pipeline(pg_pool=None)
    ctx = BookRagContext(
        envelope=_envelope(),
        config=pipeline.config,
        source_id=canonical_source_id(_HASH),
        raw_point_ids=["raw-p1"],
    )

    await pipeline._register_artifact(ctx)

    assert called is False


@pytest.mark.asyncio
async def test_register_artifact_uses_validator_pool_when_pg_pool_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []

    async def _fake_insert_artifact(pool: Any, artifact: KnowledgeArtifact) -> None:
        captured.append(pool)

    monkeypatch.setattr(workflow_mod, "insert_artifact", _fake_insert_artifact, raising=False)
    validator_pool = object()
    pipeline = _pipeline(pool=validator_pool, pg_pool=None)
    ctx = BookRagContext(
        envelope=_envelope(),
        config=pipeline.config,
        source_id=canonical_source_id(_HASH),
        raw_point_ids=["raw-p1"],
    )

    await pipeline._register_artifact(ctx)

    assert captured == [validator_pool]


# ── register_source derives the canonical id ────────────────────────────────


@pytest.mark.asyncio
async def test_register_source_derives_canonical_id() -> None:
    pipeline = _pipeline()
    ctx = BookRagContext(envelope=_envelope(), config=pipeline.config)
    await pipeline._register_source(ctx)
    assert ctx.source_id == "vo-rag-corpus-sha256:" + _HASH


@pytest.mark.asyncio
async def test_register_source_requires_file_hash() -> None:
    pipeline = _pipeline()
    env = _envelope()
    env.raw_content = {"garage_key": "drops/book.pdf"}  # no file_hash
    ctx = BookRagContext(envelope=env, config=pipeline.config)
    with pytest.raises(ValueError, match="file_hash"):
        await pipeline._register_source(ctx)


# ── failure handling ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_step_failure_marks_failed_and_reraises() -> None:
    pipeline = _pipeline()
    order: list[str] = []
    _stub_all_steps(pipeline, order)

    async def _boom(ctx: BookRagContext) -> None:
        raise RuntimeError("parse exploded")

    pipeline._parse = _boom  # type: ignore[method-assign]
    env = _envelope()

    with pytest.raises(RuntimeError, match="parse exploded"):
        await pipeline.ingest(env)

    assert env.status is EnvelopeStatus.FAILED
    assert env.error == "parse exploded"
    assert order == ["register_source"]  # stopped at the failing step


# ── advanced-understanding gate ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_advanced_understanding_noop_when_all_disabled() -> None:
    pipeline = _pipeline()
    ctx = BookRagContext(envelope=_envelope(), config=BookRagConfig())
    await pipeline._advanced_understanding(ctx)  # must not raise


@pytest.mark.asyncio
async def test_advanced_understanding_raises_when_enabled() -> None:
    pipeline = _pipeline(config=BookRagConfig(enable_dsi=True))
    ctx = BookRagContext(envelope=_envelope(), config=pipeline.config)
    with pytest.raises(RuntimeError, match="AdvancedBuildWorkflow"):
        await pipeline._advanced_understanding(ctx)


# ── sterilize first-class branch ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sterilize_skipped_when_disabled() -> None:
    pipeline = _pipeline(config=BookRagConfig(sterilize_enabled=False))
    ctx = BookRagContext(envelope=_envelope(), config=pipeline.config)
    await pipeline._sterilize(ctx)
    assert ctx.sterile_point_ids == []


@pytest.mark.asyncio
async def test_sterilize_computes_deterministic_point_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    import vo_rag.pipelines.book_rag.steps.sterilize as ster_mod
    from vo_rag.core.storage.qdrant import deterministic_point_id

    async def _fake_do_sterilize(
        *,
        raw_chunks: Any,
        llm_client: Any,
        config: Any,
        store: Any,
        embed: Any,
        **kwargs: Any,
    ) -> list[Any]:
        return [SimpleNamespace(source_id="sid"), SimpleNamespace(source_id="sid")]

    monkeypatch.setattr(ster_mod, "do_sterilize", _fake_do_sterilize)
    pipeline = _pipeline(config=BookRagConfig(sterilize_enabled=True))
    ctx = BookRagContext(envelope=_envelope(), config=pipeline.config)
    ctx.chunks = [SimpleNamespace(source_id="sid")]  # type: ignore[list-item]

    await pipeline._sterilize(ctx)

    assert ctx.sterile_point_ids == [
        deterministic_point_id("sid", 0, "sterile"),
        deterministic_point_id("sid", 1, "sterile"),
    ]


@pytest.mark.asyncio
async def test_validate_threads_sterilize_enabled_to_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeValidator:
        def __init__(
            self,
            qdrant: Any,
            pool: Any,
            garage: Any,
            retriever: Any,
            *,
            figures: Any,
            artifacts_bucket: str,
            sterilize_enabled: bool,
            raw_collection: str = RAW_COLLECTION,
        ) -> None:
            captured["qdrant"] = qdrant
            captured["pool"] = pool
            captured["garage"] = garage
            captured["retriever"] = retriever
            captured["artifacts_bucket"] = artifacts_bucket
            captured["sterilize_enabled"] = sterilize_enabled
            captured["raw_collection"] = raw_collection

        async def validate(self, *, document_id: str, source_id: str, **kwargs: Any) -> Any:
            captured["document_id"] = document_id
            captured["source_id"] = source_id
            captured["report"] = SimpleNamespace(passed=True, checks=[])
            return captured["report"]

    monkeypatch.setattr("vo_rag.pipelines.book_rag.workflow.IngestionValidator", _FakeValidator)
    pipeline = _pipeline(config=BookRagConfig(sterilize_enabled=False), pool=None)
    ctx = BookRagContext(envelope=_envelope(), config=pipeline.config, source_id="src-1")
    ctx.parse_meta = {"document_id": "doc-1"}

    await pipeline._validate(ctx)

    assert captured["sterilize_enabled"] is False
    assert captured["pool"] is None
    assert captured["document_id"] == "doc-1"
    assert captured["source_id"] == "src-1"
    assert ctx.validation_report is captured["report"]


# ── output aggregates point ids ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_output_aggregates_qdrant_point_ids() -> None:
    pipeline = _pipeline()
    order: list[str] = []
    _stub_all_steps(pipeline, order)

    async def _chunk(ctx: BookRagContext) -> None:
        ctx.raw_point_ids = ["r1", "r2"]

    async def _ster(ctx: BookRagContext) -> None:
        ctx.sterile_point_ids = ["s1"]

    pipeline._chunk_embed = _chunk  # type: ignore[method-assign]
    pipeline._sterilize = _ster  # type: ignore[method-assign]

    out = await pipeline.ingest(_envelope())
    assert out.raw_point_ids == ["r1", "r2"]
    assert out.sterile_point_ids == ["s1"]
    assert out.qdrant_point_ids == ["r1", "r2", "s1"]


# ── adapters ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_fn_extracts_chunk_ids() -> None:
    class _FakeRetriever:
        async def hybrid_search(
            self,
            query: str,
            *,
            collection: str,
            top_k: int,
            prefetch_depth: int,
            filters: dict[str, Any] | None = None,
        ) -> list[SimpleNamespace]:
            assert collection == "vo_rag_books_raw_v1"
            return [
                SimpleNamespace(id="p1", payload={"chunk_id": "c1"}),
                SimpleNamespace(id="p2", payload={}),
            ]

    pipeline = _pipeline(retriever=_FakeRetriever())
    out = await pipeline._retrieve_fn("q", 10)
    assert out == ["c1", "p2"]


# ── visual step (VOR-53) ────────────────────────────────────────────────────


def _parsed_doc_with_images(*, image_ref: str = IMG_A_REF) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        source_id="src-1",
        title="Wing Aerodynamics",
        author="Author",
        total_pages=2,
        elements=[
            ParsedElement(
                element_id="e-1",
                element_type=ElementType.NARRATIVE_TEXT,
                text="Preceding prose.",
                page_number=1,
            ),
            ParsedElement(
                element_id="e-2",
                element_type=ElementType.IMAGE,
                text="",
                page_number=1,
                metadata={"image_id": "img-aaaa", "image_ref": image_ref},
            ),
            ParsedElement(
                element_id="e-3",
                element_type=ElementType.NARRATIVE_TEXT,
                text="Following prose.",
                page_number=1,
            ),
        ],
        images=[
            ExtractedImage(
                image_id="img-aaaa",
                page_number=1,
                image_index=0,
                ref=image_ref,
                element_index=1,
            ),
        ],
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="pymupdf",
    )


@pytest.mark.asyncio
async def test_visual_skipped_when_disabled() -> None:
    pipeline = _pipeline(config=BookRagConfig(enable_visual_extraction=False))
    ctx = BookRagContext(envelope=_envelope(), config=pipeline.config)
    ctx.parsed = _parsed_doc_with_images()

    await pipeline._visual(ctx)
    assert ctx.composites == []


@pytest.mark.asyncio
async def test_visual_extraction_produces_composites_and_image_chunks() -> None:
    """With enable_visual_extraction=True (description disabled), the visual
    step must build CompositeChunks that the chunker turns into RetrievalChunk
    records with has_image=True and image_ref set."""

    pipeline = _pipeline(config=BookRagConfig(enable_visual_extraction=True))
    ctx = BookRagContext(envelope=_envelope(), config=pipeline.config)
    ctx.parsed = _parsed_doc_with_images()

    await pipeline._visual(ctx)

    assert len(ctx.composites) == 1
    assert ctx.composites[0].image_id == "img-aaaa"
    assert ctx.composites[0].image_ref == IMG_A_REF
    assert ctx.composites[0].vlm_description == ""

    # Feed composites into the real chunker
    ctx.chunks = do_build_chunks(ctx.parsed, composites=ctx.composites)
    image_chunks = [c for c in ctx.chunks if c.has_image]
    assert image_chunks, "expected at least one chunk with has_image=True"
    assert all(c.image_ref == IMG_A_REF for c in image_chunks)


@pytest.mark.asyncio
async def test_visual_description_raises_without_vlm_service() -> None:
    """When enable_visual_description=True but no vlm_service is injected,
    the visual step must raise RuntimeError loudly."""

    pipeline = _pipeline(
        config=BookRagConfig(
            enable_visual_extraction=True,
            enable_visual_description=True,
        )
    )
    ctx = BookRagContext(envelope=_envelope(), config=pipeline.config)
    ctx.parsed = _parsed_doc_with_images()
    ctx.parse_meta = {"parsed_garage_key": "parsed/doc-1/parsed.json"}

    with pytest.raises(RuntimeError, match="vlm_service"):
        await pipeline._visual(ctx)


class _FakeGarage:
    """In-memory garage for workflow visual-description tests."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self._objects: dict[str, bytes] = dict(objects or {})

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        return self._objects[key]

    async def put_bytes(
        self, bucket: str, key: str, data: bytes, content_type: str | None = None
    ) -> None:
        self._objects[key] = data

    async def list_objects(self, bucket: str, prefix: str = "") -> list[dict[str, Any]]:
        return [{"key": k} for k in sorted(self._objects) if k.startswith(prefix)]

    async def delete_object(self, bucket: str, key: str) -> None:
        self._objects.pop(key, None)


class _FakeVLMService:
    """Duck-typed VLM service for workflow tests."""

    def __init__(self, *, model: str = "qwen2-vl-7b") -> None:
        self.model = model
        self.calls: list[tuple[bytes, str, str, int]] = []
        self.aclose_called = False

    async def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        image_id: str,
        *,
        max_tokens: int = 1024,
    ) -> Any:
        from vo_rag.pipelines.book_rag.visual_sidecar import VLMDescription

        self.calls.append((image_bytes, prompt, image_id, max_tokens))
        return VLMDescription(
            description=f"diagram showing {image_id}",
            model_used=self.model,
            token_count=12,
            duration_ms=100,
            image_id=image_id,
        )

    async def aclose(self) -> None:
        self.aclose_called = True


@pytest.mark.asyncio
async def test_visual_description_populates_vlm_description(tmp_path: Path) -> None:
    """With both extraction and description enabled, the visual step must
    produce CompositeChunks whose vlm_description is non-empty."""

    figures = FigureStore(tmp_path)
    ref = figures.write(SRC, 1, 0, b"PNG-A", ext="png")
    parsed = _parsed_doc_with_images(image_ref=ref)
    garage = _FakeGarage(
        {
            "parsed/doc-1/parsed.json": parsed.model_dump_json().encode("utf-8"),
        }
    )
    vlm = _FakeVLMService()
    pipeline = _pipeline(
        garage=garage,
        config=BookRagConfig(
            enable_visual_extraction=True,
            enable_visual_description=True,
        ),
        vlm_service=vlm,
        figures=figures,
    )
    ctx = BookRagContext(envelope=_envelope(), config=pipeline.config)
    ctx.parsed = parsed
    ctx.parse_meta = {"parsed_garage_key": "parsed/doc-1/parsed.json"}

    await pipeline._visual(ctx)

    assert len(ctx.composites) == 1
    assert ctx.composites[0].vlm_description == "diagram showing img-aaaa"
    assert ctx.composites[0].image_ref == ref

    # Chunker must emit has_image=True chunks
    ctx.chunks = do_build_chunks(ctx.parsed, composites=ctx.composites)
    image_chunks = [c for c in ctx.chunks if c.has_image]
    assert image_chunks
    assert all(c.vlm_description == "diagram showing img-aaaa" for c in image_chunks)


# ── per-envelope ingest metrics (VOR-1434) ───────────────────────────


def _metrics_writes(pool: _FakeQualityPool) -> list[tuple[Any, ...]]:
    from vo_rag.core.observability.metrics import ENVELOPE_METRICS_MERGE_SQL

    return [args for query, args in pool.executed if query == ENVELOPE_METRICS_MERGE_SQL]


def _stub_steps_with_parse(pipeline: BookRagPipeline, pages: int) -> None:
    _stub_all_steps(pipeline, [])

    async def _parse(ctx: BookRagContext) -> None:
        ctx.parsed = _parsed_doc()
        ctx.parsed.total_pages = pages

    pipeline._parse = _parse  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_ingest_persists_every_stage_time_and_the_page_count() -> None:
    pool = _FakeQualityPool()
    pipeline = _pipeline(qdrant_store=_quality_store([]), pg_pool=pool)
    _stub_steps_with_parse(pipeline, pages=412)
    env = _envelope()

    await pipeline.ingest(env)

    writes = _metrics_writes(pool)
    assert writes, "no metrics were persisted"
    assert {args[0] for args in writes} == {str(env.id)}
    stages: dict[str, Any] = {}
    for args in writes:
        stages.update(args[2])
    assert set(stages) == {
        "register_source",
        "parse",
        "visual",
        "chunk_embed",
        "sterilize",
        "advanced_understanding",
        "register_artifact",
        "validate",
        "retrieval_smoke",
    }
    assert all(entry["seconds"] >= 0 for entry in stages.values())
    assert writes[-1][1]["pages"] == 412


@pytest.mark.asyncio
async def test_ingest_persists_embed_counts_made_during_a_stage() -> None:
    from vo_rag.core.observability.metrics import record_embed_call

    pool = _FakeQualityPool()
    pipeline = _pipeline(qdrant_store=_quality_store([]), pg_pool=pool)
    _stub_all_steps(pipeline, [])

    async def _chunk_embed(ctx: BookRagContext) -> None:
        record_embed_call(retries_429=2)
        record_embed_call(retries_429=0)

    pipeline._chunk_embed = _chunk_embed  # type: ignore[method-assign]

    await pipeline.ingest(_envelope())

    last = _metrics_writes(pool)[-1]  # absolute attempt totals
    assert (last[3], last[4]) == (2, 2)


@pytest.mark.asyncio
async def test_a_failed_stage_still_persists_its_time() -> None:
    pool = _FakeQualityPool()
    pipeline = _pipeline(qdrant_store=_quality_store([]), pg_pool=pool)
    _stub_all_steps(pipeline, [])

    async def _boom(ctx: BookRagContext) -> None:
        raise RuntimeError("chunk_embed exploded")

    pipeline._chunk_embed = _boom  # type: ignore[method-assign]
    env = _envelope()

    with pytest.raises(RuntimeError, match="chunk_embed exploded"):
        await pipeline.ingest(env)

    stages: dict[str, Any] = {}
    for args in _metrics_writes(pool):
        stages.update(args[2])
    assert "chunk_embed" in stages
    assert env.status is EnvelopeStatus.FAILED


@pytest.mark.asyncio
async def test_a_metrics_write_failure_never_fails_the_book() -> None:
    from vo_rag.core.observability.metrics import ENVELOPE_METRICS_MERGE_SQL

    class _Pool(_FakeQualityPool):
        async def execute(self, query: str, *args: Any) -> str:
            if query == ENVELOPE_METRICS_MERGE_SQL:
                raise RuntimeError("metrics write failed")
            return await super().execute(query, *args)

    pool = _Pool()
    pipeline = _pipeline(qdrant_store=_quality_store([]), pg_pool=pool)
    _stub_all_steps(pipeline, [])
    env = _envelope()

    out = await pipeline.ingest(env)

    assert env.status is EnvelopeStatus.COMPLETED
    assert out.status is EnvelopeStatus.COMPLETED


@pytest.mark.asyncio
async def test_ingest_without_a_pool_records_nothing_and_completes() -> None:
    pipeline = _pipeline()
    _stub_all_steps(pipeline, [])
    env = _envelope()
    await pipeline.ingest(env)
    assert env.status is EnvelopeStatus.COMPLETED


@pytest.mark.asyncio
async def test_long_stage_counts_reach_postgres_before_the_stage_ends() -> None:
    """Periodic flush: EmbedBackpressure must not be blind for a whole stage."""
    import asyncio

    from vo_rag.core.observability.metrics import record_embed_call

    pool = _FakeQualityPool()
    pipeline = _pipeline(qdrant_store=_quality_store([]), pg_pool=pool)
    pipeline.metrics_flush_interval_s = 0.01
    _stub_all_steps(pipeline, [])
    seen_mid_stage: list[tuple[Any, ...]] = []

    async def _chunk_embed(ctx: BookRagContext) -> None:
        record_embed_call(retries_429=3)
        for _ in range(200):
            writes = [w for w in _metrics_writes(pool) if w[3]]
            if writes:
                seen_mid_stage.extend(writes)
                return
            await asyncio.sleep(0.005)

    pipeline._chunk_embed = _chunk_embed  # type: ignore[method-assign]
    await pipeline.ingest(_envelope())

    assert seen_mid_stage, "no flush happened while the stage was running"
    assert seen_mid_stage[0][3:5] == (1, 3)
    assert not [t for t in asyncio.all_tasks() if "ingest-metrics-flush" in t.get_name()]


@pytest.mark.asyncio
async def test_the_running_stage_is_persisted() -> None:
    import asyncio

    pool = _FakeQualityPool()
    pipeline = _pipeline(qdrant_store=_quality_store([]), pg_pool=pool)
    pipeline.metrics_flush_interval_s = 0.01
    _stub_all_steps(pipeline, [])
    stages_seen: list[str] = []

    async def _visual(ctx: BookRagContext) -> None:
        for _ in range(200):
            stages_seen.extend(
                w[1].get("current_stage")
                for w in _metrics_writes(pool)
                if w[1].get("current_stage")
            )
            if "visual" in stages_seen:
                return
            await asyncio.sleep(0.005)

    pipeline._visual = _visual  # type: ignore[method-assign]
    await pipeline.ingest(_envelope())
    assert "visual" in stages_seen


@pytest.mark.asyncio
async def test_the_flush_task_is_stopped_when_the_book_fails() -> None:
    import asyncio

    pool = _FakeQualityPool()
    pipeline = _pipeline(qdrant_store=_quality_store([]), pg_pool=pool)
    pipeline.metrics_flush_interval_s = 0.01
    _stub_all_steps(pipeline, [])

    async def _boom(ctx: BookRagContext) -> None:
        raise RuntimeError("parse exploded")

    pipeline._parse = _boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="parse exploded"):
        await pipeline.ingest(_envelope())
    assert not [t for t in asyncio.all_tasks() if "ingest-metrics-flush" in t.get_name()]
