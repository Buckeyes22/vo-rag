from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from vo_rag.core.idempotency import deterministic_uuid
from vo_rag.core.ingest.source_publication import PublicationState, SourcePublicationJob
from vo_rag.core.models.envelope import EnvelopeStatus, IntakeEnvelope, IntakeSource
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.workflow import BookRagContext, BookRagPipeline

_SHA256 = "a" * 64
_TS = datetime(2026, 1, 1, tzinfo=UTC)


def _envelope() -> IntakeEnvelope:
    return IntakeEnvelope(
        id="01KT6XE8M1MW42M0Y0JJKRZSVP",  # pragma: allowlist secret
        source_type=IntakeSource.FILE_DROP,
        raw_content={
            "garage_key": "drops/book.pdf",
            "file_hash": _SHA256,
            "object_verification": {
                "status": "verified",
                "sha256": _SHA256,
                "size_bytes": 17,
            },
        },
        status=EnvelopeStatus.RECEIVED,
        created_at=_TS,
        updated_at=_TS,
    )


class _AdmissionStore:
    source_publication_enabled = True

    def __init__(self) -> None:
        self.events: list[str] = []
        self.admission: tuple[str, str, str, str] | None = None
        self.aborted: list[tuple[object, str]] = []
        self.job: object = object()

    async def count_source(self, _collection: str, _source_id: str) -> int:
        return 0

    async def admit_source_revision(
        self, collection: str, source_id: str, job_id: str, source_sha256: str
    ) -> SourcePublicationJob:
        self.events.append("admit")
        self.admission = (collection, source_id, job_id, source_sha256)
        job = SourcePublicationJob(
            job_id=job_id,
            corpus="books",
            collection=collection,
            source_id=source_id,
            revision=1,
            expected_active_revision=None,
            source_sha256=source_sha256,
            state=PublicationState.PREPARING,
            checkpoint={},
            retention_disposition=None,
        )
        self.job = job
        return job

    async def abort_source_revision(self, job: object, reason: str) -> None:
        self.aborted.append((job, reason))


def _pipeline(store: _AdmissionStore) -> BookRagPipeline:
    return BookRagPipeline(
        garage=SimpleNamespace(),
        qdrant_store=store,
        embedding_client=SimpleNamespace(),
        llm_client=SimpleNamespace(),
        retriever=SimpleNamespace(),
        parser_router=SimpleNamespace(),
        figures=SimpleNamespace(),
        config=BookRagConfig(),
    )


@pytest.mark.asyncio
async def test_normal_admission_precedes_parse_and_binds_verified_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _AdmissionStore()
    pipeline = _pipeline(store)

    async def _not_duplicate(_source_id: str) -> bool:
        return False

    monkeypatch.setattr(pipeline, "_source_already_ingested", _not_duplicate)

    async def _parse(ctx: BookRagContext) -> None:
        store.events.append("parse")
        assert ctx.publication_job is store.job

    monkeypatch.setattr(pipeline, "_parse", _parse)
    for name in (
        "_visual",
        "_chunk_embed",
        "_sterilize",
        "_advanced_understanding",
        "_register_artifact",
        "_validate",
        "_retrieval_smoke",
    ):
        setattr(pipeline, name, _noop)

    await pipeline.ingest(_envelope())

    assert store.events.index("admit") < store.events.index("parse")
    assert store.admission is not None
    collection, source_id, job_id, sha = store.admission
    assert collection == "vo_rag_books_raw_v1"
    assert job_id == deterministic_uuid(
        "normal-source-publication-v1",
        "01KT6XE8M1MW42M0Y0JJKRZSVP",
        collection,
        source_id,
    )
    assert sha == _SHA256
    assert store.aborted == []


@pytest.mark.asyncio
async def test_prewrite_parse_failure_aborts_admitted_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _AdmissionStore()
    pipeline = _pipeline(store)

    async def _not_duplicate(_source_id: str) -> bool:
        return False

    async def _parse(_ctx: BookRagContext) -> None:
        raise RuntimeError("parser failed before Qdrant write")

    monkeypatch.setattr(pipeline, "_source_already_ingested", _not_duplicate)
    monkeypatch.setattr(pipeline, "_parse", _parse)
    for name in (
        "_visual",
        "_chunk_embed",
        "_sterilize",
        "_advanced_understanding",
        "_register_artifact",
        "_validate",
        "_retrieval_smoke",
    ):
        setattr(pipeline, name, _noop)

    with pytest.raises(RuntimeError, match="parser failed"):
        await pipeline.ingest(_envelope())

    assert store.aborted == [(store.job, "parser failed before Qdrant write")]


@pytest.mark.asyncio
async def test_llamaindex_registration_checkpoints_admission_after_duplicate_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("llama_index")
    from vo_rag.pipelines.book_rag import workflow_li as workflow_li_mod
    from vo_rag.pipelines.book_rag.workflow_li import BookRagWorkflow, IngestStartEvent

    monkeypatch.setattr(workflow_li_mod, "_ingest_generations_enabled", lambda: False)
    store = _AdmissionStore()
    workflow = BookRagWorkflow(
        garage=SimpleNamespace(),
        qdrant_store=store,
        embedding_client=SimpleNamespace(),
        llm_client=SimpleNamespace(),
        retriever=SimpleNamespace(),
        parser_router=SimpleNamespace(),
        config=BookRagConfig(),
    )

    class Coordinator:
        async def _source_already_ingested(self, _source_id: str) -> bool:
            store.events.append("duplicate_check")
            return False

        async def _admit_source_publication(
            self, ctx: BookRagContext, collection: str
        ) -> SourcePublicationJob:
            return await store.admit_source_revision(
                collection,
                ctx.source_id,
                deterministic_uuid(
                    "normal-source-publication-v1",
                    str(ctx.envelope.id),
                    collection,
                    ctx.source_id,
                ),
                _SHA256,
            )

    monkeypatch.setattr(workflow, "_generation_coordinator", lambda: Coordinator())
    ctx = workflow_li_mod._workflow_context_for_run(workflow, "normal-admission-test")

    event = await workflow.register_source(ctx, IngestStartEvent(envelope=_envelope()))

    assert isinstance(event, workflow_li_mod.RegisteredEvent)
    assert store.events == ["duplicate_check", "admit"]
    assert await ctx.store.get("publication_job") == store.job
    assert await ctx.store.get("source_id")


async def _noop(_ctx: BookRagContext) -> None:
    return None
