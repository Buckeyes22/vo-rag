"""Streaming contract tests for the LlamaIndex ``BookRagWorkflow`` ingest path.

These tests run only when the optional ``llamaindex`` extra is present; the
shared ``conftest.py`` drops this module in the light unit job.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.models.envelope import (
    EnvelopeStatus,
    IntakeEnvelope,
    IntakeSource,
)
from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement
from vo_rag.core.storage.qdrant import deterministic_point_id
from vo_rag.pipelines.book_rag import retrieval_smoke as smoke
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.events import (
    INGEST_STEP_ORDER,
    ImageDescribedEvent,
    IngestProgress,
)
from vo_rag.pipelines.book_rag.source_identity import canonical_source_id
from vo_rag.pipelines.book_rag.validator import ValidationReport
from vo_rag.pipelines.book_rag.workflow import PipelineOutput
from vo_rag.pipelines.book_rag.workflow_li import BookRagWorkflow

_TS = datetime(2026, 1, 1, tzinfo=UTC)
_HASH = hashlib.sha256(b"plain text fixture").hexdigest()
_VERIFICATION = {"status": "verified", "sha256": _HASH, "size_bytes": len(b"plain text fixture")}


def _envelope(*, status: EnvelopeStatus = EnvelopeStatus.RECEIVED) -> IntakeEnvelope:
    return IntakeEnvelope(
        id="01KT6XE8M1MW42M0Y0JJKRZSVP",  # pragma: allowlist secret
        source_type=IntakeSource.FILE_DROP,
        raw_content={
            "garage_key": "drops/book.txt",
            "file_hash": _HASH,
            "object_verification": _VERIFICATION,
        },
        status=status,
        created_at=_TS,
        updated_at=_TS,
    )


def _parsed_doc(*, source_id: str = "src-1") -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        source_id=source_id,
        title="Parsed Book Title",
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
                text="Narrative text for the streaming contract test.",
                page_number=1,
            ),
        ],
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={"quality_warnings": [{"stage": "parse", "code": "source_quality_fixture"}]},
        parser_used="test",
    )


class _FakeGarage:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        return self.objects[key]

    async def put_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        self.objects[key] = data


class _FakeParserRouter:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def parse(self, path: str, file_hash: str) -> ParsedDocument:
        self.order.append("parse")
        assert len(file_hash) == 64
        return _parsed_doc()


class _FakeEmbeddingClient:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, Any]:
        self.order.append("chunk_embed")
        dense = [0.1] * 1024
        return {
            "dense": [dense for _ in texts],
            "sparse": [{1: 0.5} for _ in texts],
            "colbert": [[] for _ in texts],
        }


class _FakeQdrantStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[Any]]] = []

    async def upsert_vectors(self, collection: str, points: list[Any]) -> None:
        self.upserts.append((collection, points))

    async def count_source(self, _collection: str, _source_id: str) -> int:
        return 0


class _FakeLLMClient:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        extra: dict[str, Any],
    ) -> Any:
        self.order.append("llm_call")
        from vo_rag.core.llm.response import LLMResponse

        return LLMResponse(text='{"ok": true}', model="fake-llm")


class _FakeRetriever:
    def __init__(self, expected_chunk_id: str) -> None:
        self.expected_chunk_id = expected_chunk_id

    async def hybrid_search(
        self,
        query: str,
        *,
        collection: str,
        top_k: int,
        prefetch_depth: int,
    ) -> list[SimpleNamespace]:
        assert collection == "vo_rag_books_raw_v1"
        return [SimpleNamespace(id="point-1", payload={"chunk_id": self.expected_chunk_id})]


def _workflow(**overrides: Any) -> BookRagWorkflow:
    order: list[str] = overrides.pop("order")
    base: dict[str, Any] = {
        "garage": _FakeGarage({"drops/book.txt": b"plain text fixture"}),
        "qdrant_store": _FakeQdrantStore(),
        "embedding_client": _FakeEmbeddingClient(order),
        "llm_client": _FakeLLMClient(order),
        "retriever": _FakeRetriever("pending"),
        "parser_router": _FakeParserRouter(order),
        "config": BookRagConfig(),
        "timeout": 30,
    }
    base.update(overrides)
    return BookRagWorkflow(**base)


def _patch_step_fakes(
    monkeypatch: pytest.MonkeyPatch, order: list[str], expected_raw_id: str
) -> _FakeRetriever:
    """Fakes for every seam a real ``BookRagWorkflow`` run touches."""
    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod

    holder: dict[str, _FakeRetriever] = {"retriever": _FakeRetriever(expected_chunk_id="")}

    async def _fake_visual(
        parsed: ParsedDocument,
        *,
        document_metadata: dict[str, Any] | None = None,
    ) -> list[Any]:
        order.append("visual")
        return []

    class _FakeValidator:
        def __init__(
            self,
            qdrant: Any,
            pool: Any,
            garage: Any,
            retriever: Any,
            *,
            figures: Any = None,
            artifacts_bucket: str,
            sterilize_enabled: bool,
        ) -> None:
            self.sterilize_enabled = sterilize_enabled

        async def validate(self, *, document_id: str, source_id: str) -> Any:
            order.append("validate")
            return ValidationReport(
                document_id=document_id,
                source_id=source_id,
                passed=True,
            )

    async def _fake_insert_artifact(pool: Any, artifact: Any) -> None:
        order.append("register_artifact")
        assert artifact.qdrant_point_ids == [expected_raw_id]

    async def _fake_generate_self_queries(
        *,
        all_chunks: list[Any],
        n: int,
        llm_call: Any,
        seed: int,
    ) -> list[Any]:
        await llm_call("system", "user", 32, 0.0)
        return [
            SimpleNamespace(
                query_id="q-1",
                query_text="chapter query",
                expected_supporting_chunk_ids=[all_chunks[0].chunk_id],
            )
        ]

    async def _fake_score_self_retrieval(
        *,
        queries: list[Any],
        retrieve_fn: Any,
        top_k: int,
        source_id: str,
    ) -> smoke.SmokeReport:
        order.append("retrieval_smoke")
        holder["retriever"].expected_chunk_id = queries[0].expected_supporting_chunk_ids[0]
        retrieved = await retrieve_fn(queries[0].query_text, top_k)
        assert retrieved == [queries[0].expected_supporting_chunk_ids[0]]
        return smoke.SmokeReport(
            source_id=source_id,
            n_requested=1,
            n_scored=1,
            top_k=top_k,
            score=1.0,
        )

    monkeypatch.setattr(workflow_mod, "build_extraction_composites", _fake_visual)
    monkeypatch.setattr(workflow_mod, "IngestionValidator", _FakeValidator)
    monkeypatch.setattr(workflow_mod, "insert_artifact", _fake_insert_artifact)
    monkeypatch.setattr(smoke, "generate_self_queries", _fake_generate_self_queries)
    monkeypatch.setattr(smoke, "score_self_retrieval", _fake_score_self_retrieval)

    return holder["retriever"]


@pytest.mark.asyncio
async def test_stream_events_one_per_step_and_terminal_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    source_id = canonical_source_id(_HASH)
    expected_raw_id = deterministic_point_id(source_id, 0, "raw")

    retriever = _patch_step_fakes(monkeypatch, order, expected_raw_id)
    workflow = _workflow(order=order, retriever=retriever, pg_pool=object())
    envelope = _envelope()

    handler = workflow.run(envelope=envelope)
    progress_steps: list[str] = []
    async for ev in handler.stream_events():
        if isinstance(ev, IngestProgress):
            progress_steps.append(ev.step)
        elif isinstance(ev, ImageDescribedEvent):
            progress_steps.append("image_described")

    result = await handler

    assert isinstance(result, PipelineOutput)
    assert result.status is EnvelopeStatus.COMPLETED
    assert progress_steps == (
        INGEST_STEP_ORDER[: INGEST_STEP_ORDER.index("visual") + 1]
        + ["image_described"]
        + INGEST_STEP_ORDER[INGEST_STEP_ORDER.index("visual") + 1 :]
    )


class _FakeMcpContext:
    def __init__(self) -> None:
        self.progress_calls: list[tuple[int, int, str]] = []

    async def report_progress(self, current: int, total: int, message: str) -> None:
        self.progress_calls.append((current, total, message))


@pytest.mark.asyncio
async def test_mcp_stream_proxy_threads_runner_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.mcp import tools as tools_mod
    from vo_rag.pipelines.book_rag import runner as runner_mod

    envelope = object()
    settings = object()
    pg_pool = object()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def _fake_runner_stream(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        yield IngestProgress(envelope_id="env-1", step="parse", detail="done")

    monkeypatch.setattr(runner_mod, "stream_book_rag_ingest", _fake_runner_stream)

    events = [
        ev
        async for ev in tools_mod.stream_book_rag_ingest(
            envelope,
            settings=settings,
            pg_pool=pg_pool,
        )
    ]

    assert len(events) == 1
    assert calls == [((envelope,), {"settings": settings, "pg_pool": pg_pool})]


@pytest.mark.asyncio
async def test_mcp_progress_consumer_reports_per_step_no_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.mcp import tools as tools_mod

    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZSVP"  # pragma: allowlist secret
    fake_envelope = SimpleNamespace(
        id=envelope_id,
        status=EnvelopeStatus.RECEIVED,
        source_type=IntakeSource.FILE_DROP,
    )

    async def _fake_stream(*args: Any, **kwargs: Any) -> Any:
        for step in INGEST_STEP_ORDER:
            yield IngestProgress(
                envelope_id=envelope_id,
                step=step,
                detail=f"{step}-detail",
            )
        yield PipelineOutput(
            source_id="src-1",
            status=EnvelopeStatus.COMPLETED,
            qdrant_point_ids=["p1"],
            raw_point_ids=["p1"],
            sterile_point_ids=[],
            smoke_decision=smoke.GateDecision.PASS,
            chunk_count=1,
            quality_warnings=[{"stage": "validation", "code": "audit_unavailable"}],
        )

    monkeypatch.setattr(tools_mod, "stream_book_rag_ingest", _fake_stream)
    updates: list[tuple[Any, str, EnvelopeStatus]] = []

    async def _fake_update_envelope_status(
        pool: Any,
        eid: str,
        status: EnvelopeStatus,
    ) -> None:
        updates.append((pool, eid, status))

    async def _fake_get_envelope(pool: Any, eid: str) -> Any:
        assert eid == envelope_id
        return fake_envelope

    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.get_envelope", _fake_get_envelope)
    monkeypatch.setattr(
        "vo_rag.core.db.queries.envelopes.update_envelope_status",
        _fake_update_envelope_status,
    )

    services = SimpleNamespace(
        db_pool=object(),
        settings=SimpleNamespace(),
    )

    ctx = _FakeMcpContext()
    result = await tools_mod.vo_rag_ingest_stream_impl(
        services=services,
        envelope_id=envelope_id,
        ctx=ctx,
    )

    assert result.envelope_id == envelope_id
    assert result.status == EnvelopeStatus.COMPLETED.value
    assert result.dispatch_status == "streamed"
    assert result.warnings == ["validation:audit_unavailable"]
    assert updates == [(services.db_pool, envelope_id, EnvelopeStatus.COMPLETED)]
    assert len(ctx.progress_calls) == len(INGEST_STEP_ORDER)
    for idx, (current, total, message) in enumerate(ctx.progress_calls):
        assert current == idx
        assert total == len(INGEST_STEP_ORDER)
        assert message.startswith(INGEST_STEP_ORDER[idx])


@pytest.mark.asyncio
async def test_mcp_progress_consumer_requires_terminal_pipeline_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.mcp import tools as tools_mod
    from vo_rag.mcp.errors import ToolError

    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZSVP"  # pragma: allowlist secret
    fake_envelope = SimpleNamespace(
        id=envelope_id,
        status=EnvelopeStatus.COMPLETED,
        source_type=IntakeSource.FILE_DROP,
    )

    async def _fake_stream(*args: Any, **kwargs: Any) -> Any:
        yield ImageDescribedEvent(envelope_id=envelope_id, image_index=0, total=1)

    monkeypatch.setattr(tools_mod, "stream_book_rag_ingest", _fake_stream)
    failures: list[tuple[Any, str, str]] = []

    async def _fake_fail_envelope(pool: Any, eid: str, error: str) -> None:
        failures.append((pool, eid, error))

    async def _fake_get_envelope(pool: Any, eid: str) -> Any:
        assert eid == envelope_id
        return fake_envelope

    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.get_envelope", _fake_get_envelope)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.fail_envelope", _fake_fail_envelope)

    services = SimpleNamespace(
        db_pool=object(),
        settings=SimpleNamespace(),
    )

    with pytest.raises(ToolError, match="ingest_stream_finished_without_output"):
        await tools_mod.vo_rag_ingest_stream_impl(
            services=services,
            envelope_id=envelope_id,
            ctx=_FakeMcpContext(),
        )
    assert failures == [(services.db_pool, envelope_id, "ingest_stream_finished_without_output")]


@pytest.mark.asyncio
async def test_mcp_progress_consumer_persists_stream_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.mcp import tools as tools_mod

    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZSVP"  # pragma: allowlist secret
    fake_envelope = SimpleNamespace(
        id=envelope_id,
        status=EnvelopeStatus.RECEIVED,
        source_type=IntakeSource.FILE_DROP,
    )
    failures: list[tuple[Any, str, str]] = []

    async def _fake_stream(*args: Any, **kwargs: Any) -> Any:
        yield IngestProgress(envelope_id=envelope_id, step="parse", detail="started")
        raise RuntimeError("pipeline failed")

    async def _fake_get_envelope(pool: Any, eid: str) -> Any:
        assert eid == envelope_id
        return fake_envelope

    async def _fake_fail_envelope(pool: Any, eid: str, error: str) -> None:
        failures.append((pool, eid, error))

    monkeypatch.setattr(tools_mod, "stream_book_rag_ingest", _fake_stream)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.get_envelope", _fake_get_envelope)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.fail_envelope", _fake_fail_envelope)

    services = SimpleNamespace(
        db_pool=object(),
        settings=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="pipeline failed"):
        await tools_mod.vo_rag_ingest_stream_impl(
            services=services,
            envelope_id=envelope_id,
            ctx=_FakeMcpContext(),
        )

    assert failures == [(services.db_pool, envelope_id, "pipeline failed")]


@pytest.mark.asyncio
async def test_mcp_progress_consumer_preserves_stream_failure_when_fail_persist_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.mcp import tools as tools_mod

    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZSVP"  # pragma: allowlist secret
    fake_envelope = SimpleNamespace(
        id=envelope_id,
        status=EnvelopeStatus.RECEIVED,
        source_type=IntakeSource.FILE_DROP,
    )

    async def _fake_stream(*args: Any, **kwargs: Any) -> Any:
        yield IngestProgress(envelope_id=envelope_id, step="parse", detail="started")
        raise RuntimeError("pipeline failed")

    async def _fake_get_envelope(pool: Any, eid: str) -> Any:
        assert eid == envelope_id
        return fake_envelope

    async def _fake_fail_envelope(pool: Any, eid: str, error: str) -> None:
        assert error == "pipeline failed"
        raise RuntimeError("persist failed")

    logs: list[tuple[str, dict[str, Any]]] = []

    class _FakeLog:
        def error(self, event: str, **kwargs: Any) -> None:
            logs.append((event, kwargs))

    monkeypatch.setattr(tools_mod, "stream_book_rag_ingest", _fake_stream)
    monkeypatch.setattr(tools_mod, "log", _FakeLog(), raising=False)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.get_envelope", _fake_get_envelope)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.fail_envelope", _fake_fail_envelope)

    services = SimpleNamespace(
        db_pool=object(),
        settings=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="pipeline failed"):
        await tools_mod.vo_rag_ingest_stream_impl(
            services=services,
            envelope_id=envelope_id,
            ctx=_FakeMcpContext(),
        )

    assert logs == [
        (
            "mcp_ingest_fail_persist_failed",
            {
                "envelope_id": envelope_id,
                "error": "persist failed",
                "original_error": "pipeline failed",
                "exc_info": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_mcp_progress_consumer_preserves_cancelled_error_when_fail_persist_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.mcp import tools as tools_mod

    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZSVP"  # pragma: allowlist secret
    fake_envelope = SimpleNamespace(
        id=envelope_id,
        status=EnvelopeStatus.RECEIVED,
        source_type=IntakeSource.FILE_DROP,
    )

    async def _fake_stream(*args: Any, **kwargs: Any) -> Any:
        yield IngestProgress(envelope_id=envelope_id, step="parse", detail="started")
        raise asyncio.CancelledError("workflow cancelled")

    async def _fake_get_envelope(pool: Any, eid: str) -> Any:
        assert eid == envelope_id
        return fake_envelope

    async def _fake_fail_envelope(pool: Any, eid: str, error: str) -> None:
        assert error == "workflow cancelled"
        raise RuntimeError("persist failed")

    logs: list[tuple[str, dict[str, Any]]] = []

    class _FakeLog:
        def error(self, event: str, **kwargs: Any) -> None:
            logs.append((event, kwargs))

    monkeypatch.setattr(tools_mod, "stream_book_rag_ingest", _fake_stream)
    monkeypatch.setattr(tools_mod, "log", _FakeLog(), raising=False)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.get_envelope", _fake_get_envelope)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.fail_envelope", _fake_fail_envelope)

    services = SimpleNamespace(
        db_pool=object(),
        settings=SimpleNamespace(),
    )

    with pytest.raises(asyncio.CancelledError, match="workflow cancelled"):
        await tools_mod.vo_rag_ingest_stream_impl(
            services=services,
            envelope_id=envelope_id,
            ctx=_FakeMcpContext(),
        )

    assert logs == [
        (
            "mcp_ingest_fail_persist_failed",
            {
                "envelope_id": envelope_id,
                "error": "persist failed",
                "original_error": "workflow cancelled",
                "exc_info": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_mcp_progress_consumer_persists_drained_terminal_status_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow_li as workflow_li_mod
    from vo_rag.mcp import tools as tools_mod

    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZSVP"  # pragma: allowlist secret
    fake_envelope = SimpleNamespace(
        id=envelope_id,
        status=EnvelopeStatus.RECEIVED,
        source_type=IntakeSource.FILE_DROP,
    )
    release = asyncio.Event()
    first_progress_seen = asyncio.Event()
    stream_block = asyncio.Event()
    updates: list[tuple[Any, str, EnvelopeStatus]] = []
    failures: list[tuple[Any, str, str]] = []

    async def _fake_build_surface_services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=SimpleNamespace(),
            retriever=object(),
            db_pool=db_pool,
        )

    class _FakeLLMClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _FakeHandler:
        async def stream_events(self) -> Any:
            yield IngestProgress(envelope_id=envelope_id, step="parse", detail="started")
            await stream_block.wait()

        def __await__(self) -> Any:
            async def _run() -> PipelineOutput:
                await release.wait()
                return PipelineOutput(
                    source_id="src-1",
                    status=EnvelopeStatus.COMPLETED,
                    qdrant_point_ids=["p1"],
                    raw_point_ids=["p1"],
                    sterile_point_ids=[],
                    smoke_decision=smoke.GateDecision.PASS,
                    chunk_count=1,
                )

            return _run().__await__()

    class _FakeWorkflow:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, *, envelope: Any) -> _FakeHandler:
            return _FakeHandler()

    class _Ctx:
        async def report_progress(self, current: int, total: int, message: str) -> None:
            first_progress_seen.set()

    async def _fake_get_envelope(pool: Any, eid: str) -> Any:
        assert eid == envelope_id
        return fake_envelope

    async def _fake_update_envelope_status(pool: Any, eid: str, status: EnvelopeStatus) -> None:
        updates.append((pool, eid, status))

    async def _fake_fail_envelope(pool: Any, eid: str, error: str) -> None:
        failures.append((pool, eid, error))

    monkeypatch.setattr(singletons, "build_surface_services", _fake_build_surface_services)
    monkeypatch.setattr(parser_factory, "build_parser_router", lambda *args, **kwargs: object())
    monkeypatch.setattr(llm_mod, "LLMClient", _FakeLLMClient)
    monkeypatch.setattr(workflow_li_mod, "BookRagWorkflow", _FakeWorkflow)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.get_envelope", _fake_get_envelope)
    monkeypatch.setattr(
        "vo_rag.core.db.queries.envelopes.update_envelope_status",
        _fake_update_envelope_status,
    )
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.fail_envelope", _fake_fail_envelope)

    services = SimpleNamespace(
        db_pool=object(),
        settings=SimpleNamespace(
            garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
            figures=SimpleNamespace(root=str(tmp_path)),
        ),
    )
    task = asyncio.create_task(
        tools_mod.vo_rag_ingest_stream_impl(
            services=services,
            envelope_id=envelope_id,
            ctx=_Ctx(),
        )
    )

    await asyncio.wait_for(first_progress_seen.wait(), timeout=1.0)
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert updates == [(services.db_pool, envelope_id, EnvelopeStatus.COMPLETED)]
    assert failures == []


@pytest.mark.asyncio
async def test_mcp_progress_consumer_drained_persist_failure_keeps_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DB hiccup on the drained-success write must not replace the
    CancelledError flow, and must be logged (SSE-sibling alignment)."""
    from vo_rag.mcp import tools as tools_mod

    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZSVP"  # pragma: allowlist secret
    fake_envelope = SimpleNamespace(
        id=envelope_id,
        status=EnvelopeStatus.RECEIVED,
        source_type=IntakeSource.FILE_DROP,
    )
    failures: list[tuple[Any, str, str]] = []
    logs: list[tuple[str, dict[str, Any]]] = []

    drained = PipelineOutput(
        source_id="src-1",
        status=EnvelopeStatus.COMPLETED,
        qdrant_point_ids=["p1"],
        raw_point_ids=["p1"],
        sterile_point_ids=[],
        smoke_decision=smoke.GateDecision.PASS,
        chunk_count=1,
    )

    async def _fake_stream(*args: Any, **kwargs: Any) -> Any:
        yield IngestProgress(envelope_id=envelope_id, step="parse", detail="started")
        exc = asyncio.CancelledError()
        exc.terminal_output = drained  # type: ignore[attr-defined]
        raise exc

    async def _fake_get_envelope(pool: Any, eid: str) -> Any:
        return fake_envelope

    async def _fake_update_envelope_status(pool: Any, eid: str, status: EnvelopeStatus) -> None:
        raise RuntimeError("persist failed")

    async def _fake_fail_envelope(pool: Any, eid: str, error: str) -> None:
        failures.append((pool, eid, error))

    class _FakeLog:
        def error(self, event: str, **kwargs: Any) -> None:
            logs.append((event, kwargs))

    monkeypatch.setattr(tools_mod, "stream_book_rag_ingest", _fake_stream)
    monkeypatch.setattr(tools_mod, "log", _FakeLog(), raising=False)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.get_envelope", _fake_get_envelope)
    monkeypatch.setattr(
        "vo_rag.core.db.queries.envelopes.update_envelope_status",
        _fake_update_envelope_status,
    )
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.fail_envelope", _fake_fail_envelope)

    services = SimpleNamespace(
        db_pool=object(),
        settings=SimpleNamespace(),
    )

    with pytest.raises(asyncio.CancelledError):
        await tools_mod.vo_rag_ingest_stream_impl(
            services=services,
            envelope_id=envelope_id,
            ctx=_FakeMcpContext(),
        )

    assert failures == []
    assert logs == [
        (
            "mcp_ingest_drained_persist_failed",
            {
                "envelope_id": envelope_id,
                "status": "completed",
                "error": "persist failed",
                "exc_info": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_mcp_progress_consumer_cancel_failure_uses_non_empty_error_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.mcp import tools as tools_mod

    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZSVP"  # pragma: allowlist secret
    fake_envelope = SimpleNamespace(
        id=envelope_id,
        status=EnvelopeStatus.RECEIVED,
        source_type=IntakeSource.FILE_DROP,
    )
    failures: list[tuple[Any, str, str]] = []

    async def _fake_stream(*args: Any, **kwargs: Any) -> Any:
        yield IngestProgress(envelope_id=envelope_id, step="parse", detail="started")
        raise asyncio.CancelledError()

    async def _fake_get_envelope(pool: Any, eid: str) -> Any:
        assert eid == envelope_id
        return fake_envelope

    async def _fake_fail_envelope(pool: Any, eid: str, error: str) -> None:
        failures.append((pool, eid, error))

    monkeypatch.setattr(tools_mod, "stream_book_rag_ingest", _fake_stream)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.get_envelope", _fake_get_envelope)
    monkeypatch.setattr("vo_rag.core.db.queries.envelopes.fail_envelope", _fake_fail_envelope)

    services = SimpleNamespace(
        db_pool=object(),
        settings=SimpleNamespace(),
    )

    with pytest.raises(asyncio.CancelledError):
        await tools_mod.vo_rag_ingest_stream_impl(
            services=services,
            envelope_id=envelope_id,
            ctx=_FakeMcpContext(),
        )

    assert failures == [(services.db_pool, envelope_id, "CancelledError")]


@pytest.mark.asyncio
async def test_streamed_events_are_json_serializable() -> None:
    progress = IngestProgress(
        envelope_id="env-1",
        step="parse",
        detail="docling",
        pct=1.0,
    )
    image = ImageDescribedEvent(
        envelope_id="env-1",
        image_index=2,
        total=5,
    )

    for ev in (progress, image):
        payload = ev.model_dump()
        serialized = json.dumps(payload)
        restored = json.loads(serialized)
        assert isinstance(restored, dict)
        assert restored["envelope_id"] == "env-1"


@pytest.mark.asyncio
async def test_cli_streamed_consumer_returns_terminal_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`vo-rag ingest run` streamed path (extra present) renders per-step
    progress and returns the terminal :class:`PipelineOutput`.

    Covers ``cli.ingest._run_book_rag_streamed``; the base-install fallback
    (``_run_book_rag_unstreamed``) is covered by the light-job CLI test.
    """
    from vo_rag.cli import ingest as cli_ingest_mod
    from vo_rag.pipelines.book_rag import runner as runner_mod

    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZSVP"  # pragma: allowlist secret

    async def _fake_stream(*args: Any, **kwargs: Any) -> Any:
        for step in INGEST_STEP_ORDER:
            yield IngestProgress(envelope_id=envelope_id, step=step, detail=f"{step}-detail")
        yield PipelineOutput(
            source_id="src-1",
            status=EnvelopeStatus.COMPLETED,
            qdrant_point_ids=["p1"],
            raw_point_ids=["p1"],
            sterile_point_ids=[],
            smoke_decision=smoke.GateDecision.PASS,
            chunk_count=1,
        )

    monkeypatch.setattr(runner_mod, "stream_book_rag_ingest", _fake_stream)

    terminal = await cli_ingest_mod._run_book_rag_streamed(
        SimpleNamespace(id=envelope_id),
        SimpleNamespace(),
        object(),
    )

    assert isinstance(terminal, PipelineOutput)
    assert terminal.source_id == "src-1"
    assert terminal.status is EnvelopeStatus.COMPLETED


class _MetricsPool:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        return "UPDATE 1"

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        return []

    async def fetchrow(self, query: str, *args: Any) -> Any:
        return None


@pytest.mark.asyncio
async def test_streamed_ingest_persists_every_stage_time_and_the_page_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VOR-1434: a real ``BookRagWorkflow`` run through the streaming
    runner records a time for every stage in INGEST_STEP_ORDER plus the parsed
    page count, so the canary stage gate and the over-budget alert see
    streamed books too."""
    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    from vo_rag.core.observability.metrics import ENVELOPE_METRICS_MERGE_SQL
    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod
    from vo_rag.pipelines.book_rag.runner import stream_book_rag_ingest

    order: list[str] = []
    source_id = canonical_source_id(_HASH)
    expected_raw_id = deterministic_point_id(source_id, 0, "raw")
    retriever = _patch_step_fakes(monkeypatch, order, expected_raw_id)
    pool = _MetricsPool()
    real_workflow = _workflow(order=order, retriever=retriever, pg_pool=pool)

    async def _services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=SimpleNamespace(),
            retriever=object(),
            db_pool=db_pool,
        )

    monkeypatch.setattr(singletons, "build_surface_services", _services)
    monkeypatch.setattr(parser_factory, "build_parser_router", lambda *_: object())
    monkeypatch.setattr(llm_mod, "LLMClient", lambda **_: SimpleNamespace())
    monkeypatch.setattr(workflow_mod, "BookRagWorkflow", lambda **_: real_workflow)

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
    )
    events = [
        ev
        async for ev in stream_book_rag_ingest(
            _envelope(),
            settings=settings,  # type: ignore[arg-type]
            pg_pool=pool,
        )
    ]
    assert isinstance(events[-1], PipelineOutput)
    assert events[-1].status is EnvelopeStatus.COMPLETED

    merges = [args for query, args in pool.executed if query == ENVELOPE_METRICS_MERGE_SQL]
    stages: dict[str, Any] = {}
    for args in merges:
        stages.update(args[2])
    assert set(INGEST_STEP_ORDER) <= set(stages), sorted(stages)
    assert all(entry["seconds"] >= 0 for entry in stages.values())
    assert merges[-1][1]["pages"] == 1
    assert merges[-1][1]["active_stream"] is False


async def _stream_real_workflow(monkeypatch: pytest.MonkeyPatch, pool: _MetricsPool) -> list[Any]:
    """Drive a real ``BookRagWorkflow`` through the streaming runner (the
    API/MCP/``vo-rag ingest run`` path), with the harness fakes."""
    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    from vo_rag.pipelines.book_rag import workflow_li as workflow_mod
    from vo_rag.pipelines.book_rag.runner import stream_book_rag_ingest

    order: list[str] = []
    source_id = canonical_source_id(_HASH)
    expected_raw_id = deterministic_point_id(source_id, 0, "raw")
    retriever = _patch_step_fakes(monkeypatch, order, expected_raw_id)
    real_workflow = _workflow(order=order, retriever=retriever, pg_pool=pool)

    async def _services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=SimpleNamespace(),
            retriever=object(),
            db_pool=db_pool,
        )

    monkeypatch.setattr(singletons, "build_surface_services", _services)
    monkeypatch.setattr(parser_factory, "build_parser_router", lambda *_: object())
    monkeypatch.setattr(llm_mod, "LLMClient", lambda **_: SimpleNamespace())
    monkeypatch.setattr(workflow_mod, "BookRagWorkflow", lambda **_: real_workflow)

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
    )
    return [
        ev
        async for ev in stream_book_rag_ingest(
            _envelope(),
            settings=settings,  # type: ignore[arg-type]
            pg_pool=pool,
        )
    ]


def _quality_writes(pool: _MetricsPool) -> list[tuple[Any, ...]]:
    return [args for query, args in pool.executed if "'{quality}'" in query]


@pytest.mark.asyncio
async def test_streamed_ingest_records_the_book_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VOR-1434: a streamed ingest (API/MCP/CLI) records
    ``metadata.quality`` through the same per-book gate as a dispatcher
    ingest, scoped to the book's own source and collection."""
    from vo_rag.pipelines.book_rag import workflow as native_workflow

    seen: dict[str, Any] = {}

    async def _fake_collect_stats(store: Any, **kwargs: Any) -> list[Any]:
        seen.update(kwargs)
        return ["book-stats"]

    quality = {"findings": [], "stats": {"source_id": canonical_source_id(_HASH), "chunks": 1}}
    monkeypatch.setattr(native_workflow, "collect_stats", _fake_collect_stats)
    monkeypatch.setattr(native_workflow, "book_quality_metadata", lambda stats: quality)

    pool = _MetricsPool()
    events = await _stream_real_workflow(monkeypatch, pool)

    assert isinstance(events[-1], PipelineOutput)
    assert events[-1].status is EnvelopeStatus.COMPLETED
    assert seen["source_ids"] == [canonical_source_id(_HASH)]
    assert seen["collection"] == "vo_rag_books_raw_v1"
    [(envelope_id, written)] = _quality_writes(pool)
    assert envelope_id == str(_envelope().id)
    assert written == quality


@pytest.mark.asyncio
async def test_streamed_ingest_completes_when_the_quality_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import workflow as native_workflow

    async def _broken_collect_stats(store: Any, **kwargs: Any) -> list[Any]:
        raise RuntimeError("qdrant scroll exploded")

    monkeypatch.setattr(native_workflow, "collect_stats", _broken_collect_stats)

    pool = _MetricsPool()
    events = await _stream_real_workflow(monkeypatch, pool)

    assert isinstance(events[-1], PipelineOutput)
    assert events[-1].status is EnvelopeStatus.COMPLETED
    assert _quality_writes(pool) == []
    assert any(w["code"] == "source_quality_fixture" for w in events[-1].quality_warnings)
    assert any(w["code"] == "audit_unavailable" for w in events[-1].quality_warnings)
