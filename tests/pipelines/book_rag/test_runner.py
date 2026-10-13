"""Production book_rag ingest runner wiring."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from vo_rag.core.models.envelope import EnvelopeStatus, IntakeEnvelope, IntakeSource
from vo_rag.pipelines.book_rag.events import IngestProgress
from vo_rag.pipelines.book_rag.workflow import PipelineOutput

_TS = datetime(2026, 1, 1, tzinfo=UTC)


def _envelope() -> IntakeEnvelope:
    return IntakeEnvelope(
        id="01KT6XE8M1MW42M0Y0JJKRZSVP",  # pragma: allowlist secret
        source_type=IntakeSource.FILE_DROP,
        raw_content={"garage_key": "drops/book.pdf", "file_hash": "a" * 64},
        status=EnvelopeStatus.RECEIVED,
        created_at=_TS,
        updated_at=_TS,
    )


class _FakeLLMClient:
    def __init__(self, *, provider: str, role: str) -> None:
        self.provider = provider
        self.role = role
        self.aclose_called = False

    async def aclose(self) -> None:
        self.aclose_called = True


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.aclose_called = False

    async def aclose(self) -> None:
        self.aclose_called = True


@pytest.mark.asyncio
async def test_run_book_rag_ingest_constructs_pipeline_with_parser_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow as workflow_mod
    from vo_rag.pipelines.book_rag.runner import run_book_rag_ingest

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
    )
    pg_pool = object()
    garage = object()
    qdrant = object()
    embedding_client = _FakeEmbeddingClient()
    retriever = object()
    parser_router = object()
    built: dict[str, Any] = {}
    expected = PipelineOutput(
        source_id="source-1",
        status=EnvelopeStatus.COMPLETED,
        qdrant_point_ids=["p1"],
        raw_point_ids=["p1"],
        sterile_point_ids=[],
        smoke_decision=None,
        chunk_count=1,
    )

    async def _fake_build_surface_services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        built["settings"] = settings_arg
        built["db_pool"] = db_pool
        return SimpleNamespace(
            garage=garage,
            qdrant=qdrant,
            embedding_client=embedding_client,
            retriever=retriever,
            db_pool=db_pool,
        )

    def _fake_build_parser_router(settings_arg: Any, figures_arg: Any) -> object:
        built["parser_router_settings"] = settings_arg
        built["parser_router_figures"] = figures_arg
        return parser_router

    class _FakePipeline:
        def __init__(self, **kwargs: Any) -> None:
            built["pipeline_kwargs"] = kwargs

        async def ingest(self, envelope: IntakeEnvelope) -> PipelineOutput:
            built["envelope"] = envelope
            return expected

    monkeypatch.setattr(singletons, "build_surface_services", _fake_build_surface_services)
    monkeypatch.setattr(parser_factory, "build_parser_router", _fake_build_parser_router)
    monkeypatch.setattr(llm_mod, "LLMClient", _FakeLLMClient)
    monkeypatch.setattr(workflow_mod, "BookRagPipeline", _FakePipeline)
    monkeypatch.setenv("VORAG_RAW_COLLECTION", "vo_rag_books_raw_v1__ab_semantic")

    envelope = _envelope()
    out = await run_book_rag_ingest(
        envelope,
        settings=settings,  # type: ignore[arg-type]
        pg_pool=pg_pool,
    )

    assert out is expected
    assert built["settings"] is settings
    assert built["db_pool"] is pg_pool
    assert built["parser_router_settings"] is settings
    assert str(built["parser_router_figures"].root) == "/tmp/vo-rag-test-figures"
    assert built["envelope"] is envelope
    assert built["pipeline_kwargs"]["garage"] is garage
    assert built["pipeline_kwargs"]["qdrant_store"] is qdrant
    assert built["pipeline_kwargs"]["embedding_client"] is embedding_client
    assert built["pipeline_kwargs"]["retriever"] is retriever
    assert built["pipeline_kwargs"]["parser_router"] is parser_router
    assert built["pipeline_kwargs"]["pool"] is pg_pool
    assert built["pipeline_kwargs"]["pg_pool"] is pg_pool
    assert built["pipeline_kwargs"]["settings"] is settings
    assert built["pipeline_kwargs"]["raw_collection"] == "vo_rag_books_raw_v1__ab_semantic"
    assert built["pipeline_kwargs"]["llm_client"].provider == "runpod_serverless"
    assert built["pipeline_kwargs"]["llm_client"].role == "text"
    assert built["pipeline_kwargs"]["llm_client"].aclose_called is True
    assert embedding_client.aclose_called is True


@pytest.mark.asyncio
async def test_stream_book_rag_ingest_waits_for_workflow_before_cleanup_on_consumer_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("llama_index")

    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow_li as workflow_li_mod
    from vo_rag.pipelines.book_rag.runner import stream_book_rag_ingest

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
    )
    release = asyncio.Event()
    completed = asyncio.Event()
    embedding_client = _FakeEmbeddingClient()
    llm_instances: list[_FakeLLMClient] = []

    async def _fake_build_surface_services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=embedding_client,
            retriever=object(),
            db_pool=db_pool,
        )

    def _fake_build_parser_router(settings_arg: Any, figures_arg: Any) -> object:
        _ = (settings_arg, figures_arg)
        return object()

    class _RecordingLLMClient(_FakeLLMClient):
        def __init__(self, *, provider: str, role: str) -> None:
            super().__init__(provider=provider, role=role)
            llm_instances.append(self)

    class _FakeHandler:
        async def stream_events(self) -> Any:
            yield IngestProgress(envelope_id=str(_envelope().id), step="parse", detail="started")

        def __await__(self) -> Any:
            async def _run() -> PipelineOutput:
                await release.wait()
                completed.set()
                return PipelineOutput(
                    source_id="source-1",
                    status=EnvelopeStatus.COMPLETED,
                    qdrant_point_ids=[],
                    raw_point_ids=[],
                    sterile_point_ids=[],
                    smoke_decision=None,
                    chunk_count=0,
                )

            return _run().__await__()

    class _FakeWorkflow:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, *, envelope: IntakeEnvelope) -> _FakeHandler:
            return _FakeHandler()

    monkeypatch.setattr(singletons, "build_surface_services", _fake_build_surface_services)
    monkeypatch.setattr(parser_factory, "build_parser_router", _fake_build_parser_router)
    monkeypatch.setattr(llm_mod, "LLMClient", _RecordingLLMClient)
    monkeypatch.setattr(workflow_li_mod, "BookRagWorkflow", _FakeWorkflow)

    stream = stream_book_rag_ingest(_envelope(), settings=settings, pg_pool=object())  # type: ignore[arg-type]
    first = await anext(stream)
    assert isinstance(first, IngestProgress)

    close_task = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0)

    assert embedding_client.aclose_called is False
    assert llm_instances[0].aclose_called is False

    release.set()
    await asyncio.wait_for(close_task, timeout=1.0)

    assert completed.is_set()
    assert embedding_client.aclose_called is True
    assert llm_instances[0].aclose_called is True


@pytest.mark.asyncio
async def test_stream_book_rag_ingest_drains_and_closes_on_consumer_task_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("llama_index")

    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow_li as workflow_li_mod
    from vo_rag.pipelines.book_rag.runner import stream_book_rag_ingest

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
    )
    release = asyncio.Event()
    stream_block = asyncio.Event()
    first_seen = asyncio.Event()
    completed = asyncio.Event()
    embedding_client = _FakeEmbeddingClient()
    llm_instances: list[_FakeLLMClient] = []

    async def _fake_build_surface_services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=embedding_client,
            retriever=object(),
            db_pool=db_pool,
        )

    def _fake_build_parser_router(settings_arg: Any, figures_arg: Any) -> object:
        _ = (settings_arg, figures_arg)
        return object()

    class _RecordingLLMClient(_FakeLLMClient):
        def __init__(self, *, provider: str, role: str) -> None:
            super().__init__(provider=provider, role=role)
            llm_instances.append(self)

    class _FakeHandler:
        async def stream_events(self) -> Any:
            yield IngestProgress(envelope_id=str(_envelope().id), step="parse", detail="started")
            await stream_block.wait()

        def __await__(self) -> Any:
            async def _run() -> PipelineOutput:
                await release.wait()
                completed.set()
                return PipelineOutput(
                    source_id="source-1",
                    status=EnvelopeStatus.COMPLETED,
                    qdrant_point_ids=[],
                    raw_point_ids=[],
                    sterile_point_ids=[],
                    smoke_decision=None,
                    chunk_count=0,
                )

            return _run().__await__()

    class _FakeWorkflow:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, *, envelope: IntakeEnvelope) -> _FakeHandler:
            return _FakeHandler()

    monkeypatch.setattr(singletons, "build_surface_services", _fake_build_surface_services)
    monkeypatch.setattr(parser_factory, "build_parser_router", _fake_build_parser_router)
    monkeypatch.setattr(llm_mod, "LLMClient", _RecordingLLMClient)
    monkeypatch.setattr(workflow_li_mod, "BookRagWorkflow", _FakeWorkflow)

    async def _consume() -> None:
        async for ev in stream_book_rag_ingest(
            _envelope(),
            settings=settings,  # type: ignore[arg-type]
            pg_pool=object(),
        ):
            assert isinstance(ev, IngestProgress)
            first_seen.set()

    task = asyncio.create_task(_consume())
    await asyncio.wait_for(first_seen.wait(), timeout=1.0)
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert completed.is_set()
    assert embedding_client.aclose_called is True
    assert llm_instances[0].aclose_called is True


@pytest.mark.asyncio
async def test_stream_book_rag_ingest_cleanup_does_not_loop_on_cancelled_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("llama_index")

    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow_li as workflow_li_mod
    from vo_rag.pipelines.book_rag.runner import stream_book_rag_ingest

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
    )
    embedding_client = _FakeEmbeddingClient()
    llm_instances: list[_FakeLLMClient] = []

    async def _fake_build_surface_services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=embedding_client,
            retriever=object(),
            db_pool=db_pool,
        )

    def _fake_build_parser_router(settings_arg: Any, figures_arg: Any) -> object:
        _ = (settings_arg, figures_arg)
        return object()

    class _RecordingLLMClient(_FakeLLMClient):
        def __init__(self, *, provider: str, role: str) -> None:
            super().__init__(provider=provider, role=role)
            llm_instances.append(self)

    class _CancelledHandler:
        async def stream_events(self) -> Any:
            yield IngestProgress(envelope_id=str(_envelope().id), step="parse", detail="started")

        def __await__(self) -> Any:
            async def _run() -> PipelineOutput:
                raise asyncio.CancelledError

            return _run().__await__()

    class _FakeWorkflow:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, *, envelope: IntakeEnvelope) -> _CancelledHandler:
            return _CancelledHandler()

    monkeypatch.setattr(singletons, "build_surface_services", _fake_build_surface_services)
    monkeypatch.setattr(parser_factory, "build_parser_router", _fake_build_parser_router)
    monkeypatch.setattr(llm_mod, "LLMClient", _RecordingLLMClient)
    monkeypatch.setattr(workflow_li_mod, "BookRagWorkflow", _FakeWorkflow)

    stream = stream_book_rag_ingest(_envelope(), settings=settings, pg_pool=object())  # type: ignore[arg-type]
    first = await anext(stream)
    assert isinstance(first, IngestProgress)

    await asyncio.wait_for(stream.aclose(), timeout=1.0)

    assert embedding_client.aclose_called is True
    assert llm_instances[0].aclose_called is True


@pytest.mark.asyncio
async def test_stream_book_rag_ingest_logs_and_closes_when_drain_handler_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("llama_index")

    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow_li as workflow_li_mod
    from vo_rag.pipelines.book_rag import runner as runner_mod

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
    )
    embedding_client = _FakeEmbeddingClient()
    llm_instances: list[_FakeLLMClient] = []
    warnings: list[tuple[str, dict[str, Any]]] = []

    async def _fake_build_surface_services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=embedding_client,
            retriever=object(),
            db_pool=db_pool,
        )

    def _fake_build_parser_router(settings_arg: Any, figures_arg: Any) -> object:
        _ = (settings_arg, figures_arg)
        return object()

    class _RecordingLLMClient(_FakeLLMClient):
        def __init__(self, *, provider: str, role: str) -> None:
            super().__init__(provider=provider, role=role)
            llm_instances.append(self)

    class _RaisingHandler:
        async def stream_events(self) -> Any:
            yield IngestProgress(envelope_id=str(_envelope().id), step="parse", detail="started")

        def __await__(self) -> Any:
            async def _run() -> PipelineOutput:
                raise RuntimeError("workflow drain failed")

            return _run().__await__()

    class _FakeWorkflow:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, *, envelope: IntakeEnvelope) -> _RaisingHandler:
            return _RaisingHandler()

    class _FakeLog:
        def info(self, event: str, **kwargs: Any) -> None:
            pass

        def warning(self, event: str, **kwargs: Any) -> None:
            warnings.append((event, kwargs))

    monkeypatch.setattr(singletons, "build_surface_services", _fake_build_surface_services)
    monkeypatch.setattr(parser_factory, "build_parser_router", _fake_build_parser_router)
    monkeypatch.setattr(llm_mod, "LLMClient", _RecordingLLMClient)
    monkeypatch.setattr(workflow_li_mod, "BookRagWorkflow", _FakeWorkflow)
    monkeypatch.setattr(runner_mod, "log", _FakeLog(), raising=False)

    stream = runner_mod.stream_book_rag_ingest(
        _envelope(),
        settings=settings,  # type: ignore[arg-type]
        pg_pool=object(),
    )
    first = await anext(stream)
    assert isinstance(first, IngestProgress)

    await asyncio.wait_for(stream.aclose(), timeout=1.0)

    assert embedding_client.aclose_called is True
    assert llm_instances[0].aclose_called is True
    assert warnings == [
        (
            "book_rag_ingest_stream_background_failed_before_cleanup",
            {
                "envelope_id": str(_envelope().id),
                "error": "workflow drain failed",
                "exc_info": True,
            },
        )
    ]


# ----------------------------------------------------------------------
# VOR-1405 — reusable ingest session. run_book_rag_ingest builds settings,
# services, the parser router and the pipeline per call, which is right for
# the dispatcher (one envelope at a time) but pathological for a bulk batch:
# measured 12.9s of fixed setup per document on localhost against 0.1ms
# to parse a Markdown file. The session lets a batch build once and ingest
# many; run_book_rag_ingest becomes the one-shot case of it.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_session_builds_once_and_ingests_many(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow as workflow_mod
    from vo_rag.pipelines.book_rag.runner import book_rag_ingest_session

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
    )
    embedding_client = _FakeEmbeddingClient()
    counts = {"services": 0, "router": 0, "pipeline": 0}
    ingested: list[str] = []
    expected = PipelineOutput(
        source_id="source-1",
        status=EnvelopeStatus.COMPLETED,
        qdrant_point_ids=["p1"],
        raw_point_ids=["p1"],
        sterile_point_ids=[],
        smoke_decision=None,
        chunk_count=1,
    )

    async def _fake_build_surface_services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        counts["services"] += 1
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=embedding_client,
            retriever=object(),
            db_pool=db_pool,
        )

    def _fake_build_parser_router(settings_arg: Any, figures_arg: Any) -> object:
        counts["router"] += 1
        return object()

    class _FakePipeline:
        def __init__(self, **kwargs: Any) -> None:
            counts["pipeline"] += 1

        async def ingest(self, envelope: IntakeEnvelope) -> PipelineOutput:
            ingested.append(str(envelope.id))
            return expected

    monkeypatch.setattr(singletons, "build_surface_services", _fake_build_surface_services)
    monkeypatch.setattr(parser_factory, "build_parser_router", _fake_build_parser_router)
    monkeypatch.setattr(llm_mod, "LLMClient", _FakeLLMClient)
    monkeypatch.setattr(workflow_mod, "BookRagPipeline", _FakePipeline)

    async with book_rag_ingest_session(settings) as session:  # type: ignore[arg-type]
        first = await session.ingest(_envelope())
        second = await session.ingest(_envelope())

    assert first is expected
    assert second is expected
    assert len(ingested) == 2
    assert counts == {"services": 1, "router": 1, "pipeline": 1}


@pytest.mark.asyncio
async def test_ingest_session_closes_its_seams_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow as workflow_mod
    from vo_rag.pipelines.book_rag.runner import book_rag_ingest_session

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
    )
    embedding_client = _FakeEmbeddingClient()
    llm_holder: dict[str, Any] = {}

    async def _fake_build_surface_services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=embedding_client,
            retriever=object(),
            db_pool=db_pool,
        )

    class _CapturingLLMClient(_FakeLLMClient):
        def __init__(self, *, provider: str, role: str) -> None:
            super().__init__(provider=provider, role=role)
            llm_holder["client"] = self

    class _FakePipeline:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def ingest(self, envelope: IntakeEnvelope) -> PipelineOutput:  # pragma: no cover
            raise AssertionError("not used")

    monkeypatch.setattr(singletons, "build_surface_services", _fake_build_surface_services)
    monkeypatch.setattr(parser_factory, "build_parser_router", lambda s, f: object())
    monkeypatch.setattr(llm_mod, "LLMClient", _CapturingLLMClient)
    monkeypatch.setattr(workflow_mod, "BookRagPipeline", _FakePipeline)

    async with book_rag_ingest_session(settings):  # type: ignore[arg-type]
        pass

    assert embedding_client.aclose_called is True
    assert llm_holder["client"].aclose_called is True


@pytest.mark.asyncio
async def test_ingest_session_injects_and_closes_enabled_resident_vlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.visual_sidecar as visual_sidecar
    import vo_rag.pipelines.book_rag.workflow as workflow_mod
    from vo_rag.pipelines.book_rag.runner import book_rag_ingest_session

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
        vlm=SimpleNamespace(enabled=True, describe_concurrency=1),
    )
    embedding_client = _FakeEmbeddingClient()
    captured: dict[str, Any] = {}

    class _Vlm:
        model = "resident"
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    vlm = _Vlm()

    async def _fake_build_surface_services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=embedding_client,
            retriever=object(),
            db_pool=db_pool,
        )

    class _FakePipeline:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(singletons, "build_surface_services", _fake_build_surface_services)
    monkeypatch.setattr(parser_factory, "build_parser_router", lambda s, f: object())
    monkeypatch.setattr(llm_mod, "LLMClient", _FakeLLMClient)
    monkeypatch.setattr(workflow_mod, "BookRagPipeline", _FakePipeline)
    monkeypatch.setattr(visual_sidecar, "build_vlm_service", lambda settings_arg: vlm)

    async with book_rag_ingest_session(settings):  # type: ignore[arg-type]
        assert captured["vlm_service"] is vlm
        assert captured["config"].enable_visual_description is True
        assert vlm.closed is False

    assert vlm.closed is True


class _RecordingPool:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        return "UPDATE 1"


@pytest.mark.asyncio
async def test_stream_book_rag_ingest_records_metrics_for_the_workflow_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VOR-1434: the streaming (API/CLI/MCP) path stamps its start and
    persists what its workflow tasks did -- the recorder lives on the workflow's
    tasks, not across the generator's yields, so the consumer never sees it."""
    pytest.importorskip("llama_index")

    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow_li as workflow_li_mod
    from vo_rag.core.observability.metrics import (
        ENVELOPE_METRICS_MERGE_SQL,
        ENVELOPE_STREAM_START_SQL,
        current_ingest_recorder,
        record_embed_call,
        record_qdrant_call,
    )
    from vo_rag.pipelines.book_rag.runner import stream_book_rag_ingest

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
    )
    pool = _RecordingPool()
    output = PipelineOutput(
        source_id="source-1",
        status=EnvelopeStatus.COMPLETED,
        qdrant_point_ids=[],
        raw_point_ids=[],
        sterile_point_ids=[],
        smoke_decision=None,
        chunk_count=0,
    )

    async def _fake_build_surface_services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=_FakeEmbeddingClient(),
            retriever=object(),
            db_pool=db_pool,
        )

    class _Handler:
        def __init__(self, task: asyncio.Task[PipelineOutput]) -> None:
            self._task = task

        async def stream_events(self) -> Any:
            yield IngestProgress(envelope_id=str(_envelope().id), step="parse", detail="x")

        def __await__(self) -> Any:
            return self._task.__await__()

    class _FakeWorkflow:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, *, envelope: IntakeEnvelope) -> _Handler:
            # llama-index starts its step tasks inside run(): they inherit
            # the context current at that moment.
            async def _steps() -> PipelineOutput:
                await asyncio.sleep(0)
                record_embed_call(retries_429=2)
                record_qdrant_call(0.3)
                return output

            return _Handler(asyncio.create_task(_steps()))

    monkeypatch.setattr(singletons, "build_surface_services", _fake_build_surface_services)
    monkeypatch.setattr(parser_factory, "build_parser_router", lambda *_: object())
    monkeypatch.setattr(llm_mod, "LLMClient", _FakeLLMClient)
    monkeypatch.setattr(workflow_li_mod, "BookRagWorkflow", _FakeWorkflow)

    events = []
    async for ev in stream_book_rag_ingest(_envelope(), settings=settings, pg_pool=pool):  # type: ignore[arg-type]
        assert current_ingest_recorder() is None  # not leaked into the consumer
        events.append(ev)
    assert events[-1] is output

    envelope_id = str(_envelope().id)
    assert pool.executed[0] == (ENVELOPE_STREAM_START_SQL, (envelope_id,))
    merges = [args for query, args in pool.executed if query == ENVELOPE_METRICS_MERGE_SQL]
    assert merges, "stream metrics never persisted"
    assert merges[-1][3:5] == (1, 2)  # absolute attempt totals
    final_overwrite = merges[-1][1]
    assert final_overwrite["active_stream"] is False
    assert final_overwrite["qdrant"]["calls"] == 1


@pytest.mark.asyncio
async def test_stream_clears_active_stream_even_when_the_workflow_fails_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("llama_index")

    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow_li as workflow_li_mod
    from vo_rag.core.observability.metrics import ENVELOPE_METRICS_MERGE_SQL
    from vo_rag.pipelines.book_rag.runner import stream_book_rag_ingest

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
    )
    pool = _RecordingPool()

    async def _fake_build_surface_services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=_FakeEmbeddingClient(),
            retriever=object(),
            db_pool=db_pool,
        )

    class _BrokenWorkflow:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, *, envelope: IntakeEnvelope) -> Any:
            raise RuntimeError("workflow failed to start")

    monkeypatch.setattr(singletons, "build_surface_services", _fake_build_surface_services)
    monkeypatch.setattr(parser_factory, "build_parser_router", lambda *_: object())
    monkeypatch.setattr(llm_mod, "LLMClient", _FakeLLMClient)
    monkeypatch.setattr(workflow_li_mod, "BookRagWorkflow", _BrokenWorkflow)

    with pytest.raises(RuntimeError, match="failed to start"):
        async for _ in stream_book_rag_ingest(_envelope(), settings=settings, pg_pool=pool):  # type: ignore[arg-type]
            pass
    merges = [args for query, args in pool.executed if query == ENVELOPE_METRICS_MERGE_SQL]
    assert merges and merges[-1][1]["active_stream"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failed_output", "raises", "completed"])
async def test_stream_records_a_failure_exactly_when_the_book_fails(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """vo_rag_intake_failures_last_hour keys on the failure time the failure
    record stamps, so a streamed failure must record one too (once)."""
    pytest.importorskip("llama_index")

    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow_li as workflow_li_mod
    from vo_rag.core.observability.metrics import ENVELOPE_FAILURE_METRIC_SQL
    from vo_rag.pipelines.book_rag.runner import stream_book_rag_ingest

    settings = SimpleNamespace(
        garage=SimpleNamespace(buckets=SimpleNamespace(artifacts="b")),
        figures=SimpleNamespace(root="/tmp/vo-rag-test-figures"),
    )
    pool = _RecordingPool()
    status = EnvelopeStatus.FAILED if outcome == "failed_output" else EnvelopeStatus.COMPLETED

    async def _fake_build_surface_services(settings_arg: Any, *, db_pool: Any = None) -> Any:
        return SimpleNamespace(
            garage=object(),
            qdrant=object(),
            embedding_client=_FakeEmbeddingClient(),
            retriever=object(),
            db_pool=db_pool,
        )

    class _Handler:
        def __init__(self, task: asyncio.Task[PipelineOutput]) -> None:
            self._task = task

        async def stream_events(self) -> Any:
            yield IngestProgress(envelope_id=str(_envelope().id), step="parse", detail="x")

        def __await__(self) -> Any:
            return self._task.__await__()

    class _FakeWorkflow:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, *, envelope: IntakeEnvelope) -> _Handler:
            async def _steps() -> PipelineOutput:
                await asyncio.sleep(0)
                if outcome == "raises":
                    raise RuntimeError("step exploded")
                return PipelineOutput(
                    source_id="s",
                    status=status,
                    qdrant_point_ids=[],
                    raw_point_ids=[],
                    sterile_point_ids=[],
                    smoke_decision=None,
                    chunk_count=0,
                )

            return _Handler(asyncio.create_task(_steps()))

    monkeypatch.setattr(singletons, "build_surface_services", _fake_build_surface_services)
    monkeypatch.setattr(parser_factory, "build_parser_router", lambda *_: object())
    monkeypatch.setattr(llm_mod, "LLMClient", _FakeLLMClient)
    monkeypatch.setattr(workflow_li_mod, "BookRagWorkflow", _FakeWorkflow)

    async def _consume() -> None:
        async for _ in stream_book_rag_ingest(_envelope(), settings=settings, pg_pool=pool):  # type: ignore[arg-type]
            pass

    if outcome == "raises":
        with pytest.raises(RuntimeError, match="step exploded"):
            await _consume()
    else:
        await _consume()
    records = [a for q, a in pool.executed if q == ENVELOPE_FAILURE_METRIC_SQL]
    assert len(records) == (0 if outcome == "completed" else 1)
