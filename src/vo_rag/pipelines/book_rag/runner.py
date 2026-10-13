"""Production entry point for running book_rag ingest on a stored envelope."""

from __future__ import annotations

import asyncio
import contextvars
import os
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import structlog

if TYPE_CHECKING:
    from vo_rag.core.config import Settings
    from vo_rag.core.models.envelope import IntakeEnvelope
    from vo_rag.pipelines.book_rag.events import ImageDescribedEvent, IngestProgress
    from vo_rag.pipelines.book_rag.workflow import PipelineOutput

log = structlog.get_logger(__name__)


def _book_rag_config(settings: Settings) -> Any:
    """Build the production pipeline flags from resolved service settings."""
    from vo_rag.pipelines.book_rag.config import BookRagConfig

    vlm = getattr(settings, "vlm", None)
    return BookRagConfig(
        enable_visual_description=bool(getattr(vlm, "enabled", False)),
        vlm_concurrency=int(getattr(vlm, "describe_concurrency", 1)),
    )


def _build_optional_llm_client(llm_mod: Any) -> Any | None:
    """Build the text LLM client, or return None when serverless is unconfigured.

    A "defer-LLM" ingest (parse/extract-images/chunk/embed only — no visual
    descriptions, no advanced layers) does not need an LLM: the only LLM step is
    the terminal retrieval-smoke gate, which is skipped when the client is None.
    Returning None instead of raising lets such an ingest run without a RunPod /
    serverless endpoint configured.
    """
    try:
        return llm_mod.LLMClient(provider="runpod_serverless", role="text")
    except Exception as exc:  # noqa: BLE001 - any provider/config error disables smoke
        log.warning("book_rag_llm_client_unavailable_smoke_disabled", error=str(exc))
        return None


class BookRagIngestSession:
    """A built book_rag pipeline that can ingest many envelopes.

    Construction is the expensive part -- settings, Garage/Qdrant/embedding
    seams, the parser router and the pipeline. The dispatcher handles one
    envelope at a time so paying that per call is fine, but a bulk batch paid
    12.9s of setup per document (measured 2026-08-14) against 0.1ms to parse a
    Markdown file. Hold one of these open and the cost is paid once (VOR-1405).
    """

    __slots__ = ("_pipeline",)

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    async def ingest(self, envelope: IntakeEnvelope) -> PipelineOutput:
        log.info("book_rag_ingest_runner_started", envelope_id=str(envelope.id))
        output: PipelineOutput = await self._pipeline.ingest(envelope)
        log.info(
            "book_rag_ingest_runner_completed",
            envelope_id=str(envelope.id),
            source_id=output.source_id,
            chunk_count=output.chunk_count,
        )
        return output


@asynccontextmanager
async def book_rag_ingest_session(
    settings: Settings,
    *,
    pg_pool: Any | None = None,
) -> AsyncIterator[BookRagIngestSession]:
    """Build the book_rag seams once and yield a reusable ingest session."""
    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow as workflow_mod
    from vo_rag.core.storage.figures import FigureStore

    services = await singletons.build_surface_services(settings, db_pool=pg_pool)
    garage = _require_service(services, "garage")
    qdrant = _require_service(services, "qdrant")
    embedding_client = _require_service(services, "embedding_client")
    retriever = _require_service(services, "retriever")
    db_pool = getattr(services, "db_pool", None)

    llm_client: Any | None = None
    parser_router: Any | None = None
    vlm_service: Any | None = None
    try:
        from vo_rag.pipelines.book_rag.visual_sidecar import build_vlm_service

        pipeline_config = _book_rag_config(settings)
        figures = FigureStore(settings.figures.root)
        parser_router = parser_factory.build_parser_router(settings, figures)
        llm_client = _build_optional_llm_client(llm_mod)
        if pipeline_config.enable_visual_description:
            vlm_service = build_vlm_service(settings)
        pipeline = workflow_mod.BookRagPipeline(
            garage=garage,
            qdrant_store=qdrant,
            embedding_client=embedding_client,
            llm_client=llm_client,
            retriever=retriever,
            parser_router=parser_router,
            pool=db_pool,
            pg_pool=db_pool,
            bucket=_artifacts_bucket(settings),
            figures=figures,
            settings=settings,
            config=pipeline_config,
            vlm_service=vlm_service,
            raw_collection=_raw_collection_override(),
        )
        yield BookRagIngestSession(pipeline)
    finally:
        await _close_parser_router(parser_router)
        await _close_quietly(vlm_service, "vlm_service")
        await _close_quietly(llm_client, "llm_client")
        await _close_quietly(embedding_client, "embedding_client")
        if pg_pool is None and db_pool is not None:
            import vo_rag.core.db.postgres as postgres_mod

            await _close_quietly(postgres_mod, "postgres_pool", close_method="close_pool")


async def run_book_rag_ingest(
    envelope: IntakeEnvelope,
    *,
    settings: Settings,
    pg_pool: Any | None = None,
) -> PipelineOutput:
    """Build real book_rag seams and run ``BookRagPipeline.ingest(envelope)``.

    The one-shot case of :func:`book_rag_ingest_session`.
    """
    async with book_rag_ingest_session(settings, pg_pool=pg_pool) as session:
        return await session.ingest(envelope)


async def stream_book_rag_ingest(
    envelope: IntakeEnvelope,
    *,
    settings: Settings,
    pg_pool: Any | None = None,
) -> AsyncGenerator[IngestProgress | ImageDescribedEvent | PipelineOutput, None]:
    """Build real book_rag seams and stream ``BookRagWorkflow`` progress events.

    The async generator yields every :class:`IngestProgress` and
    :class:`ImageDescribedEvent` emitted by the workflow, followed by the
    terminal :class:`PipelineOutput`. Callers must distinguish the last
    item with ``isinstance(ev, PipelineOutput)``.
    """
    import vo_rag.api.singletons as singletons
    import vo_rag.core.llm.client as llm_mod
    import vo_rag.core.parsing.factory as parser_factory
    import vo_rag.pipelines.book_rag.workflow_li as workflow_li_mod
    from vo_rag.core.observability.metrics import (
        IngestMetricsRecorder,
        record_stream_start,
        set_ingest_recorder,
        start_periodic_ingest_flush,
    )
    from vo_rag.core.storage.figures import FigureStore
    from vo_rag.pipelines.book_rag.events import ImageDescribedEvent, IngestProgress

    services = await singletons.build_surface_services(settings, db_pool=pg_pool)
    garage = _require_service(services, "garage")
    qdrant = _require_service(services, "qdrant")
    embedding_client = _require_service(services, "embedding_client")
    retriever = _require_service(services, "retriever")
    db_pool = getattr(services, "db_pool", None)

    llm_client: Any | None = None
    parser_router: Any | None = None
    handler: Any | None = None
    handler_done = False
    # VOR-1434: per-envelope metrics for the streaming path. The
    # recorder is made current only in a copied context that the workflow's
    # tasks are started from -- never across this generator's yields, so the
    # consumer's context is untouched.
    envelope_id = str(envelope.id)
    recorder = IngestMetricsRecorder()
    flusher: asyncio.Task[None] | None = None
    stream_stamped = False
    stream_failed = False
    try:
        from vo_rag.pipelines.book_rag.visual_sidecar import build_vlm_service

        figures = FigureStore(settings.figures.root)
        parser_router = parser_factory.build_parser_router(settings, figures)
        llm_client = _build_optional_llm_client(llm_mod)
        workflow = workflow_li_mod.BookRagWorkflow(
            garage=garage,
            qdrant_store=qdrant,
            embedding_client=embedding_client,
            llm_client=llm_client,
            retriever=retriever,
            parser_router=parser_router,
            pool=db_pool,
            pg_pool=db_pool,
            bucket=_artifacts_bucket(settings),
            figures=figures,
            settings=settings,
            config=_book_rag_config(settings),
            vlm_service_factory=lambda: build_vlm_service(settings),
            raw_collection=_raw_collection_override(),
            timeout=None,
        )
        log.info("book_rag_ingest_stream_started", envelope_id=envelope_id)
        await record_stream_start(db_pool, envelope_id)
        stream_stamped = True
        flusher = start_periodic_ingest_flush(db_pool, envelope_id, recorder, heartbeat=True)
        run_context = contextvars.copy_context()
        run_context.run(set_ingest_recorder, recorder)
        handler = run_context.run(workflow.run, envelope=envelope)
        async for ev in handler.stream_events():
            if isinstance(ev, (IngestProgress, ImageDescribedEvent)):
                yield ev
        result = await asyncio.shield(handler)
        handler_done = True
        stream_failed = _is_failed(result)
        log.info(
            "book_rag_ingest_stream_completed",
            envelope_id=str(envelope.id),
            source_id=result.source_id,
            chunk_count=result.chunk_count,
        )
        yield result
    finally:
        active_exc = sys.exc_info()[1]
        consumer_cancelled = isinstance(active_exc, asyncio.CancelledError)
        if isinstance(active_exc, Exception):
            stream_failed = True
        cancelled = False
        drained_result: PipelineOutput | None = None
        try:
            if handler is not None and not handler_done:
                cancelled, drained_result = await _await_workflow_handler_before_cleanup(
                    handler,
                    str(envelope.id),
                )
        finally:
            try:
                await _finish_stream_metrics(
                    db_pool,
                    envelope_id,
                    recorder,
                    flusher,
                    started=stream_stamped,
                    failed=stream_failed or _is_failed(drained_result),
                )
            finally:
                await _close_parser_router(parser_router)
                await _close_quietly(llm_client, "llm_client")
                await _close_quietly(embedding_client, "embedding_client")
                if pg_pool is None and db_pool is not None:
                    import vo_rag.core.db.postgres as postgres_mod

                    await _close_quietly(postgres_mod, "postgres_pool", close_method="close_pool")
                if cancelled or (consumer_cancelled and drained_result is not None):
                    exc = asyncio.CancelledError()
                    if drained_result is not None:
                        cast(Any, exc).terminal_output = drained_result
                    raise exc


async def _finish_stream_metrics(
    pool: Any | None,
    envelope_id: str,
    recorder: Any,
    flusher: asyncio.Task[None] | None,
    *,
    started: bool,
    failed: bool = False,
) -> None:
    """Stop the periodic flush and write the stream's final metrics: close the
    open stage, clear ``active_stream``, and record the failure (once) when the
    book failed. Best-effort: never raises except on cancellation."""
    from vo_rag.core.observability.metrics import (
        persist_ingest_metrics,
        record_envelope_failure,
        stop_periodic_ingest_flush,
    )

    await stop_periodic_ingest_flush(flusher)
    if started:
        recorder.close_stage()
        recorder.active_stream = False
        await persist_ingest_metrics(pool, envelope_id, recorder)
        if failed:
            await record_envelope_failure(pool, envelope_id)


def _is_failed(output: Any | None) -> bool:
    return getattr(getattr(output, "status", None), "value", None) == "failed"


def _require_service(services: Any, name: str) -> Any:
    value = getattr(services, name, None)
    if value is None:
        reason = getattr(services, f"{name}_unavailable_reason", "") or "not_configured"
        raise RuntimeError(f"book_rag_ingest_runner service_unavailable:{name}: {reason}")
    return value


def _artifacts_bucket(settings: Any) -> str:
    garage_settings = getattr(settings, "garage", None)
    buckets = getattr(garage_settings, "buckets", None)
    return str(getattr(buckets, "artifacts", "vorag-artifacts"))


def _raw_collection_override() -> str | None:
    raw_collection = os.environ.get("VORAG_RAW_COLLECTION")
    if raw_collection is None:
        return None
    raw_collection = raw_collection.strip()
    return raw_collection or None


async def _close_parser_router(parser_router: Any | None) -> None:
    if parser_router is None:
        return
    await _close_quietly(getattr(parser_router, "docling", None), "parser_router.docling")
    await _close_quietly(getattr(parser_router, "epub", None), "parser_router.epub")


async def _await_workflow_handler_before_cleanup(
    handler: Any, envelope_id: str
) -> tuple[bool, PipelineOutput | None]:
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(handler)
            return cancelled, result
        except asyncio.CancelledError:
            task = asyncio.current_task()
            cancelling = task.cancelling() if task is not None else 0
            if cancelling == 0:
                log.warning(
                    "book_rag_ingest_stream_background_cancelled_before_cleanup",
                    envelope_id=envelope_id,
                )
                return cancelled, None
            if cancelled:
                raise
            cancelled = True
            if task is not None:
                for _ in range(cancelling):
                    task.uncancel()
            continue
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "book_rag_ingest_stream_background_failed_before_cleanup",
                envelope_id=envelope_id,
                error=str(exc),
                exc_info=True,
            )
            return cancelled, None


async def _close_quietly(
    obj: Any | None,
    name: str,
    *,
    close_method: str = "aclose",
) -> None:
    if obj is None:
        return
    close = getattr(obj, close_method, None)
    if close is None:
        return
    try:
        result = close()
        if hasattr(result, "__await__"):
            await result
    except Exception as exc:  # noqa: BLE001
        log.warning("book_rag_ingest_runner_close_failed", name=name, error=str(exc))


__all__ = [
    "BookRagIngestSession",
    "book_rag_ingest_session",
    "run_book_rag_ingest",
    "stream_book_rag_ingest",
]
