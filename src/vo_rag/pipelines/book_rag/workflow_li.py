"""LlamaIndex Workflow port of the book_rag ingest DAG.

This module sits beside the existing dataclass orchestrator in ``workflow.py``.
It rewires the same helper calls into LlamaIndex typed events without changing
the underlying parse, chunk/embed, upsert, sterilize, artifact, validation, or
retrieval-smoke behavior.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import structlog
from llama_index.core.workflow import (
    Context as LlamaIndexContext,
)
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    WorkflowCheckpointer,
    step,
)
from pydantic import BaseModel, Field
from workflows.checkpointer import Checkpoint as WorkflowCheckpoint
from workflows.context import BaseSerializer
from workflows.context import PickleSerializer as CheckpointPickleSerializer

from vo_rag.core.chunking.composite_emitter import CompositeChunk
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.db.queries.artifacts import insert_artifact
from vo_rag.core.embedding.point_vectors import include_colbert_at_ingest
from vo_rag.core.ingest.object_verification import object_is_verified
from vo_rag.core.ingest.source_publication import SourcePublicationJob
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.artifact import ArtifactType, KnowledgeArtifact, ProvenanceEntry, TrustLevel
from vo_rag.core.models.envelope import EnvelopeStatus, IntakeEnvelope, TargetPipeline
from vo_rag.core.models.ingest_generation import IngestGenerationState
from vo_rag.core.observability.metrics import enter_ingest_stage
from vo_rag.core.parsing.contract import ParsedDocument
from vo_rag.core.summary.builder import _strip_reasoning
from vo_rag.pipelines.book_rag import retrieval_smoke as smoke
from vo_rag.pipelines.book_rag.checkpoint import Checkpoint
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.events import ImageDescribedEvent, IngestProgress
from vo_rag.pipelines.book_rag.quality_warnings import collect_quality_warnings
from vo_rag.pipelines.book_rag.source_identity import canonical_source_id, resolve_canonical_source
from vo_rag.pipelines.book_rag.steps.chunk_embed import (
    do_build_chunks,
    do_embed_raw,
    do_upsert_raw,
)
from vo_rag.pipelines.book_rag.steps.parse import do_parse_document
from vo_rag.pipelines.book_rag.steps.sterilize import do_sterilize
from vo_rag.pipelines.book_rag.validator import (
    BUCKET_ARTIFACTS,
    IngestionValidator,
    ValidationReport,
)
from vo_rag.pipelines.book_rag.visual_sidecar import (
    VlmImageJob,
    VLMServiceLike,
    _build_describe_prompt,
    _context_for_image,
    _parse_concurrency,
    _parse_float_env,
    _parse_int_env,
    _vlm_image_jobs,
    build_extraction_composites,
    build_vlm_service,
    load_visual_composites,
    visual_evidence_from_vlm,
)
from vo_rag.pipelines.book_rag.workflow import (
    RAW_COLLECTION,
    STERILE_COLLECTION,
    BookRagContext,
    BookRagPipeline,
    IngestGateRejected,
    IngestGenerationClaimed,
    PipelineOutput,
    _artifact_id,
    _artifact_license_class,
    _artifact_tags,
    _artifact_title,
    _force_reingest,
    _ingest_generations_enabled,
    _raw_collection_override,
)

log = structlog.get_logger(__name__)

_VLM_CONCURRENCY = _parse_concurrency(
    os.environ.get("VO_RAG_VLM_DESCRIBE_CONCURRENCY"),
    default=1,
)

_WORKFLOW_RUN_ID_KEY = "vo_rag_workflow_run_id"
_VISUAL_VLM_SERVICE_ATTR = "visual_vlm_service"
_GENERATION_LEASE_TASK_ATTR = "generation_lease_task"


async def _context_run_id(ctx: LlamaIndexContext[Any]) -> str:
    run_id = getattr(ctx, _WORKFLOW_RUN_ID_KEY, None)
    if not run_id:
        try:
            stored_run_id = await ctx.store.get("workflow_run_id", default="")
        except Exception as exc:
            log.warning(
                "book_rag_workflow_run_id_store_read_failed",
                error=str(exc),
                exc_info=True,
            )
            raise
        if stored_run_id:
            run_id = stored_run_id
    if not run_id:
        run_id = getattr(ctx, "run_id", None)
    if not run_id:
        raise RuntimeError("BookRagWorkflow context is missing workflow_run_id")
    setattr(ctx, _WORKFLOW_RUN_ID_KEY, str(run_id))
    return str(run_id)


async def _collect_buffer_id(ctx: LlamaIndexContext[Any], step_name: str) -> str:
    return f"{step_name}:{await _context_run_id(ctx)}"


def _workflow_context_for_run(workflow: Workflow, workflow_run_id: str) -> Context[IngestState]:
    ctx: Context[IngestState] = Context(workflow)
    setattr(ctx, _WORKFLOW_RUN_ID_KEY, workflow_run_id)
    return ctx


class Context[StateT: BaseModel](LlamaIndexContext[StateT]):
    """Typed Context annotation for native LlamaIndex state-store inference."""


class IngestStartEvent(StartEvent):
    envelope: IntakeEnvelope


class ParseMeta(TypedDict, total=False):
    parsed_garage_key: str
    document_id: str
    source_id: str
    title: str | None
    total_pages: int
    image_count: int
    element_count: int
    parser_used: str
    parser_backend: str | None
    forward_path: dict[str, Any]
    duration_ms: int
    quarantined_page_count: int
    quality_warnings: list[dict[str, Any]]
    parse_provenance: dict[str, object]


class RegisteredEvent(Event):
    source_id: str


class ParsedEvent(Event):
    source_id: str
    parse_meta: ParseMeta
    parsed: ParsedDocument


class VisualEvent(Event):
    source_id: str
    parse_meta: ParseMeta
    parsed: ParsedDocument
    composites: list[CompositeChunk]


class DescribeImageEvent(Event):
    image_id: str
    image_ref: str
    image_role: str
    element_index: int


class ImageDescriptionResult(Event):
    image_id: str
    evidence: dict[str, Any] | None


class ChunkedEvent(Event):
    source_id: str
    parse_meta: ParseMeta
    chunks: list[RetrievalChunk]
    raw_point_ids: list[str]


class EmbeddedEvent(ChunkedEvent):
    raw_embeddings: dict[str, Any]


class SterilizedEvent(Event):
    source_id: str
    parse_meta: ParseMeta
    chunks: list[RetrievalChunk]
    raw_point_ids: list[str]
    sterile_point_ids: list[str]


class RegisteredArtifactEvent(SterilizedEvent):
    pass


class ValidatedEvent(SterilizedEvent):
    validation_report: ValidationReport | None


class IngestState(BaseModel):
    """Checkpointable ingest state carried across workflow steps."""

    workflow_run_id: str = ""
    envelope: IntakeEnvelope | None = None
    source_id: str = ""
    generation_id: str | None = None
    previous_generation_id: str | None = None
    generation_owner_token: str | None = None
    generation_claim_acquired: bool = False
    publication_job: SourcePublicationJob | None = None
    sterile_publication_job: SourcePublicationJob | None = None
    raw_point_ids: list[str] = Field(default_factory=list)
    sterile_point_ids: list[str] = Field(default_factory=list)
    chunk_count: int = 0
    smoke_decision: smoke.GateDecision | None = None
    visual_jobs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    visual_parsed: ParsedDocument | None = None
    visual_source_id: str = ""
    visual_parse_meta: ParseMeta = Field(default_factory=lambda: cast(ParseMeta, {}))
    visual_envelope_id: str = ""
    visual_eligible_ids: list[str] = Field(default_factory=list)
    visual_skipped: dict[str, Any] = Field(default_factory=dict)
    visual_document_metadata: dict[str, Any] = Field(default_factory=dict)
    n_pending: int = 0


async def _describe_one_image(
    *,
    envelope_id: str,
    job: VlmImageJob,
    parsed: ParsedDocument,
    vlm_service: VLMServiceLike,
    checkpoint: Checkpoint,
    max_tokens: int,
    deadline_s: float,
    document_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Describe a single image; return evidence dict or None on skip/failure."""
    img = job.image
    try:
        context = _context_for_image(
            parsed,
            img.element_index,
            document_metadata=document_metadata,
        )
        prompt = _build_describe_prompt(
            image=img,
            image_role=job.image_role,
            parsed=parsed,
            context=context,
        )
        describe_call = vlm_service.describe_image(
            job.image_bytes,
            prompt,
            img.image_id,
            max_tokens=max_tokens,
        )
        if deadline_s > 0:
            description = await asyncio.wait_for(describe_call, timeout=deadline_s)
        else:
            description = await describe_call
        evidence = visual_evidence_from_vlm(
            description,
            image=img,
            image_role=job.image_role,
        )
        evidence_payload = evidence.model_dump()
        if not evidence.admitted:
            log.info(
                "book_rag.workflow_li.visual.describe rejected",
                envelope_id=envelope_id,
                image_id=img.image_id,
                reject_reason=evidence.rejection_reason,
                figure_kind=evidence.figure_kind,
            )
        await checkpoint.record_image(img.image_id, evidence_payload)
        return evidence_payload
    except Exception as exc:
        log.warning(
            "book_rag.workflow_li.visual.describe image failed",
            envelope_id=envelope_id,
            image_id=img.image_id,
            error=str(exc),
        )
        return None


class BookRagWorkflow(Workflow):
    """Event-driven re-expression of :class:`BookRagPipeline` ingest."""

    def __init__(
        self,
        *,
        garage: Any,
        qdrant_store: Any,
        embedding_client: Any,
        llm_client: Any,
        retriever: Any,
        parser_router: Any,
        pool: Any | None = None,
        pg_pool: Any | None = None,
        config: BookRagConfig | None = None,
        settings: Any | None = None,
        raw_collection: str | None = None,
        bucket: str = BUCKET_ARTIFACTS,
        figures: Any | None = None,
        smoke_n: int = 4,
        smoke_seed: int = 42,
        vlm_service_factory: Callable[[], VLMServiceLike] = build_vlm_service,
        **workflow_kwargs: Any,
    ) -> None:
        super().__init__(**workflow_kwargs)
        self.garage = garage
        self.qdrant_store = qdrant_store
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.retriever = retriever
        self.parser_router = parser_router
        self.pool = pool
        self.pg_pool = pg_pool
        self.config = config or BookRagConfig()
        self.settings = settings
        self.raw_collection = _raw_collection_override(raw_collection)
        self.bucket = bucket
        self.figures = figures
        self.smoke_n = smoke_n
        self.smoke_seed = smoke_seed
        self.vlm_service_factory = vlm_service_factory

    def _generation_coordinator(self) -> BookRagPipeline:
        """Reuse the native workflow's generation state machine primitives."""
        return BookRagPipeline(
            garage=self.garage,
            qdrant_store=self.qdrant_store,
            embedding_client=self.embedding_client,
            llm_client=self.llm_client,
            retriever=self.retriever,
            parser_router=self.parser_router,
            pool=self.pool,
            pg_pool=self.pg_pool,
            config=self.config,
            settings=self.settings,
            raw_collection=self.raw_collection,
            bucket=self.bucket,
            figures=self.figures,
            smoke_n=self.smoke_n,
            smoke_seed=self.smoke_seed,
        )

    async def _context_generation(
        self,
        ctx: Context[IngestState],
        envelope: IntakeEnvelope,
        *,
        source_id: str | None = None,
    ) -> BookRagContext:
        """Rehydrate the checkpointed generation lease into native state."""
        return BookRagContext(
            envelope=envelope,
            config=self.config,
            source_id=source_id or str(await ctx.store.get("source_id", default="")),
            generation_id=await ctx.store.get("generation_id", default=None),
            previous_generation_id=await ctx.store.get(
                "previous_generation_id",
                default=None,
            ),
            generation_owner_token=await ctx.store.get(
                "generation_owner_token",
                default=None,
            ),
            generation_claim_acquired=bool(
                await ctx.store.get("generation_claim_acquired", default=False)
            ),
        )

    async def _assert_context_generation_lease(
        self,
        ctx: Context[IngestState],
        envelope: IntakeEnvelope,
        *,
        source_id: str | None = None,
    ) -> BookRagContext:
        generation_ctx = await self._context_generation(
            ctx,
            envelope,
            source_id=source_id,
        )
        await self._generation_coordinator()._assert_generation_lease(generation_ctx)
        if generation_ctx.generation_claim_acquired and not isinstance(
            getattr(ctx, _GENERATION_LEASE_TASK_ATTR, None), asyncio.Task
        ):
            await self._start_context_generation_heartbeat(ctx, generation_ctx)
        return generation_ctx

    async def _start_context_generation_heartbeat(
        self,
        ctx: Context[IngestState],
        generation_ctx: BookRagContext,
    ) -> None:
        """Keep the checkpointed owner alive and stop LI workers on fence loss."""
        if not generation_ctx.generation_claim_acquired:
            return

        async def stop_workflow() -> None:
            await ctx.shutdown()

        task = asyncio.create_task(
            self._generation_coordinator()._maintain_generation_lease(
                generation_ctx,
                on_lost=stop_workflow,
            )
        )
        setattr(ctx, _GENERATION_LEASE_TASK_ATTR, task)

    async def _cleanup_context_generation_lease(
        self,
        ctx: Context[IngestState],
    ) -> None:
        """Stop heartbeats and release a canceled checkpointed owner."""
        await self._stop_context_generation_heartbeat(ctx)
        envelope = await ctx.store.get("envelope", default=None)
        if not isinstance(envelope, IntakeEnvelope):
            return
        generation_ctx = await self._context_generation(ctx, envelope)
        if generation_ctx.generation_claim_acquired:
            await self._generation_coordinator()._release_generation_lease(generation_ctx)

    async def _stop_context_generation_heartbeat(
        self,
        ctx: Context[IngestState],
    ) -> None:
        """Stop lease renewal before fenced activation or terminal cleanup."""
        heartbeat = getattr(ctx, _GENERATION_LEASE_TASK_ATTR, None)
        if isinstance(heartbeat, asyncio.Task):
            setattr(ctx, _GENERATION_LEASE_TASK_ATTR, None)
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

    async def _fail_context_generation(
        self,
        ctx: Context[IngestState],
        envelope: IntakeEnvelope,
        exc: Exception,
    ) -> None:
        """Persist retry-safe terminal state for an unexpected step exception."""
        if isinstance(exc, (IngestGateRejected, IngestGenerationClaimed)):
            return
        generation_id = await ctx.store.get("generation_id", default=None)
        source_id = await ctx.store.get("source_id", default="")
        if generation_id is None or not source_id:
            return
        try:
            generation_ctx = await self._context_generation(
                ctx,
                envelope,
                source_id=str(source_id),
            )

            await self._generation_coordinator()._fail_generation(generation_ctx, str(exc))
        except Exception as persist_exc:  # noqa: BLE001
            log.error(
                "book_rag.workflow_li.generation_failure_persist_failed",
                generation_id=generation_id,
                error_type=type(persist_exc).__name__,
            )

    async def _abort_context_publication(self, ctx: Context[IngestState], reason: str) -> None:
        """Abort only provable no-write jobs; the store preserves uncertain writes."""
        abort = getattr(self.qdrant_store, "abort_source_revision", None)
        if not callable(abort):
            return
        for key in ("sterile_publication_job", "publication_job"):
            job = await ctx.store.get(key, default=None)
            if not isinstance(job, SourcePublicationJob):
                continue
            try:
                await abort(job, reason)
            except Exception as persist_exc:  # noqa: BLE001 - preserve the triggering failure
                log.error(
                    "book_rag.workflow_li.source_publication_abort_failed",
                    job_id=job.job_id,
                    error_type=type(persist_exc).__name__,
                )

    def run(
        self,
        ctx: LlamaIndexContext[Any] | None = None,
        stepwise: bool = False,
        checkpoint_callback: Any | None = None,
        start_event: StartEvent | None = None,
        **kwargs: Any,
    ) -> Any:
        if ctx is None:
            ctx = _workflow_context_for_run(self, f"workflow-{uuid.uuid4()}")
        elif not getattr(ctx, _WORKFLOW_RUN_ID_KEY, None):
            run_id = getattr(ctx, "run_id", None) or f"workflow-{uuid.uuid4()}"
            setattr(ctx, _WORKFLOW_RUN_ID_KEY, str(run_id))
        if start_event is None:
            envelope = kwargs.pop("envelope", None)
            if not isinstance(envelope, IntakeEnvelope):
                raise TypeError("BookRagWorkflow.run requires envelope=IntakeEnvelope")
            start_event = IngestStartEvent(envelope=envelope)
        handler = super().run(
            ctx=ctx,
            stepwise=stepwise,
            checkpoint_callback=checkpoint_callback,
            start_event=start_event,
            **kwargs,
        )
        return _attach_run_resource_cleanup(handler, self)

    @step
    async def register_source(
        self, ctx: Context[IngestState], ev: IngestStartEvent
    ) -> RegisteredEvent | StopEvent:
        envelope = ev.envelope
        await ctx.store.set("envelope", envelope)
        ctx.write_event_to_stream(
            IngestProgress(
                envelope_id=str(envelope.id),
                step="register_source",
                detail="canonicalising source id",
            )
        )
        enter_ingest_stage("register_source")  # VOR-1434: stage timing
        if envelope.status is EnvelopeStatus.COMPLETED:
            sid = canonical_source_id(str(envelope.raw_content.get("file_hash", "")))
            return StopEvent(
                result=PipelineOutput(
                    source_id=sid,
                    status=EnvelopeStatus.COMPLETED,
                    qdrant_point_ids=[],
                    raw_point_ids=[],
                    sterile_point_ids=[],
                    smoke_decision=None,
                    chunk_count=0,
                    idempotent_noop=True,
                )
            )
        try:
            envelope.status = EnvelopeStatus.PROCESSING
            self._raise_if_advanced_layers_enabled()
            if not object_is_verified(envelope.raw_content):
                raise ValueError(
                    "book_rag.register_source: authoritative Garage verification is required"
                )
            file_hash = str(envelope.raw_content.get("file_hash", "")).lower().strip()
            if not file_hash:
                raise ValueError(
                    "book_rag.register_source: envelope.raw_content.file_hash is required"
                )
            source_id = canonical_source_id(file_hash)
            coordinator = self._generation_coordinator()
            if not _force_reingest() and await coordinator._source_already_ingested(source_id):
                envelope.status = EnvelopeStatus.COMPLETED
                return StopEvent(
                    result=PipelineOutput(
                        source_id=source_id,
                        status=EnvelopeStatus.COMPLETED,
                        qdrant_point_ids=[],
                        raw_point_ids=[],
                        sterile_point_ids=[],
                        smoke_decision=None,
                        chunk_count=0,
                        idempotent_noop=True,
                        generation_state=(
                            IngestGenerationState.ACTIVE if _ingest_generations_enabled() else None
                        ),
                    )
                )
            workflow_run_id = await _context_run_id(ctx)
            generation_ctx = BookRagContext(
                envelope=envelope,
                config=self.config,
                source_id=source_id,
            )
            publication_job = await coordinator._admit_source_publication(
                generation_ctx, self.raw_collection or RAW_COLLECTION
            )
            if _ingest_generations_enabled():
                try:
                    await coordinator._start_generation(generation_ctx)
                except asyncio.CancelledError:
                    await coordinator._release_generation_lease(generation_ctx)
                    raise
                if generation_ctx.generation_already_active:
                    envelope.status = EnvelopeStatus.COMPLETED
                    return StopEvent(
                        result=PipelineOutput(
                            source_id=source_id,
                            status=EnvelopeStatus.COMPLETED,
                            qdrant_point_ids=[],
                            raw_point_ids=[],
                            sterile_point_ids=[],
                            smoke_decision=None,
                            chunk_count=0,
                            idempotent_noop=True,
                            generation_id=generation_ctx.generation_id,
                            generation_state=IngestGenerationState.ACTIVE,
                        )
                    )
            async with ctx.store.edit_state() as state:
                state.workflow_run_id = workflow_run_id
                state.source_id = source_id
                state.generation_id = generation_ctx.generation_id
                state.previous_generation_id = generation_ctx.previous_generation_id
                state.generation_owner_token = generation_ctx.generation_owner_token
                state.generation_claim_acquired = generation_ctx.generation_claim_acquired
                state.publication_job = publication_job
            await self._start_context_generation_heartbeat(ctx, generation_ctx)
            return RegisteredEvent(source_id=source_id)
        except IngestGenerationClaimed:
            envelope.status = EnvelopeStatus.PROCESSING
            raise
        except asyncio.CancelledError:
            await self._abort_context_publication(ctx, "workflow step cancelled")
            raise
        except Exception as exc:
            _mark_failed(envelope, exc)
            await self._abort_context_publication(ctx, str(exc))
            await self._fail_context_generation(ctx, envelope, exc)
            raise

    @step
    async def parse(self, ctx: Context[IngestState], ev: RegisteredEvent) -> ParsedEvent:
        envelope = await _context_envelope(ctx)
        ctx.write_event_to_stream(
            IngestProgress(envelope_id=str(envelope.id), step="parse", detail="docling")
        )
        enter_ingest_stage("parse")  # VOR-1434: stage timing
        try:
            await self._assert_context_generation_lease(
                ctx,
                envelope,
                source_id=ev.source_id,
            )
            garage_key = envelope.raw_content.get("garage_key")
            if not garage_key:
                raise ValueError("book_rag.parse: envelope.raw_content.garage_key is required")
            parse_meta = cast(
                ParseMeta,
                await do_parse_document(
                    envelope_id=str(envelope.id),
                    garage_key=str(garage_key),
                    source_id=ev.source_id,
                    garage=self.garage,
                    bucket=self.bucket,
                    parser_router=self.parser_router,
                ),
            )
            raw = await self.garage.get_bytes(self.bucket, parse_meta["parsed_garage_key"])
            parsed = ParsedDocument.model_validate_json(raw)
            return ParsedEvent(source_id=ev.source_id, parse_meta=parse_meta, parsed=parsed)
        except asyncio.CancelledError:
            await self._abort_context_publication(ctx, "workflow step cancelled")
            raise
        except Exception as exc:
            _mark_failed(envelope, exc)
            await self._abort_context_publication(ctx, str(exc))
            await self._fail_context_generation(ctx, envelope, exc)
            raise

    @step
    async def fan_out_images(
        self, ctx: Context[IngestState], ev: ParsedEvent
    ) -> VisualEvent | DescribeImageEvent | None:
        envelope = await _context_envelope(ctx)
        ctx.write_event_to_stream(
            IngestProgress(
                envelope_id=str(envelope.id),
                step="visual",
                detail="describing images",
            )
        )
        enter_ingest_stage("visual")  # VOR-1434: stage timing
        try:
            await self._assert_context_generation_lease(
                ctx,
                envelope,
                source_id=ev.source_id,
            )
            composites: list[CompositeChunk] = []
            if not self.config.enable_visual_extraction:
                ctx.write_event_to_stream(
                    ImageDescribedEvent(
                        envelope_id=str(envelope.id),
                        image_index=0,
                        total=len(ev.parsed.images),
                    )
                )
                return VisualEvent(
                    source_id=ev.source_id,
                    parse_meta=ev.parse_meta,
                    parsed=ev.parsed,
                    composites=composites,
                )
            document_metadata = await self._book_metadata(
                BookRagContext(
                    envelope=envelope,
                    config=self.config,
                    source_id=ev.source_id,
                    parse_meta=dict(ev.parse_meta),
                    parsed=ev.parsed,
                )
            )
            if not self.config.enable_visual_description:
                composites = await build_extraction_composites(
                    ev.parsed,
                    document_metadata=document_metadata,
                )
                ctx.write_event_to_stream(
                    ImageDescribedEvent(
                        envelope_id=str(envelope.id),
                        image_index=0,
                        total=len(ev.parsed.images),
                    )
                )
                return VisualEvent(
                    source_id=ev.source_id,
                    parse_meta=ev.parse_meta,
                    parsed=ev.parsed,
                    composites=composites,
                )
            parsed_garage_key = ev.parse_meta.get("parsed_garage_key")
            if not parsed_garage_key:
                raise RuntimeError("book_rag.visual: parsed_garage_key missing from parse_meta")
            checkpoint = Checkpoint(
                self.garage,
                self.bucket,
                workflow_run_id=await _context_run_id(ctx),
            )
            descriptions: dict[str, object] = await checkpoint.load()
            if self.figures is None:
                raise RuntimeError("book_rag.visual: figure store not configured")
            jobs, eligible_image_ids, skipped_by_reason = await _vlm_image_jobs(
                self.figures,
                ev.parsed,
                existing_descriptions=descriptions,
                document_metadata=document_metadata,
            )
            if not eligible_image_ids:
                await checkpoint.finalize()
                composites, _ = await load_visual_composites(
                    self.garage,
                    self.bucket,
                    ev.parsed,
                    descriptions_garage_key=None,
                    document_metadata=document_metadata,
                )
                ctx.write_event_to_stream(
                    ImageDescribedEvent(
                        envelope_id=str(envelope.id),
                        image_index=0,
                        total=len(ev.parsed.images),
                    )
                )
                return VisualEvent(
                    source_id=ev.source_id,
                    parse_meta=ev.parse_meta,
                    parsed=ev.parsed,
                    composites=composites,
                )
            job_map = {
                job.image.image_id: {
                    "image_ref": job.image.ref,
                    "image_role": job.image_role,
                    "element_index": job.image.element_index,
                }
                for job in jobs
            }
            await ctx.store.set("visual_jobs", job_map)
            await ctx.store.set("visual_parsed", ev.parsed)
            await ctx.store.set("visual_source_id", ev.source_id)
            await ctx.store.set("visual_parse_meta", ev.parse_meta)
            await ctx.store.set("visual_envelope_id", str(envelope.id))
            await ctx.store.set("visual_eligible_ids", sorted(eligible_image_ids))
            await ctx.store.set("visual_skipped", dict(skipped_by_reason))
            await ctx.store.set("visual_document_metadata", document_metadata)
            n_pending = len(jobs)
            if n_pending == 0:
                await checkpoint.finalize()
                output_key = f"parsed/{envelope.id}/vlm_descriptions.json"
                eligible_descriptions = {
                    image_id: descriptions[image_id]
                    for image_id in sorted(eligible_image_ids)
                    if image_id in descriptions
                }
                if eligible_descriptions:
                    await self.garage.put_bytes(
                        self.bucket,
                        output_key,
                        json.dumps(eligible_descriptions).encode("utf-8"),
                        "application/json",
                    )
                    descriptions_garage_key = output_key
                else:
                    descriptions_garage_key = None
                composites, _ = await load_visual_composites(
                    self.garage,
                    self.bucket,
                    ev.parsed,
                    descriptions_garage_key=descriptions_garage_key,
                    document_metadata=document_metadata,
                )
                ctx.write_event_to_stream(
                    ImageDescribedEvent(
                        envelope_id=str(envelope.id),
                        image_index=0,
                        total=len(ev.parsed.images),
                    )
                )
                return VisualEvent(
                    source_id=ev.source_id,
                    parse_meta=ev.parse_meta,
                    parsed=ev.parsed,
                    composites=composites,
                )
            setattr(ctx, _VISUAL_VLM_SERVICE_ATTR, self.vlm_service_factory())
            await ctx.store.set("n_pending", n_pending)
            for job in jobs:
                ctx.send_event(
                    DescribeImageEvent(
                        image_id=job.image.image_id,
                        image_ref=job.image.ref,
                        image_role=job.image_role,
                        element_index=job.image.element_index,
                    )
                )
            return None
        except asyncio.CancelledError:
            await self._abort_context_publication(ctx, "workflow step cancelled")
            raise
        except Exception as exc:
            _mark_failed(envelope, exc)
            await self._abort_context_publication(ctx, str(exc))
            await self._fail_context_generation(ctx, envelope, exc)
            raise

    @step(num_workers=_VLM_CONCURRENCY)
    async def describe_one(
        self, ctx: Context[IngestState], ev: DescribeImageEvent
    ) -> ImageDescriptionResult:
        envelope = await _context_envelope(ctx)
        try:
            envelope_id = await ctx.store.get("visual_envelope_id", default="")
            parsed = await ctx.store.get("visual_parsed")
            if not isinstance(parsed, ParsedDocument):
                raise TypeError("visual_parsed not found in context")
            image = None
            for img in parsed.images:
                if img.image_id == ev.image_id:
                    image = img
                    break
            if image is None:
                raise ValueError(
                    f"book_rag.visual: image {ev.image_id} not found in parsed document"
                )
            if self.figures is None:
                raise RuntimeError("book_rag.visual: figure store not configured")
            image_bytes = await asyncio.to_thread(self.figures.open_bytes, ev.image_ref)
            document_metadata = await ctx.store.get("visual_document_metadata", default={})
            if not isinstance(document_metadata, dict):
                document_metadata = {}
            job = VlmImageJob(
                image=image,
                image_bytes=image_bytes,
                image_role=ev.image_role,
            )
            checkpoint = Checkpoint(
                self.garage,
                self.bucket,
                workflow_run_id=await _context_run_id(ctx),
            )
            vlm_settings = getattr(self.settings, "vlm", None)
            max_tokens = _parse_int_env(
                "VO_RAG_VLM_MAX_TOKENS",
                default=int(getattr(vlm_settings, "max_tokens", 1024)),
            )
            deadline_s = _parse_float_env(
                "VO_RAG_VLM_PER_IMAGE_DEADLINE_S",
                default=float(getattr(vlm_settings, "per_image_deadline_s", 300.0)),
            )
            vlm_service = await _context_visual_vlm_service(ctx, self)
            evidence = await _describe_one_image(
                envelope_id=envelope_id,
                job=job,
                parsed=parsed,
                vlm_service=vlm_service,
                checkpoint=checkpoint,
                max_tokens=max_tokens,
                deadline_s=deadline_s,
                document_metadata=document_metadata,
            )
            return ImageDescriptionResult(image_id=ev.image_id, evidence=evidence)
        except asyncio.CancelledError:
            await self._abort_context_publication(ctx, "workflow step cancelled")
            raise
        except Exception as exc:
            _mark_failed(envelope, exc)
            await self._abort_context_publication(ctx, str(exc))
            await self._fail_context_generation(ctx, envelope, exc)
            await _close_visual_vlm_service(ctx, self)
            raise

    @step
    async def collect_visual(
        self, ctx: Context[IngestState], ev: ImageDescriptionResult
    ) -> VisualEvent | None:
        n_pending = await ctx.store.get("n_pending", default=0)
        if n_pending == 0:
            raise RuntimeError("book_rag.visual: received image result before fan-out")
        done = ctx.collect_events(
            ev,
            [ImageDescriptionResult] * n_pending,
            buffer_id=await _collect_buffer_id(ctx, "book_rag.visual.collect_visual"),
        )
        if done is None:
            return None
        envelope = await _context_envelope(ctx)
        try:
            envelope_id = await ctx.store.get("visual_envelope_id", default="")
            parsed = await ctx.store.get("visual_parsed")
            if not isinstance(parsed, ParsedDocument):
                raise TypeError("visual_parsed not found in context")
            source_id = await ctx.store.get("visual_source_id", default="")
            parse_meta = cast(ParseMeta, await ctx.store.get("visual_parse_meta", default={}))
            document_metadata = await ctx.store.get("visual_document_metadata", default={})
            if not isinstance(document_metadata, dict):
                document_metadata = {}
            eligible_ids = set(await ctx.store.get("visual_eligible_ids", default=[]))
            checkpoint = Checkpoint(
                self.garage,
                self.bucket,
                workflow_run_id=await _context_run_id(ctx),
            )
            descriptions: dict[str, object] = await checkpoint.load()
            for result in done:
                if result.evidence is not None:
                    descriptions[result.image_id] = result.evidence
            vlm_processed_count = sum(1 for image_id in eligible_ids if image_id in descriptions)
            describe_yield = vlm_processed_count / len(eligible_ids) if eligible_ids else 1.0
            min_describe_yield = _parse_float_env("VO_RAG_MIN_VLM_DESCRIBE_YIELD", default=0.90)
            if eligible_ids and describe_yield < min_describe_yield:
                raise RuntimeError(
                    f"book_rag VLM processed yield {vlm_processed_count}/{len(eligible_ids)} "
                    f"({describe_yield:.2%}) below threshold {min_describe_yield:.0%}"
                )
            output_key = f"parsed/{envelope_id}/vlm_descriptions.json"
            eligible_descriptions = {
                image_id: descriptions[image_id]
                for image_id in sorted(eligible_ids)
                if image_id in descriptions
            }
            if eligible_descriptions:
                await self.garage.put_bytes(
                    self.bucket,
                    output_key,
                    json.dumps(eligible_descriptions).encode("utf-8"),
                    "application/json",
                )
                descriptions_garage_key = output_key
            else:
                descriptions_garage_key = None
            await checkpoint.finalize()
            composites, _ = await load_visual_composites(
                self.garage,
                self.bucket,
                parsed,
                descriptions_garage_key=descriptions_garage_key,
                document_metadata=document_metadata,
            )
            ctx.write_event_to_stream(
                ImageDescribedEvent(
                    envelope_id=envelope_id,
                    image_index=0,
                    total=len(parsed.images),
                )
            )
            return VisualEvent(
                source_id=source_id,
                parse_meta=parse_meta,
                parsed=parsed,
                composites=composites,
            )
        except asyncio.CancelledError:
            await self._abort_context_publication(ctx, "workflow step cancelled")
            raise
        except Exception as exc:
            _mark_failed(envelope, exc)
            await self._abort_context_publication(ctx, str(exc))
            await self._fail_context_generation(ctx, envelope, exc)
            raise
        finally:
            await _close_visual_vlm_service(ctx, self)

    @step
    async def chunk_embed(self, ctx: Context[IngestState], ev: VisualEvent) -> EmbeddedEvent:
        envelope = await _context_envelope(ctx)
        ctx.write_event_to_stream(
            IngestProgress(
                envelope_id=str(envelope.id),
                step="chunk_embed",
                detail="chunk + embed + upsert",
            )
        )
        enter_ingest_stage("chunk_embed")  # VOR-1434: stage timing
        try:
            book_metadata = await self._book_metadata(
                BookRagContext(
                    envelope=envelope,
                    config=self.config,
                    source_id=ev.source_id,
                    parse_meta=dict(ev.parse_meta),
                    parsed=ev.parsed,
                    composites=ev.composites,
                )
            )
            chunks = do_build_chunks(
                ev.parsed,
                composites=ev.composites,
                book_metadata=book_metadata,
                settings=self.settings,
                embedding_client=self.embedding_client,
            )
            binding = ev.parse_meta.get("parse_provenance")
            if isinstance(binding, dict):
                for chunk in chunks:
                    chunk.metadata["parse_provenance"] = dict(binding)
            raw_embeddings = await do_embed_raw(chunks, self.embedding_client)
            generation_ctx = await self._assert_context_generation_lease(
                ctx,
                envelope,
                source_id=ev.source_id,
            )
            generation_id = generation_ctx.generation_id
            publication_job = await ctx.store.get("publication_job", default=None)
            raw_write = await do_upsert_raw(
                chunks,
                raw_embeddings,
                self.qdrant_store,
                ev.source_id,
                collection=self.raw_collection,
                trust_level="trusted",
                generation_id=generation_id,
                publication_job=(
                    publication_job if isinstance(publication_job, SourcePublicationJob) else None
                ),
                ingest_state=(
                    IngestGenerationState.STAGING
                    if generation_id is not None
                    else IngestGenerationState.ACTIVE
                ),
            )
            raw_point_ids = list(raw_write["point_ids"])
            if generation_id is not None:
                actual = await self.qdrant_store.count_source(
                    self.raw_collection or RAW_COLLECTION,
                    ev.source_id,
                    generation_id=generation_id,
                    ingest_state=IngestGenerationState.STAGING,
                )
                if actual != len(chunks):
                    raise RuntimeError(
                        f"raw generation count mismatch: expected={len(chunks)} actual={actual}"
                    )
                await self._generation_coordinator()._record_generation_counts(
                    generation_ctx,
                    expected_raw_points=len(chunks),
                    actual_raw_points=actual,
                )
            async with ctx.store.edit_state() as state:
                state.raw_point_ids = raw_point_ids
                state.chunk_count = len(chunks)
            return EmbeddedEvent(
                source_id=ev.source_id,
                parse_meta=ev.parse_meta,
                chunks=chunks,
                raw_point_ids=raw_point_ids,
                raw_embeddings=raw_embeddings,
            )
        except asyncio.CancelledError:
            await self._abort_context_publication(ctx, "workflow step cancelled")
            raise
        except Exception as exc:
            _mark_failed(envelope, exc)
            await self._abort_context_publication(ctx, str(exc))
            await self._fail_context_generation(ctx, envelope, exc)
            raise

    @step
    async def sterilize(self, ctx: Context[IngestState], ev: EmbeddedEvent) -> SterilizedEvent:
        envelope = await _context_envelope(ctx)
        ctx.write_event_to_stream(
            IngestProgress(
                envelope_id=str(envelope.id),
                step="sterilize",
                detail="sterilising raw chunks",
            )
        )
        enter_ingest_stage("sterilize")  # VOR-1434: stage timing
        try:
            sterile_point_ids: list[str] = []
            generation_ctx = await self._assert_context_generation_lease(
                ctx,
                envelope,
                source_id=ev.source_id,
            )
            generation_id = generation_ctx.generation_id
            if self.config.sterilize_enabled:
                expected_dependencies: dict[str, int | None] | None = None
                if getattr(self.qdrant_store, "source_publication_enabled", False):
                    raw_revision = await self.qdrant_store.active_source_revision(
                        RAW_COLLECTION, ev.source_id
                    )
                    expected_dependencies = {ev.source_id: raw_revision}
                sterile_publication_job = (
                    await self._generation_coordinator()._admit_source_publication(
                        generation_ctx,
                        STERILE_COLLECTION,
                        expected_dependencies=expected_dependencies,
                        producer_recipe={"writer": "book-rag-sterile-representation-v1"},
                    )
                )
                await ctx.store.set("sterile_publication_job", sterile_publication_job)
                written_sterile_ids: list[str] = []
                sterile_chunks = await do_sterilize(
                    raw_chunks=ev.chunks,
                    llm_client=self.llm_client,
                    config=self.config,
                    store=self.qdrant_store,
                    embed=self._embed,
                    generation_id=generation_id,
                    publication_job=sterile_publication_job,
                    expected_dependencies=expected_dependencies,
                    point_ids_out=written_sterile_ids,
                    ingest_state=(
                        IngestGenerationState.STAGING
                        if generation_id is not None
                        else IngestGenerationState.ACTIVE
                    ),
                )
                sterile_point_ids = written_sterile_ids
                if sterile_publication_job is not None and not sterile_chunks:
                    await self._abort_context_publication(
                        ctx, "sterile representation produced no points"
                    )
                if generation_id is not None:
                    actual = await self.qdrant_store.count_source(
                        STERILE_COLLECTION,
                        ev.source_id,
                        generation_id=generation_id,
                        ingest_state=IngestGenerationState.STAGING,
                    )
                    if actual != len(sterile_chunks):
                        raise RuntimeError(
                            "sterile generation count mismatch: "
                            f"expected={len(sterile_chunks)} actual={actual}"
                        )
                    await self._generation_coordinator()._record_generation_counts(
                        generation_ctx,
                        expected_sterile_points=len(sterile_chunks),
                        actual_sterile_points=actual,
                    )
            elif generation_id is not None:
                if (self.raw_collection or RAW_COLLECTION) == RAW_COLLECTION:
                    await self.qdrant_store.delete_generation(
                        STERILE_COLLECTION,
                        ev.source_id,
                        generation_id,
                    )
                await self._generation_coordinator()._record_generation_counts(
                    generation_ctx,
                    expected_sterile_points=0,
                    actual_sterile_points=0,
                )
            async with ctx.store.edit_state() as state:
                state.sterile_point_ids = sterile_point_ids
            return SterilizedEvent(
                source_id=ev.source_id,
                parse_meta=ev.parse_meta,
                chunks=ev.chunks,
                raw_point_ids=ev.raw_point_ids,
                sterile_point_ids=sterile_point_ids,
            )
        except asyncio.CancelledError:
            await self._abort_context_publication(ctx, "workflow step cancelled")
            raise
        except Exception as exc:
            _mark_failed(envelope, exc)
            await self._abort_context_publication(ctx, str(exc))
            await self._fail_context_generation(ctx, envelope, exc)
            raise

    @step
    async def register_artifact(
        self, ctx: Context[IngestState], ev: SterilizedEvent
    ) -> RegisteredArtifactEvent:
        envelope = await _context_envelope(ctx)
        ctx.write_event_to_stream(
            IngestProgress(
                envelope_id=str(envelope.id),
                step="register_artifact",
                detail="persisting artifact",
            )
        )
        enter_ingest_stage("register_artifact")  # VOR-1434: stage timing
        try:
            artifact_ctx = await self._assert_context_generation_lease(
                ctx,
                envelope,
                source_id=ev.source_id,
            )
            artifact_ctx.parse_meta = dict(ev.parse_meta)
            artifact_ctx.chunks = ev.chunks
            artifact_ctx.raw_point_ids = ev.raw_point_ids
            artifact_ctx.sterile_point_ids = ev.sterile_point_ids
            await self._register_artifact(artifact_ctx)
            return RegisteredArtifactEvent(
                source_id=ev.source_id,
                parse_meta=ev.parse_meta,
                chunks=ev.chunks,
                raw_point_ids=ev.raw_point_ids,
                sterile_point_ids=ev.sterile_point_ids,
            )
        except asyncio.CancelledError:
            await self._abort_context_publication(ctx, "workflow step cancelled")
            raise
        except Exception as exc:
            _mark_failed(envelope, exc)
            await self._abort_context_publication(ctx, str(exc))
            await self._fail_context_generation(ctx, envelope, exc)
            raise

    @step
    async def validate(
        self, ctx: Context[IngestState], ev: RegisteredArtifactEvent
    ) -> ValidatedEvent:
        envelope = await _context_envelope(ctx)
        ctx.write_event_to_stream(
            IngestProgress(
                envelope_id=str(envelope.id),
                step="validate",
                detail="running ingestion validator",
            )
        )
        enter_ingest_stage("validate")  # VOR-1434: stage timing
        try:
            validation_ctx = await self._assert_context_generation_lease(
                ctx,
                envelope,
                source_id=ev.source_id,
            )
            validation_ctx.parse_meta = dict(ev.parse_meta)
            validation_ctx.chunks = ev.chunks
            validation_ctx.raw_point_ids = ev.raw_point_ids
            validation_ctx.sterile_point_ids = ev.sterile_point_ids
            await self._run_ingest_validator(validation_ctx)
            validation_ctx.parse_meta["quality_warnings"] = collect_quality_warnings(validation_ctx)
            return ValidatedEvent(
                source_id=ev.source_id,
                parse_meta=cast(ParseMeta, validation_ctx.parse_meta),
                chunks=ev.chunks,
                raw_point_ids=ev.raw_point_ids,
                sterile_point_ids=ev.sterile_point_ids,
                validation_report=validation_ctx.validation_report,
            )
        except asyncio.CancelledError:
            await self._abort_context_publication(ctx, "workflow step cancelled")
            raise
        except Exception as exc:
            _mark_failed(envelope, exc)
            await self._abort_context_publication(ctx, str(exc))
            await self._fail_context_generation(ctx, envelope, exc)
            raise

    @step
    async def retrieval_smoke(self, ctx: Context[IngestState], ev: ValidatedEvent) -> StopEvent:
        envelope = await _context_envelope(ctx)
        ctx.write_event_to_stream(
            IngestProgress(
                envelope_id=str(envelope.id),
                step="retrieval_smoke",
                detail="self-query smoke test",
            )
        )
        enter_ingest_stage("retrieval_smoke")  # VOR-1434: stage timing
        try:
            smoke_ctx = await self._assert_context_generation_lease(
                ctx,
                envelope,
                source_id=ev.source_id,
            )
            smoke_ctx.parse_meta = dict(ev.parse_meta)
            smoke_ctx.chunks = ev.chunks
            smoke_ctx.raw_point_ids = ev.raw_point_ids
            smoke_ctx.sterile_point_ids = ev.sterile_point_ids
            if self.llm_client is not None and self.config.retrieval_smoke_enabled:
                await self._retrieval_smoke(smoke_ctx)
            else:
                # Defer-LLM ingest (or no LLM client): chunks + artifact are
                # already persisted by earlier steps; skip the LLM-dependent
                # self-query smoke gate and complete normally.
                log.info(
                    "book_rag.workflow_li.retrieval_smoke.skipped",
                    envelope_id=str(envelope.id),
                    reason="no_llm_client" if self.llm_client is None else "disabled",
                )
                if smoke_ctx.generation_id is not None:
                    await self._generation_coordinator()._transition_owned_generation(
                        smoke_ctx,
                        from_states=(IngestGenerationState.STAGING,),
                        to_state=IngestGenerationState.VALIDATED,
                        smoke_decision="skipped",
                    )
            if smoke_ctx.generation_id is not None:
                await self._stop_context_generation_heartbeat(ctx)
                await self._generation_coordinator()._assert_generation_lease(smoke_ctx)
                await self._generation_coordinator()._activate_generation(smoke_ctx)
            await self._record_book_quality(smoke_ctx)
            await self._generation_coordinator()._persist_quality_warnings(smoke_ctx)
            envelope.status = EnvelopeStatus.COMPLETED
            async with ctx.store.edit_state() as state:
                state.raw_point_ids = smoke_ctx.raw_point_ids
                state.sterile_point_ids = smoke_ctx.sterile_point_ids
                state.smoke_decision = (
                    smoke_ctx.smoke_result.decision if smoke_ctx.smoke_result else None
                )
            return StopEvent(
                result=PipelineOutput(
                    source_id=ev.source_id,
                    status=envelope.status,
                    qdrant_point_ids=[*smoke_ctx.raw_point_ids, *smoke_ctx.sterile_point_ids],
                    raw_point_ids=smoke_ctx.raw_point_ids,
                    sterile_point_ids=smoke_ctx.sterile_point_ids,
                    smoke_decision=smoke_ctx.smoke_result.decision
                    if smoke_ctx.smoke_result
                    else None,
                    chunk_count=len(ev.chunks),
                    quality_warnings=smoke_ctx.quality_warnings,
                    generation_id=smoke_ctx.generation_id,
                    generation_state=(
                        IngestGenerationState.ACTIVE
                        if smoke_ctx.generation_id is not None
                        else None
                    ),
                )
            )
        except asyncio.CancelledError:
            await self._abort_context_publication(ctx, "workflow step cancelled")
            raise
        except Exception as exc:
            _mark_failed(envelope, exc)
            await self._abort_context_publication(ctx, str(exc))
            await self._fail_context_generation(ctx, envelope, exc)
            raise

    async def _record_book_quality(self, ctx: BookRagContext) -> None:
        """The per-book quality gate for streamed ingests (VOR-1434): the
        dispatcher path records ``metadata.quality`` in
        ``BookRagPipeline._ingest``; the streaming path (API/MCP/``vo-rag
        ingest run``) completes here, so it runs the same check through the
        generation coordinator. Never fails the book: the coordinator's
        check swallows its own errors, and anything else is logged here."""
        try:
            await self._generation_coordinator()._record_book_quality(ctx)
        except Exception as exc:  # noqa: BLE001 - a quality-check bug must never fail the ingest
            log.warning(
                "book_rag.workflow_li.quality_check_failed",
                envelope_id=str(ctx.envelope.id),
                source_id=ctx.source_id,
                error=str(exc),
                exc_info=True,
            )

    async def _book_metadata(self, ctx: BookRagContext) -> dict[str, Any]:
        envelope_metadata = self._envelope_book_metadata(ctx)
        lookup_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if lookup_pool is None:
            return envelope_metadata

        try:
            async with lookup_pool.acquire() as conn:
                source = await resolve_canonical_source(ctx.source_id, conn=conn)
        except Exception as exc:
            log.warning(
                "book_rag.source_identity_lookup_failed",
                source_id=ctx.source_id,
                error=str(exc),
            )
            return envelope_metadata

        if source is None:
            log.warning("book_rag.source_identity_lookup_miss", source_id=ctx.source_id)
            return envelope_metadata

        title = str(source.title or "").strip()
        author = str(source.author or "").strip()
        source_book = f"{title} ({author})" if title and author else (title or source.canonical_id)
        return {
            "source_book": source_book,
            "title": title,
            "author": author or None,
            "domain_tags": envelope_metadata.get("domain_tags", []),
            "license_class": source.license_class,
            "canonical_source_id": source.canonical_id,
            "project_id": source.project_id,
        }

    @staticmethod
    def _envelope_book_metadata(ctx: BookRagContext) -> dict[str, Any]:
        em = ctx.envelope.metadata or {}
        parsed_title = (ctx.parsed.title if ctx.parsed else "") or ""
        title = str(em.get("title") or parsed_title or "").strip()
        author = str(em.get("author") or "").strip()
        raw_tags = em.get("tags") or em.get("domain_tags") or []
        tags = [str(t) for t in raw_tags] if isinstance(raw_tags, list) else []
        source_book = f"{title} ({author})" if title and author else (title or ctx.source_id)
        return {
            "source_book": source_book,
            "title": title,
            "author": author or None,
            "domain_tags": tags,
            "license_class": str(em.get("license_class") or "unlicensed"),
            "canonical_source_id": ctx.source_id,
        }

    async def _register_artifact(self, ctx: BookRagContext) -> None:
        artifact_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if artifact_pool is None:
            log.info(
                "book_rag.artifact_registration_skipped",
                source_id=ctx.source_id,
                reason="no_pg_pool",
            )
            return

        metadata = await self._book_metadata(ctx)
        title = _artifact_title(ctx, metadata)
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        raw_collection = self.raw_collection or RAW_COLLECTION
        content = {"source_id": ctx.source_id}
        if ctx.generation_id is not None or raw_collection != RAW_COLLECTION:
            content["raw_collection"] = raw_collection
        if ctx.generation_id is not None:
            content["generation_id"] = ctx.generation_id
        artifact = KnowledgeArtifact(
            id=_artifact_id(ctx.source_id, raw_collection, ctx.generation_id),
            envelope_id=ctx.envelope.id,
            pipeline=TargetPipeline.BOOK_RAG,
            artifact_type=ArtifactType.EXTRACTION,
            title=title,
            content=content,
            tags=_artifact_tags(metadata),
            priority=ctx.envelope.priority,
            license_class=_artifact_license_class(metadata.get("license_class")),
            status="staging" if ctx.generation_id is not None else "active",
            trust_level=(
                TrustLevel.QUARANTINED if ctx.generation_id is not None else TrustLevel.TRUSTED
            ),
            qdrant_collection=raw_collection,
            qdrant_point_ids=list(ctx.raw_point_ids),
            source_provenance=[
                ProvenanceEntry(
                    source_id=ctx.source_id,
                    source_title=title,
                    source_type=str(ctx.envelope.source_type),
                    contribution=1.0,
                )
            ],
            created_at=now,
            updated_at=now,
        )
        if ctx.generation_id is None:
            await insert_artifact(artifact_pool, artifact)
        else:
            await self._generation_coordinator()._insert_artifact_owned(ctx, artifact)

    # LlamaIndex's Workflow base class owns ``_validate()`` for graph validation,
    # so the ingest validator seam cannot keep the dataclass pipeline's private
    # method name without shadowing the runtime.
    async def _run_ingest_validator(self, ctx: BookRagContext) -> None:
        """Use the shared advisory quality policy for streamed ingestion."""
        await self._generation_coordinator()._audit_validation(
            ctx, validator_factory=IngestionValidator
        )

    async def _retrieval_smoke(self, ctx: BookRagContext) -> None:
        """Use the shared advisory quality policy for streamed ingestion."""
        await self._generation_coordinator()._retrieval_smoke(ctx)

    async def _llm_call(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        resp: LLMResponse = await self.llm_client.complete(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.model_copy(update={"text": _strip_reasoning(resp.text)})

    async def _embed(self, texts: list[str]) -> dict[str, Any]:
        resp: dict[str, Any] = await self.embedding_client.embed(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert=include_colbert_at_ingest(),
        )
        return resp

    async def _retrieve_fn(
        self,
        query: str,
        top_k: int,
        *,
        generation_id: str | None = None,
        ingest_state: str | None = None,
    ) -> list[str]:
        filters: dict[str, Any] | None = None
        if generation_id is not None:
            filters = {"generation_id": generation_id}
        search_kwargs: dict[str, Any] = {
            "collection": self.raw_collection or RAW_COLLECTION,
            "top_k": top_k,
            "prefetch_depth": 100,
        }
        if filters is not None:
            search_kwargs["filters"] = filters
        if ingest_state is not None:
            search_kwargs["ingest_state_override"] = str(ingest_state)
        results = await self.retriever.hybrid_search(query, **search_kwargs)
        out: list[str] = []
        for result in results:
            payload = getattr(result, "payload", None) or {}
            chunk_id = payload.get("chunk_id") or getattr(result, "id", None)
            if chunk_id is not None:
                out.append(str(chunk_id))
        return out

    def _raise_if_advanced_layers_enabled(self) -> None:
        enabled = [
            name
            for name, on in (
                ("summary_tree", self.config.enable_summary_tree),
                ("raptor", self.config.enable_raptor),
                ("dsi", self.config.enable_dsi),
                ("graph", self.config.enable_graph),
            )
            if on
        ]
        if not enabled:
            return
        raise RuntimeError(
            "book_rag.advanced_understanding: inline ingestion cannot build "
            f"layers {enabled} — run AdvancedBuildWorkflow after ingest "
            "(it recovers title/author/category_ids from persisted chunks and "
            "supplies BuildConfig/DsiBuildConfig/GraphBuildConfig/RaptorBuildConfig)."
        )


@dataclass
class BookRagWorkflowCheckpointRunner:
    """Run-scoped wrapper around LlamaIndex ``WorkflowCheckpointer``."""

    workflow: BookRagWorkflow
    checkpointer: WorkflowCheckpointer
    checkpoint_serializer: BaseSerializer

    @classmethod
    def create(
        cls,
        workflow: BookRagWorkflow,
        *,
        checkpoint_serializer: BaseSerializer | None = None,
    ) -> BookRagWorkflowCheckpointRunner:
        # Workflow checkpoints are pickle payloads; keep the name honest because
        # checkpoint load is deserialization of trusted local state, not JSON.
        serializer = checkpoint_serializer or CheckpointPickleSerializer()
        return cls(
            workflow=workflow,
            checkpointer=WorkflowCheckpointer(
                workflow=workflow,
                checkpoint_serializer=serializer,
            ),
            checkpoint_serializer=serializer,
        )

    def latest_checkpoint(self, workflow_run_id: str) -> WorkflowCheckpoint | None:
        checkpoints = self.checkpointer.checkpoints.get(workflow_run_id, [])
        return checkpoints[-1] if checkpoints else None

    def _checkpoint_callback_for_run(self, workflow_run_id: str) -> Any:
        callback = self.checkpointer.new_checkpoint_callback_for_run()

        async def _create_checkpoint(
            run_id: str,
            last_completed_step: str | None,
            input_ev: Event | None,
            output_ev: Event | None,
            ctx: LlamaIndexContext,
        ) -> None:
            await callback(
                run_id=workflow_run_id,
                last_completed_step=last_completed_step,
                input_ev=input_ev,
                output_ev=output_ev,
                ctx=ctx,
            )

        return _create_checkpoint

    def run_handler(self, *, envelope: IntakeEnvelope, workflow_run_id: str) -> Any:
        ctx = _workflow_context_for_run(self.workflow, workflow_run_id)
        handler = self.workflow.run(
            ctx=ctx,
            checkpoint_callback=self._checkpoint_callback_for_run(workflow_run_id),
            envelope=envelope,
        )
        return handler

    def resume_from_checkpoint(
        self,
        *,
        checkpoint: WorkflowCheckpoint,
        envelope: IntakeEnvelope,
        workflow_run_id: str,
    ) -> Any:
        handler = self.workflow.run_from(
            checkpoint=checkpoint,
            ctx_serializer=self.checkpoint_serializer,
            checkpoint_callback=self._checkpoint_callback_for_run(workflow_run_id),
            envelope=envelope,
        )
        return handler

    def resume_from_last(self, *, envelope: IntakeEnvelope, workflow_run_id: str) -> Any:
        checkpoint = self.latest_checkpoint(workflow_run_id)
        if checkpoint is None:
            raise RuntimeError(
                f"book_rag.workflow_checkpoint: no checkpoint found for {workflow_run_id}"
            )
        return self.resume_from_checkpoint(
            checkpoint=checkpoint,
            envelope=envelope,
            workflow_run_id=workflow_run_id,
        )

    async def run(
        self,
        *,
        envelope: IntakeEnvelope,
        workflow_run_id: str,
        resume_on_error: bool = True,
    ) -> PipelineOutput:
        handler = self.run_handler(envelope=envelope, workflow_run_id=workflow_run_id)
        try:
            result = await handler
        except Exception as exc:
            if not resume_on_error:
                raise
            log.warning(
                "book_rag_workflow_resume_after_error",
                workflow_run_id=workflow_run_id,
                error=str(exc),
                exc_info=True,
            )
            checkpoint = self.latest_checkpoint(workflow_run_id)
            if checkpoint is None:
                raise
            result = await self.resume_from_checkpoint(
                checkpoint=checkpoint,
                envelope=envelope,
                workflow_run_id=workflow_run_id,
            )
        if not isinstance(result, PipelineOutput):
            raise TypeError("BookRagWorkflow checkpointed run did not return PipelineOutput")
        return result


async def run_book_rag_workflow_checkpointed(
    *,
    envelope: IntakeEnvelope,
    workflow_run_id: str,
    workflow: BookRagWorkflow | None = None,
    checkpoint_serializer: BaseSerializer | None = None,
    resume_on_error: bool = True,
    **workflow_kwargs: Any,
) -> tuple[PipelineOutput, BookRagWorkflowCheckpointRunner]:
    """Run ``BookRagWorkflow`` with step checkpoints keyed by ``workflow_run_id``."""

    if workflow is None:
        workflow = BookRagWorkflow(**workflow_kwargs)
    elif workflow_kwargs:
        raise ValueError("workflow_kwargs cannot be supplied when workflow is already built")

    runner = BookRagWorkflowCheckpointRunner.create(
        workflow,
        checkpoint_serializer=checkpoint_serializer,
    )
    result = await runner.run(
        envelope=envelope,
        workflow_run_id=workflow_run_id,
        resume_on_error=resume_on_error,
    )
    return result, runner


async def _context_envelope(ctx: Context[IngestState]) -> IntakeEnvelope:
    envelope = await ctx.store.get("envelope")
    if not isinstance(envelope, IntakeEnvelope):
        raise TypeError("BookRagWorkflow context envelope is not an IntakeEnvelope")
    return envelope


async def _context_visual_vlm_service(
    ctx: Context[IngestState],
    workflow: BookRagWorkflow,
) -> VLMServiceLike:
    service = getattr(ctx, _VISUAL_VLM_SERVICE_ATTR, None)
    if service is None:
        service = workflow.vlm_service_factory()
        setattr(ctx, _VISUAL_VLM_SERVICE_ATTR, service)
    return service


async def _close_visual_vlm_service(
    ctx: Context[IngestState],
    workflow: BookRagWorkflow,
) -> None:
    service = getattr(ctx, _VISUAL_VLM_SERVICE_ATTR, None)
    if service is None:
        return
    setattr(ctx, _VISUAL_VLM_SERVICE_ATTR, None)
    close = getattr(service, "aclose", None)
    if close is not None:
        try:
            result = close()
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "book_rag.workflow_li.visual_vlm_close_failed",
                error=str(exc),
            )


class _RunCleanupHandler:
    def __init__(self, handler: Any, workflow: BookRagWorkflow) -> None:
        self._handler = handler
        self._workflow = workflow
        self._cleanup_complete = False
        self._cleanup_task: asyncio.Task[None] | None = None
        add_done_callback = getattr(handler, "add_done_callback", None)
        if add_done_callback is not None:
            add_done_callback(self._on_handler_done)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handler, name)

    def stream_events(self) -> Any:
        return self._handler.stream_events()

    async def _cleanup_once(self) -> None:
        if self._cleanup_complete:
            return
        self._cleanup_complete = True
        ctx = getattr(self._handler, "ctx", None)
        if ctx is None:
            return
        typed_ctx = cast(Context[IngestState], ctx)
        await _close_visual_vlm_service(typed_ctx, self._workflow)
        await self._workflow._cleanup_context_generation_lease(typed_ctx)

    def _on_handler_done(self, _handler: Any) -> None:
        # Hold the reference: an unreferenced pending task is GC-eligible and
        # its exceptions go unretrieved (asyncio docs).
        self._cleanup_task = asyncio.create_task(self._cleanup_once())

    def __await__(self) -> Any:
        async def _run() -> Any:
            try:
                return await self._handler
            finally:
                await self._cleanup_once()

        return _run().__await__()


def _attach_run_resource_cleanup(handler: Any, workflow: BookRagWorkflow) -> _RunCleanupHandler:
    return _RunCleanupHandler(handler, workflow)


def _mark_failed(envelope: IntakeEnvelope, exc: Exception) -> None:
    if isinstance(exc, IngestGenerationClaimed):
        envelope.status = EnvelopeStatus.PROCESSING
        return
    envelope.status = EnvelopeStatus.FAILED
    envelope.error = str(exc)


__all__ = [
    "BookRagWorkflow",
    "BookRagWorkflowCheckpointRunner",
    "PipelineOutput",
    "IngestState",
    "IngestStartEvent",
    "ParseMeta",
    "RegisteredEvent",
    "ParsedEvent",
    "VisualEvent",
    "EmbeddedEvent",
    "SterilizedEvent",
    "RegisteredArtifactEvent",
    "ValidatedEvent",
    "run_book_rag_workflow_checkpointed",
]
