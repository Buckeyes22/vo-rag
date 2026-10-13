"""book_rag ingestion DAG orchestrator.

This is a plain async Python class, NOT a Hatchet workflow. The work is split
into focused step modules (``steps/parse.py``, ``steps/chunk_embed.py``,
``steps/sterilize.py``) that this class sequences, ordered by CORRECTNESS:

    register_source -> parse -> chunk_embed -> sterilize
        -> advanced_understanding -> register_artifact -> validate -> retrieval_smoke

Each step is a coroutine sharing a typed :class:`BookRagContext`. Steps depend
only on injected seams (Garage, QdrantStore, EmbeddingClient, LLMClient,
HybridRetriever, ParserRouter, optional Postgres pool) so the whole DAG is unit
-testable with fakes — no GPU, no live services.

Idempotency: re-running a COMPLETED envelope is a no-op. On any step failure the
envelope is marked FAILED (with the error) and the exception re-raised.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import structlog

from vo_rag.core.chunking.composite_emitter import CompositeChunk
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.corpus.health import book_quality_metadata, collect_stats
from vo_rag.core.db.queries.artifacts import insert_artifact
from vo_rag.core.db.queries.ingest_generations import (
    activate_generation_record_locked,
    count_active_generation_artifacts,
    create_generation,
    generation_claim_is_current,
    get_active_generation,
    get_active_generation_id,
    get_generation,
    get_superseded_generation_ids,
    record_generation_cleanup,
    refresh_generation_claim,
    release_generation_claim,
    transition_generation,
    update_generation_counts,
)
from vo_rag.core.embedding.point_vectors import include_colbert_at_ingest
from vo_rag.core.idempotency import deterministic_hex, deterministic_uuid
from vo_rag.core.ingest.object_verification import object_is_verified
from vo_rag.core.ingest.source_publication import SourcePublicationJob
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models import ulid_new
from vo_rag.core.models.artifact import (
    ArtifactType,
    KnowledgeArtifact,
    LicenseClass,
    ProvenanceEntry,
    TrustLevel,
)
from vo_rag.core.models.envelope import EnvelopeStatus, IntakeEnvelope, TargetPipeline
from vo_rag.core.models.ingest_generation import (
    IngestGenerationState,
    SourceIngestGeneration,
)
from vo_rag.core.observability.metrics import (
    INGEST_METRICS_FLUSH_INTERVAL_S,
    current_ingest_recorder,
    ingest_recording,
    persist_ingest_metrics,
    start_periodic_ingest_flush,
    stop_periodic_ingest_flush,
)
from vo_rag.core.observability.otel import get_tracer
from vo_rag.core.parsing.contract import ParsedDocument
from vo_rag.core.storage.qdrant import deterministic_point_id
from vo_rag.core.summary.builder import _strip_reasoning
from vo_rag.pipelines.book_rag import retrieval_smoke as smoke
from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.published_points import current_published_ids
from vo_rag.pipelines.book_rag.quality_warnings import collect_quality_warnings, quality_warning
from vo_rag.pipelines.book_rag.source_identity import canonical_source_id, resolve_canonical_source
from vo_rag.pipelines.book_rag.steps.chunk_embed import (
    do_build_chunks,
    do_embed_raw,
    do_upsert_raw,
)
from vo_rag.pipelines.book_rag.steps.parse import do_parse_document
from vo_rag.pipelines.book_rag.validator import (
    BUCKET_ARTIFACTS,
    IngestionValidator,
    ValidationReport,
)
from vo_rag.pipelines.book_rag.visual_sidecar import (
    build_extraction_composites,
    describe_visual_images,
    load_visual_composites,
)

RAW_COLLECTION = "vo_rag_books_raw_v1"
STERILE_COLLECTION = "vo_rag_books_sterile_v1"
log = structlog.get_logger(__name__)
GENERATION_SCHEMA_VERSION = "book-rag-points-v1"
GENERATION_CLAIM_LEASE = timedelta(minutes=15)
GENERATION_CLAIM_HEARTBEAT_SECONDS = 60.0
QUALITY_AUDIT_TIMEOUT_SECONDS = 10.0


@asynccontextmanager
async def _reconciled_activation_transaction(
    pool: Any,
    on_error: Callable[[], Awaitable[bool]],
) -> AsyncIterator[Any]:
    """Run a transaction and reconcile external state on body/commit failure."""
    try:
        async with pool.acquire() as conn, conn.transaction():
            yield conn
    except BaseException:
        authoritative_commit = False
        reconcile_task: asyncio.Future[bool] = asyncio.ensure_future(on_error())
        try:
            authoritative_commit = await asyncio.shield(reconcile_task)
        except asyncio.CancelledError:
            # Preserve the original error/cancellation, but do not let a second
            # cancellation abandon the authoritative Postgres reconciliation.
            try:
                authoritative_commit = await reconcile_task
            except BaseException as reconcile_exc:  # noqa: BLE001
                log.error(
                    "book_rag.generation_activation_reconciliation_failed",
                    error_type=type(reconcile_exc).__name__,
                )
        except BaseException as reconcile_exc:  # noqa: BLE001
            log.error(
                "book_rag.generation_activation_reconciliation_failed",
                error_type=type(reconcile_exc).__name__,
            )
        if authoritative_commit:
            return
        raise


class IngestGateRejected(RuntimeError):
    """A staged generation failed a mandatory publication gate."""

    def __init__(self, state: IngestGenerationState, reason: str) -> None:
        self.state = state
        super().__init__(reason)


class IngestGenerationClaimed(RuntimeError):
    """Another runner owns the same deterministic generation attempt."""


def _raw_collection_override(collection: str | None = None) -> str | None:
    raw = collection if collection is not None else os.environ.get("VORAG_RAW_COLLECTION")
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _force_reingest() -> bool:
    """True when the operator wants an already-ingested source re-processed.

    Full-overwrite re-ingest is a real workflow (figure restoration, A/B chunking
    runs), and it works precisely because chunk point ids are deterministic. The
    duplicate skip must therefore be defeatable rather than absolute.
    """
    raw = os.environ.get("VORAG_BOOK_RAG_FORCE_REINGEST", "")
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _ingest_generations_enabled() -> bool:
    return os.environ.get("VORAG_INGEST_GENERATIONS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclasses.dataclass
class BookRagContext:
    """Mutable state threaded through the DAG steps."""

    envelope: IntakeEnvelope
    config: BookRagConfig
    source_id: str = ""
    generation_id: str | None = None
    previous_generation_id: str | None = None
    generation_already_active: bool = False
    generation_owner_token: str | None = None
    generation_claim_acquired: bool = False
    generation_lease_lost: bool = False
    publication_job: SourcePublicationJob | None = None
    sterile_publication_job: SourcePublicationJob | None = None
    parse_meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    parsed: ParsedDocument | None = None
    composites: list[CompositeChunk] = dataclasses.field(default_factory=list)
    chunks: list[RetrievalChunk] = dataclasses.field(default_factory=list)
    raw_embeddings: dict[str, Any] = dataclasses.field(default_factory=dict)
    raw_point_ids: list[str] = dataclasses.field(default_factory=list)
    sterile_point_ids: list[str] = dataclasses.field(default_factory=list)
    smoke_result: smoke.GateResult | None = None
    validation_report: ValidationReport | None = None
    quality_warnings: list[dict[str, Any]] = dataclasses.field(default_factory=list)


IngestStep = Callable[[BookRagContext], Awaitable[None]]


@dataclasses.dataclass
class PipelineOutput:
    """The terminal result of an ingest run."""

    source_id: str
    status: EnvelopeStatus
    qdrant_point_ids: list[str]
    raw_point_ids: list[str]
    sterile_point_ids: list[str]
    smoke_decision: smoke.GateDecision | None
    chunk_count: int
    idempotent_noop: bool = False
    generation_id: str | None = None
    generation_state: str | None = None
    quality_warnings: list[dict[str, Any]] = dataclasses.field(default_factory=list)


class BookRagPipeline:
    """Async orchestrator for one document's book_rag ingestion."""

    #: How often in-flight metrics are flushed to the envelope (VOR-1434).
    metrics_flush_interval_s: float = INGEST_METRICS_FLUSH_INTERVAL_S

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
        vlm_service: Any | None = None,
    ) -> None:
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
        self.vlm_service = vlm_service

    # ── public entry ────────────────────────────────────────────────────────

    async def ingest(self, envelope: IntakeEnvelope) -> PipelineOutput:
        """Run the DAG, recording per-stage wall time, embed calls/429s and
        Qdrant latency into ``intake_envelopes.metadata->'metrics'`` (VOR-1434;
        shape in ``core/observability/metrics.py``). Recording is
        best-effort and never changes the ingest's outcome."""
        envelope_id = str(envelope.id)
        with ingest_recording() as recorder:
            # Periodic flush so counts made inside a long stage (e.g. 429s in
            # chunk_embed) reach Postgres while it runs; never fails the book.
            flusher = start_periodic_ingest_flush(
                self._metrics_pool, envelope_id, recorder, self.metrics_flush_interval_s
            )
            try:
                return await self._ingest(envelope)
            finally:
                await stop_periodic_ingest_flush(flusher)
                await persist_ingest_metrics(self._metrics_pool, envelope_id, recorder)

    @property
    def _metrics_pool(self) -> Any | None:
        return self.pg_pool if self.pg_pool is not None else self.pool

    async def _ingest(self, envelope: IntakeEnvelope) -> PipelineOutput:
        # Idempotency: a COMPLETED envelope is never reprocessed.
        if envelope.status is EnvelopeStatus.COMPLETED:
            sid = canonical_source_id(str(envelope.raw_content.get("file_hash", "")))
            return PipelineOutput(
                source_id=sid,
                status=EnvelopeStatus.COMPLETED,
                qdrant_point_ids=[],
                raw_point_ids=[],
                sterile_point_ids=[],
                smoke_decision=None,
                chunk_count=0,
                idempotent_noop=True,
            )

        ctx = BookRagContext(envelope=envelope, config=self.config)
        claim_heartbeat: asyncio.Task[None] | None = None
        try:
            envelope.status = EnvelopeStatus.PROCESSING
            # Verification/source registration must precede duplicate detection.
            await self._run_ingest_step(ctx, "register_source", self._register_source)
            if not _force_reingest() and await self._source_already_ingested(ctx.source_id):
                envelope.status = EnvelopeStatus.COMPLETED
                log.info(
                    "book_rag.duplicate_source_skipped",
                    source_id=ctx.source_id,
                    envelope_id=str(envelope.id),
                    collection=self.raw_collection or RAW_COLLECTION,
                )
                return PipelineOutput(
                    source_id=ctx.source_id,
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
            ctx.publication_job = await self._admit_source_publication(
                ctx, self.raw_collection or RAW_COLLECTION
            )
            if _ingest_generations_enabled():
                await self._start_generation(ctx)
                if ctx.generation_already_active:
                    envelope.status = EnvelopeStatus.COMPLETED
                    return PipelineOutput(
                        source_id=ctx.source_id,
                        status=EnvelopeStatus.COMPLETED,
                        qdrant_point_ids=[],
                        raw_point_ids=[],
                        sterile_point_ids=[],
                        smoke_decision=None,
                        chunk_count=0,
                        idempotent_noop=True,
                        generation_id=ctx.generation_id,
                        generation_state=IngestGenerationState.ACTIVE,
                    )
                if ctx.generation_claim_acquired:
                    claim_heartbeat = asyncio.create_task(
                        self._maintain_generation_lease(
                            ctx,
                            owner_task=asyncio.current_task(),
                        )
                    )
            steps: tuple[tuple[str, IngestStep], ...] = (
                ("parse", self._parse),
                ("visual", self._visual),
                ("chunk_embed", self._chunk_embed),
                ("sterilize", self._sterilize),
                ("advanced_understanding", self._advanced_understanding),
                ("register_artifact", self._register_artifact),
                ("validate", self._validate),
                ("retrieval_smoke", self._retrieval_smoke),
            )
            for step_name, step in steps:
                await self._run_ingest_step(ctx, step_name, step)
            if ctx.generation_id is not None:
                if claim_heartbeat is not None:
                    claim_heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await claim_heartbeat
                    claim_heartbeat = None
                await self._assert_generation_lease(ctx)
                await self._activate_generation(ctx)
            await self._record_book_quality(ctx)
            await self._persist_quality_warnings(ctx)
            envelope.status = EnvelopeStatus.COMPLETED
        except asyncio.CancelledError:
            envelope.status = EnvelopeStatus.PROCESSING
            await self._abort_source_publication(ctx, "book_rag ingest cancelled")
            await self._release_generation_lease(ctx)
            raise
        except IngestGateRejected as exc:
            envelope.status = EnvelopeStatus.FAILED
            envelope.error = str(exc)
            await self._abort_source_publication(ctx, str(exc))
            raise
        except IngestGenerationClaimed:
            # A concurrent owner must remain the only writer. Do not fail its
            # shared generation or demote its envelope/points/artifact.
            envelope.status = EnvelopeStatus.PROCESSING
            await self._abort_source_publication(ctx, "generation is owned by another runner")
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised after marking FAILED
            envelope.status = EnvelopeStatus.FAILED
            envelope.error = str(exc)
            await self._abort_source_publication(ctx, str(exc))
            await self._fail_generation(ctx, str(exc))
            raise
        finally:
            if claim_heartbeat is not None:
                claim_heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await claim_heartbeat

        return PipelineOutput(
            source_id=ctx.source_id,
            status=envelope.status,
            qdrant_point_ids=[*ctx.raw_point_ids, *ctx.sterile_point_ids],
            raw_point_ids=ctx.raw_point_ids,
            sterile_point_ids=ctx.sterile_point_ids,
            smoke_decision=ctx.smoke_result.decision if ctx.smoke_result else None,
            chunk_count=len(ctx.chunks),
            quality_warnings=collect_quality_warnings(ctx),
            generation_id=ctx.generation_id,
            generation_state=(
                IngestGenerationState.ACTIVE if ctx.generation_id is not None else None
            ),
        )

    async def _source_already_ingested(self, source_id: str) -> bool:
        """Whether a complete compatible active generation exists for this source."""
        collection = self.raw_collection or RAW_COLLECTION
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if _ingest_generations_enabled():
            if generation_pool is None:
                raise RuntimeError("ingest generations require a PostgreSQL pool")
            active = await get_active_generation(generation_pool, source_id, collection)
            return active is not None and await self._active_generation_is_reconciled(active)
        visible_counter = getattr(
            self.qdrant_store,
            "count_active_or_legacy_source",
            None,
        )
        if visible_counter is None:
            # Compatibility for narrow injected test seams; production stores
            # always expose the visibility-aware counter.
            return bool(await self.qdrant_store.count_source(collection, source_id))
        return bool(await visible_counter(collection, source_id))

    async def _active_generation_is_reconciled(
        self,
        active: SourceIngestGeneration,
    ) -> bool:
        """Prove one active generation's exact PG, artifact, and Qdrant shape."""
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if generation_pool is None:
            raise RuntimeError("ingest generations require a PostgreSQL pool")
        collection = self.raw_collection or RAW_COLLECTION
        if active.raw_collection != collection:
            return False
        if active.schema_version != GENERATION_SCHEMA_VERSION:
            return False
        if active.chunking_version != self._chunking_version():
            return False
        if active.sterilize_enabled != self.config.sterilize_enabled:
            return False
        if active.cleanup_required:
            try:
                await self._retry_pending_generation_cleanup(active)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "book_rag.generation_cleanup_retry_failed",
                    source_id=active.source_id,
                    generation_id=active.generation_id,
                    error_type=type(exc).__name__,
                )
        if active.expected_raw_points is None:
            return False
        if active.actual_artifacts != active.expected_artifacts:
            return False
        artifact_count = await count_active_generation_artifacts(
            generation_pool,
            active.source_id,
            active.raw_collection,
            active.generation_id,
        )
        if artifact_count != active.expected_artifacts:
            return False
        raw_count = await self.qdrant_store.count_source(
            collection,
            active.source_id,
            generation_id=active.generation_id,
            ingest_state=IngestGenerationState.ACTIVE,
        )
        if raw_count != active.expected_raw_points:
            return False
        if collection == RAW_COLLECTION:
            if active.expected_sterile_points is None:
                return False
            sterile_count = await self.qdrant_store.count_source(
                STERILE_COLLECTION,
                active.source_id,
                generation_id=active.generation_id,
                ingest_state=IngestGenerationState.ACTIVE,
            )
            if sterile_count != active.expected_sterile_points:
                return False
        return True

    async def _retry_pending_generation_cleanup(
        self,
        active: SourceIngestGeneration,
    ) -> None:
        """Retry harmless post-activation residue without changing publication."""
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if generation_pool is None:
            return
        collections = self._generation_collections()
        errors: list[str] = []
        superseded = await get_superseded_generation_ids(
            generation_pool,
            active.source_id,
            active.raw_collection,
        )
        for old_generation_id in superseded:
            for collection in collections:
                try:
                    await self.qdrant_store.delete_generation(
                        collection,
                        active.source_id,
                        old_generation_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{old_generation_id}/{collection}: {type(exc).__name__}")
        for collection in collections:
            try:
                await self.qdrant_store.delete_legacy_source(collection, active.source_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"legacy/{collection}: {type(exc).__name__}")
        await record_generation_cleanup(
            generation_pool,
            active.generation_id,
            required=bool(errors),
            error="; ".join(errors) or None,
        )

    def _chunking_version(self) -> str:
        chunking = getattr(self.settings, "chunking", None)
        strategy = getattr(chunking, "strategy", None)
        return str(strategy or "default")

    def _generation_collections(self) -> list[str]:
        """Collections participating in one collection-scoped generation."""
        raw_collection = self.raw_collection or RAW_COLLECTION
        if raw_collection != RAW_COLLECTION:
            if self.config.sterilize_enabled:
                raise RuntimeError(
                    "sterilized generation ingest is only supported for the canonical "
                    f"raw collection {RAW_COLLECTION}; got {raw_collection}"
                )
            return [raw_collection]
        # Canonical replacements always retire/clean the canonical sterile
        # collection, including an enabled -> disabled configuration change.
        return [raw_collection, STERILE_COLLECTION]

    async def _start_generation(self, ctx: BookRagContext) -> None:
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if generation_pool is None:
            raise RuntimeError("ingest generations require a PostgreSQL pool")
        raw_collection = self.raw_collection or RAW_COLLECTION
        self._generation_collections()  # validate custom/sterile compatibility
        previous = await get_active_generation(
            generation_pool,
            ctx.source_id,
            raw_collection,
        )
        ctx.previous_generation_id = previous.generation_id if previous is not None else None
        ctx.generation_id = deterministic_hex(
            str(ctx.envelope.id),
            ctx.source_id,
            raw_collection,
            GENERATION_SCHEMA_VERSION,
            self._chunking_version(),
            f"sterilize={int(self.config.sterilize_enabled)}",
            length=26,
        ).upper()
        ctx.generation_owner_token = ulid_new()
        inserted = await create_generation(
            generation_pool,
            SourceIngestGeneration(
                generation_id=ctx.generation_id,
                source_id=ctx.source_id,
                raw_collection=raw_collection,
                envelope_id=str(ctx.envelope.id),
                pipeline=TargetPipeline.BOOK_RAG,
                schema_version=GENERATION_SCHEMA_VERSION,
                chunking_version=self._chunking_version(),
                sterilize_enabled=self.config.sterilize_enabled,
                owner_token=ctx.generation_owner_token,
                claim_expires_at=datetime.now(UTC) + GENERATION_CLAIM_LEASE,
            ),
        )
        if inserted:
            ctx.generation_claim_acquired = True
            return
        existing = await get_generation(generation_pool, ctx.generation_id)
        if existing is not None and existing.state is IngestGenerationState.ACTIVE:
            if await self._active_generation_is_reconciled(existing):
                ctx.generation_already_active = True
                return
            raise RuntimeError(f"active generation {ctx.generation_id} is not fully reconciled")
        raise IngestGenerationClaimed(
            f"generation {ctx.generation_id} is already owned by another runner"
        )

    async def _release_generation_lease(self, ctx: BookRagContext) -> None:
        """Make a canceled staging attempt immediately reclaimable."""
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if (
            generation_pool is None
            or ctx.generation_id is None
            or ctx.generation_owner_token is None
        ):
            return
        release_task = asyncio.create_task(
            release_generation_claim(
                generation_pool,
                ctx.generation_id,
                ctx.generation_owner_token,
            )
        )
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError:
            await release_task
        except Exception as exc:  # noqa: BLE001
            log.error(
                "book_rag.generation_claim_release_failed",
                generation_id=ctx.generation_id,
                envelope_id=str(ctx.envelope.id),
                error_type=type(exc).__name__,
            )

    async def _assert_generation_lease(self, ctx: BookRagContext) -> None:
        """Fail closed when this runner no longer owns the mutable generation."""
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if ctx.generation_id is None:
            return
        if generation_pool is None or ctx.generation_owner_token is None:
            raise IngestGenerationClaimed(
                f"generation {ctx.generation_id} has no provable owner lease"
            )
        if ctx.generation_lease_lost or not await generation_claim_is_current(
            generation_pool,
            ctx.generation_id,
            ctx.generation_owner_token,
        ):
            ctx.generation_lease_lost = True
            raise IngestGenerationClaimed(f"generation {ctx.generation_id} owner lease was lost")

    async def _maintain_generation_lease(
        self,
        ctx: BookRagContext,
        *,
        owner_task: asyncio.Task[Any] | None = None,
        on_lost: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Refresh ownership and actively stop a runner that loses its fence."""
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if (
            generation_pool is None
            or ctx.generation_id is None
            or ctx.generation_owner_token is None
        ):
            return
        while True:
            await asyncio.sleep(GENERATION_CLAIM_HEARTBEAT_SECONDS)
            try:
                refreshed = await refresh_generation_claim(
                    generation_pool,
                    ctx.generation_id,
                    ctx.generation_owner_token,
                    datetime.now(UTC) + GENERATION_CLAIM_LEASE,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "book_rag.generation_claim_heartbeat_failed",
                    generation_id=ctx.generation_id,
                    error_type=type(exc).__name__,
                )
                refreshed = False
            if not refreshed:
                ctx.generation_lease_lost = True
                if on_lost is not None:
                    await on_lost()
                if owner_task is not None and not owner_task.done():
                    owner_task.cancel()
                return

    async def _record_generation_counts(
        self,
        ctx: BookRagContext,
        *,
        expected_raw_points: int | None = None,
        actual_raw_points: int | None = None,
        expected_sterile_points: int | None = None,
        actual_sterile_points: int | None = None,
        actual_artifacts: int | None = None,
    ) -> None:
        """Persist staging counts only while this runner owns the lease."""
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if ctx.generation_id is None or generation_pool is None:
            return
        await self._assert_generation_lease(ctx)
        changed = await update_generation_counts(
            generation_pool,
            ctx.generation_id,
            expected_raw_points=expected_raw_points,
            actual_raw_points=actual_raw_points,
            expected_sterile_points=expected_sterile_points,
            actual_sterile_points=actual_sterile_points,
            actual_artifacts=actual_artifacts,
            owner_token=ctx.generation_owner_token,
        )
        if not changed:
            ctx.generation_lease_lost = True
            raise IngestGenerationClaimed(
                f"generation {ctx.generation_id} count update lost its owner fence"
            )

    async def _transition_owned_generation(
        self,
        ctx: BookRagContext,
        *,
        from_states: tuple[IngestGenerationState, ...],
        to_state: IngestGenerationState,
        validation_decision: dict[str, object] | None = None,
        smoke_decision: str | None = None,
    ) -> None:
        """Apply a non-rejection transition under the current lease fence."""
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if ctx.generation_id is None or generation_pool is None:
            return
        await self._assert_generation_lease(ctx)
        changed = await transition_generation(
            generation_pool,
            ctx.generation_id,
            from_states=from_states,
            to_state=to_state,
            validation_decision=validation_decision,
            smoke_decision=smoke_decision,
            owner_token=ctx.generation_owner_token,
        )
        if not changed:
            ctx.generation_lease_lost = True
            raise IngestGenerationClaimed(
                f"generation {ctx.generation_id} transition lost its owner fence"
            )

    async def _insert_artifact_owned(
        self,
        ctx: BookRagContext,
        artifact: KnowledgeArtifact,
    ) -> None:
        """Insert a staging artifact while holding the generation owner row lock."""
        artifact_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if artifact_pool is None:
            raise RuntimeError("generation artifact registration requires PostgreSQL")
        if ctx.generation_id is None:
            await insert_artifact(artifact_pool, artifact)
            return
        if ctx.generation_owner_token is None:
            raise IngestGenerationClaimed(
                f"generation {ctx.generation_id} has no artifact owner fence"
            )
        async with artifact_pool.acquire() as conn, conn.transaction():
            claim = await conn.fetchrow(
                """
                SELECT state FROM source_ingest_generations
                WHERE generation_id = $1
                  AND owner_token = $2
                  AND claim_expires_at > NOW()
                  AND state IN ('staging', 'validated')
                FOR UPDATE
                """,
                ctx.generation_id,
                ctx.generation_owner_token,
            )
            if claim is None:
                ctx.generation_lease_lost = True
                raise IngestGenerationClaimed(
                    f"generation {ctx.generation_id} artifact write lost its owner fence"
                )
            await insert_artifact(conn, artifact)
            changed = await update_generation_counts(
                conn,
                ctx.generation_id,
                actual_artifacts=1,
                owner_token=ctx.generation_owner_token,
            )
            if not changed:
                raise IngestGenerationClaimed(
                    f"generation {ctx.generation_id} artifact count lost its owner fence"
                )

    async def _fail_generation(self, ctx: BookRagContext, reason: str) -> None:
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if ctx.generation_id is None or generation_pool is None:
            return
        changed = await transition_generation(
            generation_pool,
            ctx.generation_id,
            from_states=(IngestGenerationState.STAGING, IngestGenerationState.VALIDATED),
            to_state=IngestGenerationState.FAILED,
            failure_reason=reason,
            owner_token=ctx.generation_owner_token,
        )
        if changed:
            await self._mark_generation_points_state(ctx, IngestGenerationState.FAILED)
            return
        # Activation may have committed even when its transaction exit raised.
        # Re-read and repair under the same collection-scoped advisory lock;
        # never hide a generation unless the conditional PG failure won.
        current = await get_generation(generation_pool, ctx.generation_id)
        if current is not None and current.state is IngestGenerationState.ACTIVE:
            await self._reconcile_activation_after_transaction_error(ctx)

    async def _mark_generation_points_state(
        self,
        ctx: BookRagContext,
        state: IngestGenerationState,
    ) -> None:
        """Best-effort Qdrant state marker; active retrieval still fails closed."""
        if ctx.generation_id is None:
            return
        # Replacement must retire both canonical collections even when the new
        # configuration disables sterile writes; otherwise old sterile points
        # remain active beside the new raw-only generation.
        collections = self._generation_collections()
        for collection in collections:
            try:
                await self.qdrant_store.set_generation_state(
                    collection,
                    ctx.source_id,
                    ctx.generation_id,
                    state,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "book_rag.generation_point_state_update_failed",
                    source_id=ctx.source_id,
                    generation_id=ctx.generation_id,
                    state=state,
                    collection=collection,
                    error_type=type(exc).__name__,
                )

    async def _reject_generation(
        self,
        ctx: BookRagContext,
        state: IngestGenerationState,
        *,
        validation_decision: dict[str, object] | None = None,
        smoke_decision: str | None = None,
        failure_reason: str,
    ) -> bool:
        """Conditionally reject in PG before hiding any Qdrant points."""
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if ctx.generation_id is None or generation_pool is None:
            return False
        changed = await transition_generation(
            generation_pool,
            ctx.generation_id,
            from_states=(IngestGenerationState.STAGING,),
            to_state=state,
            validation_decision=validation_decision,
            smoke_decision=smoke_decision,
            failure_reason=failure_reason,
            owner_token=ctx.generation_owner_token,
        )
        if changed:
            await self._mark_generation_points_state(ctx, state)
        else:
            current = await get_generation(generation_pool, ctx.generation_id)
            if current is None or current.state is not IngestGenerationState.ACTIVE:
                return False
            await self._reconcile_activation_after_transaction_error(ctx)
        return changed

    async def _record_book_quality(self, ctx: BookRagContext) -> None:
        """Best-effort per-book quality gate (VOR-1434).

        Computes `vo_rag.core.corpus.health.book_findings` over the book just
        upserted (a fresh `collect_stats` scroll filtered to `ctx.source_id`,
        the same indexed field every other per-book Qdrant read in this
        module uses, restricted to `health._STATS_PAYLOAD` and
        `with_vectors=False` -- no more than the findings need) and writes
        the result to `intake_envelopes.metadata->'quality'`
        (`book_quality_metadata`'s stable ``{"findings": [...], "stats":
        {...}}`` shape). Never deletes anything, and never raises: a bug in
        the quality check must not turn an otherwise-successful ingest into a
        failed one, so any exception here is logged and swallowed and the
        envelope proceeds to COMPLETED regardless.

        Scrolls `self.raw_collection or RAW_COLLECTION`
        -- the same override `VORAG_RAW_COLLECTION` gives every other
        Qdrant read in this pipeline -- so a bulk run against a non-default
        collection doesn't silently scroll the canonical one and record a
        false-clean ``{"findings": [], "stats": {"chunks": 0, ...}}``. A
        genuinely empty result (0 chunks, whatever the cause) is itself
        `book_findings`'s `no_chunks` finding, never reported as clean.
        """
        started = time.monotonic()
        try:
            async with asyncio.timeout(5.0):
                generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
                collection = self.raw_collection or RAW_COLLECTION
                stats = await collect_stats(
                    self.qdrant_store,
                    source_ids=[ctx.source_id],
                    pool=generation_pool,
                    collection=collection,
                )
                if not stats:
                    return
                quality = book_quality_metadata(stats[0])
                for finding in quality["findings"]:
                    ctx.quality_warnings.append(
                        quality_warning(
                            "book_quality", finding, "Stored passage statistics require review."
                        )
                    )
                if generation_pool is not None:
                    await generation_pool.execute(
                        """
                        UPDATE intake_envelopes
                        SET metadata = jsonb_set(
                            COALESCE(metadata, '{}'::jsonb), '{quality}', $2, true)
                        WHERE id = $1
                        """,
                        str(ctx.envelope.id),
                        quality,
                    )
                log.info(
                    "book_rag.quality_recorded",
                    envelope_id=str(ctx.envelope.id),
                    source_id=ctx.source_id,
                    collection=collection,
                    findings=quality["findings"],
                    duration_s=round(time.monotonic() - started, 3),
                )
        except Exception as exc:  # noqa: BLE001 - a quality-check bug must never fail the ingest
            ctx.quality_warnings.append(
                quality_warning("book_quality", "audit_unavailable", type(exc).__name__)
            )
            log.warning(
                "book_rag.quality_check_failed",
                envelope_id=str(ctx.envelope.id),
                source_id=ctx.source_id,
                error=str(exc),
                duration_s=round(time.monotonic() - started, 3),
                exc_info=True,
            )

    async def _activate_generation(self, ctx: BookRagContext) -> None:
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        generation_id = ctx.generation_id
        if generation_pool is None or generation_id is None:
            return
        raw_collection = self.raw_collection or RAW_COLLECTION
        collections = self._generation_collections()

        changed_old: list[str] = []
        changed_legacy: list[str] = []
        changed_new: list[str] = []
        async with _reconciled_activation_transaction(
            generation_pool,
            lambda: self._reconcile_activation_after_transaction_error(ctx),
        ) as conn:
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1 || chr(31) || $2, 0))",
                ctx.source_id,
                raw_collection,
            )
            current = await conn.fetchrow(
                """
                SELECT generation_id FROM source_ingest_generations
                WHERE source_id = $1 AND raw_collection = $2 AND state = 'active'
                FOR UPDATE
                """,
                ctx.source_id,
                raw_collection,
            )
            ctx.previous_generation_id = (
                str(current["generation_id"]) if current is not None else None
            )
            if ctx.previous_generation_id == generation_id:
                # A retry/forced run can deterministically target the already
                # active generation. Verify its published shape and leave both
                # stores untouched; treating it as its own predecessor would
                # delete the active points during post-switch cleanup.
                ctx.previous_generation_id = None
                await self._reconcile_activated_counts(ctx)
                return
            claim = await conn.fetchrow(
                """
                SELECT state FROM source_ingest_generations
                WHERE generation_id = $1
                  AND source_id = $2
                  AND raw_collection = $3
                  AND owner_token = $4
                  AND claim_expires_at > NOW()
                  AND state = 'validated'
                FOR UPDATE
                """,
                generation_id,
                ctx.source_id,
                raw_collection,
                ctx.generation_owner_token,
            )
            if claim is None:
                ctx.generation_lease_lost = True
                raise IngestGenerationClaimed(
                    f"generation {generation_id} activation lost its owner fence"
                )
            try:
                # Old points leave the default retrieval view first; a short
                # read gap is safer than a mixed old/new generation.
                if ctx.previous_generation_id is not None:
                    for collection in collections:
                        # Record rollback intent before the remote await: a
                        # canceled HTTP request may still have applied server-side.
                        changed_old.append(collection)
                        await self.qdrant_store.set_generation_state(
                            collection,
                            ctx.source_id,
                            ctx.previous_generation_id,
                            IngestGenerationState.SUPERSEDED,
                        )
                # Points written before generation support have no generation_id.
                # Hide them in the same source-serialized switch so the first
                # migrated activation cannot mix legacy and generated chunks.
                # A later replacement must leave already-superseded legacy
                # residue alone; otherwise rollback could resurrect it.
                if ctx.previous_generation_id is None:
                    for collection in collections:
                        changed_legacy.append(collection)
                        await self.qdrant_store.set_legacy_source_state(
                            collection,
                            ctx.source_id,
                            IngestGenerationState.SUPERSEDED,
                        )
                for collection in collections:
                    changed_new.append(collection)
                    await self.qdrant_store.set_generation_state(
                        collection,
                        ctx.source_id,
                        generation_id,
                        IngestGenerationState.ACTIVE,
                    )
                await self._reconcile_activated_counts(ctx)
                await activate_generation_record_locked(
                    conn,
                    generation_id,
                    ctx.source_id,
                    raw_collection,
                    ctx.generation_owner_token,
                )
            except BaseException:
                # Restore the prior retrieval view if the cross-store switch
                # does not reach the durable Postgres activation record.
                for collection in reversed(changed_new):
                    try:
                        await self.qdrant_store.set_generation_state(
                            collection,
                            ctx.source_id,
                            generation_id,
                            IngestGenerationState.STAGING,
                        )
                    except BaseException as rollback_exc:  # noqa: BLE001
                        log.error(
                            "book_rag.generation_activation_rollback_failed",
                            source_id=ctx.source_id,
                            generation_id=generation_id,
                            collection=collection,
                            error_type=type(rollback_exc).__name__,
                        )
                if ctx.previous_generation_id is not None:
                    for collection in reversed(changed_old):
                        try:
                            await self.qdrant_store.set_generation_state(
                                collection,
                                ctx.source_id,
                                ctx.previous_generation_id,
                                IngestGenerationState.ACTIVE,
                            )
                        except BaseException as rollback_exc:  # noqa: BLE001
                            log.error(
                                "book_rag.previous_generation_restore_failed",
                                source_id=ctx.source_id,
                                generation_id=ctx.previous_generation_id,
                                collection=collection,
                                error_type=type(rollback_exc).__name__,
                            )
                for collection in reversed(changed_legacy):
                    try:
                        await self.qdrant_store.set_legacy_source_state(
                            collection,
                            ctx.source_id,
                            IngestGenerationState.ACTIVE,
                        )
                    except BaseException as rollback_exc:  # noqa: BLE001
                        log.error(
                            "book_rag.legacy_generation_restore_failed",
                            source_id=ctx.source_id,
                            collection=collection,
                            error_type=type(rollback_exc).__name__,
                        )
                raise

        # Cleanup is intentionally last.  Failure leaves harmless superseded
        # points that a retryable janitor can delete; it never deactivates new data.
        cleanup_errors: list[str] = []
        deferred_base_error: BaseException | None = None
        if ctx.previous_generation_id is not None and ctx.previous_generation_id != generation_id:
            for collection in collections:
                try:
                    await self.qdrant_store.delete_generation(
                        collection, ctx.source_id, ctx.previous_generation_id
                    )
                except BaseException as exc:  # noqa: BLE001
                    log.warning(
                        "book_rag.superseded_generation_cleanup_failed",
                        source_id=ctx.source_id,
                        generation_id=ctx.previous_generation_id,
                        collection=collection,
                        error_type=type(exc).__name__,
                    )
                    cleanup_errors.append(f"{collection}: {type(exc).__name__}")
                    if not isinstance(exc, Exception) and deferred_base_error is None:
                        deferred_base_error = exc
        for collection in collections:
            try:
                await self.qdrant_store.delete_legacy_source(collection, ctx.source_id)
            except BaseException as exc:  # noqa: BLE001
                log.warning(
                    "book_rag.legacy_generation_cleanup_failed",
                    source_id=ctx.source_id,
                    collection=collection,
                    error_type=type(exc).__name__,
                )
                cleanup_errors.append(f"legacy/{collection}: {type(exc).__name__}")
                if not isinstance(exc, Exception) and deferred_base_error is None:
                    deferred_base_error = exc
        cleanup_status_task = asyncio.create_task(
            record_generation_cleanup(
                generation_pool,
                generation_id,
                required=bool(cleanup_errors),
                error="; ".join(cleanup_errors) or None,
            )
        )
        try:
            await asyncio.shield(cleanup_status_task)
        except asyncio.CancelledError as exc:
            # A second cancellation while persisting the marker must not orphan
            # residue with cleanup_required=false. The shielded DB write is
            # awaited to completion before cancellation is propagated.
            if deferred_base_error is None:
                deferred_base_error = exc
            try:
                await cleanup_status_task
            except Exception as persist_exc:  # noqa: BLE001
                log.error(
                    "book_rag.generation_cleanup_status_persist_failed",
                    source_id=ctx.source_id,
                    generation_id=generation_id,
                    cleanup_required=bool(cleanup_errors),
                    error_type=type(persist_exc).__name__,
                )
        except Exception as exc:  # noqa: BLE001
            # Publication already committed. Metadata trouble must not turn a
            # successful active generation into a false failed envelope.
            log.error(
                "book_rag.generation_cleanup_status_persist_failed",
                source_id=ctx.source_id,
                generation_id=generation_id,
                cleanup_required=bool(cleanup_errors),
                error_type=type(exc).__name__,
            )
        if deferred_base_error is not None:
            raise deferred_base_error

    async def _reconcile_activation_after_transaction_error(
        self,
        ctx: BookRagContext,
    ) -> bool:
        """Repair under the source lock; return true when the new commit won."""
        generation_pool = self.pg_pool if self.pg_pool is not None else self.pool
        generation_id = ctx.generation_id
        if generation_pool is None or generation_id is None:
            return False
        authoritative_commit = False
        try:
            async with generation_pool.acquire() as conn, conn.transaction():
                raw_collection = self.raw_collection or RAW_COLLECTION
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1 || chr(31) || $2, 0))",
                    ctx.source_id,
                    raw_collection,
                )
                durable_active_id = await get_active_generation_id(
                    conn,
                    ctx.source_id,
                    raw_collection,
                )
                authoritative_commit = durable_active_id == generation_id
                await self._apply_activation_reconciliation(
                    ctx,
                    durable_active_id=durable_active_id,
                    record_connection=conn,
                )
        except BaseException as exc:  # noqa: BLE001
            log.error(
                "book_rag.generation_activation_durable_read_failed",
                source_id=ctx.source_id,
                generation_id=generation_id,
                error_type=type(exc).__name__,
            )
            if authoritative_commit:
                try:
                    await record_generation_cleanup(
                        generation_pool,
                        generation_id,
                        required=True,
                        error="activation reconciliation exit interrupted",
                    )
                except BaseException as persist_exc:  # noqa: BLE001
                    log.error(
                        "book_rag.generation_cleanup_status_persist_failed",
                        source_id=ctx.source_id,
                        generation_id=generation_id,
                        cleanup_required=True,
                        error_type=type(persist_exc).__name__,
                    )
        return authoritative_commit

    async def _apply_activation_reconciliation(
        self,
        ctx: BookRagContext,
        *,
        durable_active_id: str | None,
        record_connection: Any,
    ) -> bool:
        """Apply the Qdrant half while the caller holds the source lock."""
        generation_id = ctx.generation_id
        if generation_id is None:
            raise RuntimeError("activation reconciliation requires generation_id")
        collections = self._generation_collections()

        async def _set_generation(target_id: str, state: IngestGenerationState) -> None:
            for collection in collections:
                try:
                    await self.qdrant_store.set_generation_state(
                        collection,
                        ctx.source_id,
                        target_id,
                        state,
                    )
                except BaseException as exc:  # noqa: BLE001
                    log.error(
                        "book_rag.generation_activation_reconcile_state_failed",
                        source_id=ctx.source_id,
                        generation_id=target_id,
                        state=state,
                        collection=collection,
                        error_type=type(exc).__name__,
                    )

        if durable_active_id == generation_id:
            if (
                ctx.previous_generation_id is not None
                and ctx.previous_generation_id != generation_id
            ):
                await _set_generation(
                    ctx.previous_generation_id,
                    IngestGenerationState.SUPERSEDED,
                )
            else:
                for collection in collections:
                    try:
                        await self.qdrant_store.set_legacy_source_state(
                            collection,
                            ctx.source_id,
                            IngestGenerationState.SUPERSEDED,
                        )
                    except BaseException as exc:  # noqa: BLE001
                        log.error(
                            "book_rag.legacy_activation_reconcile_failed",
                            source_id=ctx.source_id,
                            collection=collection,
                            error_type=type(exc).__name__,
                        )
            await _set_generation(generation_id, IngestGenerationState.ACTIVE)
            try:
                await record_generation_cleanup(
                    record_connection,
                    generation_id,
                    required=True,
                    error="activation transaction exit interrupted before cleanup",
                )
            except BaseException as exc:  # noqa: BLE001
                log.error(
                    "book_rag.generation_cleanup_status_persist_failed",
                    source_id=ctx.source_id,
                    generation_id=generation_id,
                    cleanup_required=True,
                    error_type=type(exc).__name__,
                )
            return True

        # The new Postgres activation did not commit. Hide its points and
        # restore exactly the generation (or legacy view) Postgres says is active.
        await _set_generation(generation_id, IngestGenerationState.STAGING)
        if ctx.previous_generation_id is not None and ctx.previous_generation_id != generation_id:
            await _set_generation(
                ctx.previous_generation_id,
                (
                    IngestGenerationState.ACTIVE
                    if durable_active_id == ctx.previous_generation_id
                    else IngestGenerationState.SUPERSEDED
                ),
            )
        if durable_active_id is not None and durable_active_id not in {
            generation_id,
            ctx.previous_generation_id,
        }:
            await _set_generation(durable_active_id, IngestGenerationState.ACTIVE)
        if ctx.previous_generation_id is None:
            legacy_state = (
                IngestGenerationState.ACTIVE
                if durable_active_id is None
                else IngestGenerationState.SUPERSEDED
            )
            for collection in collections:
                try:
                    await self.qdrant_store.set_legacy_source_state(
                        collection,
                        ctx.source_id,
                        legacy_state,
                    )
                except BaseException as exc:  # noqa: BLE001
                    log.error(
                        "book_rag.legacy_activation_reconcile_failed",
                        source_id=ctx.source_id,
                        collection=collection,
                        error_type=type(exc).__name__,
                    )
        return False

    async def _admit_source_publication(
        self,
        ctx: BookRagContext,
        collection: str,
        *,
        expected_dependencies: dict[str, int | None] | None = None,
        producer_recipe: dict[str, Any] | None = None,
    ) -> SourcePublicationJob | None:
        """Pin a verified immutable source before parser or model work begins."""
        admit = getattr(self.qdrant_store, "admit_source_revision", None)
        if not callable(admit):
            return None
        publication_enabled = bool(
            getattr(
                self.qdrant_store,
                "source_publication_enabled",
                getattr(self.qdrant_store, "_publication_enabled", False),
            )
        )
        if publication_enabled and _ingest_generations_enabled():
            raise RuntimeError(
                "source publication and generation-scoped ingest cannot be enabled together"
            )
        file_hash = str(ctx.envelope.raw_content.get("file_hash", "")).lower().strip()
        job_id = deterministic_uuid(
            "normal-source-publication-v1",
            str(ctx.envelope.id),
            collection,
            ctx.source_id,
        )
        if publication_enabled and (
            expected_dependencies is not None or producer_recipe is not None
        ):
            result = await admit(
                collection,
                ctx.source_id,
                job_id,
                file_hash,
                expected_dependencies=expected_dependencies,
                producer_recipe=producer_recipe,
            )
        else:
            result = await admit(collection, ctx.source_id, job_id, file_hash)
        return cast(SourcePublicationJob | None, result)

    async def _abort_source_publication(self, ctx: BookRagContext, reason: str) -> None:
        """Persist no-write disposition where provable; partial jobs remain recoverable."""
        abort = getattr(self.qdrant_store, "abort_source_revision", None)
        if not callable(abort):
            return
        for job in (ctx.sterile_publication_job, ctx.publication_job):
            if job is None:
                continue
            try:
                await abort(job, reason)
            except Exception as exc:  # noqa: BLE001 - retain original ingest failure
                log.error(
                    "book_rag.source_publication_abort_failed",
                    job_id=job.job_id,
                    error_type=type(exc).__name__,
                )

    async def _reconcile_activated_counts(self, ctx: BookRagContext) -> None:
        """Refuse publication unless active Qdrant counts match the staged write."""
        if ctx.generation_id is None:
            raise RuntimeError("activated count reconciliation requires generation_id")
        expected = [(self.raw_collection or RAW_COLLECTION, len(ctx.raw_point_ids))]
        if (self.raw_collection or RAW_COLLECTION) == RAW_COLLECTION:
            # Always prove absence when sterilization is disabled. This catches
            # same-generation residue left by an interrupted differently
            # configured attempt instead of publishing it accidentally.
            expected.append((STERILE_COLLECTION, len(ctx.sterile_point_ids)))
        for collection, expected_count in expected:
            actual = await self.qdrant_store.count_source(
                collection,
                ctx.source_id,
                generation_id=ctx.generation_id,
                ingest_state=IngestGenerationState.ACTIVE,
            )
            if actual != expected_count:
                raise RuntimeError(
                    "activated generation count mismatch: "
                    f"collection={collection} expected={expected_count} actual={actual}"
                )

    async def _run_ingest_step(
        self,
        ctx: BookRagContext,
        step_name: str,
        step: IngestStep,
    ) -> None:
        tracer = get_tracer("vo_rag.ingest")
        recorder = current_ingest_recorder()
        if recorder is not None:
            recorder.current_stage = step_name
        started = time.monotonic()
        with tracer.start_as_current_span(f"ingest.{step_name}") as span:
            span.set_attribute("vo_rag.source_id", _span_source_id(ctx))
            try:
                await self._assert_generation_lease(ctx)
                await step(ctx)
                await self._assert_generation_lease(ctx)
            finally:
                span.set_attribute("vo_rag.source_id", _span_source_id(ctx))
                if recorder is not None:
                    recorder.record_stage(step_name, time.monotonic() - started)
                    if ctx.parsed is not None and ctx.parsed.total_pages:
                        recorder.pages = int(ctx.parsed.total_pages)
        if recorder is not None:
            # Flush per stage so a long book's progress (and its 429s) is
            # visible before it finishes; failures flush from ``ingest``.
            await persist_ingest_metrics(self._metrics_pool, str(ctx.envelope.id), recorder)

    # ── steps ───────────────────────────────────────────────────────────────

    async def _register_source(self, ctx: BookRagContext) -> None:
        if not object_is_verified(ctx.envelope.raw_content):
            raise ValueError(
                "book_rag.register_source: authoritative Garage verification and file_hash "
                "are required"
            )
        file_hash = str(ctx.envelope.raw_content.get("file_hash", "")).lower().strip()
        if not file_hash:
            raise ValueError("book_rag.register_source: envelope.raw_content.file_hash is required")
        ctx.source_id = canonical_source_id(file_hash)

    async def _parse(self, ctx: BookRagContext) -> None:
        garage_key = ctx.envelope.raw_content.get("garage_key")
        if not garage_key:
            raise ValueError("book_rag.parse: envelope.raw_content.garage_key is required")
        ctx.parse_meta = await do_parse_document(
            envelope_id=str(ctx.envelope.id),
            garage_key=str(garage_key),
            source_id=ctx.source_id,
            garage=self.garage,
            bucket=self.bucket,
            parser_router=self.parser_router,
        )
        raw = await self.garage.get_bytes(self.bucket, ctx.parse_meta["parsed_garage_key"])
        ctx.parsed = ParsedDocument.model_validate_json(raw)

    async def _visual(self, ctx: BookRagContext) -> None:
        if not ctx.config.enable_visual_extraction:
            return
        if ctx.parsed is None:
            raise RuntimeError("book_rag.visual: parse step did not produce a ParsedDocument")
        document_metadata = await self._book_metadata(ctx)
        if ctx.config.enable_visual_description:
            if self.vlm_service is None:
                raise RuntimeError(
                    "book_rag.visual: enable_visual_description=True "
                    "but no vlm_service was injected"
                )
            parsed_garage_key = ctx.parse_meta.get("parsed_garage_key")
            if not parsed_garage_key:
                raise RuntimeError("book_rag.visual: parsed_garage_key missing from parse_meta")
            describe_result = await describe_visual_images(
                envelope_id=str(ctx.envelope.id),
                parsed_garage_key=str(parsed_garage_key),
                garage=self.garage,
                bucket=self.bucket,
                figures=self.figures,
                vlm_service=self.vlm_service,
                document_metadata=document_metadata,
            )
            ctx.composites, _ = await load_visual_composites(
                self.garage,
                self.bucket,
                ctx.parsed,
                descriptions_garage_key=describe_result.get("descriptions_garage_key"),
                document_metadata=document_metadata,
            )
        else:
            ctx.composites = await build_extraction_composites(
                ctx.parsed,
                document_metadata=document_metadata,
            )

    async def _book_metadata(self, ctx: BookRagContext) -> dict[str, Any]:
        envelope_metadata = self._envelope_book_metadata(ctx)
        lookup_pool = self.pg_pool if self.pg_pool is not None else self.pool
        if lookup_pool is None:
            return envelope_metadata

        try:
            async with lookup_pool.acquire() as conn:
                source = await resolve_canonical_source(ctx.source_id, conn=conn)
        except Exception as exc:  # noqa: BLE001 - identity lookup must never fail ingest
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
        """Book identity for chunk metadata, sourced from the intake envelope
        (falling back to the parsed PDF title). Stamped onto every chunk so DSI/
        graph/validator/retrieval have it — see do_build_chunks."""
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

    async def _chunk_embed(self, ctx: BookRagContext) -> None:
        if ctx.parsed is None:
            raise RuntimeError("book_rag.chunk_embed: parse step did not produce a ParsedDocument")
        book_metadata = await self._book_metadata(ctx)
        ctx.chunks = do_build_chunks(
            ctx.parsed,
            composites=ctx.composites,
            book_metadata=book_metadata,
            settings=self.settings,
            embedding_client=self.embedding_client,
        )
        binding = ctx.parse_meta.get("parse_provenance")
        if isinstance(binding, dict):
            for chunk in ctx.chunks:
                chunk.metadata["parse_provenance"] = dict(binding)
        ctx.raw_embeddings = await do_embed_raw(ctx.chunks, self.embedding_client)
        # VOR-57: raw chunks must be queryable, so they are upserted as "trusted";
        # the default "quarantined" makes HybridRetriever filter them out.
        raw_write = await do_upsert_raw(
            ctx.chunks,
            ctx.raw_embeddings,
            self.qdrant_store,
            ctx.source_id,
            collection=self.raw_collection,
            trust_level="trusted",
            generation_id=ctx.generation_id,
            publication_job=ctx.publication_job,
            ingest_state=(
                IngestGenerationState.STAGING
                if ctx.generation_id is not None
                else IngestGenerationState.ACTIVE
            ),
        )
        ctx.raw_point_ids = list(raw_write["point_ids"])
        if ctx.generation_id is not None:
            actual = await self.qdrant_store.count_source(
                self.raw_collection or RAW_COLLECTION,
                ctx.source_id,
                generation_id=ctx.generation_id,
                ingest_state=IngestGenerationState.STAGING,
            )
            if actual != len(ctx.chunks):
                raise RuntimeError(
                    f"raw generation count mismatch: expected={len(ctx.chunks)} actual={actual}"
                )
            await self._record_generation_counts(
                ctx,
                expected_raw_points=len(ctx.chunks),
                actual_raw_points=actual,
            )

    async def _sterilize(self, ctx: BookRagContext) -> None:
        # First-class branch (resolved open question): sterilize is a config bool,
        # not an env flag. The DAG node always exists; when disabled it logs a skip
        # and the sterile collection simply stays empty for this source.
        if not ctx.config.sterilize_enabled:
            ctx.sterile_point_ids = []
            if ctx.generation_id is not None:
                if (self.raw_collection or RAW_COLLECTION) == RAW_COLLECTION:
                    await self.qdrant_store.delete_generation(
                        STERILE_COLLECTION,
                        ctx.source_id,
                        ctx.generation_id,
                    )
                await self._record_generation_counts(
                    ctx,
                    expected_sterile_points=0,
                    actual_sterile_points=0,
                )
            return
        from vo_rag.pipelines.book_rag.steps.sterilize import do_sterilize

        expected_dependencies: dict[str, int | None] | None = None
        if getattr(self.qdrant_store, "source_publication_enabled", False):
            active_revision = await self.qdrant_store.active_source_revision(
                RAW_COLLECTION, ctx.source_id
            )
            expected_dependencies = {ctx.source_id: active_revision}
        ctx.sterile_publication_job = await self._admit_source_publication(
            ctx,
            STERILE_COLLECTION,
            expected_dependencies=expected_dependencies,
            producer_recipe={"writer": "book-rag-sterile-representation-v1"},
        )
        written_sterile_ids: list[str] = []
        sterile_chunks = await do_sterilize(
            raw_chunks=ctx.chunks,
            llm_client=self.llm_client,
            config=ctx.config,
            store=self.qdrant_store,
            embed=self._embed,
            generation_id=ctx.generation_id,
            publication_job=ctx.sterile_publication_job,
            expected_dependencies=expected_dependencies,
            point_ids_out=written_sterile_ids,
            ingest_state=(
                IngestGenerationState.STAGING
                if ctx.generation_id is not None
                else IngestGenerationState.ACTIVE
            ),
        )
        # do_sterilize upserts to the sterile collection internally; the point
        # ids are deterministic (source_id, chunk_index, "sterile").
        ctx.sterile_point_ids = written_sterile_ids or (
            [
                deterministic_point_id(chunk.source_id, index, "sterile", ctx.generation_id)
                for index, chunk in enumerate(sterile_chunks)
            ]
            if ctx.sterile_publication_job is None
            else []
        )
        if ctx.sterile_publication_job is not None and not sterile_chunks:
            await self._abort_source_publication(ctx, "sterile representation produced no points")
        if ctx.generation_id is not None:
            actual = await self.qdrant_store.count_source(
                STERILE_COLLECTION,
                ctx.source_id,
                generation_id=ctx.generation_id,
                ingest_state=IngestGenerationState.STAGING,
            )
            if actual != len(sterile_chunks):
                raise RuntimeError(
                    "sterile generation count mismatch: "
                    f"expected={len(sterile_chunks)} actual={actual}"
                )
            await self._record_generation_counts(
                ctx,
                expected_sterile_points=len(sterile_chunks),
                actual_sterile_points=actual,
            )

    async def _advanced_understanding(self, ctx: BookRagContext) -> None:
        # No-op when every layer is disabled (the default ingest path). The
        # summary/raptor/dsi/graph builders require corpus-build configs + book
        # metadata (author, category_ids, BuildConfig) that are not part of the
        # ingest envelope; they run in the dedicated advanced-understanding build
        # harness, not inline ingestion. Enabling a flag here is therefore an
        # explicit error rather than a silently-skipped no-op.
        enabled = [
            name
            for name, on in (
                ("summary_tree", ctx.config.enable_summary_tree),
                ("raptor", ctx.config.enable_raptor),
                ("dsi", ctx.config.enable_dsi),
                ("graph", ctx.config.enable_graph),
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
        await self._insert_artifact_owned(ctx, artifact)

    async def _validate(self, ctx: BookRagContext) -> None:
        await self._audit_validation(ctx)

    async def _audit_validation(
        self, ctx: BookRagContext, *, validator_factory: Any = None
    ) -> None:
        validator_kwargs: dict[str, Any] = {
            "figures": self.figures,
            "artifacts_bucket": self.bucket,
            "sterilize_enabled": ctx.config.sterilize_enabled,
        }
        if self.raw_collection is not None:
            validator_kwargs["raw_collection"] = self.raw_collection
        validation_kwargs: dict[str, Any] = {
            "document_id": str(ctx.parse_meta.get("document_id", "")),
            "source_id": ctx.source_id,
        }
        if ctx.generation_id is not None:
            validation_kwargs.update(
                artifact_id=_artifact_id(
                    ctx.source_id,
                    self.raw_collection or RAW_COLLECTION,
                    ctx.generation_id,
                ),
                generation_id=ctx.generation_id,
                ingest_state=IngestGenerationState.STAGING,
            )
        try:
            async with asyncio.timeout(QUALITY_AUDIT_TIMEOUT_SECONDS):
                validator = (validator_factory or IngestionValidator)(
                    self.qdrant_store,
                    self.pool,
                    self.garage,
                    self.retriever,
                    **validator_kwargs,
                )
                ctx.validation_report = await validator.validate(**validation_kwargs)
        except Exception as exc:  # noqa: BLE001 - quality audit must not stop usable ingestion
            ctx.quality_warnings.append(
                quality_warning(
                    "validation",
                    "audit_unavailable",
                    type(exc).__name__,
                )
            )
        if ctx.validation_report is not None:
            if not ctx.validation_report.passed and not ctx.validation_report.checks:
                ctx.quality_warnings.append(
                    quality_warning(
                        "validation",
                        "audit_failed",
                        "Validation reported failure without check details.",
                    )
                )
            for check in ctx.validation_report.checks:
                if not check.passed:
                    ctx.quality_warnings.append(
                        quality_warning("validation", check.name, check.detail)
                    )
        if ctx.generation_id is not None:
            decision = (
                ctx.validation_report.model_dump(mode="json")
                if ctx.validation_report
                else {
                    "passed": False,
                    "checks": [],
                    "unavailable": True,
                }
            )
            decision["quality_warnings"] = collect_quality_warnings(ctx)
            await self._transition_owned_generation(
                ctx,
                from_states=(IngestGenerationState.STAGING,),
                to_state=IngestGenerationState.STAGING,
                validation_decision=decision,
            )

    async def _retrieval_smoke(self, ctx: BookRagContext) -> None:
        if self.llm_client is None or not self.config.retrieval_smoke_enabled:
            # Defer-LLM ingest (or no LLM client): chunks are already persisted;
            # skip the LLM-dependent self-query smoke gate.
            if ctx.generation_id is not None:
                await self._transition_owned_generation(
                    ctx,
                    from_states=(IngestGenerationState.STAGING,),
                    to_state=IngestGenerationState.VALIDATED,
                    smoke_decision="skipped",
                )
            return
        try:
            async with asyncio.timeout(QUALITY_AUDIT_TIMEOUT_SECONDS):
                queries = await smoke.generate_self_queries(
                    all_chunks=ctx.chunks,
                    n=self.smoke_n,
                    llm_call=self._llm_call,
                    seed=self.smoke_seed,
                )
                report = await smoke.score_self_retrieval(
                    queries=queries,
                    retrieve_fn=lambda query, top_k: self._retrieve_fn(
                        query,
                        top_k,
                        generation_id=ctx.generation_id,
                        ingest_state=IngestGenerationState.STAGING
                        if ctx.generation_id is not None
                        else None,
                    ),
                    top_k=10,
                    source_id=ctx.source_id,
                )
                ctx.smoke_result = smoke.gate_decision(report=report)
                if ctx.smoke_result.decision is not smoke.GateDecision.PASS:
                    ctx.quality_warnings.append(
                        quality_warning(
                            "retrieval_smoke",
                            str(ctx.smoke_result.decision),
                            ctx.smoke_result.reason,
                        )
                    )
        except Exception as exc:  # noqa: BLE001 - optional model check is advisory
            ctx.quality_warnings.append(
                quality_warning(
                    "retrieval_smoke",
                    "audit_unavailable",
                    type(exc).__name__,
                )
            )
        if ctx.generation_id is not None:
            await self._transition_owned_generation(
                ctx,
                from_states=(IngestGenerationState.STAGING,),
                to_state=IngestGenerationState.VALIDATED,
                smoke_decision=ctx.smoke_result.decision if ctx.smoke_result else "skipped",
            )

    async def _persist_quality_warnings(self, ctx: BookRagContext) -> None:
        """Expose quality warnings on intake status and searchable passage payloads."""
        warnings = collect_quality_warnings(ctx)
        ctx.quality_warnings = warnings
        ctx.envelope.metadata["quality_warnings"] = warnings
        pool = self.pg_pool if self.pg_pool is not None else self.pool

        async def persist(surface: str, operation: Any) -> None:
            try:
                async with asyncio.timeout(5.0):
                    await operation()
            except Exception as exc:  # noqa: BLE001 - warning delivery is advisory
                warning = quality_warning(
                    "quality", "warning_persistence_pending", f"{surface}:{type(exc).__name__}"
                )
                if warning not in ctx.quality_warnings:
                    ctx.quality_warnings.append(warning)
                ctx.envelope.metadata["quality_warnings"] = ctx.quality_warnings
                log.warning(
                    "book_rag.quality_warning_persist_failed",
                    envelope_id=str(ctx.envelope.id),
                    surface=surface,
                    error_type=type(exc).__name__,
                )

        if pool is not None:

            async def write_envelope() -> None:
                await pool.execute(
                    "UPDATE intake_envelopes SET metadata = jsonb_set("
                    "COALESCE(metadata, '{}'::jsonb), '{quality_warnings}', $2::jsonb, true) "
                    "WHERE id = $1",
                    str(ctx.envelope.id),
                    list(ctx.quality_warnings),
                )

            await persist("envelope", write_envelope)
        writer = getattr(self.qdrant_store, "set_source_quality_warnings", None)
        if writer is not None and warnings:
            collections = (
                self._generation_collections()
                if ctx.generation_id
                else [
                    self.raw_collection or RAW_COLLECTION,
                    *([STERILE_COLLECTION] if ctx.sterile_point_ids else []),
                ]
            )
            for collection in collections:

                async def write_passages(collection: str = collection) -> None:
                    await writer(
                        collection,
                        ctx.source_id,
                        list(ctx.quality_warnings),
                        generation_id=ctx.generation_id,
                    )

                await persist(collection, write_passages)

        prior_raw_ids = list(ctx.raw_point_ids)
        for attribute, collection in (
            ("raw_point_ids", self.raw_collection or RAW_COLLECTION),
            ("sterile_point_ids", STERILE_COLLECTION),
        ):
            prior_ids = list(getattr(ctx, attribute))
            if not prior_ids:
                continue

            async def refresh_ids(
                attribute: str = attribute,
                collection: str = collection,
                prior_ids: list[str] = prior_ids,
            ) -> None:
                # An unverified stale physical ID must not escape on an
                # advisory delivery failure; passages remain source-queryable.
                setattr(ctx, attribute, [])
                current_ids = await current_published_ids(
                    store=self.qdrant_store,
                    collection=collection,
                    source_id=ctx.source_id,
                    point_ids=prior_ids,
                )
                setattr(ctx, attribute, current_ids)

            await persist(f"{collection}:point_ids", refresh_ids)

        if pool is not None and ctx.raw_point_ids != prior_raw_ids:

            async def refresh_artifact() -> None:
                await pool.execute(
                    "UPDATE knowledge_artifacts SET qdrant_point_ids = $2::text[], "
                    "updated_at = NOW() WHERE id = $1 AND envelope_id = $3",
                    _artifact_id(
                        ctx.source_id, self.raw_collection or RAW_COLLECTION, ctx.generation_id
                    ),
                    list(ctx.raw_point_ids),
                    str(ctx.envelope.id),
                )

            await persist("artifact:point_ids", refresh_artifact)

    # ── adapters (live seams -> harness callables) ──────────────────────────

    async def _llm_call(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        # Structured-output discipline for the synthetic-query + retrieval-smoke
        # seam: disable Qwen3 thinking so chain-of-thought can't crowd out /
        # corrupt the JSON, and strip any leaked reasoning as a backstop before
        # the caller parses the response.
        resp: LLMResponse = await self.llm_client.complete(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.model_copy(update={"text": _strip_reasoning(resp.text)})

    async def _embed(self, texts: list[str]) -> dict[str, Any]:
        # Sterile-collection embed (feeds do_sterilize, which falls back to a
        # dense-derived ColBERT for schema compat). ColBERT at ingest is opt-in.
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
        for r in results:
            payload = getattr(r, "payload", None) or {}
            chunk_id = payload.get("chunk_id") or getattr(r, "id", None)
            if chunk_id is not None:
                out.append(str(chunk_id))
        return out


def _artifact_id(
    source_id: str,
    raw_collection: str,
    generation_id: str | None = None,
) -> str:
    parts = [
        source_id,
        TargetPipeline.BOOK_RAG.value,
        ArtifactType.EXTRACTION.value,
    ]
    # Preserve the pre-generation canonical artifact identity for flag-off
    # compatibility. Every generation-scoped or custom-collection artifact is
    # collection-qualified so the same source can coexist independently.
    if generation_id is not None or raw_collection != RAW_COLLECTION:
        parts.append(raw_collection)
    if generation_id is not None:
        parts.append(generation_id)
    return deterministic_hex(
        *parts,
        length=26,
    ).upper()


def _span_source_id(ctx: BookRagContext) -> str:
    if ctx.source_id:
        return ctx.source_id

    file_hash = str(ctx.envelope.raw_content.get("file_hash", "")).lower().strip()
    if file_hash:
        return canonical_source_id(file_hash)

    return str(ctx.envelope.id)


def _artifact_title(ctx: BookRagContext, metadata: dict[str, Any]) -> str:
    parsed_title = (ctx.parsed.title if ctx.parsed else "") or ""
    return str(
        metadata.get("title")
        or metadata.get("source_book")
        or ctx.envelope.metadata.get("title")
        or parsed_title
        or ctx.source_id
    )


def _artifact_tags(metadata: dict[str, Any]) -> list[str]:
    raw_tags = metadata.get("domain_tags") or []
    if not isinstance(raw_tags, list):
        return []
    return [str(tag) for tag in raw_tags]


def _artifact_license_class(value: Any) -> LicenseClass:
    if isinstance(value, LicenseClass):
        return value
    text = str(value or LicenseClass.UNKNOWN.value).strip().lower()
    try:
        return LicenseClass(text)
    except ValueError:
        return LicenseClass.UNKNOWN


__all__ = [
    "BookRagContext",
    "BookRagPipeline",
    "IngestGenerationClaimed",
    "PipelineOutput",
]
