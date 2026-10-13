"""Postgres-polling dispatcher for book_rag ingest workers."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager, suppress
from datetime import UTC, datetime, timedelta
from types import FrameType, TracebackType
from typing import Final, Protocol

import asyncpg  # type: ignore[import-untyped]
import structlog

from vo_rag.core.config import Settings, get_settings
from vo_rag.core.db.postgres import close_pool, get_pool
from vo_rag.core.models.envelope import (
    EnvelopeStatus,
    IntakeEnvelope,
    IntakeSource,
    TargetPipeline,
)
from vo_rag.core.observability.metrics import (
    STREAM_HEARTBEAT_STALE_S,
    dispatcher_active_ingest,
    dispatcher_cancellations_total,
    dispatcher_claims_total,
    dispatcher_heartbeat_failures_total,
    dispatcher_stop_requests_total,
    record_envelope_failure,
)
from vo_rag.pipelines.book_rag.dispatcher_health import (
    DispatcherState,
    write_dispatcher_heartbeat,
)
from vo_rag.pipelines.book_rag.runner import run_book_rag_ingest
from vo_rag.pipelines.book_rag.workflow import IngestGenerationClaimed

log = structlog.get_logger(__name__)

DEFAULT_POLL_INTERVAL_S: Final = 5.0
DEFAULT_STALE_TIMEOUT: Final = timedelta(minutes=15)
DEFAULT_HEARTBEAT_INTERVAL_S: Final = 60.0
DEFAULT_SHUTDOWN_GRACE_S: Final = 240.0
#: VOR-1434: a normal lane may claim a starving `route=giant` envelope
#: after it has waited this long with no giant lane configured to take it.
DEFAULT_GIANT_FALLBACK_AFTER_S: Final = 6 * 60 * 60.0

_TERMINAL_STATUSES: Final = frozenset({EnvelopeStatus.COMPLETED, EnvelopeStatus.FAILED})

_CLAIM_SQL: Final = """
WITH candidate AS (
    SELECT id, status AS claimed_from_status
    FROM intake_envelopes
    WHERE (
            status = $1
            OR (
            status = $2
            AND updated_at < NOW() - ($3::double precision * INTERVAL '1 second')
            )
          )
      AND ($4::timestamptz IS NULL OR created_at >= $4)
      AND source_type = $5
      AND (target_pipeline IS NULL OR target_pipeline = $6)
      AND NULLIF(BTRIM(raw_content->>'garage_key'), '') IS NOT NULL
      AND NULLIF(BTRIM(raw_content->>'file_hash'), '') IS NOT NULL
      AND raw_content #>> '{object_verification,status}' = 'verified'
      AND raw_content->>'file_hash' = raw_content #>> '{object_verification,sha256}'
      AND (
            $8::text[] IS NULL
            OR lower(substring(raw_content->>'garage_key' from '[.]([^./]+)$')) = ANY($8::text[])
          )
      -- VOR-1434: never take an envelope a streaming (API/CLI/MCP) ingest is
      -- running, i.e. active_stream with a heartbeat newer than $9 seconds.
      -- A stale stream (crashed API) is claimable and the claim clears it.
      -- (COALESCE: a row without these keys yields NULL, which must not exclude it.)
      AND NOT COALESCE(
            metadata #> '{metrics,active_stream}' = 'true'::jsonb
            AND jsonb_typeof(metadata #> '{metrics,stream_heartbeat_epoch_s}') = 'number'
            AND (metadata #>> '{metrics,stream_heartbeat_epoch_s}')::double precision
                >= EXTRACT(EPOCH FROM NOW())::double precision - $9::double precision,
            false
          )
      -- VOR-1434: lane class. $10 is NULL (unset lane: claim anything,
      -- today's behaviour), 'giant' (claim only route=giant), or 'normal'
      -- (claim everything except route=giant). A route-less envelope
      -- (no preflight metadata) is DISTINCT FROM 'giant', so it stays
      -- claimable by normal and unset lanes even though the comparison
      -- against 'giant' would otherwise be NULL/unknown. A normal lane may
      -- still take a starving giant once it has waited past $11 seconds, so
      -- giants are never stuck forever when no giant lane is configured --
      -- but only while no giant lane is alive, i.e. no giant envelope is
      -- processing with an update newer than the stale timeout ($3). In a
      -- long bulk run every queued giant passes the fallback age, and a
      -- working giant lane must keep them all.
      AND (
            $10::text IS NULL
            OR ($10 = 'giant' AND metadata->'preflight'->>'route' = 'giant')
            OR (
                  $10 = 'normal'
                  AND (
                        metadata->'preflight'->>'route' IS DISTINCT FROM 'giant'
                        OR (
                              created_at < NOW() - ($11::double precision * INTERVAL '1 second')
                              AND NOT EXISTS (
                                    SELECT 1
                                    FROM intake_envelopes AS giant_lane
                                    WHERE giant_lane.status = $2
                                      AND giant_lane.metadata->'preflight'->>'route' = 'giant'
                                      AND giant_lane.updated_at
                                          >= NOW() - ($3::double precision * INTERVAL '1 second')
                                  )
                           )
                      )
                )
          )
    ORDER BY created_at ASC, id ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
UPDATE intake_envelopes AS env
SET status = $7,
    error = NULL,
    -- VOR-1434: stamp the claim time and count stale reclaims under
    -- metadata->'metrics' (see observability/metrics.py): 'stale_reclaims' is
    -- consecutive (a fresh claim resets it), 'stale_reclaims_total' never
    -- resets (crash, requeue, crash still adds up). Written so it cannot
    -- raise: a non-object metadata/metrics is left alone or replaced, and a
    -- non-numeric count restarts from 0.
    metadata = CASE WHEN jsonb_typeof(env.metadata) = 'object' THEN
        env.metadata || jsonb_build_object('metrics',
            (CASE WHEN jsonb_typeof(env.metadata->'metrics') = 'object'
                  THEN env.metadata->'metrics' ELSE '{}'::jsonb END)
            || jsonb_build_object(
                'claimed_at_epoch_s', EXTRACT(EPOCH FROM NOW()),
                -- a dispatcher owns it now: any streaming ingest's flag is stale
                'active_stream', false,
                'stale_reclaims', CASE WHEN candidate.claimed_from_status = $2 THEN
                    (CASE WHEN jsonb_typeof(env.metadata #> '{metrics,stale_reclaims}') = 'number'
                          THEN (env.metadata #> '{metrics,stale_reclaims}')::numeric ELSE 0 END) + 1
                    ELSE 0 END,
                'stale_reclaims_total',
                    (CASE
                        WHEN jsonb_typeof(env.metadata #> '{metrics,stale_reclaims_total}')
                             = 'number'
                        THEN (env.metadata #> '{metrics,stale_reclaims_total}')::numeric
                        ELSE 0 END)
                    + CASE WHEN candidate.claimed_from_status = $2 THEN 1 ELSE 0 END
            ))
        ELSE env.metadata END
FROM candidate
WHERE env.id = candidate.id
RETURNING env.*, candidate.claimed_from_status
"""

_CUTOFF_AUDIT_SQL: Final = """
SELECT
    COUNT(*) FILTER (WHERE created_at >= $4) AS claimable_with_cutoff,
    COUNT(*) AS claimable_ignoring_cutoff,
    MAX(created_at) AS newest_created_at
FROM intake_envelopes
WHERE (
        status = $1
        OR (
        status = $2
        AND updated_at < NOW() - ($3::double precision * INTERVAL '1 second')
        )
      )
  AND source_type = $5
  AND (target_pipeline IS NULL OR target_pipeline = $6)
  AND NULLIF(BTRIM(raw_content->>'garage_key'), '') IS NOT NULL
  AND NULLIF(BTRIM(raw_content->>'file_hash'), '') IS NOT NULL
  AND raw_content #>> '{object_verification,status}' = 'verified'
  AND raw_content->>'file_hash' = raw_content #>> '{object_verification,sha256}'
"""

_UPDATE_STATUS_SQL: Final = "UPDATE intake_envelopes SET status = $2 WHERE id = $1"
_FAIL_ENVELOPE_SQL: Final = "UPDATE intake_envelopes SET status = $2, error = $3 WHERE id = $1"
_RETRY_ENVELOPE_SQL: Final = """
UPDATE intake_envelopes
SET status = $2, error = $3
WHERE id = $1 AND status = $4
"""
_HEARTBEAT_SQL: Final = """
UPDATE intake_envelopes
SET status = $2
WHERE id = $1
  AND status = $2
"""


class _Transaction(Protocol):
    async def __aenter__(self) -> object: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> object | None: ...


class _Connection(Protocol):
    def transaction(self) -> _Transaction: ...

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None: ...


class _Pool(Protocol):
    def acquire(self) -> AbstractAsyncContextManager[_Connection]: ...

    async def execute(self, query: str, *args: object) -> object: ...


async def claim_next_envelope(
    conn: _Connection,
    *,
    stale_timeout: timedelta = DEFAULT_STALE_TIMEOUT,
    claim_not_before: datetime | None = None,
    only_extensions: tuple[str, ...] | None = None,
    lane_class: str | None = None,
    giant_fallback_after_s: float = DEFAULT_GIANT_FALLBACK_AFTER_S,
) -> IntakeEnvelope | None:
    """Atomically claim one admitted received or stale-processing envelope.

    ``only_extensions`` limits the claim to source files with those extensions
    (lowercase, no dot). A GPU-free lane sets it to ebook/text formats so it
    never takes a PDF that needs docling on a GPU (VOR-1427).

    ``lane_class`` (VOR-1434) is ``None`` (claim anything, today's
    behaviour), ``"giant"`` (claim only ``preflight.route == "giant"``), or
    ``"normal"`` (claim everything except a giant). ``giant_fallback_after_s``
    lets a normal lane take a giant anyway once it has waited that long, so a
    giant is never stuck forever when no giant lane is running -- but never
    while a giant lane is alive (a giant envelope processing with an update
    newer than ``stale_timeout``).
    """
    row = await conn.fetchrow(
        _CLAIM_SQL,
        EnvelopeStatus.RECEIVED,
        EnvelopeStatus.PROCESSING,
        stale_timeout.total_seconds(),
        claim_not_before,
        IntakeSource.FILE_DROP,
        TargetPipeline.BOOK_RAG,
        EnvelopeStatus.PROCESSING,
        list(only_extensions) if only_extensions else None,
        STREAM_HEARTBEAT_STALE_S,
        lane_class,
        giant_fallback_after_s,
    )
    if row is None:
        return None
    values = dict(row)
    claimed_from = values.pop("claimed_from_status", EnvelopeStatus.RECEIVED)
    kind = "stale_reclaim" if claimed_from == EnvelopeStatus.PROCESSING else "new"
    dispatcher_claims_total.labels(kind=kind).inc()
    return IntakeEnvelope.model_validate(values)


async def audit_claim_cutoff(
    conn: _Connection,
    *,
    stale_timeout: timedelta = DEFAULT_STALE_TIMEOUT,
    claim_not_before: datetime,
) -> None:
    """Warn when ``claim_not_before`` excludes every otherwise-claimable envelope.

    A cutoff newer than the whole backlog leaves the dispatcher polling forever
    with nothing to claim while systemd still reports the unit ``active`` and the
    heartbeat still reports ``idle`` — indistinguishable from an empty queue.
    That stalled ingest for four weeks (VOR-1402); saying so at startup is what
    makes the state legible.
    """
    row = await conn.fetchrow(
        _CUTOFF_AUDIT_SQL,
        EnvelopeStatus.RECEIVED,
        EnvelopeStatus.PROCESSING,
        stale_timeout.total_seconds(),
        claim_not_before,
        IntakeSource.FILE_DROP,
        TargetPipeline.BOOK_RAG,
    )
    if row is None:
        return
    raw_with_cutoff = row["claimable_with_cutoff"]
    raw_ignoring_cutoff = row["claimable_ignoring_cutoff"]
    with_cutoff = raw_with_cutoff if isinstance(raw_with_cutoff, int) else 0
    ignoring_cutoff = raw_ignoring_cutoff if isinstance(raw_ignoring_cutoff, int) else 0
    if with_cutoff or not ignoring_cutoff:
        return
    newest = row["newest_created_at"]
    log.warning(
        "book_rag_dispatcher_claim_cutoff_excludes_all",
        claim_not_before=claim_not_before.isoformat(),
        excluded_envelopes=ignoring_cutoff,
        newest_envelope_created_at=(newest.isoformat() if isinstance(newest, datetime) else None),
        remedy=(
            "clear VORAG_BOOK_RAG_DISPATCHER_CLAIM_NOT_BEFORE (empty = no date gate) "
            "to admit the backlog; the dispatcher will otherwise idle indefinitely"
        ),
    )


async def _audit_cutoff_from_pool(
    pool: _Pool,
    *,
    stale_timeout: timedelta,
    claim_not_before: datetime,
) -> None:
    async with pool.acquire() as conn:
        await audit_claim_cutoff(
            conn,
            stale_timeout=stale_timeout,
            claim_not_before=claim_not_before,
        )


async def run_dispatcher_loop(
    pool: _Pool,
    settings: Settings,
    *,
    poll_interval: float,
    stop_event: asyncio.Event,
    stale_timeout: timedelta = DEFAULT_STALE_TIMEOUT,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    shutdown_grace: float = DEFAULT_SHUTDOWN_GRACE_S,
    claim_not_before: datetime | None = None,
    heartbeat_dsn: str | None = None,
    only_extensions: tuple[str, ...] | None = None,
    lane_class: str | None = None,
    giant_fallback_after_s: float = DEFAULT_GIANT_FALLBACK_AFTER_S,
) -> None:
    """Poll Postgres, claim one envelope at a time, and run book_rag ingest.

    With ``heartbeat_dsn`` the claimed envelope's heartbeat runs on its own
    thread and connection (see ``_ThreadedHeartbeat``), so CPU-bound ingest
    steps cannot starve it into a stale reclaim.
    """
    log.info(
        "book_rag_dispatcher_loop_started",
        poll_interval=poll_interval,
        stale_timeout_s=stale_timeout.total_seconds(),
        heartbeat_interval=heartbeat_interval,
        shutdown_grace_s=shutdown_grace,
        claim_not_before=claim_not_before.isoformat() if claim_not_before else None,
        only_extensions=list(only_extensions) if only_extensions else None,
        lane_class=lane_class,
        giant_fallback_after_s=giant_fallback_after_s,
    )
    if claim_not_before is not None:
        await _audit_cutoff_from_pool(
            pool,
            stale_timeout=stale_timeout,
            claim_not_before=claim_not_before,
        )
    _publish_heartbeat("idle")
    try:
        while not stop_event.is_set():
            envelope = await _claim_from_pool(
                pool,
                stale_timeout=stale_timeout,
                claim_not_before=claim_not_before,
                only_extensions=only_extensions,
                lane_class=lane_class,
                giant_fallback_after_s=giant_fallback_after_s,
            )
            if envelope is None:
                _publish_heartbeat("idle")
                await _wait_for_next_poll(stop_event, poll_interval)
                continue
            _publish_heartbeat("processing")
            await _run_claimed_until_complete_or_stopped(
                pool,
                settings,
                envelope,
                stop_event=stop_event,
                heartbeat_interval=heartbeat_interval,
                shutdown_grace=shutdown_grace,
                heartbeat_dsn=heartbeat_dsn,
            )
            _publish_heartbeat("idle")
    finally:
        _publish_heartbeat("stopping")
        log.info("book_rag_dispatcher_loop_stopped")


async def _run_claimed_until_complete_or_stopped(
    pool: _Pool,
    settings: Settings,
    envelope: IntakeEnvelope,
    *,
    stop_event: asyncio.Event,
    heartbeat_interval: float,
    shutdown_grace: float,
    heartbeat_dsn: str | None = None,
) -> None:
    """Give one claimed ingest bounded time to checkpoint after a stop request."""
    ingest_task = asyncio.create_task(
        _run_claimed_envelope(
            pool,
            settings,
            envelope,
            heartbeat_interval=heartbeat_interval,
            heartbeat_dsn=heartbeat_dsn,
        )
    )
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {ingest_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if ingest_task in done:
            await ingest_task
            return

        log.info(
            "book_rag_dispatcher_ingest_drain_started",
            envelope_id=str(envelope.id),
            shutdown_grace_s=shutdown_grace,
        )
        try:
            await asyncio.wait_for(asyncio.shield(ingest_task), timeout=shutdown_grace)
        except TimeoutError:
            log.warning(
                "book_rag_dispatcher_ingest_drain_expired",
                envelope_id=str(envelope.id),
                shutdown_grace_s=shutdown_grace,
            )
            ingest_task.cancel()
            with suppress(asyncio.CancelledError):
                await ingest_task
    finally:
        stop_task.cancel()
        with suppress(asyncio.CancelledError):
            await stop_task
        if not ingest_task.done():
            ingest_task.cancel()
            with suppress(asyncio.CancelledError):
                await ingest_task


async def _claim_from_pool(
    pool: _Pool,
    *,
    stale_timeout: timedelta,
    claim_not_before: datetime | None,
    only_extensions: tuple[str, ...] | None = None,
    lane_class: str | None = None,
    giant_fallback_after_s: float = DEFAULT_GIANT_FALLBACK_AFTER_S,
) -> IntakeEnvelope | None:
    async with pool.acquire() as conn, conn.transaction():
        return await claim_next_envelope(
            conn,
            stale_timeout=stale_timeout,
            claim_not_before=claim_not_before,
            only_extensions=only_extensions,
            lane_class=lane_class,
            giant_fallback_after_s=giant_fallback_after_s,
        )


async def _run_claimed_envelope(
    pool: _Pool,
    settings: Settings,
    envelope: IntakeEnvelope,
    *,
    heartbeat_interval: float,
    heartbeat_dsn: str | None = None,
) -> None:
    envelope_id = str(envelope.id)
    heartbeat_stop = asyncio.Event()
    heartbeat_task: asyncio.Task[None] | None = None
    threaded_heartbeat: _ThreadedHeartbeat | None = None
    if heartbeat_interval > 0 and heartbeat_dsn:
        threaded_heartbeat = _ThreadedHeartbeat(heartbeat_dsn, envelope_id, heartbeat_interval)
        threaded_heartbeat.start()
    elif heartbeat_interval > 0:
        heartbeat_task = asyncio.create_task(
            _heartbeat_processing_envelope(
                pool,
                envelope_id,
                interval=heartbeat_interval,
                stop_event=heartbeat_stop,
            )
        )

    # Set once the envelope is persisted FAILED; the failure is counted once,
    # in ``finally``, however many handlers below persisted it (VOR-1434).
    failure_persisted = False
    dispatcher_active_ingest.inc()
    try:
        log.info("book_rag_dispatcher_ingest_started", envelope_id=envelope_id)
        output = await run_book_rag_ingest(envelope, settings=settings, pg_pool=pool)
        await _persist_terminal_status(pool, envelope_id, output.status)
        failure_persisted = output.status is EnvelopeStatus.FAILED
        log.info(
            "book_rag_dispatcher_ingest_completed",
            envelope_id=envelope_id,
            status=output.status.value,
            source_id=output.source_id,
            chunk_count=output.chunk_count,
        )
    except asyncio.CancelledError:
        dispatcher_cancellations_total.inc()
        log.warning(
            "book_rag_dispatcher_ingest_interrupted",
            envelope_id=envelope_id,
            retryable=True,
        )
        try:
            await _retry_envelope(pool, envelope_id)
        except Exception as persist_exc:  # noqa: BLE001
            log.error(
                "book_rag_dispatcher_ingest_retry_persist_failed",
                envelope_id=envelope_id,
                error=str(persist_exc),
                exc_info=True,
            )
        raise
    except IngestGenerationClaimed as exc:
        # A concurrent runner owns this deterministic generation. Leave the
        # envelope processing; that owner (or stale reclaim after a crash) is
        # authoritative, and failing/requeueing here would create more overlap.
        log.warning(
            "book_rag_dispatcher_generation_already_claimed",
            envelope_id=envelope_id,
            error=str(exc),
        )
    except Exception as exc:
        log.error(
            "book_rag_dispatcher_ingest_failed",
            envelope_id=envelope_id,
            error=str(exc),
            exc_info=True,
        )
        try:
            await _fail_envelope(pool, envelope_id, str(exc))
            failure_persisted = True
        except Exception as persist_exc:  # noqa: BLE001
            log.error(
                "book_rag_dispatcher_ingest_fail_persist_failed",
                envelope_id=envelope_id,
                error=str(persist_exc),
                original_error=str(exc),
                exc_info=True,
            )
    finally:
        if failure_persisted:
            await record_envelope_failure(pool, envelope_id)
        heartbeat_stop.set()
        if heartbeat_task is not None:
            await heartbeat_task
        if threaded_heartbeat is not None:
            await asyncio.to_thread(threaded_heartbeat.stop)
        dispatcher_active_ingest.dec()


async def _persist_terminal_status(
    pool: _Pool,
    envelope_id: str,
    status: EnvelopeStatus,
) -> None:
    if status not in _TERMINAL_STATUSES:
        raise RuntimeError(f"book_rag_dispatcher_non_terminal_status:{status.value}")
    try:
        await _update_envelope_status(pool, envelope_id, status)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "book_rag_dispatcher_success_persist_failed",
            envelope_id=envelope_id,
            status=status.value,
            error=str(exc),
            exc_info=True,
        )
        raise


async def _heartbeat_processing_envelope(
    pool: _Pool,
    envelope_id: str,
    *,
    interval: float,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        await _wait_for_next_poll(stop_event, interval)
        if stop_event.is_set():
            return
        try:
            await pool.execute(_HEARTBEAT_SQL, envelope_id, EnvelopeStatus.PROCESSING)
            _publish_heartbeat("processing")
        except Exception as exc:  # noqa: BLE001
            dispatcher_heartbeat_failures_total.labels(target="database").inc()
            log.error(
                "book_rag_dispatcher_heartbeat_failed",
                envelope_id=envelope_id,
                error=str(exc),
                exc_info=True,
            )


class _ThreadedHeartbeat:
    """Keep a claimed envelope's heartbeat alive from its own thread (VOR-1429).

    The asyncio heartbeat shares the event loop with the ingest, so any long
    synchronous step (a figure-heavy visual build ran 3+ h) stopped it; after
    the stale timeout another lane reclaimed the *live* book, and in turn every
    lane ended up on the same book. A thread keeps beating through CPU-bound
    Python (the GIL is released between bytecode slices), using a short-lived
    connection per beat because asyncpg pools are bound to their own loop.
    """

    def __init__(self, dsn: str, envelope_id: str, interval: float) -> None:
        self._dsn = dsn
        self._envelope_id = envelope_id
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"book-rag-heartbeat-{envelope_id}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                asyncio.run(self._beat())
                _publish_heartbeat("processing")
            except Exception as exc:  # noqa: BLE001
                dispatcher_heartbeat_failures_total.labels(target="database").inc()
                log.error(
                    "book_rag_dispatcher_heartbeat_failed",
                    envelope_id=self._envelope_id,
                    error=str(exc),
                    exc_info=True,
                )

    async def _beat(self) -> None:
        conn = await asyncpg.connect(self._dsn, statement_cache_size=0, timeout=30)
        try:
            await conn.execute(_HEARTBEAT_SQL, self._envelope_id, EnvelopeStatus.PROCESSING.value)
        finally:
            await conn.close()


def _publish_heartbeat(state: DispatcherState) -> None:
    try:
        write_dispatcher_heartbeat(state)
    except Exception as exc:  # noqa: BLE001
        dispatcher_heartbeat_failures_total.labels(target="file").inc()
        log.error(
            "book_rag_dispatcher_heartbeat_publish_failed",
            state=state,
            error=str(exc),
        )


async def _update_envelope_status(
    pool: _Pool,
    envelope_id: str,
    status: EnvelopeStatus,
) -> None:
    await pool.execute(_UPDATE_STATUS_SQL, envelope_id, status)


async def _fail_envelope(pool: _Pool, envelope_id: str, error: str) -> None:
    await pool.execute(_FAIL_ENVELOPE_SQL, envelope_id, EnvelopeStatus.FAILED, error)


async def _retry_envelope(pool: _Pool, envelope_id: str) -> None:
    await pool.execute(
        _RETRY_ENVELOPE_SQL,
        envelope_id,
        EnvelopeStatus.RECEIVED,
        "dispatcher interrupted during shutdown; retryable from checkpoint",
        EnvelopeStatus.PROCESSING,
    )


async def _wait_for_next_poll(stop_event: asyncio.Event, interval: float) -> None:
    if interval <= 0:
        await asyncio.sleep(0)
        return
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=interval)
    except TimeoutError:
        return


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning("book_rag_dispatcher_invalid_float_env", name=name, value=raw)
        return default
    if value < 0:
        log.warning("book_rag_dispatcher_negative_float_env", name=name, value=raw)
        return default
    return value


def _env_datetime(name: str) -> datetime | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if value.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return value.astimezone(UTC)


def _env_extensions(name: str) -> tuple[str, ...] | None:
    """Comma-separated extensions (``epub,mobi``), lowercased; unset or empty = any."""
    raw = os.environ.get(name, "")
    values = tuple(
        sorted({part.strip().lower().lstrip(".") for part in raw.split(",") if part.strip()})
    )
    return values or None


_VALID_LANE_CLASSES: Final = frozenset({"giant", "normal"})


def _env_lane_class(name: str) -> str | None:
    """``VORAG_BOOK_RAG_LANE_CLASS`` (VOR-1434): ``"giant"``, ``"normal"``,
    or unset/empty/invalid, which means claim anything (today's behaviour)."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw not in _VALID_LANE_CLASSES:
        log.warning("book_rag_dispatcher_invalid_lane_class_env", name=name, value=raw)
        return None
    return raw


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def _request_stop(signum: int | None = None, frame: FrameType | None = None) -> None:
        del frame
        dispatcher_stop_requests_total.inc()
        log.info("book_rag_dispatcher_stop_requested", signal=signum)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop, sig)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, _request_stop)


async def _amain() -> None:
    settings = get_settings()
    pool = await get_pool(
        settings.database.url,
        min_size=settings.database.pool_min,
        max_size=settings.database.pool_max,
    )
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    try:
        await run_dispatcher_loop(
            pool,
            settings,
            poll_interval=_env_float(
                "VORAG_BOOK_RAG_DISPATCHER_POLL_INTERVAL_S",
                DEFAULT_POLL_INTERVAL_S,
            ),
            stop_event=stop_event,
            stale_timeout=timedelta(
                seconds=_env_float(
                    "VORAG_BOOK_RAG_DISPATCHER_STALE_TIMEOUT_S",
                    DEFAULT_STALE_TIMEOUT.total_seconds(),
                )
            ),
            heartbeat_interval=_env_float(
                "VORAG_BOOK_RAG_DISPATCHER_HEARTBEAT_INTERVAL_S",
                DEFAULT_HEARTBEAT_INTERVAL_S,
            ),
            shutdown_grace=_env_float(
                "VORAG_BOOK_RAG_DISPATCHER_SHUTDOWN_GRACE_S",
                DEFAULT_SHUTDOWN_GRACE_S,
            ),
            claim_not_before=_env_datetime("VORAG_BOOK_RAG_DISPATCHER_CLAIM_NOT_BEFORE"),
            heartbeat_dsn=settings.database.url,
            # GPU-free lanes (CUDA_VISIBLE_DEVICES empty) set this to ebook and
            # text formats; GPU lanes leave it unset and take everything.
            only_extensions=_env_extensions("VORAG_BOOK_RAG_DISPATCHER_ONLY_EXTENSIONS"),
            # VOR-1434: a dedicated giant lane sets LANE_CLASS=giant;
            # everything else stays unset (claim anything) or is set to
            # normal (claim everything except a giant, with a starvation
            # fallback after GIANT_FALLBACK_AFTER_S).
            lane_class=_env_lane_class("VORAG_BOOK_RAG_LANE_CLASS"),
            giant_fallback_after_s=_env_float(
                "VORAG_BOOK_RAG_GIANT_FALLBACK_AFTER_S",
                DEFAULT_GIANT_FALLBACK_AFTER_S,
            ),
        )
    finally:
        await close_pool()


def main() -> None:
    """Run the dispatcher as a long-lived systemd process."""
    asyncio.run(_amain())


__all__ = [
    "DEFAULT_GIANT_FALLBACK_AFTER_S",
    "DEFAULT_HEARTBEAT_INTERVAL_S",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_SHUTDOWN_GRACE_S",
    "DEFAULT_STALE_TIMEOUT",
    "audit_claim_cutoff",
    "claim_next_envelope",
    "main",
    "run_dispatcher_loop",
]


if __name__ == "__main__":
    main()
