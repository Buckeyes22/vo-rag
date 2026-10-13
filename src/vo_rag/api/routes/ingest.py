"""API ingest routes — POST /api/v1/intake and POST /api/v1/intake/upload-url.

Behavior notes:

- The /api/v1/intake route is the *only* path that does core ingest — it
  converts the API-side ``IntakeSubmission`` to the core shape, calls
  ``normalize()`` exactly once, persists the resulting ``IntakeEnvelope``
  through ``vo_rag.core.db.queries.envelopes.insert_envelope`` (no inline
  SQL), and returns the envelope + ``dispatch_status``.

- The ``dispatch_status`` field is explicit: a fresh supervised PostgreSQL
  dispatcher reports ``queued`` only for complete book-rag file-drop envelopes.
  Other source/pipeline shapes remain ``received`` with
  ``dispatch_status="not_configured"`` rather than being claimed into a known
  validation failure.

- For ``source_type="url"`` the route runs the SSRF gate (F-014) via
  ``vo_rag.web.url_safety.validate_external_url`` BEFORE the DB insert.
  ``SSRFBlocked`` is translated to a 422 with an operator-readable detail.
  The SSRF gate is the single chokepoint for all URL intake — every later
  pipeline that fetches the URL must reuse the resolved IP for connect
  pinning (F-048) and re-validate the final URL after redirects (F-049).

- For ``source_type="file_drop"`` the route hashes the stored Garage bytes,
  checks optional client size/type/hash assertions, and conditionally promotes
  the verified ETag to a server-owned content-addressed key before normalization
  or persistence. Client hashes never own canonical identity.

- The /upload-url route sanitizes filenames through
  ``vo_rag.api.utils.secure_filename``, mints a ULID-scoped key
  ``intake/{ulid}/{filename}``, asks ``GarageClient.generate_presigned_upload``
  for an S3 PUT URL with a 6h TTL, and respects
  ``settings.portal.upload_max_bytes`` (smaller wins over the hard 2 GiB
  schema ceiling).

- GitHub star webhook is dropped: ``repo_analysis`` is still a planned
  pipeline and is not built. ``api/middleware.py:EXEMPT_PATHS``
  already does not include ``/webhooks/github/star``.

``extra='forbid'`` on every request model (the API-side
``IntakeSubmission``, ``UploadUrlRequest``) is what makes unknown-field
tests pass at the boundary — typos like ``source_typey`` are 422'd by
Pydantic, not silently dropped.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from vo_rag.api.schemas import (
    EnvelopeResponse,
    IntakeSubmission,
    UploadUrlRequest,
    UploadUrlResponse,
)
from vo_rag.api.utils import secure_filename
from vo_rag.core.db.queries.envelopes import (
    fail_envelope,
    get_envelope,
    insert_envelope,
    update_envelope_status,
)
from vo_rag.core.ingest.normalize import (
    IntakeSubmission as CoreIntakeSubmission,
)
from vo_rag.core.ingest.normalize import (
    normalize as core_normalize,
)
from vo_rag.core.ingest.object_verification import (
    ObjectVerificationError,
    delete_rejected_staged_object,
    object_is_verified,
    verify_file_drop_object,
)
from vo_rag.core.models.common import ulid_new
from vo_rag.core.models.envelope import (
    IntakeSource,
    Priority,
    Sovereignty,
    TargetPipeline,
)
from vo_rag.pipelines.book_rag.dispatcher_health import dispatcher_health
from vo_rag.web.url_safety import SSRFBlocked, validate_external_url

router = APIRouter()
log = structlog.get_logger(__name__)


# Dispatch status values returned to the client. The durable Postgres row is
# queue state, but only an envelope satisfying the book-rag worker's input
# contract is eligible for this dispatcher.
_DISPATCH_NOT_CONFIGURED = "not_configured"
_DISPATCH_QUEUED = "queued"


def _book_rag_dispatchable(envelope: object) -> bool:
    source_type = getattr(envelope, "source_type", None)
    target_pipeline = getattr(envelope, "target_pipeline", None)
    raw_content = getattr(envelope, "raw_content", {})
    if source_type is not IntakeSource.FILE_DROP:
        return False
    if target_pipeline not in {None, TargetPipeline.BOOK_RAG}:
        return False
    if not isinstance(raw_content, dict):
        return False
    garage_key = raw_content.get("garage_key")
    return (
        isinstance(garage_key, str) and bool(garage_key.strip()) and object_is_verified(raw_content)
    )


async def stream_book_rag_ingest(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
    """Lazy proxy kept as the stable monkeypatch seam for stream route tests."""
    from vo_rag.pipelines.book_rag.runner import stream_book_rag_ingest as _stream_book_rag_ingest

    async for event in _stream_book_rag_ingest(*args, **kwargs):
        yield event


def _to_core_submission(submission: IntakeSubmission) -> CoreIntakeSubmission:
    """Convert the API-side ``IntakeSubmission`` to the core shape.

    The core ``IntakeSubmission`` requires ``IntakeSource`` / ``Priority`` /
    ``Sovereignty`` enums; the API-side model carries them as plain
    strings so Pydantic ``extra='forbid'`` can reject unknown source
    types at the boundary without bringing the enum in. Construction
    errors are converted to a 422 with the offending field name.
    """
    try:
        return CoreIntakeSubmission(
            source_type=IntakeSource(submission.source_type),
            source_data=dict(submission.source_data),
            raw_content=dict(submission.raw_content),
            metadata=dict(submission.metadata),
            sovereignty=Sovereignty(submission.sovereignty),
            priority=Priority(submission.priority),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid_intake_field:{exc}") from exc


async def _validate_url_for_intake(url: str) -> None:
    """Run the F-014 SSRF gate on a URL submission BEFORE normalization.

    Raises:
        HTTPException: 422 with an operator-readable detail when the
            URL fails the gate. The DB insert never runs on failure.
    """
    try:
        await validate_external_url(url)
    except SSRFBlocked as exc:
        raise HTTPException(
            status_code=422,
            detail=f"ssrf_blocked:{exc.reason}",
        ) from exc


@router.post("/api/v1/intake")
async def submit_intake(submission: IntakeSubmission, request: Request) -> EnvelopeResponse:
    """Submit an intake envelope and persist it.

    The route is intentionally narrow:

    1. If ``source_type="url"``, run the SSRF gate first. No DB write
       happens on a blocked URL.
    2. Convert the API submission to the core shape and call
       ``normalize()`` exactly once.
    3. Persist the resulting ``IntakeEnvelope`` through
       ``vo_rag.core.db.queries.envelopes.insert_envelope``.
    4. Report queued only for a complete book-rag file drop while the
       independently supervised dispatcher heartbeat is fresh. Other shapes
       remain durably registered and explicitly ``not_configured``.
    """
    pool = request.app.state.db_pool
    if pool is None:
        # Do not copy/promote a file before knowing its envelope can be stored.
        raise HTTPException(status_code=503, detail="db_pool_unavailable: cannot persist envelope")

    if submission.source_type == IntakeSource.URL.value:
        url = submission.raw_content.get("url")
        if not isinstance(url, str) or not url:
            raise HTTPException(
                status_code=422, detail="url requires raw_content.url with http(s) scheme"
            )
        await _validate_url_for_intake(url)

    if submission.source_type == IntakeSource.FILE_DROP.value:
        await _verify_file_drop_for_intake(submission, request)

    core_submission = _to_core_submission(submission)
    envelope = core_normalize(core_submission)

    try:
        await insert_envelope(pool, envelope)
    except Exception as exc:
        log.error(
            "intake_envelope_persist_failed",
            envelope_id=str(envelope.id),
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="db_persist_unavailable: envelope was not stored; upload remains retryable",
        ) from exc
    if submission.source_type == IntakeSource.FILE_DROP.value:
        await _delete_staged_after_envelope_commit(envelope.raw_content, request)

    # The durable Postgres row is the queue. Report queued only when the
    # independently supervised poller has a fresh content-free heartbeat.
    dispatch_status = _DISPATCH_NOT_CONFIGURED
    if _book_rag_dispatchable(envelope) and dispatcher_health().get("ok") is True:
        dispatch_status = _DISPATCH_QUEUED

    response = EnvelopeResponse.from_envelope(envelope)
    dumped = response.model_dump()
    dumped["dispatch_status"] = dispatch_status
    return EnvelopeResponse.model_validate(dumped)


async def _delete_staged_after_envelope_commit(
    raw_content: dict[str, Any],
    request: Request,
) -> None:
    """Best-effort staged-key cleanup after the durable envelope exists."""
    verification = raw_content.get("object_verification")
    if not isinstance(verification, dict):
        return
    staged_key = verification.get("staged_key")
    authoritative_key = verification.get("authoritative_key")
    if staged_key == authoritative_key:
        return
    settings = request.app.state.settings
    garage = request.app.state.garage
    try:
        await delete_rejected_staged_object(
            garage,
            settings.garage.buckets.artifacts,
            staged_key,
        )
    except Exception as exc:  # noqa: BLE001 - envelope is already durable
        log.warning(
            "intake_staged_object_cleanup_failed",
            envelope_garage_key=authoritative_key,
            error_type=type(exc).__name__,
        )


async def _verify_file_drop_for_intake(
    submission: IntakeSubmission,
    request: Request,
) -> None:
    """Replace client identity fields with metadata derived from Garage bytes."""
    settings = getattr(request.app.state, "settings", None)
    garage = getattr(request.app.state, "garage", None)
    if settings is None or garage is None:
        raise HTTPException(status_code=503, detail="garage_unavailable: cannot verify file drop")
    bucket = settings.garage.buckets.artifacts
    max_bytes = int(getattr(settings.portal, "upload_max_bytes", 2 * 1024 * 1024 * 1024))
    key = submission.raw_content.get("garage_key")
    try:
        submission.raw_content = await verify_file_drop_object(
            submission.raw_content,
            garage=garage,
            bucket=bucket,
            max_bytes=max_bytes,
        )
        submission.source_data = dict(submission.source_data)
        submission.source_data["garage_key"] = submission.raw_content["garage_key"]
    except ObjectVerificationError as exc:
        try:
            await delete_rejected_staged_object(garage, bucket, key)
        except Exception as cleanup_exc:  # noqa: BLE001 - preserve verification error
            log.warning(
                "intake_rejected_object_cleanup_failed",
                garage_key=key,
                error_type=type(cleanup_exc).__name__,
            )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.error(
            "intake_object_verification_unavailable",
            garage_key=key,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="garage_verification_unavailable: stored bytes could not be verified",
        ) from exc


@router.post("/api/v1/intake/upload-url")
async def upload_url(
    body: UploadUrlRequest,
    request: Request,
) -> UploadUrlResponse:
    """Return a presigned S3 PUT URL for an intake file drop.

    The filename is sanitized through ``secure_filename`` (already
    invoked by the Pydantic field validator — the value on ``body`` is
    safe by the time it reaches here), the key is ULID-scoped
    (``intake/{ulid}/{filename}``), and the bucket is
    ``settings.garage.buckets.artifacts``.

    The hard 2 GiB schema ceiling on ``size_bytes`` is enforced by
    ``UploadUrlRequest``. The route additionally checks
    ``settings.portal.upload_max_bytes`` so a smaller portal cap wins
    (enforced whenever ``settings.portal.upload_max_bytes`` is available).
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is not None:
        portal = getattr(settings, "portal", None)
        max_bytes = getattr(portal, "upload_max_bytes", None)
        if isinstance(max_bytes, int) and max_bytes > 0 and body.size_bytes > max_bytes:
            raise HTTPException(
                status_code=422,
                detail=f"size_exceeds_max: size_bytes={body.size_bytes} max={max_bytes}",
            )

    garage = getattr(request.app.state, "garage", None)
    if garage is None:
        raise HTTPException(status_code=503, detail="garage_unavailable: cannot mint presigned URL")

    safe_filename = secure_filename(body.filename)
    if not safe_filename:
        safe_filename = "upload.bin"
    bucket = settings.garage.buckets.artifacts  # type: ignore[union-attr]
    key = f"intake/{ulid_new()}/{safe_filename}"
    upload_url = await garage.generate_presigned_upload(
        bucket=bucket,
        key=key,
        ttl=21600,
        content_type=body.content_type,
        content_length=body.size_bytes,
    )
    return UploadUrlResponse(
        upload_url=upload_url,
        garage_key=key,
        bucket=bucket,
        expires_in_s=21600,
    )


@router.get("/api/v1/intake/{envelope_id}/stream")
async def stream_intake(envelope_id: str, request: Request) -> EventSourceResponse:
    """Server-Sent Events stream of one ingest run's per-step progress.

    This GET starts a new in-process ingest runner for the envelope; it is not
    an attach-only status view. A client disconnect only closes the SSE
    generator and does not abort the runner. The envelope status remains the
    source of truth.
    """
    services = getattr(request.app.state, "surface_services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="surface_services_unavailable")

    pool = services.db_pool
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail=f"db_pool_unavailable: {services.db_pool_unavailable_reason}",
        )

    envelope = await get_envelope(pool, envelope_id)
    if envelope is None:
        raise HTTPException(status_code=404, detail=f"envelope_not_found: {envelope_id}")

    # Lazy imports: stream requests are allowed to load the book_rag runtime,
    # but importing the FastAPI app must stay lightweight for the base surface.
    # The module-level ``stream_book_rag_ingest`` proxy above remains the stable
    # monkeypatch seam for consumer tests.
    from vo_rag.pipelines.book_rag.events import ImageDescribedEvent, IngestProgress
    from vo_rag.pipelines.book_rag.workflow import PipelineOutput

    def _output_json(output: PipelineOutput) -> str:
        return json.dumps(
            {
                "source_id": output.source_id,
                "status": output.status.value,
                "qdrant_point_ids": output.qdrant_point_ids,
                "raw_point_ids": output.raw_point_ids,
                "sterile_point_ids": output.sterile_point_ids,
                "smoke_decision": (
                    output.smoke_decision.value if output.smoke_decision is not None else None
                ),
                "chunk_count": output.chunk_count,
                "idempotent_noop": output.idempotent_noop,
                "quality_warnings": output.quality_warnings,
            }
        )

    queue: asyncio.Queue[dict[str, str] | BaseException | None] = asyncio.Queue()

    def _event_to_sse(ev: object) -> dict[str, str] | None:
        if isinstance(ev, PipelineOutput):
            return {"event": "done", "data": _output_json(ev)}
        if isinstance(ev, IngestProgress):
            return {"event": ev.step, "data": ev.model_dump_json()}
        if isinstance(ev, ImageDescribedEvent):
            return {"event": "image_described", "data": ev.model_dump_json()}
        return None

    async def _run_ingest_to_queue() -> None:
        try:
            terminal_seen = False
            async for ev in stream_book_rag_ingest(
                envelope,
                settings=services.settings,
                pg_pool=pool,
            ):
                if isinstance(ev, PipelineOutput):
                    terminal_seen = True
                    try:
                        await update_envelope_status(pool, str(envelope.id), ev.status)
                    except Exception as persist_exc:  # noqa: BLE001
                        log.error(
                            "book_rag_sse_success_persist_failed",
                            envelope_id=str(envelope.id),
                            status=ev.status.value,
                            error=str(persist_exc),
                            exc_info=True,
                        )
                item = _event_to_sse(ev)
                if item is not None:
                    await queue.put(item)
            if not terminal_seen:
                raise RuntimeError("ingest stream completed without output")
        except (Exception, asyncio.CancelledError) as exc:
            log.error(
                "book_rag_sse_ingest_failed",
                envelope_id=str(envelope.id),
                error=str(exc),
                exc_info=True,
            )
            try:
                await fail_envelope(pool, str(envelope.id), str(exc))
            except Exception as persist_exc:
                log.error(
                    "book_rag_sse_ingest_fail_persist_failed",
                    envelope_id=str(envelope.id),
                    error=str(persist_exc),
                    original_error=str(exc),
                    exc_info=True,
                )
            await queue.put(exc)
        finally:
            await queue.put(None)

    ingest_task = asyncio.create_task(_run_ingest_to_queue())

    async def event_gen() -> AsyncGenerator[dict[str, str], None]:
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if ingest_task.done() and not ingest_task.cancelled():
                ingest_task.exception()

    return EventSourceResponse(event_gen())


__all__ = ["router", "submit_intake", "upload_url", "stream_intake"]
