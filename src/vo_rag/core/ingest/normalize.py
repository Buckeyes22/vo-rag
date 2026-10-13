"""Intake submission normalization.

The ``_validate_source_data`` helper is intentionally narrow: it only
checks that an optional ``source_data["pipeline_hint"]`` (when present)
is a valid ``TargetPipeline`` string. Triage, disposition and priority
logic live in the pipelines (book_rag), not here.

Notes:

- ``IntakeSubmission`` uses ``extra='forbid'`` (inherited from
  ``VoBaseModel``) so unknown fields fail loudly at the API boundary.
- ``FILE_DROP`` requires ``raw_content.garage_key``; optional
  ``file_hash`` is lowercased and stripped so the book_rag register
  step can use it for canonical ``source_id`` derivation without
  re-downloading the file from Garage.
- ``URL`` requires ``http://`` or ``https://``; ``javascript:``,
  ``ftp://``, etc. are rejected.
- ``TEXT_PROMPT`` requires non-empty ``raw_content.text``.
- ``API`` and ``GITHUB_STAR`` pass ``raw_content`` through unchanged
  — their content shape is opaque to the ingest layer.
- All validation failures raise ``fastapi.HTTPException(status_code=422, ...)``
  (a 422-shaped contract), so the API layer can render
  them without translation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from pydantic import Field

from vo_rag.core.ingest.object_verification import (
    ObjectVerificationError,
    normalize_sha256_assertion,
)
from vo_rag.core.models.common import VoBaseModel, ulid_new
from vo_rag.core.models.envelope import (
    EnvelopeStatus,
    IntakeEnvelope,
    IntakeSource,
    Priority,
    Sovereignty,
    TargetPipeline,
)


class InvalidPipelineHint(ValueError):
    """Raised when ``source_data.pipeline_hint`` is present but unsupported.

    The detail string is the API's 422 response body, so it must stay
    byte-for-byte stable for client tests.
    """

    def __init__(self, value: object) -> None:
        self.value = value
        super().__init__(f"invalid_pipeline_hint:{value}")


def _resolve_pipeline_hint_from_source_data(
    source_data: dict[str, Any],
) -> TargetPipeline | None:
    """Narrow local pipeline-hint validator.

    Returns the validated ``TargetPipeline`` if a hint is present, or
    ``None`` if absent. Raises ``InvalidPipelineHint`` for any value
    that is not a valid ``TargetPipeline`` string. Non-string values
    are also rejected — the hint is a string enum key, period.

    This deliberately does NOT route, do triage, or assign a disposition.
    That logic lives in a later task.
    """
    if "pipeline_hint" not in source_data:
        return None
    value = source_data["pipeline_hint"]
    if not isinstance(value, str):
        raise InvalidPipelineHint(value)
    try:
        return TargetPipeline(value)
    except ValueError as exc:
        raise InvalidPipelineHint(value) from exc


#: ``metadata`` keys only the server writes (VOR-1434):
#: ``metrics`` drives claim eligibility (``active_stream`` plus a far-future
#: heartbeat would hide an envelope from every dispatcher) and ``quality`` is
#: the per-book gate's verdict. A submitted copy is dropped here, the one
#: normalization step every intake surface (API, CLI, MCP) runs;
#: ``preflight`` stays client-settable.
SERVER_OWNED_METADATA_KEYS = frozenset({"metrics", "quality"})


class IntakeSubmission(VoBaseModel):
    """Submitted intake envelope, before normalization."""

    source_type: IntakeSource
    source_data: dict[str, Any] = Field(default_factory=dict)
    raw_content: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    sovereignty: Sovereignty = Sovereignty.CLOUD_OK
    priority: Priority = Priority.MEDIUM


def normalize(submission: IntakeSubmission) -> IntakeEnvelope:
    """Normalize a submission into an envelope ready for triage + dispatch.

    Steps:
        1. Validate ``source_data.pipeline_hint`` (narrow, see helper).
        2. Run source-type-specific ``raw_content`` normalization
           (FILE_DROP, URL, TEXT_PROMPT have required fields; API and
           GITHUB_STAR pass through).
        3. Drop server-owned ``metadata`` keys (``SERVER_OWNED_METADATA_KEYS``).
        4. Mint a new ULID, set status to ``RECEIVED``, stamp UTC
           created_at and updated_at.

    Raises:
        HTTPException(422): any source-type-specific validation failure
            or invalid pipeline hint.
    """
    now = datetime.now(UTC)
    _validate_source_data(submission.source_data)
    raw_content = _normalize_raw(submission.source_type, submission.raw_content)
    return IntakeEnvelope(
        id=ulid_new(),
        source_type=submission.source_type,
        source_data=submission.source_data,
        raw_content=raw_content,
        metadata={
            key: value
            for key, value in submission.metadata.items()
            if key not in SERVER_OWNED_METADATA_KEYS
        },
        sovereignty=submission.sovereignty,
        priority=submission.priority,
        status=EnvelopeStatus.RECEIVED,
        created_at=now,
        updated_at=now,
    )


def _validate_source_data(source_data: dict[str, Any]) -> None:
    try:
        _resolve_pipeline_hint_from_source_data(source_data)
    except InvalidPipelineHint as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _normalize_raw(source_type: IntakeSource, raw_content: dict[str, Any]) -> dict[str, Any]:
    if source_type is IntakeSource.FILE_DROP:
        if "garage_key" not in raw_content:
            raise HTTPException(status_code=422, detail="file_drop requires raw_content.garage_key")
        normalized: dict[str, Any] = {
            "garage_key": raw_content["garage_key"],
            "filename": raw_content.get("filename"),
            "content_type": raw_content.get("content_type"),
        }
        try:
            file_hash = normalize_sha256_assertion(raw_content.get("file_hash"))
        except ObjectVerificationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if file_hash is not None:
            normalized["file_hash"] = file_hash
        # These fields are produced by the authoritative Garage verifier.  The
        # intake API always performs that verification before normalization.
        if "size_bytes" in raw_content:
            normalized["size_bytes"] = raw_content["size_bytes"]
        if "object_verification" in raw_content:
            normalized["object_verification"] = raw_content["object_verification"]
        return normalized

    if source_type is IntakeSource.URL:
        url = (raw_content.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=422,
                detail="url requires raw_content.url with http(s) scheme",
            )
        return {"url": url, "title": raw_content.get("title")}

    if source_type is IntakeSource.TEXT_PROMPT:
        text = (raw_content.get("text") or "").strip()
        if not text:
            raise HTTPException(
                status_code=422,
                detail="text_prompt requires non-empty raw_content.text",
            )
        return {"text": text, "title": raw_content.get("title")}

    if source_type is IntakeSource.API:
        return raw_content

    if source_type is IntakeSource.GITHUB_STAR:
        return raw_content

    raise HTTPException(status_code=422, detail=f"unsupported source_type {source_type}")


__all__ = [
    "IntakeSubmission",
    "InvalidPipelineHint",
    "normalize",
]
