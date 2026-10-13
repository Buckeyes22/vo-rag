"""Generation-scoped publication state for one source ingest attempt."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from .common import VoBaseModel


class IngestGenerationState(StrEnum):
    STAGING = "staging"
    VALIDATED = "validated"
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class SourceIngestGeneration(VoBaseModel):
    generation_id: str
    source_id: str
    raw_collection: str
    envelope_id: str
    run_id: str | None = None
    pipeline: str = "book_rag"
    schema_version: str
    chunking_version: str
    sterilize_enabled: bool = False
    owner_token: str | None = None
    claim_expires_at: datetime | None = None
    state: IngestGenerationState = IngestGenerationState.STAGING
    expected_raw_points: int | None = None
    actual_raw_points: int = 0
    expected_sterile_points: int | None = None
    actual_sterile_points: int = 0
    expected_artifacts: int = 1
    actual_artifacts: int = 0
    validation_decision: dict[str, object] | None = None
    smoke_decision: str | None = None
    failure_reason: str | None = None
    cleanup_required: bool = False
    cleanup_error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    activated_at: datetime | None = None
    completed_at: datetime | None = None


__all__ = ["IngestGenerationState", "SourceIngestGeneration"]
