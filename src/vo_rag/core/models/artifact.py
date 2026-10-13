"""KnowledgeArtifact, IdeaRecord, and related types."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from .common import ULID, VoBaseModel
from .envelope import Priority, TargetPipeline


class LicenseClass(StrEnum):
    CLEARED = "cleared"
    PENDING = "pending"
    UNLICENSED = "unlicensed"
    PUBLIC_DOMAIN = "public_domain"
    UNKNOWN = "unknown"


class TrustLevel(StrEnum):
    """Injection-defense trust tag on any stored artifact / chunk."""

    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    QUARANTINED = "quarantined"


class ArtifactType(StrEnum):
    SUMMARY = "summary"
    EVALUATION = "evaluation"
    IDEA = "idea"
    RESEARCH = "research"
    EXTRACTION = "extraction"
    SNAPSHOT = "snapshot"
    COACHING_RECORD = "coaching_record"


class ProvenanceEntry(VoBaseModel):
    source_id: str
    source_title: str | None = None
    source_type: str
    contribution: float = Field(ge=0.0, le=1.0)
    page_refs: list[int] = Field(default_factory=list)


class KnowledgeArtifact(VoBaseModel):
    id: ULID
    envelope_id: ULID
    pipeline: TargetPipeline
    artifact_type: ArtifactType
    title: str
    content: dict[str, Any]
    tags: list[str] = Field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    status: str = "active"
    license_class: LicenseClass = LicenseClass.UNKNOWN
    # F-031: fail-closed — defaults to QUARANTINED. Callers that have run the
    # compliance gate must opt IN to TRUSTED explicitly. Forgetting to pass
    # trust_level on a new pipeline now puts artifacts in quarantine instead
    # of granting trust.
    trust_level: TrustLevel = TrustLevel.QUARANTINED
    qdrant_collection: str | None = None
    qdrant_point_ids: list[str] = Field(default_factory=list)
    source_provenance: list[ProvenanceEntry] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class IdeaStatus(StrEnum):
    CAPTURED = "captured"
    EXPLORING = "exploring"
    BUILDING = "building"
    PARKED = "parked"
    DISCARDED = "discarded"


class IdeaRecord(VoBaseModel):
    id: ULID
    artifact_id: ULID | None = None
    core_concept: str
    domain: str | None = None
    applications: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    related_ids: list[ULID] = Field(default_factory=list)
    status: IdeaStatus = IdeaStatus.CAPTURED
    project_brief: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
