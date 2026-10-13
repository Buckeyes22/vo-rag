"""Compliance log + similarity / citation primitives."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from .common import ULID, VoBaseModel


class CitationTier(StrEnum):
    GENERAL_KNOWLEDGE = "general_knowledge"
    AUTHOR_ASSOCIATED = "author_associated"
    PARAPHRASED = "paraphrased"
    DIRECT_QUOTE = "direct_quote"


class ComplianceMode(StrEnum):
    INTERNAL = "internal"
    PUBLICATION = "publication"
    PARTNER = "partner"


class SimilarityScore(VoBaseModel):
    source_id: str
    cosine: float = Field(ge=-1.0, le=1.0)
    ngram: float = Field(ge=0.0, le=1.0)
    winnow: float = Field(ge=0.0, le=1.0)


class ComplianceLog(VoBaseModel):
    id: ULID
    artifact_id: ULID | None = None
    query_text: str | None = None
    output_text: str | None = None
    similarity_scores: list[SimilarityScore] = Field(default_factory=list)
    source_contributions: dict[str, float] = Field(default_factory=dict)
    citation_tiers: dict[str, CitationTier] = Field(default_factory=dict)
    passed: bool
    failure_reasons: list[str] = Field(default_factory=list)
    mode: ComplianceMode
    created_at: datetime
