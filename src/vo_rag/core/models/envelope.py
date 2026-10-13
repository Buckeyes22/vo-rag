"""IntakeEnvelope and related enums."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from .common import ULID, VoBaseModel


class IntakeSource(StrEnum):
    GITHUB_STAR = "github_star"
    FILE_DROP = "file_drop"
    URL = "url"
    TEXT_PROMPT = "text_prompt"
    API = "api"


class InputType(StrEnum):
    BOOK = "book"
    REPOSITORY = "repository"
    ARTICLE = "article"
    TOOL_OR_LIBRARY = "tool_or_library"
    RESEARCH_QUESTION = "research_question"
    RAW_IDEA = "raw_idea"
    DATASET = "dataset"
    REFERENCE_URL = "reference_url"
    CODE_SNIPPET = "code_snippet"
    UNKNOWN = "unknown"


class Disposition(StrEnum):
    PROCESS = "process"
    BOOKMARK_ONLY = "bookmark_only"
    DISCARD = "discard"
    NEEDS_HUMAN = "needs_human"


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOISE = "noise"


class Sovereignty(StrEnum):
    HOMELAB_ONLY = "homelab_only"
    CLOUD_OK = "cloud_ok"
    CLOUD_OK_NO_PERSIST = "cloud_ok_no_persist"


class EnvelopeStatus(StrEnum):
    RECEIVED = "received"
    TRIAGING = "triaging"
    TRIAGED = "triaged"
    DISPATCHED = "dispatched"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DISCARDED = "discarded"


class TargetPipeline(StrEnum):
    BOOK_RAG = "book_rag"
    REPO_ANALYSIS = "repo_analysis"
    RESEARCH_AGENT = "research_agent"
    IDEA_CAPTURE = "idea_capture"
    ARTICLE_EXTRACTION = "article_extraction"
    TOOL_EVALUATION = "tool_evaluation"
    URL_SNAPSHOT = "url_snapshot"
    STRUCTURED_SCRAPE = "structured_scrape"
    VCE = "vce"
    CODE_RAG = "code_rag"


class TriageResult(VoBaseModel):
    input_type: InputType
    disposition: Disposition
    priority: Priority
    tags: list[str] = Field(default_factory=list)
    target_pipeline: TargetPipeline | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    used_fallback: bool = False


class IntakeEnvelope(VoBaseModel):
    id: ULID
    source_type: IntakeSource
    source_data: dict[str, Any] = Field(default_factory=dict)
    raw_content: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    triage_result: TriageResult | None = None
    status: EnvelopeStatus = EnvelopeStatus.RECEIVED
    target_pipeline: TargetPipeline | None = None
    priority: Priority = Priority.MEDIUM
    sovereignty: Sovereignty = Sovereignty.CLOUD_OK
    error: str | None = None
    created_at: datetime
    updated_at: datetime
