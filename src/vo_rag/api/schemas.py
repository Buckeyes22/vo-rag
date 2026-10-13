"""Centralized Pydantic request/response models for the API surface.

These schemas are shared by the API routes, the CLI and the MCP server.
Field names here are part of the public contract — do not rename
without updating all three surfaces.

The split is intentional:

- Request models validate user input at the boundary (collections
  must be non-empty when provided, ``visual_mode`` must come from the
  ``VISUAL_MODES`` allowlist, ``top_k`` bounds, etc.).
- Response models preserve core ids and payloads byte-for-byte. In
  particular ``QueryHit`` and ``EnvelopeResponse`` are thin
  serializations of the landed core types — no field remapping.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vo_rag.core.answer.context import ContextScope
from vo_rag.core.answer.synthesizer import AnswerProfile
from vo_rag.core.chunking.models import hit_payload
from vo_rag.core.models.common import ULID, VoBaseModel
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import SearchResult, validate_filters
from vo_rag.core.retrieval.visual_intent import VISUAL_MODES

# Fallback upload limit when ``settings.portal.upload_max_bytes`` is unset.
# 2 GiB matches the documented spec fallback.
_DEFAULT_UPLOAD_MAX_BYTES = 2 * 1024 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
class QueryRequest(VoBaseModel):
    """Top-level /api/v1/query body."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    collections: list[str] | None = Field(default=None, min_length=1)
    top_k: int = Field(default=20, ge=1, le=200)
    filters: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Payload filters. A scalar matches exactly, a list matches any value, and "
            '{"gte"|"gt"|"lte"|"lt": n} is a numeric range, e.g. {"page_number": {"gte": 10}}.'
        ),
    )
    rerank: bool = True
    mode: QueryMode = QueryMode.INTERNAL
    visual_mode: str | None = None

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("collections")
    @classmethod
    def _collections_non_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (not value or any(not name.strip() for name in value)):
            raise ValueError("collections must contain nonempty names when provided")
        return value

    @field_validator("visual_mode")
    @classmethod
    def _visual_mode_allowed(cls, value: str | None) -> str | None:
        if value is not None and value not in VISUAL_MODES:
            raise ValueError(f"visual_mode must be one of {VISUAL_MODES}; got {value!r}")
        return value

    @field_validator("filters")
    @classmethod
    def _filters_translatable(
        cls,
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Reject a filter the retriever cannot translate as a 422, not a 500."""
        return validate_filters(value)


class QueryHit(BaseModel):
    """A single retrieval hit — preserved from core ``SearchResult``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    score: float
    collection: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_search_result(cls, result: SearchResult) -> QueryHit:
        return cls(
            id=str(result.id),
            score=float(result.score),
            collection=str(result.collection),
            payload=hit_payload(result.payload),
        )


class QueryResponse(BaseModel):
    """Top-level /api/v1/query response."""

    model_config = ConfigDict(extra="forbid")

    results: list[QueryHit]


# --------------------------------------------------------------------------- #
# Answer
# --------------------------------------------------------------------------- #
class AnswerRequest(VoBaseModel):
    """Top-level /api/v1/answer body. F-045: default ``context_scope`` is sterile."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    collections: list[str] | None = Field(default=None, min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)
    filters: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Payload filters. A scalar matches exactly, a list matches any value, and "
            '{"gte"|"gt"|"lte"|"lt": n} is a numeric range, e.g. {"page_number": {"gte": 10}}.'
        ),
    )
    rerank: bool = True
    mode: QueryMode = QueryMode.INTERNAL
    profile: AnswerProfile = AnswerProfile.QUALITY
    context_scope: ContextScope = ContextScope.STERILE
    include_retrieval: bool = True
    visual_mode: str | None = None

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("collections")
    @classmethod
    def _collections_non_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (not value or any(not name.strip() for name in value)):
            raise ValueError("collections must contain nonempty names when provided")
        return value

    @field_validator("visual_mode")
    @classmethod
    def _visual_mode_allowed(cls, value: str | None) -> str | None:
        if value is not None and value not in VISUAL_MODES:
            raise ValueError(f"visual_mode must be one of {VISUAL_MODES}; got {value!r}")
        return value

    @field_validator("filters")
    @classmethod
    def _filters_translatable(
        cls,
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Reject a filter the retriever cannot translate as a 422, not a 500."""
        return validate_filters(value)


class ComplianceInfo(BaseModel):
    """Quarantine / canary-trace payload for the answer response."""

    model_config = ConfigDict(extra="forbid")

    status: str = "not_configured"
    should_quarantine: bool = False
    reason: str | None = None
    pipeline: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    """Top-level /api/v1/answer response."""

    model_config = ConfigDict(extra="forbid")

    answer_id: ULID
    answer: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    retrieval_results: list[QueryHit] = Field(default_factory=list)
    compliance: ComplianceInfo = Field(default_factory=ComplianceInfo)
    warnings: list[str] = Field(default_factory=list)
    model: str
    mode: str


# --------------------------------------------------------------------------- #
# Ingest — used by /api/v1/intake/upload-url.
# --------------------------------------------------------------------------- #
class UploadUrlRequest(VoBaseModel):
    """Request body for the presigned upload-url endpoint."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=1, le=_DEFAULT_UPLOAD_MAX_BYTES)
    sovereignty: str = "cloud_ok"

    @field_validator("filename")
    @classmethod
    def _filename_safe(cls, value: str) -> str:
        from vo_rag.api.utils import secure_filename

        return secure_filename(value)

    @field_validator("content_type")
    @classmethod
    def _content_type_safe(cls, value: str) -> str:
        # Strip any semicolon-suffixed charset parameters; we only need
        # the bare mime for the bucket policy.
        return value.split(";", 1)[0].strip().lower()


class UploadUrlResponse(BaseModel):
    """Response body for the presigned upload-url endpoint."""

    model_config = ConfigDict(extra="forbid")

    upload_url: str
    garage_key: str
    bucket: str
    expires_in_s: int = 21600


class EnvelopeResponse(BaseModel):
    """Response body for /api/v1/intake. Preserves core envelope ids."""

    model_config = ConfigDict(extra="forbid")

    envelope_id: ULID
    status: str
    source_type: str
    target_pipeline: str | None = None
    priority: str
    sovereignty: str
    created_at: datetime
    updated_at: datetime
    raw_content: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    dispatch_status: str = "not_configured"

    @classmethod
    def from_envelope(cls, envelope: Any) -> EnvelopeResponse:
        """Build a response from the landed core ``IntakeEnvelope``.

        Importing ``IntakeEnvelope`` here would create a cycle through
        ``vo_rag.core.models``; the type is loose (``Any``) and the
        fields we read are documented in the core model.
        """
        from vo_rag.core.models.envelope import IntakeEnvelope

        if not isinstance(envelope, IntakeEnvelope):
            raise TypeError("envelope must be an IntakeEnvelope")

        def _value(value: Any) -> str:
            return str(value.value if hasattr(value, "value") else value)

        return cls(
            envelope_id=envelope.id,
            status=_value(envelope.status),
            source_type=_value(envelope.source_type),
            target_pipeline=(
                _value(envelope.target_pipeline) if envelope.target_pipeline is not None else None
            ),
            priority=_value(envelope.priority),
            sovereignty=_value(envelope.sovereignty),
            created_at=envelope.created_at,
            updated_at=envelope.updated_at,
            raw_content=dict(envelope.raw_content),
            metadata=dict(envelope.metadata),
        )


# --------------------------------------------------------------------------- #
# Status / readiness / admin routes.
# --------------------------------------------------------------------------- #
class ServiceHealth(BaseModel):
    """One row in a /ready, /status, or /api/v1/admin/status body."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    """Liveness / health response shape."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    service: str = "vo-rag"
    version: str = "0.1.0"
    services: dict[str, ServiceHealth] = Field(default_factory=dict)
    pipeline_roster: list[dict[str, Any]] = Field(default_factory=list)
    eval_gate: dict[str, Any] | None = None
    cloud: dict[str, Any] = Field(default_factory=dict)


class ReadinessResponse(BaseModel):
    """Readiness probe — 503 when any required service is unavailable."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    services: dict[str, ServiceHealth] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class AdminStatusResponse(BaseModel):
    """Operator-facing status — pipeline roster + compute placement + cost."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    services: dict[str, ServiceHealth] = Field(default_factory=dict)
    pipeline_roster: list[dict[str, Any]] = Field(default_factory=list)
    eval_gate: dict[str, Any] = Field(default_factory=dict)
    cloud: dict[str, Any] = Field(default_factory=dict)
    compute_placement: dict[str, str] = Field(default_factory=dict)
    cost: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Books — used by the book-reader API surface.
# --------------------------------------------------------------------------- #
class BookStats(BaseModel):
    """Stats read from the book_rag knowledge artifact content."""

    model_config = ConfigDict(extra="forbid")

    text_chunks: int
    figure_chunks: int
    pages: int | None = None
    domain_tags: list[str] = Field(default_factory=list)


class BookLayers(BaseModel):
    """Per-book point counts for advanced Qdrant layers."""

    model_config = ConfigDict(extra="forbid")

    summary_tree: int
    raptor: int
    dsi: int
    graph_nodes: int


class BookOverviewResponse(BaseModel):
    """GET /api/v1/books/{source_id} response."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    title: str | None = None
    author: str | None = None
    license_class: str | None = None
    artifact_id: str | None = None
    stats: BookStats | None = None
    layers: BookLayers


class BookPageChunk(BaseModel):
    """One raw text chunk on a rendered book page."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    section_title: str | None = None
    text: str
    has_image: bool
    image_id: str | None = None
    vlm_description: str | None = None
    token_count: int | None = None


class BookPageResponse(BaseModel):
    """GET /api/v1/books/{source_id}/pages/{page} response."""

    model_config = ConfigDict(extra="forbid")

    page: int
    max_page: int | None = None
    chunks: list[BookPageChunk] = Field(default_factory=list)


class BookDsiSummary(BaseModel):
    """Flattened one-book DSI summary payload."""

    model_config = ConfigDict(extra="forbid")

    chapter_outline: str | None = None
    key_concepts: list[str] = Field(default_factory=list)
    key_methods: list[str] = Field(default_factory=list)
    key_equations_or_models: list[str] = Field(default_factory=list)
    prerequisite_topics: list[str] = Field(default_factory=list)
    figure_heavy_topics: list[str] = Field(default_factory=list)


class BookSummaryTreeNode(BaseModel):
    """Public projection of a summary-tree node."""

    model_config = ConfigDict(extra="forbid")

    display_path: str | None = None
    key_points: list[str] = Field(default_factory=list)


class BookSummaryTreeSummary(BaseModel):
    """Book/chapter summary-tree projection."""

    model_config = ConfigDict(extra="forbid")

    book: list[BookSummaryTreeNode] = Field(default_factory=list)
    chapter: list[BookSummaryTreeNode] = Field(default_factory=list)
    section_count: int


class BookGraphNodeSummary(BaseModel):
    """One graph node ranked by supporting chunk mentions."""

    model_config = ConfigDict(extra="forbid")

    label: str
    node_type: str
    mentions: int


class BookGraphSummary(BaseModel):
    """Graph layer summary for one book."""

    model_config = ConfigDict(extra="forbid")

    node_count: int
    top_nodes: list[BookGraphNodeSummary] = Field(default_factory=list)


class BookRaptorSummary(BaseModel):
    """RAPTOR layer summary for one book."""

    model_config = ConfigDict(extra="forbid")

    node_count: int


class BookSummariesResponse(BaseModel):
    """GET /api/v1/books/{source_id}/summaries response."""

    model_config = ConfigDict(extra="forbid")

    dsi: BookDsiSummary | None = None
    summary_tree: BookSummaryTreeSummary | None = None
    graph: BookGraphSummary | None = None
    raptor: BookRaptorSummary | None = None


# --------------------------------------------------------------------------- #
# IntakeSubmission — exposed at the API surface for /api/v1/intake.
# We re-declare the schema here (rather than re-exporting from
# ``vo_rag.core.ingest.normalize``) so the API boundary owns its own
# extra='forbid' shape and the route layer can add API-specific
# validators without touching core.
# --------------------------------------------------------------------------- #
class IntakeSubmission(VoBaseModel):
    """API-shaped intake submission. Field names must match the core
    IntakeSubmission used by ``normalize()``."""

    model_config = ConfigDict(extra="forbid")

    source_type: str = Field(min_length=1)
    source_data: dict[str, Any] = Field(default_factory=dict)
    raw_content: dict[str, Any] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    sovereignty: str = "cloud_ok"
    priority: str = "medium"
    target_pipeline: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_empty_raw_content(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw = value.get("raw_content")
        if raw is not None and not isinstance(raw, dict):
            raise ValueError("raw_content must be an object")
        return value
