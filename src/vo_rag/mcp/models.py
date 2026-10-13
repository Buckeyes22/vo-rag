"""Pydantic models for the Python MCP surface.

Field names are aligned with the API surface schemas in
``vo_rag.api.schemas`` and the core models in
``vo_rag.core.models.envelope`` / ``vo_rag.core.retrieval.search``
— no parallel vocabulary. The MCP wire format is JSON-RPC, so every
model is ``extra='forbid'`` to catch client typos at the boundary
the same way the API surface does.

Why a separate models module?  The three surface layers (API / CLI /
MCP) all read the same core seams. Keeping the MCP-specific
Pydantic shapes in this leaf module means:

- ``tools.py`` can stay focused on orchestration.
- ``server.py`` can re-use the same types as the structured-content
  part of an MCP ``CallToolResult`` (which the FastMCP layer
  serializes for the client).
- Tests can import the models without triggering the FastMCP
  module's lifespan / settings logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vo_rag.core.models.common import ULID, VoBaseModel

MCP_DISPATCH_STATUSES: Final[frozenset[str]] = frozenset(
    {"submitted", "streamed", "not_configured"}
)


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
class McpQueryArgs(VoBaseModel):
    """Arguments to the ``vo_rag_query`` MCP tool.

    ``top_k`` has no Pydantic upper bound on purpose: the boundary
    clamp lives in the tool function so a too-large value is
    silently capped at the MCP ceiling rather than surfaced as a
    wire-format Pydantic error. The model only rejects a non-positive
    integer.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    collections: list[str] | None = None
    top_k: int = Field(default=5, ge=1)
    mode: str = "internal"
    filters: dict[str, Any] | None = None
    rerank: bool = True
    visual_mode: str | None = None

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("mode")
    @classmethod
    def _mode_known(cls, value: str) -> str:
        # The actual enum coercion lives in the tool (so a bad value
        # surfaces as a redacted tool error, not a Pydantic wire error).
        # We only enforce "string-ness" here.
        if not isinstance(value, str) or not value.strip():
            raise ValueError("mode must be a non-empty string")
        return value


class McpQueryHit(BaseModel):
    """A single retrieval hit — thin serialisation of core ``SearchResult``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    score: float
    collection: str
    title: str | None = None
    source: str | None = None
    page: int | None = None
    image: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class McpQueryResult(BaseModel):
    """Structured result for the ``vo_rag_query`` MCP tool."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    code: str | None = None
    error: str | None = None
    command: str = "vo_rag_query"
    schema_version: int = 1
    results: list[McpQueryHit] = Field(default_factory=list)
    mode: str = "internal"
    top_k: int = 5
    reranked: bool = False
    warnings: list[str] = Field(default_factory=list)


class McpToolErrorResult(BaseModel):
    """Structured failure returned by a registered MCP tool."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = False
    code: str
    error: str
    schema_version: int = 1


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
class McpIngestArgs(VoBaseModel):
    """Arguments to the ``vo_rag_ingest`` MCP tool.

    ``content`` is the flat payload whose shape depends on
    ``source_type``:
      - ``url`` / ``github_star`` → a URL string.
      - ``text_prompt`` → free-form text.
      - ``file_drop`` → a Garage key returned by the presigned
        upload-url endpoint.
    The tool converts ``content`` into the core
    ``raw_content`` dict and validates URLs through the SSRF gate.
    """

    model_config = ConfigDict(extra="forbid")

    source_type: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    sovereignty: str = "cloud_ok"

    @field_validator("source_type")
    @classmethod
    def _source_type_known(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("source_type must be a non-empty string")
        return value


class McpIngestResult(BaseModel):
    """Structured result for the ``vo_rag_ingest`` MCP tool."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    code: str | None = None
    error: str | None = None
    envelope_id: ULID | None = None
    status: str = "unavailable"
    source_type: str = "unknown"
    dispatch_status: str = "not_configured"
    warnings: list[str] = Field(default_factory=list)
    schema_version: int = 1
    command: str = "vo_rag_ingest"

    @field_validator("dispatch_status")
    @classmethod
    def _dispatch_status_known(cls, value: str) -> str:
        if value in MCP_DISPATCH_STATUSES or value.endswith("_unrecognized"):
            return value
        raise ValueError(
            "dispatch_status must be one of "
            f"{', '.join(sorted(MCP_DISPATCH_STATUSES))} or end with _unrecognized"
        )


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
class McpStatusArgs(VoBaseModel):
    """Arguments to the ``vo_rag_status`` MCP tool."""

    model_config = ConfigDict(extra="forbid")

    envelope_id: ULID | None = None


class McpEnvelopeView(BaseModel):
    """Thin serialisation of an ``IntakeEnvelope`` for the status tool.

    Same shape as the API ``EnvelopeResponse`` minus the deep
    raw_content / metadata (which are surfaced through the get-by-id
    endpoint on a real dispatcher).
    """

    model_config = ConfigDict(extra="forbid")

    envelope_id: str
    status: str
    source_type: str
    priority: str
    sovereignty: str
    created_at: datetime
    updated_at: datetime


class McpStatusResult(BaseModel):
    """Structured result for the ``vo_rag_status`` MCP tool.

    ``scope`` is ``"system"`` when no envelope id was given, and
    ``"envelope"`` when one was. Exactly one of ``system`` and
    ``envelope`` is populated.
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    code: str | None = None
    error: str | None = None
    scope: str = "system"  # "system" | "envelope"
    system: dict[str, Any] | None = None
    envelope: McpEnvelopeView | None = None
    warnings: list[str] = Field(default_factory=list)
    schema_version: int = 1
    command: str = "vo_rag_status"


__all__ = [
    "MCP_DISPATCH_STATUSES",
    "McpEnvelopeView",
    "McpIngestArgs",
    "McpIngestResult",
    "McpQueryArgs",
    "McpQueryHit",
    "McpQueryResult",
    "McpStatusArgs",
    "McpStatusResult",
    "McpToolErrorResult",
]
