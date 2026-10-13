"""vo-rag Model Context Protocol surface.

This package exposes the vo-rag knowledge layer as four MCP tools:
``vo_rag_query``, ``vo_rag_ingest``, ``vo_rag_ingest_stream``, and
``vo_rag_status``.
The tools call the core retrieval / ingest / status seams
directly through an injected :class:`SurfaceServices` — no HTTP
round-trip to the API surface.

The transport assembly lives in :mod:`vo_rag.mcp.server`; the
pure-async tool implementations live in :mod:`vo_rag.mcp.tools`;
the wire-format Pydantic models live in :mod:`vo_rag.mcp.models`;
the error redaction helpers live in :mod:`vo_rag.mcp.errors`.

Public entry points:

- :func:`build_mcp_server` — assembles the FastMCP server with the
  four tool names. Hermetic when ``services`` is injected.
- :data:`SERVER_NAME` — the server name reported to MCP clients
  in the ``initialize`` handshake.
- :class:`McpQueryResult`, :class:`McpIngestResult`,
  :class:`McpStatusResult` — the wire-format result models.
- :class:`ToolError` — the tool-level error type. Translated into
  an ``isError: True`` MCP response; messages are redacted to
  keep API keys, Garage secrets, and stack traces off the wire.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from vo_rag.mcp.errors import ToolError, redact_error_message
from vo_rag.mcp.models import (
    McpEnvelopeView,
    McpIngestArgs,
    McpIngestResult,
    McpQueryArgs,
    McpQueryHit,
    McpQueryResult,
    McpStatusArgs,
    McpStatusResult,
)

if TYPE_CHECKING:
    from vo_rag.mcp import server as server
    from vo_rag.mcp import tools as tools
    from vo_rag.mcp.server import (
        SERVER_NAME as SERVER_NAME,
    )
    from vo_rag.mcp.server import (
        build_mcp_server as build_mcp_server,
    )
    from vo_rag.mcp.server import (
        main as main,
    )
    from vo_rag.mcp.server import (
        resolve_services as resolve_services,
    )
    from vo_rag.mcp.tools import (
        render_query_text as render_query_text,
    )
    from vo_rag.mcp.tools import (
        vo_rag_ingest_impl as vo_rag_ingest_impl,
    )
    from vo_rag.mcp.tools import (
        vo_rag_ingest_stream_impl as vo_rag_ingest_stream_impl,
    )
    from vo_rag.mcp.tools import (
        vo_rag_query_impl as vo_rag_query_impl,
    )
    from vo_rag.mcp.tools import (
        vo_rag_status_impl as vo_rag_status_impl,
    )

_SERVER_EXPORTS = frozenset(
    {
        "SERVER_NAME",
        "build_mcp_server",
        "main",
        "resolve_services",
    }
)
_TOOL_EXPORTS = frozenset(
    {
        "render_query_text",
        "vo_rag_ingest_impl",
        "vo_rag_ingest_stream_impl",
        "vo_rag_query_impl",
        "vo_rag_status_impl",
    }
)
_SUBMODULE_EXPORTS = frozenset({"server", "tools"})


def __getattr__(name: str) -> Any:
    """Lazily resolve server-backed public entry points (PEP 562)."""
    if name in _SERVER_EXPORTS:
        return getattr(import_module("vo_rag.mcp.server"), name)
    if name in _TOOL_EXPORTS:
        return getattr(import_module("vo_rag.mcp.tools"), name)
    if name in _SUBMODULE_EXPORTS:
        return import_module(f"vo_rag.mcp.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "McpEnvelopeView",
    "McpIngestArgs",
    "McpIngestResult",
    "McpQueryArgs",
    "McpQueryHit",
    "McpQueryResult",
    "McpStatusArgs",
    "McpStatusResult",
    "SERVER_NAME",
    "ToolError",
    "build_mcp_server",
    "main",
    "redact_error_message",
    "render_query_text",
    "resolve_services",
    "vo_rag_ingest_impl",
    "vo_rag_ingest_stream_impl",
    "vo_rag_query_impl",
    "vo_rag_status_impl",
]
