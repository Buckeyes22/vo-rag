"""Assemble the native MCP server and its tool schemas.

``build_mcp_server`` registers the core tools. Runtime-owned
services are constructed inside the FastMCP lifespan on its event loop and
closed there; injected service containers remain caller-owned. Explicit
inspection-only servers expose schemas without opening runtime resources.

The HTTP transport requires an API key from ``settings.api.api_key`` or
``VORAG_MCP_API_KEY``. Stdio remains the default for local callers.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
from typing import Annotated, Any

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, ToolAnnotations
from pydantic import AnyHttpUrl, TypeAdapter

from vo_rag.mcp.models import (
    McpIngestResult,
    McpQueryResult,
    McpStatusResult,
)
from vo_rag.mcp.result import to_call_tool_result

log = logging.getLogger("vo_rag.mcp.server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8200
DEFAULT_TRANSPORT = "stdio"
ALLOWED_TRANSPORTS = ("stdio", "streamable-http")
_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)

#: The MCP server name reported to clients in the ``initialize``
#: handshake. Pinned in tests — do not change without an
#: integration test.
SERVER_NAME = "vo-rag"

#: Module-level FastMCP instance so ``python -m vo_rag.mcp.server``
#: starts the right object. ``build_mcp_server`` rebuilds it from
#: scratch when a test wants a hermetic instance.
_server: FastMCP | None = None


class _StaticApiKeyVerifier:
    """FastMCP token verifier for a single operator-provided API key."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def verify_token(self, token: str) -> AccessToken | None:
        if not secrets.compare_digest(token.encode(), self._api_key.encode()):
            return None
        return AccessToken(
            token=token,
            client_id="vo-rag-mcp-static-key",
            scopes=[],
        )


def build_mcp_server(
    services: Any = None,
    *,
    api_key: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    inspection_only: bool = False,
    runtime_builder: Any = None,
    close_services: Any = None,
) -> FastMCP:
    """Assemble the FastMCP server with core tools.

    Parameters
    ----------
    services:
        Optional injected service container. Injected services are never closed
        by the MCP server. Otherwise services are built inside FastMCP lifespan.
    api_key:
        Optional bearer token for request-time FastMCP HTTP auth.
        Stdio callers leave this unset; ``main(streamable-http)``
        passes the configured key after startup validation.
    inspection_only:
        Build a socket-free server for schema and tool inventory inspection.
        Calls that need a runtime service return a clear not-configured error.
    runtime_builder, close_services:
        Test seams for lifespan construction and teardown.
    """
    if services is not None and inspection_only:
        raise ValueError("inspection_only cannot be combined with injected services")
    lifespan = None
    if services is None and not inspection_only:
        from vo_rag.mcp.runtime import make_lifespan

        lifespan = make_lifespan(service_builder=runtime_builder, close_services=close_services)
    elif services is None:
        services = resolve_services()

    if api_key is not None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("api_key must be non-empty when provided")

    auth = _auth_settings(host=host, port=port) if api_key is not None else None
    token_verifier = _StaticApiKeyVerifier(api_key) if api_key is not None else None
    server = FastMCP(
        name=SERVER_NAME,
        host=host,
        port=port,
        auth=auth,
        token_verifier=token_verifier,
        lifespan=lifespan,
    )

    _register_tools(server, services, dynamic_services=services is None)
    return server


def _register_tools(server: FastMCP, services: Any, *, dynamic_services: bool = False) -> None:
    """Register native core tools."""
    from vo_rag.mcp.errors import ToolError
    from vo_rag.mcp.runtime import lifespan_services

    def current_services(ctx: Any) -> Any:
        if dynamic_services:
            return lifespan_services(ctx)
        return services

    def failed(exc: Exception, output_model: Any, **defaults: Any) -> CallToolResult:
        if isinstance(exc, ToolError):
            code, message = exc.code, str(exc)
        else:
            log.error("mcp_tool_failed", extra={"error_type": type(exc).__name__})
            code, message = "internal_error", "The tool failed while processing this request"
        payload = output_model(ok=False, code=code, error=message, **defaults)
        return to_call_tool_result(payload, f"{code}: {message}", error=True)

    from vo_rag.mcp.tools import (
        render_query_text,
        vo_rag_ingest_impl,
        vo_rag_ingest_stream_impl,
        vo_rag_query_impl,
        vo_rag_status_impl,
    )

    @server.tool(
        name="vo_rag_query",
        structured_output=True,
        annotations=ToolAnnotations(readOnlyHint=True),
        description=(
            "Search the vo-rag knowledge layer. Returns ranked hits with source "
            "attribution. filters: scalar = exact match, list = any-of, "
            '{"gte"|"gt"|"lte"|"lt": n} = numeric range '
            '(e.g. {"page_number": {"gte": 10}}).'
        ),
    )
    async def vo_rag_query(  # noqa: D401 - tool name is the contract
        query: str,
        collections: list[str] | None = None,
        top_k: int = 5,
        mode: str = "internal",
        filters: dict[str, Any] | None = None,
        rerank: bool = True,
        visual_mode: str | None = None,
        *,
        ctx: Context[Any, Any, Any],
    ) -> Annotated[CallToolResult, McpQueryResult]:
        try:
            result = await vo_rag_query_impl(
                services=current_services(ctx),
                query=query,
                collections=collections,
                top_k=top_k,
                mode=mode,
                filters=filters,
                rerank=rerank,
                visual_mode=visual_mode,
            )
            text = render_query_text([_search_result_from_hit(hit) for hit in result.results])
            return to_call_tool_result(result, text)
        except Exception as exc:
            return failed(exc, McpQueryResult, mode=mode, top_k=top_k)

    @server.tool(
        name="vo_rag_ingest",
        description="Submit content to vo-rag. Returns envelope_id and dispatch status.",
        structured_output=True,
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
    )
    async def vo_rag_ingest(  # noqa: D401 - tool name is the contract
        source_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        sovereignty: str = "cloud_ok",
        *,
        ctx: Context[Any, Any, Any],
    ) -> Annotated[CallToolResult, McpIngestResult]:
        try:
            result = await vo_rag_ingest_impl(
                services=current_services(ctx),
                source_type=source_type,
                content=content,
                metadata=metadata,
                sovereignty=sovereignty,
            )
            text_lines = [
                "Ingest accepted.",
                f"envelope_id: {result.envelope_id}",
                f"status: {result.status}",
                f"source_type: {result.source_type}",
                f"dispatch_status: {result.dispatch_status}",
            ]
            text_lines.extend(f"warning: {warning}" for warning in result.warnings)
            return to_call_tool_result(result, "\n".join(text_lines))
        except Exception as exc:
            return failed(exc, McpIngestResult)

    @server.tool(
        name="vo_rag_ingest_stream",
        description="Stream ingest progress for an envelope. Reports per-step progress via MCP notifications.",  # noqa: E501
        structured_output=True,
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
    )
    async def vo_rag_ingest_stream(  # noqa: D401 - tool name is the contract
        envelope_id: str,
        ctx: Context[Any, Any, Any],
    ) -> Annotated[CallToolResult, McpIngestResult]:
        try:
            result = await vo_rag_ingest_stream_impl(
                services=current_services(ctx),
                envelope_id=envelope_id,
                ctx=ctx,
            )
            text = "\n".join(
                [
                    "Ingest stream completed.",
                    f"envelope_id: {result.envelope_id}",
                    f"status: {result.status}",
                    f"dispatch_status: {result.dispatch_status}",
                    *(f"warning: {warning}" for warning in result.warnings),
                ]
            )
            return to_call_tool_result(result, text)
        except Exception as exc:
            return failed(exc, McpIngestResult)

    @server.tool(
        name="vo_rag_status",
        description="Check vo-rag system status. No args: system status. With envelope_id: envelope progress.",  # noqa: E501
        structured_output=True,
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def vo_rag_status(  # noqa: D401 - tool name is the contract
        ctx: Context[Any, Any, Any],
        envelope_id: str | None = None,
    ) -> Annotated[CallToolResult, McpStatusResult]:
        try:
            result = await vo_rag_status_impl(
                services=current_services(ctx), envelope_id=envelope_id
            )
            return to_call_tool_result(result, _render_status_text(result))
        except Exception as exc:
            return failed(exc, McpStatusResult)


def _search_result_from_hit(hit: Any) -> Any:
    """Adapt a ``McpQueryHit`` back to the shape ``render_query_text``
    consumes (``SearchResult``). The tool layer keeps
    ``SearchResult`` as the rendering input so the field-name
    contract is one-way (core → wire, never wire → core).
    """
    from vo_rag.core.retrieval.search import SearchResult

    payload: dict[str, Any] = dict(getattr(hit, "payload", {}) or {})
    if getattr(hit, "title", None) is not None and "section_title" not in payload:
        payload["section_title"] = hit.title
    if getattr(hit, "source", None) is not None and "source_title" not in payload:
        payload["source_title"] = hit.source
    if getattr(hit, "page", None) is not None and "page_number" not in payload:
        payload["page_number"] = hit.page
    if getattr(hit, "image", None) is not None and "image_ref" not in payload:
        payload["image_ref"] = hit.image
    return SearchResult(
        id=hit.id,
        score=hit.score,
        collection=hit.collection,
        payload=payload,
    )


def _render_status_text(result: McpStatusResult) -> str:
    """Render a status result as a human-readable text block.

    Two branches: system status (per-slot health lines) and envelope
    status (envelope metadata block).
    """
    if result.scope == "envelope" and result.envelope is not None:
        env = result.envelope
        return "\n".join(
            [
                f"Envelope: {env.envelope_id}",
                f"Source:   {env.source_type}",
                f"Status:   {env.status}",
                f"Priority: {env.priority}",
                f"Sovereignty: {env.sovereignty}",
            ]
        )

    system = result.system or {}
    services_block = system.get("services", {}) or {}
    lines = [
        f"Service: {system.get('service', 'vo-rag')}",
        f"Version: {system.get('version', '')}",
        f"OK:      {system.get('ok', False)}",
        "",
        "Services:",
    ]
    for name, block in services_block.items():
        ok = block.get("ok", False) if isinstance(block, dict) else False
        err = block.get("error") if isinstance(block, dict) else None
        marker = "ok" if ok else f"failed: {err}" if err else "not configured"
        lines.append(f"  {name:<20} {marker}")
    capabilities = system.get("capabilities", {}) or {}
    if isinstance(capabilities, dict) and capabilities:
        lines.extend(["", "Capability availability:"])
        for name, block in capabilities.items():
            if not isinstance(block, dict):
                lines.append(f"  {name:<20} unavailable")
                continue
            ready = block.get("ready", False)
            state = "configured" if ready else f"unavailable: {block.get('reason', 'unknown')}"
            backend = block.get("backend")
            if backend:
                state = f"{backend}: {state}"
            lines.append(f"  {name:<20} {state}")
    availability = system.get("availability")
    if availability:
        lines.append(f"Availability: {availability} (configuration only, not live health)")
    for warning in result.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI / entry point
# --------------------------------------------------------------------------- #
def resolve_services() -> Any:
    """Return a socket-free service container for schema inspection."""
    from vo_rag.api.singletons import SurfaceServices

    return SurfaceServices(settings=get_settings())


def get_settings() -> Any:
    """Return the active :class:`Settings` instance.

    Lazily imports to keep ``import vo_rag.mcp.server`` cheap for
    callers that only want ``build_mcp_server`` with an injected
    service container.
    """
    from vo_rag.core.config import get_settings as _get_settings

    return _get_settings()


def _auth_settings(*, host: str, port: int) -> AuthSettings:
    base_url = _http_base_url(host=host, port=port)
    return AuthSettings(issuer_url=base_url, resource_server_url=base_url)


def _http_base_url(*, host: str, port: int) -> AnyHttpUrl:
    display_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return _HTTP_URL_ADAPTER.validate_python(f"http://{display_host}:{port}")


def _require_api_key(transport: str) -> str | None:
    """Refuse the HTTP transport when no API key is configured.

    The check is split out so the test suite can pin it. The
    env-var fallback (``VORAG_MCP_API_KEY``) lets operators wire
    the key without touching the YAML config — useful in
    container deployments where the key is mounted as a secret
    and the YAML is baked into the image.
    """
    if transport != "streamable-http":
        return None

    env_key = os.environ.get("VORAG_MCP_API_KEY", "").strip()
    settings_key = ""
    try:
        settings = get_settings()
        settings_key = str(getattr(getattr(settings, "api", None), "api_key", "") or "")
    except Exception as exc:  # noqa: BLE001
        # If the settings loader is misconfigured (e.g. dev box
        # with no YAML), we still want to surface the missing key,
        # not a settings import error.
        log.warning("mcp_settings_api_key_load_failed: %s", exc, exc_info=True)
        settings_key = ""

    api_key = env_key or settings_key.strip()
    if api_key:
        return api_key

    raise PermissionError(
        "streamable-http transport requires VORAG_MCP_API_KEY or settings.api.api_key; "
        "use the stdio transport for unauthenticated local callers"
    )


def main(
    transport: str = DEFAULT_TRANSPORT, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> None:
    """CLI entry point: build the server, validate the transport, run it."""
    if transport not in ALLOWED_TRANSPORTS:
        raise ValueError(f"transport must be one of {ALLOWED_TRANSPORTS!r}; got {transport!r}")

    api_key = _require_api_key(transport)

    server = build_mcp_server(api_key=api_key, host=host, port=port)
    log.info(
        "mcp_server_starting",
        extra={
            "transport": transport,
            "host": host,
            "port": port,
            "server_name": SERVER_NAME,
        },
    )
    server.run(transport=transport)  # type: ignore[arg-type]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vo-rag-mcp",
        description="vo-rag Model Context Protocol server.",
    )
    parser.add_argument(
        "--transport",
        choices=ALLOWED_TRANSPORTS,
        default=DEFAULT_TRANSPORT,
        help="MCP transport (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Host to bind for HTTP transports (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to bind for HTTP transports (default: 8200).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover - module entry point
    args = _parse_args()
    main(transport=args.transport, host=args.host, port=args.port)


__all__ = [
    "ALLOWED_TRANSPORTS",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_TRANSPORT",
    "SERVER_NAME",
    "build_mcp_server",
    "main",
    "resolve_services",
]
