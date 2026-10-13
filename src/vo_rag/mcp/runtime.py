"""Build and close MCP services inside the server's event loop."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

ServiceBuilder = Callable[[Any], Awaitable[Any]]


@dataclass
class McpRuntime:
    """Own the service container created for one MCP server lifespan."""

    services: Any
    close: Callable[[Any], Awaitable[None]]


async def build_runtime(
    *,
    settings: Any | None = None,
    service_builder: ServiceBuilder | None = None,
    close_services: Callable[[Any], Awaitable[None]] | None = None,
) -> McpRuntime:
    """Create initialized runtime services and attach MCP core callbacks."""
    if settings is None:
        from vo_rag.core.config import get_settings

        settings = get_settings()
    if service_builder is None:
        from vo_rag.api.singletons import build_surface_services

        service_builder = build_surface_services
    if close_services is None:
        from vo_rag.api.singletons import close_surface_services

        close_services = close_surface_services

    services = await service_builder(settings)
    try:
        _attach_core_contract(services)
    except Exception:
        await close_services(services)
        raise
    return McpRuntime(services=services, close=close_services)


def _attach_core_contract(services: Any) -> None:
    """Attach canonical intake, envelope lookup, and readiness callbacks."""

    async def submit(envelope: Any) -> dict[str, str]:
        from vo_rag.core.db.queries.envelopes import insert_envelope
        from vo_rag.mcp.errors import ToolError

        pool = getattr(services, "db_pool", None)
        if pool is None:
            raise ToolError(
                "intake_unavailable: envelope persistence is unavailable",
                code="intake_unavailable",
            )
        await insert_envelope(pool, envelope)
        return {"envelope_id": str(envelope.id), "status": "received"}

    async def lookup(envelope_id: str) -> Any:
        from vo_rag.core.db.queries.envelopes import get_envelope
        from vo_rag.mcp.errors import ToolError

        pool = getattr(services, "db_pool", None)
        if pool is None:
            raise ToolError(
                "envelope_status_unavailable: envelope status storage is unavailable",
                code="envelope_status_unavailable",
            )
        return await get_envelope(pool, envelope_id)

    async def status() -> dict[str, Any]:
        db_ready = getattr(services, "db_pool", None) is not None
        retrieval_ready = getattr(services, "retriever", None) is not None
        streaming_ready = (
            db_ready
            and retrieval_ready
            and getattr(services, "garage", None) is not None
            and getattr(services, "embedding_client", None) is not None
        )
        capabilities: dict[str, dict[str, Any]] = {
            "retrieval": _capability(
                retrieval_ready,
                _reason(services, "qdrant_unavailable_reason")
                if getattr(services, "qdrant", None) is None
                else "retriever_unavailable",
            ),
            "intake": _capability(db_ready, _reason(services, "db_pool_unavailable_reason")),
            "workflow_streaming": _capability(
                streaming_ready, "required_ingest_dependencies_unavailable"
            ),
        }
        return {
            "service": "vo-rag",
            "version": str(getattr(getattr(services, "settings", None), "version", "") or ""),
            "ok": all(item["ready"] for item in capabilities.values()),
            "availability": "configured_dependencies",
            "capabilities": capabilities,
            "warnings": [
                "readiness reports configured dependencies; it is not a live health probe"
            ],
        }

    services.ingest_submit = submit
    services.envelope_lookup = lookup
    services.status_provider = status


def _reason(services: Any, field: str) -> str:
    return str(getattr(services, field, "") or "dependency unavailable")


def _capability(ready: bool, reason: str) -> dict[str, Any]:
    return {"ready": ready, **({} if ready else {"reason": reason})}


def make_lifespan(
    *,
    settings: Any | None = None,
    service_builder: ServiceBuilder | None = None,
    close_services: Callable[[Any], Awaitable[None]] | None = None,
) -> Callable[[Any], Any]:
    """Return FastMCP lifespan that owns services on the server event loop."""

    @asynccontextmanager
    async def lifespan(_server: Any) -> AsyncIterator[McpRuntime]:
        runtime = await build_runtime(
            settings=settings,
            service_builder=service_builder,
            close_services=close_services,
        )
        try:
            yield runtime
        finally:
            await runtime.close(runtime.services)

    return lifespan


def lifespan_services(ctx: Any) -> Any:
    """Return services from the active FastMCP lifespan context."""
    request_context = getattr(ctx, "request_context", None)
    runtime = getattr(request_context, "lifespan_context", None)
    services = getattr(runtime, "services", None)
    if services is None:
        raise RuntimeError("mcp_runtime_unavailable: server lifespan has not initialized")
    return services


__all__ = ["McpRuntime", "build_runtime", "lifespan_services", "make_lifespan"]
