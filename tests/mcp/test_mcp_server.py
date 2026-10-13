"""Tests for the FastMCP server assembly in ``vo_rag.mcp.server``.

The server module owns:

- ``build_mcp_server(services=None) -> FastMCP`` — the assembly point
  shared by the stdio entry point (``vo_rag.mcp.server:main``) and by
  tests.
- ``main(transport='stdio', host='127.0.0.1', port=8200)`` — the
  CLI entry point. The default is stdio; the streamable-http
  transport requires an API key (``settings.api.api_key`` or the
  ``VORAG_MCP_API_KEY`` env var) so unauthenticated operators cannot
  accidentally publish the surface to the LAN.
- The four-tool registration: ``vo_rag_query``, ``vo_rag_ingest``,
  ``vo_rag_ingest_stream``, ``vo_rag_status``. No more, no less —
  the public surface area is the contract the harness pins.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
from tests.mcp._fakes import (
    FakeReranker,
    FakeRetriever,
    find_tool,
    make_fake_services,
    make_search_result,
)

pytestmark = [pytest.mark.asyncio]


# --------------------------------------------------------------------------- #
# Server assembly
# --------------------------------------------------------------------------- #
async def test_build_mcp_server_returns_fastmcp_named_vo_rag() -> None:
    """The server returned by build_mcp_server is a FastMCP and is
    named 'vo-rag' (the value MCP clients use as the
    ``ServerInfo.name`` in the ``initialize`` handshake)."""
    from mcp.server.fastmcp import FastMCP

    from vo_rag.mcp.server import build_mcp_server

    services = make_fake_services()
    server = build_mcp_server(services=services)

    assert isinstance(server, FastMCP)
    assert server.name == "vo-rag"


async def test_build_mcp_server_registers_exactly_four_tools() -> None:
    """The surface area is exactly four tools — no more, no less.
    Drift here is a public-contract change."""
    from vo_rag.mcp.server import build_mcp_server

    services = make_fake_services()
    server = build_mcp_server(services=services)

    names = sorted(server._tool_manager._tools.keys())
    assert names == [
        "vo_rag_ingest",
        "vo_rag_ingest_stream",
        "vo_rag_query",
        "vo_rag_status",
    ]


async def test_build_mcp_server_no_services_builds_minimal_container() -> None:
    """When called without services (the CLI default), the server
    must still build — the lifespan is what opens sockets at run
    time, not construction."""
    from vo_rag.mcp.server import build_mcp_server

    server = build_mcp_server()
    assert server.name == "vo-rag"
    assert "vo_rag_query" in server._tool_manager._tools


async def test_build_mcp_server_with_api_key_installs_request_auth() -> None:
    """HTTP bearer auth must be mounted on the FastMCP server, not just
    checked before startup."""
    from mcp.server.auth.provider import AccessToken

    from vo_rag.mcp.server import build_mcp_server

    server = build_mcp_server(services=make_fake_services(), api_key="test-key")

    assert server.settings.auth is not None
    assert server._token_verifier is not None
    assert await server._token_verifier.verify_token("wrong-key") is None

    access_token = await server._token_verifier.verify_token("test-key")
    assert isinstance(access_token, AccessToken)
    assert access_token.token == "test-key"
    assert access_token.client_id == "vo-rag-mcp-static-key"


async def test_build_mcp_server_rejects_empty_api_key() -> None:
    from vo_rag.mcp.server import build_mcp_server

    with pytest.raises(ValueError, match="api_key must be non-empty"):
        build_mcp_server(services=make_fake_services(), api_key="")

    with pytest.raises(ValueError, match="api_key must be non-empty"):
        build_mcp_server(services=make_fake_services(), api_key="   ")


async def test_build_mcp_server_strips_api_key_before_auth() -> None:
    from vo_rag.mcp.server import build_mcp_server

    server = build_mcp_server(services=make_fake_services(), api_key="  test-key  ")

    assert server._token_verifier is not None
    assert await server._token_verifier.verify_token("  test-key  ") is None
    assert await server._token_verifier.verify_token("test-key") is not None


async def test_mcp_streamable_http_rejects_missing_and_wrong_bearer_tokens() -> None:
    from vo_rag.mcp.server import build_mcp_server

    server = build_mcp_server(services=make_fake_services(), api_key="test-key")
    app = server.streamable_http_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        absent = await client.post("/mcp", content=b"{}")
        wrong = await client.post(
            "/mcp",
            headers={"Authorization": "Bearer wrong-key"},
            content=b"{}",
        )
        correct = await client.post(
            "/mcp",
            headers={"Authorization": "Bearer test-key"},
            content=b"{}",
        )

    assert absent.status_code == 401
    assert wrong.status_code == 401
    assert correct.status_code != 401


async def test_vo_rag_query_tool_description_documents_filter_ranges() -> None:
    """``filters`` is the only way a caller narrows a payload by field, and the
    tool description is all an MCP model caller sees — the range syntax, with a
    real chunk field as its example, has to be in the advertised description,
    not only in the Python docstring (VOR-1443)."""
    from vo_rag.mcp.server import build_mcp_server

    server = build_mcp_server(services=make_fake_services())

    description = find_tool(server, "vo_rag_query").description
    assert "gte" in description
    assert "lte" in description
    assert '{"page_number": {"gte": 10}}' in description


# --------------------------------------------------------------------------- #
# Tool smoke tests through the FastMCP tool manager
# --------------------------------------------------------------------------- #
async def test_vo_rag_query_runs_through_fastmcp_tool_manager() -> None:
    """The end-to-end path: the FastMCP tool manager invokes
    vo_rag_query, which in turn calls vo_rag_query_impl, which
    reaches the retriever. This is the contract the MCP client
    harness sees."""
    from vo_rag.mcp.server import build_mcp_server

    retriever = FakeRetriever(scripted=[make_search_result()])
    services = make_fake_services(retriever=retriever, reranker=FakeReranker())
    server = build_mcp_server(services=services)

    tool = find_tool(server, "vo_rag_query")
    # FastMCP tool functions return the native ``CallToolResult`` with
    # ``content`` and ``structuredContent``. The structured payload
    # is what model-side clients see.
    result = await tool.run({"query": "trail braking", "top_k": 5})

    # retriever was hit
    assert retriever.calls
    assert retriever.calls[0]["query"] == "trail braking"
    # structured payload present at the MCP wire-format path
    from mcp.types import CallToolResult

    assert isinstance(result, CallToolResult)
    structured = result.structuredContent
    assert isinstance(structured, dict)
    assert "results" in structured
    assert structured["results"][0]["id"] == "point-1"


async def test_vo_rag_ingest_runs_through_fastmcp_tool_manager() -> None:
    """End-to-end ingest path through the FastMCP tool manager."""
    from vo_rag.mcp.server import build_mcp_server

    submitted: list[Any] = []

    def _submit(envelope: Any) -> dict[str, str]:
        submitted.append(envelope)
        return {
            "envelope_id": envelope.id,
            "status": "received",
            "dispatch_status": "submitted",
        }

    services = make_fake_services(ingest_submit=_submit)
    server = build_mcp_server(services=services)
    tool = find_tool(server, "vo_rag_ingest")

    result = await tool.run(
        {
            "source_type": "text_prompt",
            "content": "what is trail braking?",
        }
    )
    assert submitted, "ingest_submit was not called"
    from mcp.types import CallToolResult

    assert isinstance(result, CallToolResult)
    structured = result.structuredContent
    assert isinstance(structured, dict)
    assert structured["dispatch_status"] == "submitted"
    assert structured["status"] == "received"


async def test_vo_rag_status_runs_through_fastmcp_tool_manager() -> None:
    """End-to-end status path: no envelope_id → system status."""
    from vo_rag.mcp.server import build_mcp_server

    def _provider() -> dict[str, Any]:
        return {"service": "vo-rag", "ok": True, "services": {}}

    services = make_fake_services(status_provider=_provider)
    server = build_mcp_server(services=services)
    tool = find_tool(server, "vo_rag_status")

    result = await tool.run({})
    from mcp.types import CallToolResult

    assert isinstance(result, CallToolResult)
    structured = result.structuredContent
    assert isinstance(structured, dict)
    assert structured["scope"] == "system"
    assert structured["system"]["service"] == "vo-rag"


async def test_core_tools_publish_typed_output_schemas() -> None:
    from vo_rag.mcp.server import build_mcp_server

    server = build_mcp_server(services=make_fake_services())
    listed = {tool.name: tool for tool in await server.list_tools()}
    for name in ("vo_rag_query", "vo_rag_ingest", "vo_rag_ingest_stream", "vo_rag_status"):
        schema = listed[name].outputSchema
        assert schema is not None
        assert schema["type"] == "object"
        assert "result" not in schema["properties"]


# --------------------------------------------------------------------------- #
# main() and HTTP transport authorization
# --------------------------------------------------------------------------- #
async def test_main_streamable_http_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """streamable-http must refuse to start without VORAG_MCP_API_KEY
    or settings.api.api_key. Stdio is the safe default and must
    always be permitted."""
    from mcp.server.fastmcp import FastMCP

    from vo_rag.mcp import server as server_mod

    captured: dict[str, Any] = {}

    def _fake_run(*args: Any, **kwargs: Any) -> None:
        captured["kwargs"] = kwargs

    monkeypatch.setattr(server_mod, "resolve_services", lambda: make_fake_services())
    monkeypatch.setattr(FastMCP, "run", _fake_run)

    # Ensure no env var or settings key
    monkeypatch.delenv("VORAG_MCP_API_KEY", raising=False)
    monkeypatch.setattr(
        server_mod,
        "get_settings",
        lambda: _settings_with_api_key(""),
    )

    # http without key → PermissionError
    with pytest.raises(PermissionError) as excinfo:
        server_mod.main(transport="streamable-http", host="127.0.0.1", port=8200)
    assert "api_key" in str(excinfo.value)

    # stdio with no key → fine, calls mcp.run(transport='stdio')
    server_mod.main(transport="stdio", host="127.0.0.1", port=8200)
    assert captured.get("kwargs", {}).get("transport") == "stdio"


async def test_main_streamable_http_logs_settings_loader_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from vo_rag.mcp import server as server_mod

    monkeypatch.delenv("VORAG_MCP_API_KEY", raising=False)
    monkeypatch.setattr(
        server_mod,
        "get_settings",
        lambda: (_ for _ in ()).throw(RuntimeError("settings yaml broken")),
    )

    with (
        caplog.at_level(logging.WARNING, logger="vo_rag.mcp.server"),
        pytest.raises(PermissionError, match="api_key"),
    ):
        server_mod.main(transport="streamable-http", host="127.0.0.1", port=8200)

    assert "mcp_settings_api_key_load_failed" in caplog.text
    assert "settings yaml broken" in caplog.text


async def test_main_streamable_http_with_api_key_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When VORAG_MCP_API_KEY is set, the HTTP transport is allowed
    to start and the API key is passed into build_mcp_server so
    request-time bearer auth is mounted."""
    from vo_rag.mcp import server as server_mod

    captured: dict[str, Any] = {}

    class _FakeServer:
        def run(self, *args: Any, **kwargs: Any) -> None:
            captured["run_kwargs"] = kwargs

    def _fake_build_mcp_server(**kwargs: Any) -> _FakeServer:
        captured["build_kwargs"] = kwargs
        return _FakeServer()

    monkeypatch.setattr(server_mod, "build_mcp_server", _fake_build_mcp_server)
    monkeypatch.setenv("VORAG_MCP_API_KEY", "test-key")  # pragma: allowlist secret

    server_mod.main(transport="streamable-http", host="127.0.0.1", port=8200)
    assert captured["build_kwargs"]["api_key"] == "test-key"  # pragma: allowlist secret
    assert captured["build_kwargs"]["host"] == "127.0.0.1"
    assert captured["build_kwargs"]["port"] == 8200
    assert captured["run_kwargs"]["transport"] == "streamable-http"
    monkeypatch.delenv("VORAG_MCP_API_KEY", raising=False)


async def test_main_defers_service_construction_to_the_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no pre-injected services, main() must not resolve (or open) any
    service at startup: the runtime builds them inside the FastMCP lifespan.
    ``resolve_services`` is only the socket-free inspection container."""
    from mcp.server.fastmcp import FastMCP

    from vo_rag.mcp import server as server_mod

    seen: dict[str, Any] = {}

    def _fake_resolve() -> Any:
        seen["called"] = True
        return make_fake_services()

    def _fake_run(self: FastMCP, *a: Any, **kw: Any) -> None:
        seen["lifespan"] = self.settings.lifespan

    monkeypatch.setattr(server_mod, "resolve_services", _fake_resolve)
    monkeypatch.setattr(FastMCP, "run", _fake_run)

    server_mod.main(transport="stdio", host="127.0.0.1", port=8200)
    assert "called" not in seen
    assert seen["lifespan"] is not None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _settings_with_api_key(key: str) -> Any:
    class _Api:
        api_key = key
        host = "127.0.0.1"
        port = 8200

    class _S:
        api = _Api()

    return _S()
