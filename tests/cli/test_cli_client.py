from __future__ import annotations

import httpx
import pytest
import respx
from click.testing import CliRunner

from vo_rag.cli.client import VoRagApiClient, VoRagApiError
from vo_rag.cli.main import cli


class TestVoRagApiClient:
    def test_init_sets_headers(self) -> None:
        client = VoRagApiClient(base_url="http://localhost:8000", api_key="test-key")
        assert client.base_url == "http://localhost:8000"
        assert client.api_key == "test-key"  # pragma: allowlist secret
        assert client.headers["X-VO-RAG-Caller"] == "cli"
        assert client.headers["X-API-Key"] == "test-key"  # pragma: allowlist secret

    def test_init_no_key(self) -> None:
        client = VoRagApiClient(base_url="http://localhost:8000", api_key=None)
        assert "X-API-Key" not in client.headers

    @respx.mock
    async def test_get_json_success(self) -> None:
        route = respx.get("http://localhost:8000/api/v1/test").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        client = VoRagApiClient(base_url="http://localhost:8000", api_key="test-key")
        result = await client.get_json("/api/v1/test")
        assert result == {"ok": True}
        assert route.called

    @respx.mock
    async def test_post_json_success(self) -> None:
        route = respx.post("http://localhost:8000/api/v1/test").mock(
            return_value=httpx.Response(200, json={"id": "123"})
        )
        client = VoRagApiClient(base_url="http://localhost:8000", api_key="test-key")
        result = await client.post_json("/api/v1/test", {"foo": "bar"})
        assert result == {"id": "123"}
        assert route.called

    @respx.mock
    async def test_retry_on_429(self) -> None:
        route = respx.get("http://localhost:8000/api/v1/test").mock(
            side_effect=[
                httpx.Response(429, json={"detail": "rate limited"}),
                httpx.Response(429, json={"detail": "rate limited"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        client = VoRagApiClient(base_url="http://localhost:8000", api_key="test-key")
        result = await client.get_json("/api/v1/test")
        assert result == {"ok": True}
        assert route.call_count == 3

    @respx.mock
    async def test_retry_on_503(self) -> None:
        route = respx.get("http://localhost:8000/api/v1/test").mock(
            side_effect=[
                httpx.Response(503, json={"detail": "unavailable"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        client = VoRagApiClient(base_url="http://localhost:8000", api_key="test-key")
        result = await client.get_json("/api/v1/test")
        assert result == {"ok": True}
        assert route.call_count == 2

    @respx.mock
    async def test_no_retry_on_400(self) -> None:
        route = respx.get("http://localhost:8000/api/v1/test").mock(
            return_value=httpx.Response(400, json={"detail": "bad request"})
        )
        client = VoRagApiClient(base_url="http://localhost:8000", api_key="test-key")
        with pytest.raises(VoRagApiError) as exc_info:
            await client.get_json("/api/v1/test")
        assert exc_info.value.status_code == 400
        assert route.call_count == 1

    @respx.mock
    async def test_transport_error_retry(self) -> None:
        route = respx.get("http://localhost:8000/api/v1/test").mock(
            side_effect=[
                httpx.ConnectError("connection failed"),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        client = VoRagApiClient(base_url="http://localhost:8000", api_key="test-key")
        result = await client.get_json("/api/v1/test")
        assert result == {"ok": True}
        assert route.call_count == 2


def test_client_from_ctx_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("VORAG_API_KEY", raising=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["--url", "http://localhost:8000", "admin", "status"])
    assert result.exit_code != 0
    assert "VORAG_API_KEY is required" in result.output


def test_client_from_ctx_allow_unauth_no_key(monkeypatch) -> None:
    monkeypatch.delenv("VORAG_API_KEY", raising=False)
    with respx.mock:
        respx.get("http://localhost:8000/health").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["--url", "http://localhost:8000", "health"])
        assert result.exit_code == 0
        assert "ok" in result.output


def test_client_survives_separate_event_loops_like_the_cli() -> None:
    """The CLI runs each request in its own ``asyncio.run`` loop and closes the
    client in yet another one (``ctx.call_on_close``). Pooled connections bound
    to a dead loop used to crash that close with 'Event loop is closed' after a
    successful request, turning a success into exit 1. Needs real sockets, so
    it talks to a throwaway local HTTP server instead of respx."""
    import asyncio
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, so connections stay pooled

        def do_GET(self) -> None:
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = VoRagApiClient(base_url=f"http://127.0.0.1:{server.server_port}", api_key="k")
        assert asyncio.run(client.get_json("/a")) == {"ok": True}
        assert asyncio.run(client.get_json("/b")) == {"ok": True}
        asyncio.run(client.aclose())
    finally:
        server.shutdown()
        server.server_close()
