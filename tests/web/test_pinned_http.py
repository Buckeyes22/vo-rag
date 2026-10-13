from __future__ import annotations

import ssl
from ipaddress import ip_address
from typing import Any

import httpcore
import pytest

from vo_rag.web.pinned_http import PinnedAsyncNetworkBackend


class FakeAsyncNetworkStream(httpcore.AsyncNetworkStream):
    def __init__(self) -> None:
        self.server_hostnames: list[str | None] = []
        self.writes: list[bytes] = []
        self.closed = False
        self._reads = [b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"]

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if not self._reads:
            return b""
        chunk = self._reads.pop(0)
        return chunk[:max_bytes]

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.writes.append(buffer)

    async def aclose(self) -> None:
        self.closed = True

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.server_hostnames.append(server_hostname)
        return self

    def get_extra_info(self, info: str) -> Any:
        return None


class RecordingBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, stream: FakeAsyncNetworkStream) -> None:
        self.stream = stream
        self.calls: list[tuple[str, int]] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        self.calls.append((host, port))
        return self.stream

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        raise AssertionError("not used")

    async def sleep(self, seconds: float) -> None:
        return None


@pytest.mark.asyncio
async def test_pinned_backend_connects_to_resolved_ip_but_tls_uses_original_host() -> None:
    stream = FakeAsyncNetworkStream()
    delegate = RecordingBackend(stream)
    backend = PinnedAsyncNetworkBackend(
        pins={("example.com", 443): ip_address("93.184.216.34")},
        delegate=delegate,
    )
    pool = httpcore.AsyncConnectionPool(network_backend=backend)

    try:
        response = await pool.request("GET", "https://example.com/article")
    finally:
        await pool.aclose()

    assert response.status == 200
    assert delegate.calls == [("93.184.216.34", 443)]
    assert stream.server_hostnames == ["example.com"]
