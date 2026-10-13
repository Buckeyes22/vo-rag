"""httpx/httpcore transport helpers for connect-time DNS pinning."""

from __future__ import annotations

import ipaddress
import ssl
from collections.abc import Mapping
from typing import Any

import httpcore
import httpx
from httpcore._backends.auto import AutoBackend

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
PinKey = tuple[str, int]
PinValue = str | IPAddress


class PinnedAsyncNetworkBackend(httpcore.AsyncNetworkBackend):
    """Network backend that dials validated IPs without rewriting request URLs."""

    def __init__(
        self,
        *,
        pins: Mapping[PinKey, PinValue],
        delegate: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self._pins = {
            (host.lower(), port): str(ipaddress.ip_address(ip)) for (host, port), ip in pins.items()
        }
        self._delegate = delegate or AutoBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        target_host = self._pins.get((host.lower(), port), host)
        return await self._delegate.connect_tcp(
            target_host,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        return await self._delegate.connect_unix_socket(
            path,
            timeout=timeout,
            socket_options=socket_options,
        )

    async def sleep(self, seconds: float) -> None:
        await self._delegate.sleep(seconds)


class PinnedAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """httpx transport that preserves URL hostnames while pinning TCP dials."""

    def __init__(
        self,
        *,
        pins: Mapping[PinKey, PinValue],
        verify: ssl.SSLContext | str | bool = True,
        limits: httpx.Limits | None = None,
        http1: bool = True,
        http2: bool = False,
        retries: int = 0,
    ) -> None:
        active_limits = limits or httpx.Limits(max_connections=10, max_keepalive_connections=0)
        super().__init__(
            verify=verify,
            limits=active_limits,
            http1=http1,
            http2=http2,
            retries=retries,
        )
        ssl_context = httpx.create_ssl_context(verify=verify)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_context,
            max_connections=active_limits.max_connections,
            max_keepalive_connections=active_limits.max_keepalive_connections,
            keepalive_expiry=active_limits.keepalive_expiry,
            http1=http1,
            http2=http2,
            retries=retries,
            network_backend=PinnedAsyncNetworkBackend(pins=pins),
        )


__all__ = ["PinnedAsyncHTTPTransport", "PinnedAsyncNetworkBackend"]
