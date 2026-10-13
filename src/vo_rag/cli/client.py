"""Async HTTP client for the vo-rag CLI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import click
import httpx


class VoRagApiError(RuntimeError):
    def __init__(
        self, message: str, *, status_code: int | None = None, payload: Any = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


@dataclass(slots=True)
class VoRagApiClient:
    base_url: str
    api_key: str | None
    verbose: bool = False
    headers: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.headers = {"X-VO-RAG-Caller": "cli"}
        if self.api_key:
            self.headers["X-API-Key"] = self.api_key

    def _http(self) -> httpx.AsyncClient:
        # One short-lived client per request, opened and closed inside the
        # caller's event loop: the CLI drives each command through its own
        # asyncio.run(), and pooled connections must never outlive that loop.
        return httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            headers=self.headers,
            timeout=httpx.Timeout(30.0),
        )

    async def aclose(self) -> None:
        """No pooled state to release; kept for call-site compatibility."""

    async def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request_json("GET", path, params=params)

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request_json("POST", path, json=payload)

    async def _request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        delay = 0.5
        last_error: Exception | None = None

        for attempt in range(1, 4):
            try:
                async with self._http() as http:
                    response = await http.request(method, path, **kwargs)
                if response.status_code >= 500 or response.status_code == 429:
                    raise VoRagApiError(
                        f"server returned retryable status {response.status_code}",
                        status_code=response.status_code,
                        payload=_safe_json(response),
                    )

                if response.status_code >= 400:
                    payload = _safe_json(response)
                    message = payload.get("detail") if isinstance(payload, dict) else response.text
                    raise VoRagApiError(
                        f"request failed ({response.status_code}): {message}",
                        status_code=response.status_code,
                        payload=payload,
                    )

                body = _safe_json(response)
                if not isinstance(body, dict):
                    raise VoRagApiError(
                        "response body was not a JSON object", status_code=response.status_code
                    )
                return body
            except (httpx.TransportError, VoRagApiError) as exc:
                retryable = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, VoRagApiError) and exc.status_code in {429, 500, 502, 503, 504}
                )
                if attempt == 3 or not retryable:
                    raise exc
                last_error = exc
                await asyncio.sleep(delay)
                delay *= 2

        if last_error is None:
            raise VoRagApiError("request failed")
        raise last_error


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"detail": response.text}


def _client_from_ctx(ctx: click.Context) -> VoRagApiClient:
    client = ctx.find_root().obj.get("client")
    if not isinstance(client, VoRagApiClient):
        raise click.ClickException("VO-RAG API client not initialized")
    if not client.api_key:
        raise click.ClickException("VORAG_API_KEY is required")
    return client


def _client_from_ctx_allow_unauth(ctx: click.Context) -> VoRagApiClient:
    client = ctx.find_root().obj.get("client")
    if not isinstance(client, VoRagApiClient):
        raise click.ClickException("VO-RAG API client not initialized")
    return client
