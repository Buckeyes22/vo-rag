"""RunPod Serverless / local vLLM OpenAI-compatible client.

Async httpx wrapper around the chat-completions endpoint exposed by RunPod
Serverless workers built from the official worker-vllm hub, or a local vLLM
OpenAI-compatible server whose base URL ends in ``/v1``. See:
- https://docs.runpod.io/serverless/vllm/openai-compatibility
- https://docs.runpod.io/serverless/vllm/environment-variables

The RunPod endpoint URL pattern is
    https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/chat/completions
and local vLLM uses
    http://<host>:<port>/v1/chat/completions.
RunPod uses a bearer token. Trusted-local OpenAI-compatible servers may omit
authorization instead of inventing a placeholder credential.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from pydantic import BaseModel


class ServerlessResponse(BaseModel):
    """Parsed chat-completion response."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    duration_ms: int
    raw: dict[str, Any]


class ServerlessClient:
    """OpenAI-compatible /v1/chat/completions client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_s: int = 600,
        retry_delays: tuple[float, ...] = (1.0, 3.0, 9.0),
    ) -> None:
        if not (base_url.endswith("/openai/v1") or base_url.endswith("/v1")):
            raise ValueError(f"base_url must end with /openai/v1 or /v1; got {base_url!r}")
        headers = {"Content-Type": "application/json"}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, headers=headers)
        self._model = model
        self._retry_delays = retry_delays

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        """Return the normalized OpenAI-compatible endpoint root."""
        return str(self._client.base_url).rstrip("/")

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> ServerlessResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra:
            payload.update(extra)

        t0 = time.perf_counter()
        response = await self._retry_post("/chat/completions", json=payload)
        data = response.json()
        duration_ms = max(1, int((time.perf_counter() - t0) * 1000))

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        usage = data.get("usage", {}) or {}

        return ServerlessResponse(
            content=str(message.get("content", "")),
            model=str(data.get("model", self._model)),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=str(choice.get("finish_reason", "")),
            duration_ms=duration_ms,
            raw=data,
        )

    async def _retry_post(self, path: str, *, json: dict[str, Any]) -> httpx.Response:
        delays = (0.0,) + self._retry_delays
        last_error: Exception | None = None
        for delay in delays:
            if delay:
                await asyncio.sleep(delay)
            try:
                response = await self._client.post(path, json=json)
            except httpx.HTTPError as exc:
                last_error = exc
                continue
            if response.status_code < 500:
                response.raise_for_status()
                return response
            last_error = httpx.HTTPStatusError(
                message=f"Server error: {response.status_code}",
                request=response.request,
                response=response,
            )
        if last_error is None:
            raise RuntimeError("retry loop exited with no error — this should be impossible")
        raise last_error

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["ServerlessClient", "ServerlessResponse"]
