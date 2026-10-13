"""Llama.cpp / OpenAI-compatible local provider."""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from vo_rag.core.llm.errors import LLMBadRequestError, LLMRateLimitError, LLMServerError
from vo_rag.core.llm.response import LLMResponse

_log = structlog.get_logger(__name__)


class LlamaCppProvider:
    name: str = "llama_cpp"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout_s)
        self.model = model

    async def complete_raw(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> LLMResponse:
        payload_messages: list[dict[str, Any]] = []
        if system is not None:
            payload_messages.append({"role": "system", "content": _flatten_system(system)})
        payload_messages.extend(messages)

        t0 = time.perf_counter()
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": payload_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                raise LLMRateLimitError(str(exc)) from exc
            if 500 <= status < 600:
                raise LLMServerError(str(exc)) from exc
            raise LLMBadRequestError(str(exc)) from exc
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            raise LLMServerError(str(exc)) from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        data = response.json()
        text = str(data["choices"][0]["message"]["content"])
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=latency_ms,
            provider=self.name,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _flatten_system(system: str | list[dict[str, Any]]) -> str:
    if isinstance(system, str):
        return system
    return "\n".join(str(b.get("text", "")) for b in system if isinstance(b, dict))


__all__ = ["LlamaCppProvider"]
