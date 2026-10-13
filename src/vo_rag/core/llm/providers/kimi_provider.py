"""Kimi API provider using the OpenAI-compatible chat/completions shape."""

from __future__ import annotations

import time
from typing import Any

import httpx

from vo_rag.core.llm.errors import LLMBadRequestError, LLMRateLimitError, LLMServerError
from vo_rag.core.llm.response import LLMResponse


class KimiProvider:
    name: str = "kimi"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.moonshot.ai/v1",
        timeout_s: float = 90.0,
        thinking_enabled: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._thinking_enabled = thinking_enabled
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )

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

        body: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.model.startswith("kimi-k2."):
            body["thinking"] = {"type": "enabled" if self._thinking_enabled else "disabled"}
        else:
            body["temperature"] = temperature

        t0 = time.perf_counter()
        try:
            response = await self._client.post("/chat/completions", json=body)
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

        data = response.json()
        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        return LLMResponse(
            text=str(message.get("content") or ""),
            model=self.model,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            provider=self.name,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _flatten_system(system: str | list[dict[str, Any]]) -> str:
    if isinstance(system, str):
        return system
    return "\n".join(str(block.get("text", "")) for block in system if isinstance(block, dict))


__all__ = ["KimiProvider"]
