"""Anthropic AsyncAnthropic provider with typed-error mapping."""

from __future__ import annotations

import time
from typing import Any

import anthropic
import structlog

from vo_rag.core.llm.errors import LLMBadRequestError, LLMRateLimitError, LLMServerError
from vo_rag.core.llm.response import LLMResponse

_log = structlog.get_logger(__name__)

_RETRIABLE_STATUSES = {429, 529}


class AnthropicProvider:
    name: str = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 60.0,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s)
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
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system is not None:
            kwargs["system"] = system

        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            status = getattr(exc, "status_code", None) or 0
            if status in _RETRIABLE_STATUSES:
                raise LLMRateLimitError(str(exc)) from exc
            if 500 <= status < 600:
                raise LLMServerError(str(exc)) from exc
            raise LLMBadRequestError(str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMServerError(str(exc)) from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = _extract_text(response)
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            latency_ms=latency_ms,
            provider=self.name,
        )

    async def aclose(self) -> None:
        await self._client.close()


def _extract_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


__all__ = ["AnthropicProvider"]
