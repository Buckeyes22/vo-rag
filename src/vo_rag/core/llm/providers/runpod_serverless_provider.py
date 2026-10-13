"""LLMClient provider for RunPod Serverless OpenAI-compatible endpoints.

The ServerlessClient import is lazy (inside methods) so this module imports
cleanly even when `vo_rag.core.cloud.serverless` has not been built yet.
"""

from __future__ import annotations

from typing import Any

from vo_rag.core.llm.response import LLMResponse


class RunpodServerlessProvider:
    """Provider implementation backed by a single ServerlessClient."""

    name: str = "runpod_serverless"

    def __init__(self, *, client: Any) -> None:
        self._client = client

    @property
    def model(self) -> str:
        return str(self._client.model)

    async def complete_raw(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        result = await self._client.chat_completion(
            messages=self._prepare_messages(system, messages),
            max_tokens=max_tokens,
            temperature=temperature,
            extra=extra,
        )
        return LLMResponse(
            text=result.content,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_read_input_tokens=0,
            cache_write_input_tokens=0,
            latency_ms=float(result.duration_ms),
            provider=self.name,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _prepare_messages(
        system: str | list[dict[str, Any]] | None,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if system is None:
            return list(messages)
        if isinstance(system, str):
            return [{"role": "system", "content": system}, *messages]

        flattened = "\n".join(
            str(block.get("text") or block.get("content") or "")
            for block in system
            if isinstance(block, dict)
        )
        return [{"role": "system", "content": flattened}, *messages]


__all__ = ["RunpodServerlessProvider"]
