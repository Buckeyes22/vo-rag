"""Provider Protocol consumed by LLMClient."""

from __future__ import annotations

from typing import Any, Protocol

from vo_rag.core.llm.response import LLMResponse


class Provider(Protocol):
    name: str

    @property
    def model(self) -> str: ...

    async def complete_raw(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse: ...

    async def aclose(self) -> None: ...


__all__ = ["Provider"]
