"""Tests for structured output completion."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from vo_rag.core.llm.errors import StructuredOutputError
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.llm.structured import complete_with_schema


class _Answer(BaseModel):
    value: int
    label: str


class _ScriptedProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0
        self.last_messages: list[dict[str, Any]] | None = None

    async def complete_raw(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> LLMResponse:
        self.calls += 1
        self.last_messages = list(messages)
        text = self._texts.pop(0)
        return LLMResponse(text=text, model=self.model, provider=self.name)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_valid_first_try_attempts_1() -> None:
    provider = _ScriptedProvider(['{"value": 7, "label": "ok"}'])

    parsed, resp = await complete_with_schema(
        provider, schema=_Answer, messages=[{"role": "user", "content": "hi"}]
    )

    assert parsed.value == 7
    assert parsed.label == "ok"
    assert resp.attempts == 1
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_malformed_then_valid_attempts_2() -> None:
    provider = _ScriptedProvider(
        [
            '{"value": "not-an-int", "label": "x"}',
            '{"value": 3, "label": "ok"}',
        ]
    )

    parsed, resp = await complete_with_schema(
        provider,
        schema=_Answer,
        messages=[{"role": "user", "content": "hi"}],
        max_retries=2,
    )

    assert parsed.value == 3
    assert resp.attempts == 2
    assert provider.calls == 2
    assert provider.last_messages is not None
    assert any("did not validate" in (m.get("content", "") or "") for m in provider.last_messages)


@pytest.mark.asyncio
async def test_always_malformed_raises_structured_output_error() -> None:
    provider = _ScriptedProvider(["garbage", "still garbage", "yet more garbage"])

    with pytest.raises(StructuredOutputError) as exc_info:
        await complete_with_schema(
            provider,
            schema=_Answer,
            messages=[{"role": "user", "content": "hi"}],
            max_retries=2,
            prompt_name="answer_v1",
        )

    assert exc_info.value.attempts == 3
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_fenced_json_isolated() -> None:
    provider = _ScriptedProvider(['```json\n{"value": 1, "label": "x"}\n```'])

    parsed, _ = await complete_with_schema(provider, schema=_Answer, messages=[])

    assert parsed.value == 1
