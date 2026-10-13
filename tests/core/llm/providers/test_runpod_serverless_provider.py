"""Tests for RunpodServerlessProvider."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from vo_rag.core.llm.providers.runpod_serverless_provider import RunpodServerlessProvider
from vo_rag.core.llm.response import LLMResponse


class _FakeServerlessResponse:
    def __init__(
        self,
        content: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        finish_reason: str,
        duration_ms: int,
        raw: dict[str, Any],
    ) -> None:
        self.content = content
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.finish_reason = finish_reason
        self.duration_ms = duration_ms
        self.raw = raw


@pytest.mark.asyncio
async def test_complete_raw_returns_llm_response() -> None:
    client = AsyncMock()
    client.model = "qwen3-6-27b"
    client.chat_completion.return_value = _FakeServerlessResponse(
        content="hello",
        model="qwen3-6-27b",
        input_tokens=10,
        output_tokens=3,
        finish_reason="stop",
        duration_ms=420,
        raw={"id": "x"},
    )
    provider = RunpodServerlessProvider(client=client)

    response = await provider.complete_raw(
        system="be brief",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=8,
    )

    assert isinstance(response, LLMResponse)
    assert response.text == "hello"
    assert response.model == "qwen3-6-27b"
    assert response.input_tokens == 10
    assert response.output_tokens == 3
    assert response.latency_ms == 420.0
    assert response.provider == "runpod_serverless"

    call = client.chat_completion.call_args
    messages = call.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[1] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_complete_raw_omits_system_when_none() -> None:
    client = AsyncMock()
    client.model = "qwen3-6-27b"
    client.chat_completion.return_value = _FakeServerlessResponse(
        content="x",
        model="m",
        input_tokens=1,
        output_tokens=1,
        finish_reason="stop",
        duration_ms=10,
        raw={},
    )
    provider = RunpodServerlessProvider(client=client)

    await provider.complete_raw(
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=8,
    )

    messages = client.chat_completion.call_args.kwargs["messages"]
    assert all(message["role"] != "system" for message in messages)
