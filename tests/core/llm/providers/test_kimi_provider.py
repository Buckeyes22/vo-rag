"""Tests for Kimi provider."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from vo_rag.core.llm.providers.kimi_provider import KimiProvider


@pytest.mark.asyncio
async def test_kimi_provider_keeps_thinking_enabled_for_k26() -> None:
    requests: list[dict[str, Any]] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler),
        base_url="https://api.moonshot.ai/v1",
    )
    provider = KimiProvider(
        api_key="test-key",
        model="kimi-k2.6",
        client=client,
    )

    response = await provider.complete_raw(messages=[{"role": "user", "content": "hi"}])

    assert response.text == '{"answer":"ok"}'
    assert requests[0]["thinking"] == {"type": "enabled"}
    assert "temperature" not in requests[0]
    await provider.aclose()
