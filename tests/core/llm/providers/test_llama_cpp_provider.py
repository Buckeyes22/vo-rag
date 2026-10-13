"""Tests for LlamaCpp provider error mapping."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from vo_rag.core.llm.errors import LLMBadRequestError, LLMRateLimitError, LLMServerError
from vo_rag.core.llm.providers.llama_cpp_provider import LlamaCppProvider


def _mock_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://llm")


@pytest.mark.asyncio
async def test_complete_raw_success() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello world"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    provider = LlamaCppProvider(
        base_url="http://llm", model="qwen3-x", client=_mock_client(handler)
    )

    resp = await provider.complete_raw(
        messages=[{"role": "user", "content": "hi"}],
        system="be terse",
    )

    assert resp.text == "hello world"
    assert resp.model == "qwen3-x"
    assert resp.input_tokens == 7
    assert resp.output_tokens == 3
    assert resp.provider == "llama_cpp"
    assert captured["path"] == "/v1/chat/completions"
    assert b"be terse" in captured["body"]


@pytest.mark.asyncio
async def test_429_mapped_to_rate_limit() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limit"})

    provider = LlamaCppProvider(base_url="http://llm", model="qwen3", client=_mock_client(handler))
    with pytest.raises(LLMRateLimitError):
        await provider.complete_raw(messages=[])


@pytest.mark.asyncio
async def test_503_mapped_to_server_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "busy"})

    provider = LlamaCppProvider(base_url="http://llm", model="qwen3", client=_mock_client(handler))
    with pytest.raises(LLMServerError):
        await provider.complete_raw(messages=[])


@pytest.mark.asyncio
async def test_400_mapped_to_bad_request() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad"})

    provider = LlamaCppProvider(base_url="http://llm", model="qwen3", client=_mock_client(handler))
    with pytest.raises(LLMBadRequestError):
        await provider.complete_raw(messages=[])


@pytest.mark.asyncio
async def test_connection_error_mapped_to_server() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = LlamaCppProvider(base_url="http://llm", model="qwen3", client=_mock_client(handler))
    with pytest.raises(LLMServerError):
        await provider.complete_raw(messages=[])
