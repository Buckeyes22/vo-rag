"""Tests for Anthropic provider error mapping."""

from __future__ import annotations

from typing import Any

import anthropic
import httpx
import pytest

from vo_rag.core.llm.errors import LLMBadRequestError, LLMRateLimitError, LLMServerError
from vo_rag.core.llm.providers.anthropic_provider import AnthropicProvider


class _FakeContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeUsage:
    def __init__(
        self,
        input_tokens: int = 10,
        output_tokens: int = 5,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_write


class _FakeResponse:
    def __init__(self, text: str, **usage_kwargs: int) -> None:
        self.content = [_FakeContent(text)]
        self.usage = _FakeUsage(**usage_kwargs)


class _FakeMessages:
    def __init__(
        self,
        *,
        raise_exc: BaseException | None = None,
        response: _FakeResponse | None = None,
    ) -> None:
        self._raise = raise_exc
        self._response = response
        self.calls = 0
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls += 1
        self.last_kwargs = kwargs
        if self._raise is not None:
            raise self._raise
        assert self._response is not None
        return self._response


class _FakeAnthropicClient:
    def __init__(self, messages: _FakeMessages) -> None:
        self.messages = messages
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _status_error(status: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request, json={"error": {"message": "x"}})
    return anthropic.APIStatusError("status error", response=response, body=None)


def _rate_limit_error() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request, json={"error": {"message": "rl"}})
    return anthropic.RateLimitError("rate limited", response=response, body=None)


@pytest.mark.asyncio
async def test_complete_raw_success_populates_fields() -> None:
    fake = _FakeMessages(
        response=_FakeResponse(
            "hello",
            input_tokens=12,
            output_tokens=4,
            cache_read=7,
            cache_write=3,
        )
    )
    client = _FakeAnthropicClient(fake)
    provider = AnthropicProvider(api_key="x", model="claude-x", client=client)  # type: ignore[arg-type]

    resp = await provider.complete_raw(
        messages=[{"role": "user", "content": "hi"}],
        system="be helpful",
    )

    assert resp.text == "hello"
    assert resp.model == "claude-x"
    assert resp.input_tokens == 12
    assert resp.output_tokens == 4
    assert resp.cache_read_input_tokens == 7
    assert resp.cache_write_input_tokens == 3
    assert resp.provider == "anthropic"
    assert resp.latency_ms >= 0.0
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["system"] == "be helpful"


@pytest.mark.asyncio
async def test_rate_limit_mapped() -> None:
    client = _FakeAnthropicClient(_FakeMessages(raise_exc=_rate_limit_error()))
    provider = AnthropicProvider(api_key="x", model="claude-x", client=client)  # type: ignore[arg-type]
    with pytest.raises(LLMRateLimitError):
        await provider.complete_raw(messages=[])


@pytest.mark.asyncio
async def test_529_mapped_to_rate_limit() -> None:
    client = _FakeAnthropicClient(_FakeMessages(raise_exc=_status_error(529)))
    provider = AnthropicProvider(api_key="x", model="claude-x", client=client)  # type: ignore[arg-type]
    with pytest.raises(LLMRateLimitError):
        await provider.complete_raw(messages=[])


@pytest.mark.asyncio
async def test_500_mapped_to_server_error() -> None:
    client = _FakeAnthropicClient(_FakeMessages(raise_exc=_status_error(500)))
    provider = AnthropicProvider(api_key="x", model="claude-x", client=client)  # type: ignore[arg-type]
    with pytest.raises(LLMServerError):
        await provider.complete_raw(messages=[])


@pytest.mark.asyncio
async def test_400_mapped_to_bad_request() -> None:
    client = _FakeAnthropicClient(_FakeMessages(raise_exc=_status_error(400)))
    provider = AnthropicProvider(api_key="x", model="claude-x", client=client)  # type: ignore[arg-type]
    with pytest.raises(LLMBadRequestError):
        await provider.complete_raw(messages=[])


@pytest.mark.asyncio
async def test_connection_error_mapped_to_server() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    conn_err = anthropic.APIConnectionError(request=request)
    client = _FakeAnthropicClient(_FakeMessages(raise_exc=conn_err))
    provider = AnthropicProvider(api_key="x", model="claude-x", client=client)  # type: ignore[arg-type]
    with pytest.raises(LLMServerError):
        await provider.complete_raw(messages=[])
