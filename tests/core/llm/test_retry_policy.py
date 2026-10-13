"""Tests for retry policy."""

from __future__ import annotations

import asyncio

import pytest

from vo_rag.core.llm.errors import LLMBadRequestError, LLMRateLimitError, LLMServerError
from vo_rag.core.llm.retry import with_retry


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real wall-clock waits in retry tests."""

    async def _fast(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _fast)


@pytest.mark.asyncio
async def test_rate_limit_then_success() -> None:
    calls = 0

    async def coro() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise LLMRateLimitError("429")
        return "ok"

    result = await with_retry("anthropic", "claude-x", coro)

    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_bad_request_not_retried() -> None:
    calls = 0

    async def coro() -> str:
        nonlocal calls
        calls += 1
        raise LLMBadRequestError("400 bad payload")

    with pytest.raises(LLMBadRequestError):
        await with_retry("anthropic", "claude-x", coro)

    assert calls == 1


@pytest.mark.asyncio
async def test_529_retried_as_rate_limit() -> None:
    calls = 0

    async def coro() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise LLMRateLimitError("529 overloaded")
        return "ok"

    result = await with_retry("anthropic", "claude-x", coro)

    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_server_error_exhausts_after_five_attempts() -> None:
    calls = 0

    async def coro() -> str:
        nonlocal calls
        calls += 1
        raise LLMServerError("503")

    with pytest.raises(LLMServerError):
        await with_retry("anthropic", "claude-x", coro)

    assert calls == 5
