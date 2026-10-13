"""Standard retry policy for LLM calls.

- 5 attempts max
- Exponential backoff 1s → 30s
- Retries LLMRateLimitError + LLMServerError only; 4xx fails fast
- Emits vo_rag_llm_retries_total per retry with (provider, model, reason)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from vo_rag.core.llm.errors import LLMRateLimitError, LLMServerError
from vo_rag.core.observability.metrics import llm_retries_total

_log = structlog.get_logger(__name__)


def make_retry(provider: str, model: str) -> AsyncRetrying:
    """Build an AsyncRetrying instance tagged with (provider, model) for metrics."""

    def _before_sleep(retry_state) -> None:  # type: ignore[no-untyped-def]
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if exc is None:
            return
        reason = "rate_limit" if isinstance(exc, LLMRateLimitError) else "server_error"
        llm_retries_total.labels(provider=provider, model=model, reason=reason).inc()
        _log.info(
            "llm_retry",
            attempt=retry_state.attempt_number,
            reason=reason,
            provider=provider,
            model=model,
        )

    return AsyncRetrying(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((LLMRateLimitError, LLMServerError)),
        before_sleep=_before_sleep,
        reraise=True,
    )


async def with_retry[T](
    provider: str,
    model: str,
    coro_factory: Callable[[], Awaitable[T]],
) -> T:
    """Execute `coro_factory()` under the standard LLM retry policy."""
    retrying = make_retry(provider, model)
    async for attempt in retrying:
        with attempt:
            return await coro_factory()
    raise RuntimeError("unreachable: retry exhausted without raising")


__all__ = ["make_retry", "with_retry"]
