"""Tests for LLM error hierarchy."""

from __future__ import annotations

from vo_rag.core.llm.errors import (
    LLMBadRequestError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    StructuredOutputError,
)


def test_hierarchy() -> None:
    for cls in (LLMRateLimitError, LLMServerError, LLMBadRequestError, StructuredOutputError):
        assert issubclass(cls, LLMError)


def test_structured_carries_attempts_and_raw() -> None:
    err = StructuredOutputError(attempts=3, last_error="missing field 'x'", last_raw="{}")
    assert err.attempts == 3
    assert err.last_error == "missing field 'x'"
    assert err.last_raw == "{}"
    assert "3 attempts" in str(err)


def test_errors_raisable_and_catchable_as_base() -> None:
    for cls in (LLMRateLimitError, LLMServerError, LLMBadRequestError):
        try:
            raise cls("boom")
        except LLMError as caught:
            assert isinstance(caught, cls)
