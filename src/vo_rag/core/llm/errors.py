"""Typed error hierarchy for the LLM call path.

Providers map native SDK exceptions into this hierarchy so the retry policy
can distinguish retriable (RateLimit/Server) from fatal (BadRequest).
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM-call errors."""


class LLMRateLimitError(LLMError):
    """429 or 529 — rate-limited or overloaded. Retriable."""


class LLMServerError(LLMError):
    """5xx (except 529) or connection failure. Retriable."""


class LLMBadRequestError(LLMError):
    """4xx other than 429. NOT retriable — fix the request."""


class StructuredOutputError(LLMError):
    """Typed-output retries exhausted without a schema-valid parse."""

    def __init__(self, attempts: int, last_error: str, last_raw: str) -> None:
        self.attempts = attempts
        self.last_error = last_error
        self.last_raw = last_raw
        super().__init__(f"Structured output failed after {attempts} attempts: {last_error}")


__all__ = [
    "LLMBadRequestError",
    "LLMError",
    "LLMRateLimitError",
    "LLMServerError",
    "StructuredOutputError",
]
