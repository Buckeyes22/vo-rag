"""Typed-output completion with retry-on-validation.

Explicit retry loop (not instructor-backed) so we observe attempts and emit
our own Prometheus counter on exhaustion. The retry reprompts with the
validation error text so the model can self-correct.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ValidationError

from vo_rag.core.llm.errors import StructuredOutputError
from vo_rag.core.llm.providers.base import Provider
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.observability.metrics import llm_structured_parse_failures_total

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.MULTILINE)
_FIRST_JSON = re.compile(r"[{\[]\[\s\S\]*[}\]]")


async def complete_with_schema[T: BaseModel](
    provider: Provider,
    *,
    schema: type[T],
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 2,
    prompt_name: str | None = None,
) -> tuple[T, LLMResponse]:
    """Call provider, parse into `schema`. On ValidationError, reprompt with the
    error text; retry up to `max_retries` times. Raises StructuredOutputError
    after `max_retries + 1` total attempts.
    """
    current_messages: list[dict[str, Any]] = list(messages)
    last_raw = ""
    last_err = ""

    for attempt in range(max_retries + 1):
        resp = await provider.complete_raw(
            messages=current_messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        last_raw = resp.text
        try:
            parsed = schema.model_validate_json(_isolate_json(resp.text))
            return parsed, resp.model_copy(update={"attempts": attempt + 1})
        except ValidationError as exc:
            last_err = str(exc)
            current_messages = [
                *current_messages,
                {"role": "assistant", "content": resp.text},
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not validate against the required "
                        f"schema.\nError:\n{exc}\n"
                        "Return ONLY corrected JSON — no prose, no markdown fences."
                    ),
                },
            ]

    llm_structured_parse_failures_total.labels(prompt_name=prompt_name or "unknown").inc()
    raise StructuredOutputError(attempts=max_retries + 1, last_error=last_err, last_raw=last_raw)


def _isolate_json(text: str) -> str:
    """Strip markdown fences and isolate the first JSON body. Returns text as-is
    if no fence/object markers present so the parser sees the raw stream."""
    cleaned = _FENCE.sub("", text.strip())
    stripped = cleaned.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    match = _FIRST_JSON.search(cleaned)
    if match:
        return match.group(0)
    return cleaned


__all__ = ["complete_with_schema"]
