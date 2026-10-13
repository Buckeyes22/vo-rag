"""Protected completion — wraps untrusted content, runs the LLM, validates
the response. Pipelines consuming external data go through this helper instead
of `LLMClient.complete_json` directly.

Contract:
- `untrusted_content` is wrapped in <document> tags with a fresh per-call
  canary. A hostile closing tag or embedded canary raises
  `UntrustedContentRejected` before the LLM sees the payload.
- The rendered system prompt gets the injection-defense suffix appended
  with that call's canary so a leak in the response is unambiguous.
- After the call, `validate_output` runs against the response text.
- Any signal increments `vo_rag_injection_suspected_total{pipeline,kind}`;
  a signal at confidence >= `quarantine_threshold` (default 0.8) tags
  the return as `should_quarantine=True`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.output_validator import ValidationResult, validate_output
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.llm.untrusted import build_system_suffix, generate_canary, wrap_untrusted
from vo_rag.core.observability.metrics import (
    injection_quarantined_total,
    injection_suspected_total,
)


async def complete_json_protected[T: BaseModel](
    client: LLMClient,
    *,
    schema: type[T],
    untrusted_content: str,
    pipeline: str,
    allowed_domains: list[str] | None = None,
    prompt_name: str | None = None,
    prompt_variables: dict[str, Any] | None = None,
    system: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    quarantine_threshold: float = 0.8,
    **complete_kwargs: Any,
) -> tuple[T, LLMResponse, ValidationResult, bool]:
    """Run a JSON-mode completion over untrusted content with injection defense.

    Returns (parsed, raw_response, validation, should_quarantine). Pipelines
    should upgrade `trust_level` and persist to `*_quarantine_v1` when
    `should_quarantine` is True.

    Raises:
        UntrustedContentRejected: via wrap_untrusted on hostile content
            (closing-tag injection, embedded canary). Caller should treat
            this as a hard rejection — never retry.
    """
    if prompt_name is not None and system is not None:
        raise ValueError("pass either prompt_name or system, not both")

    canary = generate_canary()
    wrapped = wrap_untrusted(untrusted_content, canary=canary)
    user_payload = {"role": "user", "content": wrapped}
    all_messages: list[dict[str, Any]] = list(messages or []) + [user_payload]

    if prompt_name is not None:
        if client.prompt_registry is None:
            raise ValueError("prompt_name requires a prompt_registry on LLMClient")
        base_system = await client.prompt_registry.render(prompt_name, prompt_variables or {})
    else:
        base_system = system or ""

    effective_system = base_system + build_system_suffix(canary=canary)

    parsed, resp = await client.complete_json(
        schema=schema,
        system=effective_system,
        messages=all_messages,
        **complete_kwargs,
    )

    validation = validate_output(
        resp.text,
        untrusted_input=untrusted_content,
        system_prompt=effective_system,
        canary=canary,
        allowed_domains=allowed_domains,
    )

    should_quarantine = False
    for signal in validation.signals:
        injection_suspected_total.labels(pipeline=pipeline, kind=signal.kind).inc()
        if signal.confidence >= quarantine_threshold:
            should_quarantine = True
    if should_quarantine:
        injection_quarantined_total.labels(pipeline=pipeline).inc()

    return parsed, resp, validation, should_quarantine


__all__ = ["complete_json_protected"]
