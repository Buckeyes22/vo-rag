"""LLMResponse — the single return shape for every LLM call.

Provider layer fills provider-observable fields (tokens, latency, cache
counters). LLMClient later overlays caller-layer fields (attempts,
prompt_name, prompt_version, variant, cache_hit, cost_usd).
"""

from __future__ import annotations

from vo_rag.core.models.common import VoBaseModel


class LLMResponse(VoBaseModel):
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    attempts: int = 1
    prompt_name: str | None = None
    prompt_version: int | None = None
    variant: str | None = None
    cache_hit: bool = False
    provider: str = ""


__all__ = ["LLMResponse"]
