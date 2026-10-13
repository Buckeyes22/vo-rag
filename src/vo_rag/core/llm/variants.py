"""Weighted variant selection for prompt canary / A-B rollouts.

- Weights must sum to 1.0 within 1e-9.
- When `envelope_id` is provided, selection is deterministic per envelope
  (consistent-hashing) so retries and sub-pipelines see the same variant.
- Otherwise uses random.random() for per-call stochastic selection.
"""

from __future__ import annotations

import hashlib
import random

from vo_rag.core.observability.metrics import llm_variant_selections_total


def select_variant(
    prompt_name: str,
    weights: dict[str, float],
    *,
    envelope_id: str | None = None,
    rng: random.Random | None = None,
) -> str:
    if not weights:
        raise ValueError("variant weights must be non-empty")
    if any(w < 0 for w in weights.values()):
        raise ValueError("variant weights must be non-negative")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"variant weights must sum to 1.0 (±1e-9); got {total}")

    if envelope_id is not None:
        digest = hashlib.sha256(f"{prompt_name}:{envelope_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / (1 << 64)
    else:
        value = (rng or random).random()

    cumulative = 0.0
    chosen: str | None = None
    for name in sorted(weights):
        cumulative += weights[name]
        if value < cumulative:
            chosen = name
            break
    if chosen is None:
        chosen = sorted(weights)[-1]

    llm_variant_selections_total.labels(prompt_name=prompt_name, variant=chosen).inc()
    return chosen


__all__ = ["select_variant"]
