from __future__ import annotations

from collections.abc import Callable

TokenCounterFn = Callable[[str], int]


def naive_counter(text: str) -> int:
    """Shared char/4 fallback counter used when tokenizer construction is unavailable."""
    return max(1, len(text) // 4)
