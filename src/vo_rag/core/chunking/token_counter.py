from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any


class TokenCounter:
    """BGE-M3 XLM-RoBERTa tokenizer wrapper for VORAG chunking modules."""

    def __init__(self, model: str | None = None) -> None:
        from transformers import AutoTokenizer

        model = model or os.environ.get("VORAG_CHUNK_TOKENIZER_MODEL", "BAAI/bge-m3")
        from_pretrained: Any = AutoTokenizer.from_pretrained
        self._tok: Any = from_pretrained(model)

        @lru_cache(maxsize=4096)
        def cached_count(text: str) -> int:
            return len(self._tok.encode(text, add_special_tokens=False))

        self._cached_count: Callable[[str], int] = cached_count

    def count(self, text: str) -> int:
        if not text:
            return 0
        return self._cached_count(text)

    def __call__(self, text: str) -> int:
        return self.count(text)


@lru_cache(maxsize=1)
def default_token_counter() -> TokenCounter:
    return TokenCounter()
