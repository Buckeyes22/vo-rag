"""Production BM25 tokenizer leaf.

**PERSISTENCE CONTRACT (load-bearing — do NOT change):**

    _stable_token_id(tok) == int.from_bytes(
        hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest(), "big"
    )

This is the token id used at BOTH query time and write time across every
Qdrant writer (book_rag chunk_embed and sterilize, dsi, summary, raptor,
graph, and the ops passage_migration and brain_sync tools). Changing the
digest size, the algorithm, or the endianness means re-ingesting every
existing Qdrant point.

Tests in ``tests/core/retrieval/test_bm25_tokenizer.py`` pin hard-coded
ids for representative tokens (preserved SAE/ISO pattern, preserved
Eq./Figure pattern, camelcase-num pattern). Drift in those values is
caught by the test gate.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

_PRESERVE_PATTERNS = [
    re.compile(r"\b(?:SAE|ISO|IEC|ASTM|DIN|EN|FIA)\s*[A-Z]?\d+(?:[\.\-]\d+)*\b"),
    re.compile(
        r"\b(?:Eq\.?|Equation|Fig\.?|Figure|Sec\.?|Section|Ch\.?|Chapter|Table)\s*\d+(?:\.\d+)*\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:P/N|PN|SKU|Part\s*Number)\s*[A-Z0-9]+(?:[-\.][A-Z0-9]+)*\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2,}\d+[A-Z]?\b"),
    re.compile(r"\b\d{3,4}(?:-[A-Z]\d+)?\b"),
]

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "by",
    "with",
    "is",
    "are",
    "was",
    "were",
}
_MIN_TOKEN_LEN = 2
_MAX_TOKENS_PER_DOC = 16_384


def tokenize(text: str) -> list[str]:
    preserved: list[str] = []

    def _capture(m: re.Match[str]) -> str:
        preserved.append(m.group(0).lower().replace(" ", "_"))
        return " "

    remaining = text
    for pat in _PRESERVE_PATTERNS:
        remaining = pat.sub(_capture, remaining)

    general = [
        t.lower()
        for t in re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", remaining)
        if len(t) >= _MIN_TOKEN_LEN and t.lower() not in _STOPWORDS
    ]
    return (preserved + general)[:_MAX_TOKENS_PER_DOC]


def _stable_token_id(tok: str) -> int:
    digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def tokens_to_bm25_sparse(tokens: list[str]) -> dict[int, float]:
    counts = Counter(tokens)
    return {_stable_token_id(tok): float(tf) for tok, tf in counts.items()}


__all__ = [
    "tokenize",
    "tokens_to_bm25_sparse",
    "_stable_token_id",
]
