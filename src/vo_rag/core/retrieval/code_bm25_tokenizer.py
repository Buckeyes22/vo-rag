"""Code-aware BM25 tokenizer leaf.

It reuses ``tokens_to_bm25_sparse``, so the blake2b-4 id scheme is shared
(see ``bm25_tokenizer`` docstring for the contract).

Used by code-aware RAG pipelines. Camel/snake/dot/dash splitter; 16k token
cap; stopword list trimmed to the set that does not appear in code identifiers.
"""

from __future__ import annotations

import re

from vo_rag.core.retrieval.bm25_tokenizer import tokens_to_bm25_sparse

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
    "for",
    "to",
    "is",
    "are",
    "was",
    "were",
}
_MAX_TOKENS_PER_DOC = 16_384

_TOKEN_GROUP_RE = re.compile(r"[A-Za-z0-9]+(?:[._/\-][A-Za-z0-9]+)*")
_SEPARATOR_RE = re.compile(r"[._/\-]+")
_CAMEL_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|\b)|[A-Z]?[a-z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Return the lowercased, deduplicated-by-position token list."""
    tokens: list[str] = []

    for match in _TOKEN_GROUP_RE.finditer(text):
        if len(tokens) >= _MAX_TOKENS_PER_DOC:
            break

        group = match.group(0)
        group_tokens: list[str] = []

        if "." in group and "/" not in group:
            group_tokens.append(group.lower())

        for segment in _SEPARATOR_RE.split(group):
            if not segment:
                continue
            group_tokens.extend(part.lower() for part in _CAMEL_PART_RE.findall(segment))

        seen_at_position: set[str] = set()
        for tok in group_tokens:
            if tok in seen_at_position or tok in _STOPWORDS:
                continue
            seen_at_position.add(tok)
            tokens.append(tok)
            if len(tokens) >= _MAX_TOKENS_PER_DOC:
                break

    return tokens


def to_sparse(tokens: list[str]) -> dict[int, float]:
    """Convert tokens to {token_id: tf} sparse vector matching the existing tokenizer's shape."""
    return tokens_to_bm25_sparse(tokens)


__all__ = ["tokenize", "to_sparse"]
