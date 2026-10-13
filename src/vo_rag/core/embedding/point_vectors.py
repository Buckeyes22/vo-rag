"""Shared BGE-M3 embed-response -> Qdrant point-vector helpers.

These pure functions convert a raw BGE-M3 embedding-server response row into the
vector fields a VoPoint carries (sparse int map, ColBERT downsample, dense
fallback). They live in one module
so every writer of VORAG vectors — book_rag raw/sterile ingestion AND dsi
record persistence — shares one implementation instead of drifting copies.

Configuration and defaults:
- Env vars are `VORAG_QDRANT_MAX_COLBERT_VECTORS` and
  `VORAG_COLBERT_EMPTY_FALLBACK`.
- Default ColBERT cap stays 192 (Qdrant 1 MiB/point byte budget). Bumping it is a deliberate
  operator action, never a default.
- The `dense` fallback for empty ColBERT remains the default. A silent
  drop is forbidden — a quality regression would not surface.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "sparse_payload_to_int_map",
    "include_colbert_at_ingest",
    "max_colbert_vectors_per_point",
    "downsample_colbert_vectors",
    "fallback_colbert_vectors",
]


def include_colbert_at_ingest() -> bool:
    """Whether to compute + store real ColBERT multivectors at INGEST time.

    Default OFF (``VORAG_BOOK_RAG_INCLUDE_COLBERT=0``): real per-chunk ColBERT is ~10x
    the Qdrant storage and ~30-40% of the embed compute. Schema compatibility
    is retained by the dense-derived fallback vector (see
    ``fallback_colbert_vectors``). Shared retrieval deliberately fuses
    genuine dense/sparse/BM25 channels; token-query late interaction against
    dense substitutes is not compatible scoring. Set to ``1`` only when a corpus
    genuinely needs full ColBERT fidelity at rest. This gates WRITE/ingest paths
    only. Query embeddings retain their existing wire bundle for compatibility;
    the default shared store scorer does not consume ColBERT.
    """
    return os.environ.get("VORAG_BOOK_RAG_INCLUDE_COLBERT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def sparse_payload_to_int_map(value: Any) -> dict[int, float]:
    """Normalise a BGE-M3 sparse payload to ``{token_id: weight}``.

    Accepts either the ``{"indices": [...], "values": [...]}`` shape or a plain
    ``{token: weight}`` mapping.
    """
    if isinstance(value, dict) and "indices" in value:
        return {
            int(idx): float(val)
            for idx, val in zip(value.get("indices", []), value.get("values", []), strict=False)
        }
    return {int(k): float(v) for k, v in dict(value or {}).items()}


def max_colbert_vectors_per_point() -> int:
    """Per-point ColBERT multivector cap (``VORAG_QDRANT_MAX_COLBERT_VECTORS``)."""
    raw = os.environ.get("VORAG_QDRANT_MAX_COLBERT_VECTORS", "192")
    try:
        return max(0, int(raw))
    except ValueError:
        return 192


def downsample_colbert_vectors(
    vectors: Any,
    *,
    max_vectors: int | None = None,
) -> list[list[float]]:
    """Keep Qdrant ColBERT multivectors below the per-point vector byte limit.

    Qdrant rejects a single point when the total stored vector bytes exceed
    1 MiB. A long BGE-M3 ColBERT sequence can exceed that even when upserted
    one point at a time, so batch splitting cannot recover. Deterministically
    sample across the token-vector sequence to preserve coverage of the chunk.
    """
    if not vectors:
        return []
    limit = max_colbert_vectors_per_point() if max_vectors is None else max(0, max_vectors)
    if limit <= 0:
        return []

    normalized = [list(vector) for vector in vectors if vector]
    if len(normalized) <= limit:
        return normalized
    if limit == 1:
        return [normalized[0]]

    max_index = len(normalized) - 1
    sampled: list[list[float]] = []
    last_index = -1
    for out_index in range(limit):
        source_index = round(out_index * max_index / (limit - 1))
        if source_index <= last_index:
            source_index = last_index + 1
        if source_index > max_index:
            source_index = max_index
        sampled.append(normalized[source_index])
        last_index = source_index
    return sampled


def fallback_colbert_vectors(dense: list[float]) -> list[list[float]]:
    """When BGE-M3 returns no ColBERT vectors, fall back to the dense vector.

    Controlled by ``VORAG_COLBERT_EMPTY_FALLBACK`` (default ``dense``); any other
    value hard-fails so a silent quality regression cannot slip through. The
    Qdrant ``colbert`` named vector is required, so a point must carry at least
    one ColBERT vector.
    """
    mode = os.environ.get("VORAG_COLBERT_EMPTY_FALLBACK", "dense").strip().lower()
    if mode != "dense":
        raise RuntimeError("BGE-M3 produced no ColBERT vectors and fallback is disabled")
    if not dense:
        raise RuntimeError("BGE-M3 produced no ColBERT vectors and no dense fallback vector")
    return [dense]
