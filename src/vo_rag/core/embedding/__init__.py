"""Embedding client factory.

`build_embedding_client(url, timeout_s)` returns either a raw
`EmbeddingClient` or a `CachingEmbeddingClient` wrapping it, based on
the `VORAG_EMBEDDING_CACHE_ENABLED` env flag. Default: enabled.

Callers should prefer this factory over constructing `EmbeddingClient`
directly so the cache can be rolled out (or disabled for debugging)
without editing every pipeline.

The cache wrap is best-effort: expected filesystem or SQLite failures switch
the wrapper to uncached provider calls rather than breaking pipelines. If the
optional cache dependency cannot be imported, the factory returns the raw
provider client.
"""

from __future__ import annotations

import os
from typing import Any

from vo_rag.core.embedding.client import EmbeddingClient


def build_embedding_client(url: str, *, timeout_s: float = 30.0) -> Any:
    """Return an embed(texts, ...) -> dict client, cached by default."""
    inner = EmbeddingClient(url, timeout_s=timeout_s)
    if os.environ.get("VORAG_EMBEDDING_CACHE_ENABLED", "1").strip() != "1":
        return inner
    try:
        from vo_rag.core.embedding.cache import CachingEmbeddingClient

        return CachingEmbeddingClient(inner)
    except ImportError:
        # The cache is optional in minimal installs; a missing dependency must
        # not prevent the raw provider client from serving embeddings.
        return inner


__all__ = ["EmbeddingClient", "build_embedding_client"]
