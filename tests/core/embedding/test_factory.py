"""Contract tests for vo_rag.core.embedding.build_embedding_client.

Verifies:
- default (VORAG_EMBEDDING_CACHE_ENABLED unset or "1") → CachingEmbeddingClient
  wraps the raw EmbeddingClient
- VORAG_EMBEDDING_CACHE_ENABLED=0 → raw EmbeddingClient
- VORAG_EMBEDDING_CACHE_ENABLED != "1" (any non-"1" value, e.g. "false",
  "0", "no") → raw EmbeddingClient (env var is opt-OUT-by-default-disabled)
"""

from __future__ import annotations

import pytest

from vo_rag.core.embedding import EmbeddingClient, build_embedding_client
from vo_rag.core.embedding.cache import CachingEmbeddingClient


def test_build_embedding_client_default_is_caching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("VORAG_EMBEDDING_CACHE_ENABLED", raising=False)
    monkeypatch.setenv("VORAG_EMBEDDING_CACHE_DIR", str(tmp_path / "emb"))
    client = build_embedding_client("http://embed.local", timeout_s=5.0)
    assert isinstance(client, CachingEmbeddingClient)
    assert isinstance(client._inner, EmbeddingClient)


def test_build_embedding_client_cache_enabled_1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("VORAG_EMBEDDING_CACHE_ENABLED", "1")
    monkeypatch.setenv("VORAG_EMBEDDING_CACHE_DIR", str(tmp_path / "emb"))
    client = build_embedding_client("http://embed.local", timeout_s=5.0)
    assert isinstance(client, CachingEmbeddingClient)


def test_build_embedding_client_cache_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VORAG_EMBEDDING_CACHE_ENABLED", "0")
    client = build_embedding_client("http://embed.local", timeout_s=5.0)
    assert isinstance(client, EmbeddingClient)
    assert not isinstance(client, CachingEmbeddingClient)


def test_build_embedding_client_cache_disabled_non_one_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Anything not equal to "1" (after strip) opts out — this is intentional
    # so a misconfigured string ("false", "no", "off") safely disables.
    monkeypatch.setenv("VORAG_EMBEDDING_CACHE_ENABLED", "false")
    client = build_embedding_client("http://embed.local", timeout_s=5.0)
    assert isinstance(client, EmbeddingClient)
    assert not isinstance(client, CachingEmbeddingClient)
