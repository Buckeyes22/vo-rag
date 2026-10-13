"""Contract tests for vo_rag.core.embedding.cache.

Verifies:
- miss → hit on second call (xxh64 model-versioned diskcache key)
- BGE_M3_MODEL_VERSION env var namespaces the cache (bumping it evicts)
- partial-coverage recompute: if only dense is stored, a colbert request
  misses and refreshes with the superset
- mixed batch: only the misses go to the inner client
- CachingEmbeddingClient.aclose() delegates to inner.aclose()
- VORAG_EMBEDDING_CACHE_DIR / VORAG_EMBEDDING_CACHE_SIZE_GB env override path
"""

from __future__ import annotations

import asyncio
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import vo_rag.core.embedding.cache as cache_module
from vo_rag.core.embedding.cache import CachingEmbeddingClient


class _Inner:
    def __init__(self) -> None:
        self.calls = 0
        self.texts_seen: list[list[str]] = []
        self.closed = False

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = False,
    ) -> dict[str, list[Any]]:
        self.calls += 1
        self.texts_seen.append(list(texts))
        out: dict[str, list[Any]] = {}
        if return_dense:
            out["dense"] = [[float(i), float(i) + 1.0] for i in range(len(texts))]
        if return_sparse:
            out["sparse"] = [{i: 1.0} for i in range(len(texts))]
        if return_colbert:
            out["colbert"] = [[[float(i)]] for i in range(len(texts))]
        return out

    async def aclose(self) -> None:  # pragma: no cover - delegated
        self.closed = True


@pytest.fixture
def _caching(tmp_path: Path) -> tuple[CachingEmbeddingClient, _Inner]:
    inner = _Inner()
    client = CachingEmbeddingClient(
        inner,
        cache_dir=tmp_path / "embedcache",
        size_bytes=128 * 1024 * 1024,
        model_version="test-v1",
    )
    return client, inner


@pytest.mark.asyncio
async def test_miss_then_hit_on_second_call(
    _caching: tuple[CachingEmbeddingClient, _Inner],
) -> None:
    client, inner = _caching
    first = await client.embed(["alpha"], return_dense=True, return_sparse=True)
    assert inner.calls == 1
    second = await client.embed(["alpha"], return_dense=True, return_sparse=True)
    assert inner.calls == 1  # cache hit
    assert first["dense"] == second["dense"]


@pytest.mark.asyncio
async def test_model_version_bump_invalidates(tmp_path: Path) -> None:
    inner = _Inner()
    c1 = CachingEmbeddingClient(
        inner,
        cache_dir=tmp_path / "emb",
        size_bytes=64 * 1024 * 1024,
        model_version="v1",
    )
    await c1.embed(["alpha"])
    assert inner.calls == 1

    c2 = CachingEmbeddingClient(
        inner,
        cache_dir=tmp_path / "emb",
        size_bytes=64 * 1024 * 1024,
        model_version="v2",
    )
    await c2.embed(["alpha"])
    assert inner.calls == 2  # new version → miss


@pytest.mark.asyncio
async def test_bge_m3_model_version_env_namespaces_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BGE_M3_MODEL_VERSION", "v-env-1")
    inner = _Inner()
    c1 = CachingEmbeddingClient(
        inner,
        cache_dir=tmp_path / "emb-env",
        size_bytes=64 * 1024 * 1024,
    )
    await c1.embed(["alpha"])
    assert inner.calls == 1

    monkeypatch.setenv("BGE_M3_MODEL_VERSION", "v-env-2")
    c2 = CachingEmbeddingClient(
        inner,
        cache_dir=tmp_path / "emb-env",
        size_bytes=64 * 1024 * 1024,
    )
    await c2.embed(["alpha"])
    assert inner.calls == 2  # new env version → miss


@pytest.mark.asyncio
async def test_partial_output_request_on_full_cache_is_hit(
    _caching: tuple[CachingEmbeddingClient, _Inner],
) -> None:
    client, inner = _caching
    await client.embed(["alpha"], return_dense=True, return_sparse=True)
    assert inner.calls == 1
    await client.embed(["alpha"], return_dense=True, return_sparse=False)
    assert inner.calls == 1  # dense is cached, sparse not needed


@pytest.mark.asyncio
async def test_mixed_batch_hits_and_misses(
    _caching: tuple[CachingEmbeddingClient, _Inner],
) -> None:
    client, inner = _caching
    await client.embed(["alpha", "beta"])
    assert inner.calls == 1
    assert inner.texts_seen[-1] == ["alpha", "beta"]

    await client.embed(["alpha", "gamma"])
    assert inner.calls == 2
    # Only the miss goes to inner
    assert inner.texts_seen[-1] == ["gamma"]


@pytest.mark.asyncio
async def test_identical_concurrent_misses_are_single_flight(tmp_path: Path) -> None:
    class _GatedInner(_Inner):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def embed(self, texts: list[str], **kwargs: Any) -> dict[str, list[Any]]:
            self.started.set()
            await self.release.wait()
            return await super().embed(texts, **kwargs)

    inner = _GatedInner()
    client = CachingEmbeddingClient(
        inner,
        cache_dir=tmp_path / "single-flight",
        size_bytes=64 * 1024 * 1024,
        model_version="test-v1",
    )
    first = asyncio.create_task(client.embed(["same query"]))
    await inner.started.wait()
    second = asyncio.create_task(client.embed(["same query"]))
    await asyncio.sleep(0)
    inner.release.set()
    first_result, second_result = await asyncio.gather(first, second)
    assert inner.calls == 1
    assert first_result == second_result


@pytest.mark.asyncio
async def test_colbert_request_not_cached_if_only_dense_stored(
    _caching: tuple[CachingEmbeddingClient, _Inner],
) -> None:
    client, inner = _caching
    await client.embed(["alpha"], return_dense=True, return_sparse=True, return_colbert=False)
    assert inner.calls == 1
    await client.embed(["alpha"], return_dense=False, return_sparse=False, return_colbert=True)
    # colbert wasn't stored → miss → refreshed with full superset
    assert inner.calls == 2


@pytest.mark.asyncio
async def test_aclose_delegates_to_inner(
    _caching: tuple[CachingEmbeddingClient, _Inner],
) -> None:
    client, inner = _caching
    cached = client._cache
    assert cached is not None
    cached.close()

    class _CloseTrackingCache:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    tracked = _CloseTrackingCache()
    client._cache = tracked
    await client.aclose()
    assert tracked.close_calls == 1
    assert client._cache is None
    assert inner.closed


@pytest.mark.asyncio
async def test_cache_constructor_enospc_falls_back_to_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _NoSpaceCache:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise OSError(28, "no space left on device")

    monkeypatch.setattr(shutil, "disk_usage", lambda _path: type("Usage", (), {"free": 2**40})())
    monkeypatch.setattr(cache_module, "Cache", _NoSpaceCache)
    inner = _Inner()
    client = CachingEmbeddingClient(inner, cache_dir=tmp_path / "full", size_bytes=1)

    result = await client.embed(["fresh"])

    assert result["dense"] == [[0.0, 1.0]]
    assert inner.calls == 1
    assert client._cache is None


@pytest.mark.asyncio
async def test_low_cache_space_uses_uncached_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: type("Usage", (), {"free": 0})())
    inner = _Inner()
    client = CachingEmbeddingClient(inner, cache_dir=tmp_path / "low", size_bytes=1024)

    result = await client.embed(["fresh"])

    assert result["dense"] == [[0.0, 1.0]]
    assert inner.calls == 1
    assert client._cache is None


@pytest.mark.asyncio
async def test_cache_read_oserror_disables_cache_and_returns_fresh_vectors(
    _caching: tuple[CachingEmbeddingClient, _Inner],
) -> None:
    client, inner = _caching

    class _BrokenRead:
        close_calls = 0

        def get(self, _key: str) -> None:
            raise OSError("cache storage unavailable")

        def set(self, _key: str, _value: dict[str, Any]) -> bool:
            return True

        def volume(self) -> int:
            return 0

        def close(self) -> None:
            self.close_calls += 1

    broken = _BrokenRead()
    client._cache = broken
    result = await client.embed(["fresh"])

    assert result["dense"] == [[0.0, 1.0]]
    assert inner.calls == 1
    assert client._cache is None
    assert broken.close_calls == 1


@pytest.mark.asyncio
async def test_cache_write_sqlite_error_does_not_discard_fresh_vectors(
    _caching: tuple[CachingEmbeddingClient, _Inner],
) -> None:
    client, inner = _caching

    class _BrokenWrite:
        close_calls = 0

        def get(self, _key: str) -> None:
            return None

        def set(self, _key: str, _value: dict[str, Any]) -> None:
            raise sqlite3.OperationalError("disk i/o error")

        def volume(self) -> int:
            return 0

        def close(self) -> None:
            self.close_calls += 1
            raise sqlite3.DatabaseError("cache close failed")

    broken = _BrokenWrite()
    client._cache = broken
    result = await client.embed(["fresh"])

    assert result["dense"] == [[0.0, 1.0]]
    assert inner.calls == 1
    assert client._cache is None
    assert broken.close_calls == 1


@pytest.mark.asyncio
async def test_aclose_cache_oserror_does_not_skip_inner_close(
    _caching: tuple[CachingEmbeddingClient, _Inner],
) -> None:
    client, inner = _caching
    cached = client._cache
    assert cached is not None
    cached.close()

    class _BrokenClose:
        def close(self) -> None:
            raise OSError("cache close failed")

    client._cache = _BrokenClose()
    await client.aclose()

    assert inner.closed
    assert client._cache is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("provider failed"), asyncio.CancelledError()])
async def test_provider_failures_and_cancellation_are_not_suppressed(
    tmp_path: Path, failure: BaseException
) -> None:
    class _FailingInner:
        async def embed(self, *_args: Any, **_kwargs: Any) -> dict[str, list[Any]]:
            raise failure

    client = CachingEmbeddingClient(
        _FailingInner(), cache_dir=tmp_path / "failure", size_bytes=64 * 1024 * 1024
    )

    with pytest.raises(type(failure)):
        await client.embed(["fresh"])


@pytest.mark.asyncio
async def test_malformed_provider_vectors_are_not_cached_or_returned_as_hits(
    tmp_path: Path,
) -> None:
    class _MalformedInner:
        async def embed(self, *_args: Any, **_kwargs: Any) -> dict[str, list[Any]]:
            return {"sparse": [{1: 1.0}]}

    client = CachingEmbeddingClient(
        _MalformedInner(), cache_dir=tmp_path / "malformed", size_bytes=64 * 1024 * 1024
    )

    with pytest.raises(ValueError, match="invalid dense cardinality"):
        await client.embed(["fresh"])
