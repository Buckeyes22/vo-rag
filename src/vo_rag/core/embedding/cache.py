"""Content-hash embedding cache (xxh64 model-versioned diskcache).

Wraps any object that exposes
`embed(texts, *, return_dense, return_sparse, return_colbert) -> dict` with
a diskcache-backed layer keyed on `(xxh64(text), model_version, output_tuple)`.

Cache hit path:
- Zero embedding compute.
- Returns the subset of output types the caller asked for, reconstructed
  from the stored superset record.

Cache miss path:
- Delegates to the inner client.
- Writes the resulting dense/sparse/colbert vectors back under the
  superset key so partial-output hits are free later.

Invalidation: bump `model_version` (via `BGE_M3_MODEL_VERSION` env var) to
evict the entire namespace without flushing the directory.

Configuration:
- Env vars are `VORAG_EMBEDDING_CACHE_DIR` and
  `VORAG_EMBEDDING_CACHE_SIZE_GB`.
- Cache dir default is vo-rag-relative (`/var/lib/vo_rag/cache/embeddings`).
- Default model version stays `bge-m3@unversioned` (matches the
  worker-side default and keeps the namespace stable until bumped).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
from contextlib import suppress
from pathlib import Path
from typing import Any

import structlog
import xxhash
from diskcache import Cache  # type: ignore[import-untyped]

from vo_rag.core.observability.metrics import cache_requests_total, cache_size_bytes

_log = structlog.get_logger(__name__)
_CACHE_RESERVE_MIN_BYTES = 256 * 1024**2
_CACHE_RESERVE_FRACTION = 0.25
_CACHE_FAILURES = (OSError, sqlite3.Error)


class CachingEmbeddingClient:
    def __init__(
        self,
        inner: Any,
        *,
        cache_dir: str | Path | None = None,
        size_bytes: int | None = None,
        model_version: str | None = None,
    ) -> None:
        self._inner = inner
        self._model_version = (
            model_version or os.environ.get("BGE_M3_MODEL_VERSION") or "bge-m3@unversioned"
        )
        resolved_dir = str(
            cache_dir
            or os.environ.get("VORAG_EMBEDDING_CACHE_DIR", "/var/lib/vo_rag/cache/embeddings")
        )
        resolved_size = (
            size_bytes
            if size_bytes is not None
            else int(os.environ.get("VORAG_EMBEDDING_CACHE_SIZE_GB", "20")) * 1024**3
        )
        self._cache: Cache | None = None
        if resolved_size <= 0:
            self._report_cache_failure("configuration", ValueError("cache size limit disabled"))
        else:
            try:
                Path(resolved_dir).mkdir(parents=True, exist_ok=True)
                free_bytes = shutil.disk_usage(resolved_dir).free
                reserve_bytes = max(
                    _CACHE_RESERVE_MIN_BYTES, int(resolved_size * _CACHE_RESERVE_FRACTION)
                )
                if free_bytes < resolved_size + reserve_bytes:
                    self._report_cache_failure(
                        "capacity", OSError("insufficient free space for cache and reserve")
                    )
                else:
                    self._cache = Cache(resolved_dir, size_limit=resolved_size)
            except _CACHE_FAILURES as exc:
                self._report_cache_failure("initialize", exc)
        self._inflight: dict[
            tuple[tuple[str, ...], bool, bool, bool], asyncio.Task[dict[str, list[Any]]]
        ] = {}

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = False,
    ) -> dict[str, list[Any]]:
        # Coalesce identical concurrent cache misses.  The content hashes include
        # the model-version namespace, and the output flags distinguish requests
        # whose cache-coverage requirements differ.
        flight_key = (
            tuple(self._make_key(text) for text in texts),
            return_dense,
            return_sparse,
            return_colbert,
        )
        task = self._inflight.get(flight_key)
        if task is None:
            task = asyncio.create_task(
                self._embed_impl(
                    texts,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert=return_colbert,
                )
            )
            self._inflight[flight_key] = task

            def _clear(done: asyncio.Task[dict[str, list[Any]]]) -> None:
                if self._inflight.get(flight_key) is done:
                    self._inflight.pop(flight_key, None)

            task.add_done_callback(_clear)
        return await asyncio.shield(task)

    async def _embed_impl(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, list[Any]]:
        out: dict[str, Any] = {}
        if return_dense:
            out["dense"] = [None] * len(texts)
        if return_sparse:
            out["sparse"] = [None] * len(texts)
        if return_colbert:
            out["colbert"] = [None] * len(texts)
        # Vector cache hits have no response receipt, so their runtime and
        # tokenizer facts remain unknown. Metadata from a fresh worker response
        # is aligned back to the original input positions below.
        out["runtime_identity_per_text"] = [None] * len(texts)
        out["input_token_counts_per_text"] = [None] * len(texts)
        out["input_truncated_per_text"] = [None] * len(texts)

        miss_indices: list[int] = []
        miss_texts: list[str] = []
        cache = self._cache
        if cache is None:
            miss_indices.extend(range(len(texts)))
            miss_texts.extend(texts)
            cache_requests_total.labels(cache="embedding", result="skip").inc(len(texts))
        else:
            for i, text in enumerate(texts):
                try:
                    entry = cache.get(self._make_key(text))
                except _CACHE_FAILURES as exc:
                    self._disable_cache("read", exc)
                    remaining = range(i, len(texts))
                    miss_indices.extend(remaining)
                    miss_texts.extend(texts[index] for index in remaining)
                    cache_requests_total.labels(cache="embedding", result="skip").inc(
                        len(texts) - i
                    )
                    break
                if entry is None:
                    miss_indices.append(i)
                    miss_texts.append(text)
                    continue
                if self._satisfies(entry, return_dense, return_sparse, return_colbert):
                    if return_dense:
                        out["dense"][i] = entry["dense"]
                    if return_sparse:
                        out["sparse"][i] = entry["sparse"]
                    if return_colbert:
                        out["colbert"][i] = entry["colbert"]
                    cache_requests_total.labels(cache="embedding", result="hit").inc()
                else:
                    # Partial coverage — treat as a miss so we refresh the entry
                    # with the superset of output types.
                    miss_indices.append(i)
                    miss_texts.append(text)

        if miss_texts:
            cache_requests_total.labels(cache="embedding", result="miss").inc(len(miss_texts))
            fresh = await self._inner.embed(
                miss_texts,
                return_dense=True,
                return_sparse=True,
                return_colbert=return_colbert,
            )
            self._validate_fresh_vectors(fresh, len(miss_texts), return_colbert)
            runtime_values = fresh.get("runtime_identity_per_text")
            token_values = fresh.get("input_token_counts_per_text")
            truncated_values = fresh.get("input_truncated_per_text")
            if not isinstance(runtime_values, list) or len(runtime_values) != len(miss_texts):
                runtime_values = [fresh.get("runtime_identity")] * len(miss_texts)
            if not isinstance(token_values, list) or len(token_values) != len(miss_texts):
                token_values = [None] * len(miss_texts)
            if not isinstance(truncated_values, list) or len(truncated_values) != len(miss_texts):
                truncated_values = [None] * len(miss_texts)
            out["embedding_metadata_errors"] = list(
                fresh.get("embedding_metadata_errors", [])
                if isinstance(fresh.get("embedding_metadata_errors", []), list)
                else ["embedding_metadata_errors_invalid"]
            )
            for pos, src_idx in enumerate(miss_indices):
                entry = {
                    "dense": fresh.get("dense", [None] * len(miss_texts))[pos],
                    "sparse": fresh.get("sparse", [None] * len(miss_texts))[pos],
                    "colbert": (
                        fresh.get("colbert", [None] * len(miss_texts))[pos]
                        if return_colbert
                        else None
                    ),
                }
                cache = self._cache
                if cache is not None:
                    try:
                        cache.set(self._make_key(texts[src_idx]), entry)
                    except _CACHE_FAILURES as exc:
                        self._disable_cache("write", exc)
                if return_dense:
                    out["dense"][src_idx] = entry["dense"]
                if return_sparse:
                    out["sparse"][src_idx] = entry["sparse"]
                if return_colbert:
                    out["colbert"][src_idx] = entry["colbert"]
                out["runtime_identity_per_text"][src_idx] = runtime_values[pos]
                out["input_token_counts_per_text"][src_idx] = token_values[pos]
                out["input_truncated_per_text"][src_idx] = truncated_values[pos]

        identities = {
            repr(sorted(value.items()))
            for value in out["runtime_identity_per_text"]
            if isinstance(value, dict)
        }
        out["runtime_identity_mixed"] = len(identities) > 1
        out["runtime_identity"] = (
            next(value for value in out["runtime_identity_per_text"] if isinstance(value, dict))
            if len(identities) == 1
            and all(isinstance(v, dict) for v in out["runtime_identity_per_text"])
            else None
        )

        cache = self._cache
        if cache is not None:
            try:
                cache_size_bytes.labels(cache="embedding").set(cache.volume())
            except _CACHE_FAILURES as exc:
                self._disable_cache("size", exc)

        return out

    def _make_key(self, text: str) -> str:
        content_hash = xxhash.xxh64(text.encode("utf-8")).hexdigest()
        return f"{self._model_version}|{content_hash}"

    @staticmethod
    def _validate_fresh_vectors(
        response: dict[str, Any], expected_count: int, return_colbert: bool
    ) -> None:
        """Reject malformed provider vector arrays before they can be cached."""
        required = ("dense", "sparse", "colbert") if return_colbert else ("dense", "sparse")
        for name in required:
            vectors = response.get(name)
            if not isinstance(vectors, list) or len(vectors) != expected_count:
                raise ValueError(f"embedding provider returned invalid {name} cardinality")

    def _disable_cache(self, operation: str, exc: OSError | sqlite3.Error) -> None:
        """Disable an unhealthy cache while keeping fresh embeddings usable."""
        if self._cache is None:
            return
        cache = self._cache
        self._cache = None
        with suppress(*_CACHE_FAILURES):
            cache.close()
        cache_size_bytes.labels(cache="embedding").set(0)
        self._report_cache_failure(operation, exc)

    @staticmethod
    def _report_cache_failure(operation: str, exc: BaseException) -> None:
        """Record a cache fault without logging paths or exception contents."""
        _log.warning(
            "embedding_cache_unavailable",
            operation=operation,
            error_type=type(exc).__name__,
            fallback="uncached_embeddings",
        )

    async def aclose(self) -> None:
        cache = self._cache
        self._cache = None
        if cache is not None:
            try:
                cache.close()
            except _CACHE_FAILURES as exc:
                self._report_cache_failure("close", exc)
        close = getattr(self._inner, "aclose", None)
        if callable(close):
            await close()

    @staticmethod
    def _satisfies(
        entry: dict[str, Any],
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> bool:
        if return_dense and entry.get("dense") is None:
            return False
        if return_sparse and entry.get("sparse") is None:
            return False
        return not (return_colbert and entry.get("colbert") is None)


__all__ = ["CachingEmbeddingClient"]
