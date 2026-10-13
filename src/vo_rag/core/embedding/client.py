"""httpx async client for the BGE-M3 embedding server.

- 30 MB body guard prevents RunPod load-balancer 413s.
- 502/503/504 retry with exponential backoff (4 attempts, 2s base) — these
  codes are emitted during worker boot / cycle windows.
- The API server builds its client through `build_embedding_client` in
  `embedding/__init__.py`, which wraps this client in the content-hash cache
  unless `VORAG_EMBEDDING_CACHE_ENABLED` is off. A rolling model deploy must
  bump the cache's model version so stale vectors are not served.
- Trailing `/embed` is stripped from the base URL so the env var can
  include OR omit it without bricking.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from collections.abc import Callable
from typing import Any

import httpx

from vo_rag.core.observability.metrics import record_embed_call

MAX_REQUEST_BODY_BYTES = 30_000_000
DEFAULT_EMBED_MAX_BATCH_TEXTS = 32
MAX_EMBED_BATCH_TEXTS = 32
DEFAULT_EMBED_MAX_LENGTH = 8192
EMBED_MAX_BATCH_TEXTS_ENV = "EMBED_MAX_BATCH_TEXTS"
_DEFAULT_RETRY_ATTEMPTS = 4
_DEFAULT_RETRY_BACKOFF_S = 2.0
#: 429 is the inference worker's admission backpressure, not an error: it has
#: its own, longer budget so a busy embedder slows an ingest instead of failing
#: the book (VOR-1427). Waits are jittered to half..full of the capped backoff
#: so parallel lanes stop retrying in lockstep; at the default backoff the
#: budget is ~5.5-11 min, which outlasts the saturation bursts a 10-lane wave
#: produced (the earlier 12-attempt, ~2 min budget lost books; VOR-1440).
OVERLOAD_RETRY_ATTEMPTS = 48
_OVERLOAD_MAX_WAIT_S = 15.0
_jitter = random.SystemRandom()


def _embed_max_batch_texts_from_env() -> int:
    raw = os.environ.get(EMBED_MAX_BATCH_TEXTS_ENV, str(DEFAULT_EMBED_MAX_BATCH_TEXTS))
    value = int(raw)
    if value < 1:
        raise ValueError(f"{EMBED_MAX_BATCH_TEXTS_ENV} must be >= 1, got {value}")
    return min(value, MAX_EMBED_BATCH_TEXTS)


def _empty_embedding_result(
    *,
    return_dense: bool,
    return_sparse: bool,
    return_colbert: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if return_dense:
        result["dense"] = []
    if return_sparse:
        result["sparse"] = []
    if return_colbert:
        result["colbert"] = []
    return result


def _merge_embedding_results(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for chunk in chunks:
        for key, value in chunk.items():
            if isinstance(value, list):
                existing = merged.setdefault(key, [])
                if isinstance(existing, list):
                    existing.extend(value)
                continue
            if key == "runtime_identity_mixed":
                merged[key] = bool(merged.get(key)) or bool(value)
            elif key not in merged:
                merged[key] = value
            elif key == "runtime_identity" and merged[key] != value:
                merged[key] = None
                merged["runtime_identity_mixed"] = True
    return merged


def _per_text_metadata(response: dict[str, Any], text_count: int) -> dict[str, Any]:
    """Normalize optional worker metadata without inventing missing values."""
    result = dict(response)
    raw_errors = response.get("embedding_metadata_errors", [])
    errors = (
        [error for error in raw_errors if isinstance(error, str)]
        if isinstance(raw_errors, list)
        else []
    )
    if raw_errors and not isinstance(raw_errors, list):
        errors.append("embedding_metadata_errors_invalid")

    identity = response.get("runtime_identity")
    if isinstance(identity, dict) and all(
        isinstance(key, str) and (value is None or isinstance(value, str))
        for key, value in identity.items()
    ):
        result["runtime_identity_per_text"] = [dict(identity) for _ in range(text_count)]
    else:
        result["runtime_identity_per_text"] = [None] * text_count
        if identity is not None:
            errors.append("runtime_identity_invalid")

    fields: tuple[tuple[str, str, Callable[[Any], bool]], ...] = (
        ("input_token_counts", "input_token_counts_per_text", lambda v: type(v) is int and v >= 0),
        ("input_truncated", "input_truncated_per_text", lambda v: type(v) is bool),
    )
    for field, item_key, accepted in fields:
        values = response.get(field)
        if values is None:
            result[item_key] = [None] * text_count
        elif (
            isinstance(values, list)
            and len(values) == text_count
            and all(accepted(value) for value in values)
        ):
            result[item_key] = list(values)
        else:
            result[item_key] = [None] * text_count
            errors.append(f"{field}_invalid_or_cardinality_mismatch")

    if errors:
        result["embedding_metadata_errors"] = sorted(set(errors))
    return result


def _summarize_runtime_identity(response: dict[str, Any]) -> dict[str, Any]:
    """Mark mixed per-request runtime metadata instead of keeping the first batch's value."""
    identities = response.get("runtime_identity_per_text")
    if not isinstance(identities, list):
        return response
    observed = {
        json.dumps(value, sort_keys=True) for value in identities if isinstance(value, dict)
    }
    if len(observed) > 1:
        response["runtime_identity"] = None
        response["runtime_identity_mixed"] = True
        response["runtime_identity_coverage"] = "mixed"
    elif len(observed) == 1 and all(isinstance(value, dict) for value in identities):
        response["runtime_identity"] = json.loads(next(iter(observed)))
        response["runtime_identity_coverage"] = "complete"
    elif len(observed) == 1:
        response["runtime_identity"] = None
        response["runtime_identity_coverage"] = "partial"
    else:
        response["runtime_identity"] = None
        response["runtime_identity_coverage"] = "none"
    return response


class EmbeddingClient:
    def __init__(
        self,
        url: str,
        timeout_s: float = 30.0,
        api_key: str | None = None,
        retry_attempts: int = _DEFAULT_RETRY_ATTEMPTS,
        retry_backoff_s: float = _DEFAULT_RETRY_BACKOFF_S,
        max_length: int = DEFAULT_EMBED_MAX_LENGTH,
    ) -> None:
        if retry_attempts < 1:
            raise ValueError(f"retry_attempts must be >= 1, got {retry_attempts}")
        if retry_backoff_s < 0:
            raise ValueError(f"retry_backoff_s must be >= 0, got {retry_backoff_s}")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")
        if max_length > DEFAULT_EMBED_MAX_LENGTH:
            raise ValueError(f"max_length must be <= {DEFAULT_EMBED_MAX_LENGTH}, got {max_length}")
        normalized_url = url.rstrip("/")
        if normalized_url.endswith("/embed"):
            normalized_url = normalized_url[: -len("/embed")]
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=normalized_url, timeout=timeout_s, headers=headers
        )
        self._retry_attempts = retry_attempts
        self._retry_backoff_s = retry_backoff_s
        self._max_length = max_length
        self._max_batch_texts = _embed_max_batch_texts_from_env()

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = False,
    ) -> dict[str, Any]:
        if not texts:
            return _empty_embedding_result(
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert=return_colbert,
            )

        if len(texts) <= self._max_batch_texts:
            response = await self._embed_adaptive(
                texts,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert=return_colbert,
            )
            return _summarize_runtime_identity(response)

        results: list[dict[str, Any]] = []
        for start in range(0, len(texts), self._max_batch_texts):
            batch = texts[start : start + self._max_batch_texts]
            results.append(
                await self._embed_adaptive(
                    batch,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert=return_colbert,
                )
            )
        return _summarize_runtime_identity(_merge_embedding_results(results))

    async def _embed_adaptive(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, Any]:
        try:
            response = await self._embed_one(
                texts,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert=return_colbert,
            )
            return _per_text_metadata(response, len(texts))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 413:
                raise
            if len(texts) == 1:
                raise RuntimeError("single text exceeds embed service limit") from exc
            midpoint = len(texts) // 2
            left = await self._embed_adaptive(
                texts[:midpoint],
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert=return_colbert,
            )
            right = await self._embed_adaptive(
                texts[midpoint:],
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert=return_colbert,
            )
            return _summarize_runtime_identity(_merge_embedding_results([left, right]))

    async def _embed_one(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, Any]:
        payload = {
            "texts": texts,
            "return_dense": return_dense,
            "return_sparse": return_sparse,
            "return_colbert": return_colbert,
            "max_length": self._max_length,
        }
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(body) >= MAX_REQUEST_BODY_BYTES:
            raise ValueError(
                "Embedding request payload exceeds the RunPod load-balancer 30 MB limit; "
                "chunk smaller before embedding."
            )

        overloads = [0]
        try:
            return await self._post_with_retries(body, overloads)
        finally:
            # One logical embed request and the 429s it
            # absorbed, onto the running ingest's envelope (no-op elsewhere).
            record_embed_call(retries_429=overloads[0])

    async def _post_with_retries(self, body: bytes, overload_count: list[int]) -> dict[str, Any]:
        """POST ``body`` with the retry budgets; ``overload_count[0]`` tracks 429s seen."""
        last_exc: Exception | None = None
        errors = overloads = 0
        while True:
            try:
                response = await self._client.post(
                    "/embed",
                    content=body,
                    headers={"Content-Type": "application/json"},
                )
                # Retry on transient gateway / overloaded responses from the
                # load-balancer (worker boot/cycle windows return 502/503/504).
                if response.status_code in (502, 503, 504):
                    response.raise_for_status()
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                return result
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                status = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                if status == 413:
                    raise
                if status == 429:
                    overloads += 1
                    overload_count[0] = overloads
                    if overloads >= OVERLOAD_RETRY_ATTEMPTS:
                        break
                    ceiling = min(
                        self._retry_backoff_s * (2 ** (overloads - 1)), _OVERLOAD_MAX_WAIT_S
                    )
                    await asyncio.sleep(_jitter.uniform(ceiling / 2, ceiling))
                    continue
                errors += 1
                if errors >= self._retry_attempts:
                    break
                await asyncio.sleep(self._retry_backoff_s * (2 ** (errors - 1)))
        # Loop exhausted without returning. `last_exc` is the last exception
        # captured above; if somehow None, surface a sentinel so callers see
        # a real error rather than a silent return.
        if last_exc is None:
            raise RuntimeError("embedding request exited retry loop without result")
        raise last_exc

    async def aclose(self) -> None:
        await self._client.aclose()
