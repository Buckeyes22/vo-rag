"""httpx async client for the remote /rerank endpoint (VOR-8).

Mirrors ``EmbeddingClient``: URL normalization (strip trailing ``/rerank``),
``httpx.AsyncClient(base_url=..., timeout=..., headers=...)``, retry on
HTTP 502/503/504 and transport errors with exponential backoff.
"""

from __future__ import annotations

import asyncio
import math

import httpx

from vo_rag.core.reranking.render import _render_candidate
from vo_rag.core.reranking.schemas import RerankCandidate, RerankRequest, RerankResponse
from vo_rag.core.retrieval.search import SearchResult

_DEFAULT_RETRY_ATTEMPTS = 4
_DEFAULT_RETRY_BACKOFF_S = 2.0


class RerankerClient:
    def __init__(
        self,
        url: str,
        *,
        timeout_s: float = 30.0,
        api_key: str = "",
    ) -> None:
        normalized_url = url.rstrip("/")
        if normalized_url.endswith("/rerank"):
            normalized_url = normalized_url[: -len("/rerank")]
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=normalized_url, timeout=timeout_s, headers=headers
        )
        self._retry_attempts = _DEFAULT_RETRY_ATTEMPTS
        self._retry_backoff_s = _DEFAULT_RETRY_BACKOFF_S

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 20,
    ) -> list[SearchResult]:
        if not candidates:
            return []
        if len(query) > 4000:
            raise ValueError("query exceeds 4000-character retrieval budget")
        if len({candidate.id for candidate in candidates}) != len(candidates):
            raise ValueError("rerank candidate ids must be unique")
        # Send only rendered text: the full payload doubles source text on the
        # wire and is not consumed by the server. Restore payloads locally.
        id_to_candidate = {c.id: c for c in candidates}
        rows = [self._bounded_candidate(query, c) for c in candidates]
        batches: list[list[RerankCandidate]] = []
        current: list[RerankCandidate] = []
        for row in rows:
            trial = current + [row]
            body = RerankRequest(query=query, top_k=len(trial), candidates=trial).model_dump_json()
            if current and (len(trial) > 32 or len(body.encode("utf-8")) > 48_000):
                batches.append(current)
                current = []
            current.append(row)
        if current:
            batches.append(current)
        results: list[SearchResult] = []
        for batch in batches:
            parsed = await self._request_batch(query, batch)
            for row in parsed.results:
                orig = id_to_candidate.get(row.id)
                if orig is not None:
                    results.append(
                        SearchResult(
                            id=orig.id,
                            score=row.score or 0.0,
                            collection=orig.collection,
                            payload=orig.payload,
                        )
                    )
        # Cross-encoder scores are comparable across batches, unlike ranks.
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    @staticmethod
    def _bounded_candidate(query: str, candidate: SearchResult) -> RerankCandidate:
        text = _render_candidate(candidate)[:8192]
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            row = RerankCandidate(id=candidate.id, text=text[:middle])
            body = RerankRequest(query=query, top_k=1, candidates=[row]).model_dump_json()
            if len(body.encode("utf-8")) <= 48_000:
                low = middle
            else:
                high = middle - 1
        return RerankCandidate(id=candidate.id, text=text[:low])

    async def _request_batch(self, query: str, batch: list[RerankCandidate]) -> RerankResponse:
        body = (
            RerankRequest(query=query, top_k=len(batch), candidates=batch)
            .model_dump_json()
            .encode()
        )
        if len(body) > 48_000:
            raise ValueError("query/candidate exceeds rerank request byte budget")
        last_exc: Exception | None = None
        for attempt in range(self._retry_attempts):
            try:
                response = await self._client.post(
                    "/rerank", content=body, headers={"Content-Type": "application/json"}
                )
                if response.status_code == 413 and len(batch) > 1:
                    # A deployment can advertise a smaller bound. Split once
                    # per level without retrying an unchanged oversized body.
                    middle = len(batch) // 2
                    left = await self._request_batch(query, batch[:middle])
                    right = await self._request_batch(query, batch[middle:])
                    return RerankResponse(results=left.results + right.results)
                response.raise_for_status()
                parsed = RerankResponse.model_validate(response.json())
                actual_ids = [row.id for row in parsed.results]
                expected_ids = {row.id for row in batch}
                if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
                    raise ValueError(
                        "reranker returned missing, duplicate, or foreign candidate ids"
                    )
                if any(row.score is None or not math.isfinite(row.score) for row in parsed.results):
                    raise ValueError("reranker returned missing or non-finite scores")
                return parsed
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in (
                    429,
                    502,
                    503,
                    504,
                ):
                    raise
                if attempt == self._retry_attempts - 1:
                    break
                await asyncio.sleep(self._retry_backoff_s * (2**attempt))
        if last_exc is None:
            raise RuntimeError("rerank request exited retry loop without result")
        raise last_exc

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["RerankerClient"]
