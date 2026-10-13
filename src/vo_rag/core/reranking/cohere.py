"""Cohere Rerank v3.5 HTTP client."""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from typing import Any, cast

import httpx

from vo_rag.core.reranking.render import _render_candidate
from vo_rag.core.retrieval.search import SearchResult

_DEFAULT_ENDPOINT = "https://api.cohere.com/v2/rerank"
_DEFAULT_MODEL = "rerank-v3.5"
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_MAX_CHUNKS = 1000
_DEFAULT_MAX_RETRIES = 3
_RETRY_DELAYS_S = (0.5, 1.0, 2.0)


class CohereReranker:
    """Async wrapper around Cohere's Rerank v3.5 API.

    The constructor reads ``COHERE_API_KEY`` by default and creates an
    ``httpx.AsyncClient`` without issuing network traffic. ``rerank`` posts the
    rendered search candidates to Cohere and maps result indexes back onto the
    original ``SearchResult`` instances with updated scores.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = _DEFAULT_ENDPOINT,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_chunks_per_query: int = _DEFAULT_MAX_CHUNKS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_api_key = api_key if api_key is not None else os.environ.get("COHERE_API_KEY", "")
        if not resolved_api_key:
            raise RuntimeError("COHERE_API_KEY is required for the Cohere reranker backend.")

        self.endpoint = endpoint
        self.model = model
        self.timeout_s = timeout_s
        self.max_chunks_per_query = max_chunks_per_query
        self.max_retries = max_retries
        self._client = http_client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = http_client is None
        self._headers = {
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 20,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        truncated = candidates[: self.max_chunks_per_query]
        top_n = min(top_k, len(truncated))
        body: dict[str, object] = {
            "model": self.model,
            "query": query,
            "documents": [_render_candidate(candidate) for candidate in truncated],
            "top_n": top_n,
        }

        raw = await self._post_with_retries(body)
        results = cast(list[dict[str, Any]], raw.get("results", []))
        ranked: list[SearchResult] = []
        for item in results:
            index = int(cast(int, item["index"]))
            if index < 0 or index >= len(truncated):
                continue
            score = float(cast(float, item["relevance_score"]))
            ranked.append(replace(truncated[index], score=score))
        return ranked

    async def _post_with_retries(self, body: dict[str, object]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    self.endpoint,
                    json=body,
                    headers=self._headers,
                    timeout=self.timeout_s,
                )
                if (response.status_code == 429 or response.status_code >= 500) and (
                    attempt < self.max_retries
                ):
                    await asyncio.sleep(_retry_delay(attempt))
                    continue
                response.raise_for_status()
                return cast(dict[str, Any], response.json())
            except (httpx.HTTPStatusError, httpx.TransportError, KeyError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(_retry_delay(attempt))

        if last_exc is None:
            raise RuntimeError("cohere rerank request exited retry loop without result")
        raise last_exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _retry_delay(attempt: int) -> float:
    return _RETRY_DELAYS_S[attempt] if attempt < len(_RETRY_DELAYS_S) else _RETRY_DELAYS_S[-1]


__all__ = ["CohereReranker"]
