"""DSI query-time router — embed + search DSI collection + allocate budgets.

Purely additive to the existing retrieval stack.  Does not modify
`vo_rag.core.dsi.routing` or `vo_rag.core.retrieval.search`.
"""

from __future__ import annotations

from typing import Any, Protocol

from qdrant_client.models import Filter, ScoredPoint

from vo_rag.core.dsi.builder import _DSI_COLLECTION_DEFAULT
from vo_rag.core.dsi.routing import RoutingDecision, allocate_budgets, normalize_dsi_scores
from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse


class _EmbedClientProtocol(Protocol):
    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, Any]: ...


class _QdrantStoreProtocol(Protocol):
    async def search_hybrid(
        self,
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        *,
        top_k: int = 20,
        filters: Filter | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]: ...


class DsiRouter:
    """Query-time DSI router.

    Responsibilities:
      1. Embed the query once (dense / sparse / ColBERT / BM25).
      2. Search the DSI collection via the injected store's
         ``search_hybrid`` seam.
      3. Build a ``book_id -> score`` map from the returned payloads.
      4. Normalize scores to confidences and allocate per-book chunk
         budgets via the canonical ``allocate_budgets`` function.
    """

    def __init__(
        self,
        store: _QdrantStoreProtocol,
        embedding_client: _EmbedClientProtocol,
        dsi_collection: str = _DSI_COLLECTION_DEFAULT,
    ) -> None:
        self.store = store
        self.embed = embedding_client
        self.dsi_collection = dsi_collection

    async def route(
        self,
        query: str,
        *,
        total_budget_chunks: int = 20,
        top_k: int = 8,
        fallback_share: float = 0.05,
    ) -> RoutingDecision:
        """Return a ``RoutingDecision`` for *query*.

        The decision contains per-book primary/fallback chunk budgets
        derived from DSI facet scores.
        """
        emb = await self.embed.embed(
            [query], return_dense=True, return_sparse=True, return_colbert=True
        )
        q_dense = emb["dense"][0]
        q_sparse = dict(zip(emb["sparse"][0]["indices"], emb["sparse"][0]["values"], strict=True))
        q_colbert = emb["colbert"][0]
        q_bm25 = tokens_to_bm25_sparse(tokenize(query))

        scored_points = await self.store.search_hybrid(
            collection=self.dsi_collection,
            query_dense=q_dense,
            query_sparse=q_sparse,
            query_bm25=q_bm25,
            query_colbert=q_colbert,
            top_k=top_k,
        )

        scores: dict[str, float] = {}
        for point in scored_points:
            payload = point.payload or {}
            book_id = payload.get("book_id")
            if isinstance(book_id, str):
                scores[book_id] = max(scores.get(book_id, float("-inf")), point.score)

        confidences = normalize_dsi_scores(scores)
        return allocate_budgets(
            confidences=confidences,
            total_budget_chunks=total_budget_chunks,
            top_k=top_k,
            fallback_share=fallback_share,
        )
