"""Tests for ``DsiRouter`` — query-time DSI routing path.

Purely additive; does not exercise ``HybridRetriever`` or the raw
chunk retrieval path (those live in ``tests/core/retrieval/test_search.py``).
"""

from __future__ import annotations

from typing import Any

import pytest
from qdrant_client.models import ScoredPoint

from vo_rag.core.dsi.router import DsiRouter
from vo_rag.core.dsi.routing import RoutingDecision

# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class FakeEmbedClient:
    """Deterministic 4-vector response."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "texts": list(texts),
                "return_dense": return_dense,
                "return_sparse": return_sparse,
                "return_colbert": return_colbert,
            }
        )
        return {
            "dense": [[0.1, 0.2, 0.3]],
            "sparse": [{"indices": [1, 2, 3], "values": [0.4, 0.5, 0.6]}],
            "colbert": [[[0.7, 0.8]]],
        }


class FakeDsiStore:
    """Records ``search_hybrid`` calls and replays canned results."""

    def __init__(self, next_results: list[ScoredPoint] | None = None) -> None:
        self.search_hybrid_calls: list[dict[str, Any]] = []
        self.next_results = next_results or []

    async def search_hybrid(
        self,
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        *,
        top_k: int = 20,
        filters: Any | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]:
        self.search_hybrid_calls.append(
            {
                "collection": collection,
                "query_dense": list(query_dense),
                "query_sparse": dict(query_sparse),
                "query_bm25": dict(query_bm25),
                "query_colbert": [list(v) for v in query_colbert],
                "top_k": top_k,
                "filters": filters,
                "prefetch_depth": prefetch_depth,
            }
        )
        return list(self.next_results)


# ----------------------------------------------------------------------
# DsiRouter.route
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dsi_router_embeds_query_and_searches_dsi_collection() -> None:
    store = FakeDsiStore()
    embed = FakeEmbedClient()
    router = DsiRouter(store=store, embedding_client=embed)

    await router.route("trail braking")

    assert len(embed.calls) == 1
    assert embed.calls[0]["texts"] == ["trail braking"]

    assert len(store.search_hybrid_calls) == 1
    call = store.search_hybrid_calls[0]
    assert call["collection"] == "vo_rag_books_dsi_v1"
    assert call["top_k"] == 8
    assert call["query_dense"] == [0.1, 0.2, 0.3]
    assert call["query_bm25"]  # non-empty — query was tokenized


@pytest.mark.asyncio
async def test_dsi_router_produces_routing_decision_with_sane_budgets() -> None:
    store = FakeDsiStore(
        next_results=[
            ScoredPoint(
                id="dsi-a",
                score=0.9,
                payload={"book_id": "book-a"},
                vector=None,
                version=0,
                shard_key=None,
            ),
            ScoredPoint(
                id="dsi-b",
                score=0.7,
                payload={"book_id": "book-b"},
                vector=None,
                version=0,
                shard_key=None,
            ),
            ScoredPoint(
                id="dsi-c",
                score=0.5,
                payload={"book_id": "book-c"},
                vector=None,
                version=0,
                shard_key=None,
            ),
        ]
    )
    embed = FakeEmbedClient()
    router = DsiRouter(store=store, embedding_client=embed)

    decision = await router.route(
        "test query",
        total_budget_chunks=20,
        top_k=2,
        fallback_share=0.05,
    )

    assert isinstance(decision, RoutingDecision)
    assert decision.total_budget_chunks == 20
    assert decision.top_k == 2

    routes_by_id = {r.book_id: r for r in decision.book_routes}
    assert "book-a" in routes_by_id
    assert "book-b" in routes_by_id
    assert "book-c" in routes_by_id

    # All books should have a non-negative budget
    for route in decision.book_routes:
        assert route.total_budget >= 0
        assert route.total_budget == route.primary_budget + route.fallback_budget

    # Top-k books get primary budget; non-top-k gets fallback
    assert routes_by_id["book-a"].primary_budget > 0
    assert routes_by_id["book-b"].primary_budget > 0

    # Budgets should sum close to total_budget_chunks.  The canonical
    # allocate_budgets gives fallback budget to top-k books as well as
    # non-top-k books when n_non_top_k > 0, so the total may slightly
    # exceed total_budget_chunks.
    total_allocated = sum(r.total_budget for r in decision.book_routes)
    assert total_allocated >= 20 - len(decision.book_routes)
    assert total_allocated <= 20 + decision.top_k


@pytest.mark.asyncio
async def test_dsi_router_uses_max_score_when_book_appears_multiple_times() -> None:
    """If a book_id appears in multiple scored points, the highest score
    should be kept."""
    store = FakeDsiStore(
        next_results=[
            ScoredPoint(
                id="dsi-a1",
                score=0.6,
                payload={"book_id": "book-a"},
                vector=None,
                version=0,
                shard_key=None,
            ),
            ScoredPoint(
                id="dsi-a2",
                score=0.9,
                payload={"book_id": "book-a"},
                vector=None,
                version=0,
                shard_key=None,
            ),
        ]
    )
    embed = FakeEmbedClient()
    router = DsiRouter(store=store, embedding_client=embed)

    decision = await router.route("test query", total_budget_chunks=10, top_k=1)

    routes_by_id = {r.book_id: r for r in decision.book_routes}
    assert routes_by_id["book-a"].confidence == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_dsi_router_empty_results_returns_empty_routing_decision() -> None:
    store = FakeDsiStore(next_results=[])
    embed = FakeEmbedClient()
    router = DsiRouter(store=store, embedding_client=embed)

    decision = await router.route("test query")

    assert decision.book_routes == []
    assert decision.total_budget_chunks == 20
    assert decision.top_k == 8


@pytest.mark.asyncio
async def test_dsi_router_ignores_payloads_without_book_id() -> None:
    store = FakeDsiStore(
        next_results=[
            ScoredPoint(
                id="dsi-x",
                score=0.9,
                payload={"other_key": "value"},
                vector=None,
                version=0,
                shard_key=None,
            ),
        ]
    )
    embed = FakeEmbedClient()
    router = DsiRouter(store=store, embedding_client=embed)

    decision = await router.route("test query")

    assert decision.book_routes == []
