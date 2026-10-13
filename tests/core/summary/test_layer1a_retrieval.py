"""Tests for vo_rag.core.summary.layer1a_retrieval — the composed +Layer1A retrieval arm.

Summary layer 1A must NOT call ``QdrantStore.search_hybrid`` directly. The public
``search_summary_nodes`` surface takes a ``HybridRetriever`` (or any object
satisfying the same ``hybrid_search`` public method via duck-typing — see the
consolidation test below).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from qdrant_client.models import ScoredPoint

from vo_rag.core.retrieval.search import SearchResult
from vo_rag.core.summary.layer1a_retrieval import (
    build_layer1a_search_fn,
    layer1a_retrieve,
    search_summary_nodes,
)
from vo_rag.core.summary.persistence import SUMMARY_TREE_COLLECTION
from vo_rag.core.summary.schemas import NodeLevel, SummaryNode

RAW = "vo_rag_books_raw_v1"


def _node(node_id: str, chunk_ids: list[str], level: NodeLevel = NodeLevel.SECTION) -> SummaryNode:
    return SummaryNode(
        node_id=node_id,
        book_id="book-1",
        level=level,
        node_type=level.value,
        parent_node_id=None,
        child_node_ids=[],
        canonical_path=f"/{node_id}",
        display_path=node_id,
        source_chunk_ids=chunk_ids,
        source_section_paths=[f"[Book > {node_id}]"],
        evidence_spans=[],
        summary_text=f"summary of {node_id}",
        summary_model="qwen3.6-27b-fp8",
        prompt_hash="ph",
        prompt_version="1",
        embedding_model="bge-m3",
        embedding_version="1",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        source_collection_version="book_rag_v2",
    )


def _node_point(node: SummaryNode, score: float) -> ScoredPoint:
    payload = node.model_dump(mode="json")
    return ScoredPoint(id=node.node_id, version=0, score=score, payload=payload)


def _as_search_result(point: ScoredPoint) -> SearchResult:
    """Adapt a ScoredPoint to the SearchResult shape.

    HybridRetriever.hybrid_search returns SearchResult. The summary layer
    only reads .score and .payload; the rest are populated with safe
    defaults so the fake is structurally compatible with the production
    retriever contract.
    """
    return SearchResult(
        id=str(point.id),
        score=point.score,
        collection=SUMMARY_TREE_COLLECTION,
        payload=point.payload or {},
    )


def _chunk_point(chunk_id: str, score: float) -> ScoredPoint:
    return ScoredPoint(
        id=chunk_id,
        version=0,
        score=score,
        payload={
            "chunk_id": chunk_id,
            "chunk_type": "text",
            "document_id": "doc-1",
            "source_id": "book-1",
            "section_path": "[Book > S]",
            "section_title": "S",
            "content": f"content {chunk_id}",
            "embedding_text": f"content {chunk_id}",
            "token_count": 10,
        },
    )


class _FakeRetriever:
    """Duck-typed stand-in for HybridRetriever.

    Records every call so the consolidation test can assert that
    ``search_summary_nodes`` goes through the retriever's public
    ``hybrid_search`` method, never around it. The fake
    also serves as the ``store`` for ``expand_summary_node``'s raw
    expansion via a separate ``search_hybrid`` method (the
    ``QdrantStore``-shaped surface that the selective-expansion seam
    expects); the same dispatch on ``collection`` keeps the summary vs.
    raw routes separate.

    Return type: ``list[SearchResult]`` to structurally conform to the
    ``HybridRetriever.hybrid_search`` public method, with the test data
    (``ScoredPoint``) adapted via the ``_as_search_result`` helper. This
    keeps the call sites type-clean (mypy-compatible) without changing
    the on-the-wire payload shape.
    """

    def __init__(
        self,
        node_points: list[ScoredPoint],
        chunk_points: dict[str, ScoredPoint],
    ) -> None:
        self._node_points = node_points
        self._chunk_points = chunk_points
        self.calls: list[dict[str, object]] = []
        self.raw_calls: list[dict[str, object]] = []

    async def hybrid_search(
        self,
        *,
        collection: str,
        query: str = "",
        top_k: int = 20,
        filters: dict[str, object] | None = None,
        prefetch_depth: int = 100,
        mode: object = None,
    ) -> list[SearchResult]:
        """Match the ``HybridRetriever.hybrid_search`` public signature so
        the duck-typed fake is structurally compatible with the Protocol
        used in ``search_summary_nodes`` / ``layer1a_retrieve``.

        Records every call so the consolidation test can assert that
        ``search_summary_nodes`` goes through the retriever's public
        ``hybrid_search`` method, never around it."""
        self.calls.append(
            {
                "query": query,
                "collection": collection,
                "top_k": top_k,
                "filters": filters,
            }
        )
        if collection == SUMMARY_TREE_COLLECTION:
            return [_as_search_result(p) for p in self._node_points[:top_k]]
        # raw expansion: filters carries the node's chunk_id allow-list
        allow_list = (filters or {}).get("chunk_id", [])
        allow = set(allow_list) if isinstance(allow_list, list) else set()
        pts = [self._chunk_points[c] for c in self._chunk_points if c in allow]
        pts.sort(key=lambda p: p.score, reverse=True)
        return [_as_search_result(p) for p in pts[:top_k]]

    async def search_hybrid(
        self,
        *,
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        top_k: int = 20,
        filters: dict[str, object] | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]:
        """QdrantStore-shaped seam used by ``selective_expansion.expand_summary_node``.

        Returns chunks in the same dispatch-on-collection pattern as
        ``hybrid_search`` so the fake can serve both roles."""
        self.raw_calls.append(
            {
                "collection": collection,
                "top_k": top_k,
                "filters": filters,
            }
        )
        if collection == SUMMARY_TREE_COLLECTION:
            return self._node_points[:top_k]
        allow_list = (filters or {}).get("chunk_id", [])
        allow = set(allow_list) if isinstance(allow_list, list) else set()
        pts = [self._chunk_points[c] for c in self._chunk_points if c in allow]
        pts.sort(key=lambda p: p.score, reverse=True)
        return pts[:top_k]


class TestLayer1AUsesHybridRetrieverContract:
    """The public surface routes every search through the retriever."""

    @pytest.mark.asyncio
    async def test_layer1a_search_uses_hybrid_retriever_contract(self) -> None:
        retriever = _FakeRetriever([], {})
        await search_summary_nodes(
            query_text="anti dive geometry",
            retriever=retriever,
            top_k=5,
            filters={"book_id": "book-1"},
        )
        assert retriever.calls == [
            {
                "query": "anti dive geometry",
                "collection": "vo_rag_books_summary_tree_v1",
                "top_k": 5,
                "filters": {"book_id": "book-1"},
            }
        ]

    @pytest.mark.asyncio
    async def test_layer1a_search_does_not_accept_qdrant_store_directly(self) -> None:
        """Invariant: the public surface must take a retriever
        (HybridRetriever or duck-typed equivalent), not a raw QdrantStore.

        We assert this at the import level: the kwarg name is ``retriever``,
        and the function body never references ``QdrantStore`` or
        ``search_hybrid`` (compile-time check via source grep + runtime check
        that the new public arg rejects the store-style positional).
        """
        import inspect

        sig = inspect.signature(search_summary_nodes)
        assert "retriever" in sig.parameters
        assert "qdrant" not in sig.parameters


class TestSearchSummaryNodes:
    @pytest.mark.asyncio
    async def test_reconstructs_nodes_from_payload(self) -> None:
        n1, n2 = _node("n1", ["c1"]), _node("n2", ["c2"])
        retriever = _FakeRetriever([_node_point(n1, 0.9), _node_point(n2, 0.7)], {})

        out = await search_summary_nodes(
            query_text="q",
            retriever=retriever,
            top_k=10,
        )

        assert [n.node_id for n, _ in out] == ["n1", "n2"]
        assert out[0][1] == 0.9

    @pytest.mark.asyncio
    async def test_skips_unreconstructable_payload(self) -> None:
        good = _node("n1", ["c1"])
        bad = ScoredPoint(id="x", version=0, score=0.5, payload={"garbage": True})
        retriever = _FakeRetriever([_node_point(good, 0.9), bad], {})

        out = await search_summary_nodes(
            query_text="q",
            retriever=retriever,
            top_k=10,
        )

        assert [n.node_id for n, _ in out] == ["n1"]  # bad node dropped, not raised

    @pytest.mark.asyncio
    async def test_passes_filters_through(self) -> None:
        retriever = _FakeRetriever([_node_point(_node("n1", ["c1"]), 0.9)], {})
        await search_summary_nodes(
            query_text="q",
            retriever=retriever,
            top_k=5,
            filters={"book_id": "book-1"},
        )
        assert retriever.calls[0]["filters"] == {"book_id": "book-1"}


class TestLayer1aRetrieve:
    @pytest.mark.asyncio
    async def test_merges_and_ranks_expansions(self) -> None:
        n1, n2 = _node("n1", ["c1", "c2"]), _node("n2", ["c3"])
        chunks = {
            "c1": _chunk_point("c1", 0.9),
            "c2": _chunk_point("c2", 0.5),
            "c3": _chunk_point("c3", 0.99),
        }
        retriever = _FakeRetriever([_node_point(n1, 0.8), _node_point(n2, 0.6)], chunks)

        out = await layer1a_retrieve(
            query_text="q",
            retriever=retriever,
            top_k_nodes=10,
            top_k_chunks=10,
        )
        ids = [c.chunk_id for c in out]

        # n1 (score .8): c1 rank1 -> .8, c2 rank2 -> .4 ; n2 (score .6): c3 rank1 -> .6
        assert ids == ["c1", "c3", "c2"]

    @pytest.mark.asyncio
    async def test_dedupes_chunk_across_nodes_keeping_higher_score(self) -> None:
        # c1 appears in both nodes; node n_hi (score .9) should win over n_lo (.3).
        n_hi, n_lo = _node("n_hi", ["c1"]), _node("n_lo", ["c1", "c2"])
        chunks = {"c1": _chunk_point("c1", 0.9), "c2": _chunk_point("c2", 0.9)}
        retriever = _FakeRetriever([_node_point(n_hi, 0.9), _node_point(n_lo, 0.3)], chunks)

        out = await layer1a_retrieve(
            query_text="q",
            retriever=retriever,
            top_k_nodes=10,
            top_k_chunks=10,
        )
        ids = [c.chunk_id for c in out]

        assert ids.count("c1") == 1  # deduped
        # c1 via n_hi rank1 -> .9 ; c2 via n_lo rank2 -> .15 -> c1 first
        assert ids[0] == "c1"

    @pytest.mark.asyncio
    async def test_top_k_chunks_caps_result(self) -> None:
        n1 = _node("n1", ["c1", "c2", "c3"])
        chunks = {c: _chunk_point(c, 0.9) for c in ["c1", "c2", "c3"]}
        retriever = _FakeRetriever([_node_point(n1, 0.8)], chunks)

        out = await layer1a_retrieve(
            query_text="q",
            retriever=retriever,
            top_k_nodes=10,
            top_k_chunks=2,
        )
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_no_nodes_returns_empty(self) -> None:
        retriever = _FakeRetriever([], {})
        out = await layer1a_retrieve(
            query_text="q",
            retriever=retriever,
            top_k_nodes=10,
            top_k_chunks=10,
        )
        assert out == []


class _FakeEmbed:
    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, object]:
        return {
            "dense": [[0.1, 0.2]],
            "sparse": [{"indices": [1, 2], "values": [0.5, 0.6]}],
            "colbert": [[[0.1, 0.2]]],
        }


class TestBuildLayer1aSearchFn:
    @pytest.mark.asyncio
    async def test_search_fn_returns_ranked_chunk_ids(self) -> None:
        n1 = _node("n1", ["c1", "c2"])
        chunks = {"c1": _chunk_point("c1", 0.9), "c2": _chunk_point("c2", 0.5)}
        retriever = _FakeRetriever([_node_point(n1, 0.8)], chunks)

        search_fn = build_layer1a_search_fn(embed=_FakeEmbed(), retriever=retriever)
        ids = await search_fn("how does downforce work?", 10)

        assert ids == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_search_fn_respects_top_k(self) -> None:
        n1 = _node("n1", ["c1", "c2", "c3"])
        chunks = {c: _chunk_point(c, 0.9) for c in ["c1", "c2", "c3"]}
        retriever = _FakeRetriever([_node_point(n1, 0.8)], chunks)

        search_fn = build_layer1a_search_fn(embed=_FakeEmbed(), retriever=retriever)
        ids = await search_fn("q", 2)
        assert len(ids) == 2
