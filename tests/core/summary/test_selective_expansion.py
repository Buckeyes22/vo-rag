"""Deep tests for vo_rag.core.summary.selective_expansion — source-chunk containment contract.

``expand_summary_node`` filters to
``node.source_chunk_ids`` BEFORE returning chunks (defense in depth: even
if a future hybrid-retrieval misfire returned an extra, the expansion
must NOT include it).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from qdrant_client.models import ScoredPoint

from vo_rag.core.summary.schemas import NodeLevel, SummaryNode
from vo_rag.core.summary.selective_expansion import default_m_for, expand_summary_node


def _make_node(
    *,
    node_id: str = "test-node-001",
    book_id: str = "book-001",
    level: NodeLevel = NodeLevel.SECTION,
    canonical_path: str = "Book > Ch1 > S1",
    display_path: str = "Chapter 1 > Section 1",
    source_chunk_ids: list[str] | None = None,
    prompt_hash: str = "a" * 64,
) -> SummaryNode:
    return SummaryNode(
        node_id=node_id,
        book_id=book_id,
        level=level,
        node_type=f"{level.value}_summary",
        parent_node_id=None,
        child_node_ids=[],
        canonical_path=canonical_path,
        display_path=display_path,
        source_chunk_ids=sorted(source_chunk_ids)
        if source_chunk_ids is not None
        else ["chunk-A", "chunk-B", "chunk-C"],
        source_section_paths=[canonical_path],
        evidence_spans=[],
        summary_text="Test summary text.",
        summary_model="test-model",
        prompt_hash=prompt_hash,
        prompt_version="v1",
        embedding_model="BGE-M3",
        embedding_version="test-v1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_collection_version="test-v1",
    )


def _scored_point(chunk_id: str, score: float, extra_field: str = "text") -> ScoredPoint:
    return ScoredPoint(
        id=abs(hash(chunk_id)) % (2**63),
        version=0,
        score=score,
        payload={
            "chunk_id": chunk_id,
            "chunk_type": "text",
            "document_id": "doc-001",
            "source_id": "book-001",
            "page_number": 1,
            "page_range": "1",
            "section_path": "Book > Ch1 > S1",
            "section_title": "Section 1",
            "content": f"Content of {chunk_id}",
            "embedding_text": f"Book > Ch1 > S1\n\nContent of {chunk_id}",
            "token_count": 50,
            "has_image": False,
        },
    )


_DENSE: list[float] = [0.1] * 1024
_SPARSE: dict[int, float] = {0: 0.5}
_BM25: dict[int, float] = {0: 0.3}
_COLBERT: list[list[float]] = [[0.1] * 1024]


class InMemoryQdrantStore:
    """Dict-backed QdrantStore test double — no unittest.mock."""

    def __init__(self, points: list[ScoredPoint] | None = None) -> None:
        self._points = points or []
        self._search_calls: list[dict[str, Any]] = []

    async def search_hybrid(
        self,
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]:
        self._search_calls.append(
            {
                "collection": collection,
                "query_dense": query_dense,
                "top_k": top_k,
                "filters": filters,
            }
        )
        allowed_ids = None
        if filters and "chunk_id" in filters:
            allowed_ids = set(filters["chunk_id"])
        candidates = self._points
        if allowed_ids is not None:
            candidates = [
                p
                for p in candidates
                if (p.payload is not None and p.payload.get("chunk_id") in allowed_ids)
            ]
        sorted_candidates = sorted(candidates, key=lambda p: -p.score)
        return sorted_candidates[:top_k]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_m_for_level_0_returns_8() -> None:
    assert default_m_for(NodeLevel.SECTION) == 8


def test_default_m_for_level_1_returns_15() -> None:
    assert default_m_for(NodeLevel.CHAPTER) == 15


def test_default_m_for_level_2_returns_25() -> None:
    assert default_m_for(NodeLevel.BOOK) == 25


async def test_expand_summary_node_returns_only_source_chunk_ids_chunks() -> None:
    node = _make_node(source_chunk_ids=["chunk-A", "chunk-B", "chunk-C"])

    qdrant = InMemoryQdrantStore(
        points=[
            _scored_point("chunk-A", 0.95),
            _scored_point("chunk-B", 0.85),
            _scored_point("chunk-C", 0.75),
            _scored_point("chunk-X", 0.65),
        ]
    )

    result = await expand_summary_node(
        node=node,
        query_dense=_DENSE,
        query_sparse=_SPARSE,
        query_bm25=_BM25,
        query_colbert=_COLBERT,
        m=10,
        qdrant=qdrant,  # type: ignore[arg-type]
    )

    result_ids = {chunk.chunk_id for chunk in result}
    assert result_ids == {"chunk-A", "chunk-B", "chunk-C"}
    assert "chunk-X" not in result_ids


async def test_expand_summary_node_filters_out_extras_returned_by_hybrid_retrieval() -> None:
    node = _make_node(source_chunk_ids=["chunk-A", "chunk-B"])

    qdrant = InMemoryQdrantStore(
        points=[
            _scored_point("chunk-A", 0.99),
            _scored_point("chunk-B", 0.88),
            _scored_point("chunk-Y", 0.77),
            _scored_point("chunk-Z", 0.66),
        ]
    )

    result = await expand_summary_node(
        node=node,
        query_dense=_DENSE,
        query_sparse=_SPARSE,
        query_bm25=_BM25,
        query_colbert=_COLBERT,
        m=10,
        qdrant=qdrant,  # type: ignore[arg-type]
    )

    result_ids = {chunk.chunk_id for chunk in result}
    assert result_ids == {"chunk-A", "chunk-B"}


async def test_expand_summary_node_returns_top_m_by_score() -> None:
    node = _make_node(source_chunk_ids=["c1", "c2", "c3", "c4", "c5"])

    qdrant = InMemoryQdrantStore(
        points=[
            _scored_point("c1", 0.60),
            _scored_point("c2", 0.90),
            _scored_point("c3", 0.70),
            _scored_point("c4", 0.80),
            _scored_point("c5", 0.50),
        ]
    )

    result = await expand_summary_node(
        node=node,
        query_dense=_DENSE,
        query_sparse=_SPARSE,
        query_bm25=_BM25,
        query_colbert=_COLBERT,
        m=3,
        qdrant=qdrant,  # type: ignore[arg-type]
    )

    assert len(result) == 3
    raw_scores = [r.metadata.get("score", 0.0) for r in result]
    scores = [float(s) for s in raw_scores]  # type: ignore[arg-type]
    assert scores == sorted(scores, reverse=True)
    top_ids = {r.chunk_id for r in result}
    assert top_ids == {"c2", "c4", "c3"}


async def test_expand_summary_node_handles_empty_source_chunk_ids() -> None:
    node = _make_node(source_chunk_ids=[])

    qdrant = InMemoryQdrantStore(
        points=[_scored_point("chunk-A", 0.95)],
    )

    result = await expand_summary_node(
        node=node,
        query_dense=_DENSE,
        query_sparse=_SPARSE,
        query_bm25=_BM25,
        query_colbert=_COLBERT,
        m=10,
        qdrant=qdrant,  # type: ignore[arg-type]
    )

    assert result == []
    assert len(qdrant._search_calls) == 0


# ---------------------------------------------------------------------------
# Expansion is limited to the node's source chunks
# ---------------------------------------------------------------------------


async def test_expand_summary_node_filters_to_source_chunk_ids_before_returning() -> None:
    """Source-chunk containment — even if the hybrid retrieval somehow
    returns a chunk NOT in ``node.source_chunk_ids`` (defense in depth), the
    function MUST filter it out before returning. The returned chunk-id set
    is a strict subset of ``node.source_chunk_ids``."""

    node = _make_node(source_chunk_ids=["chunk-A", "chunk-B"])

    # Hybrid retrieval returns one allowed and one rogue point.
    qdrant = InMemoryQdrantStore(
        points=[
            _scored_point("chunk-A", 0.95),
            _scored_point("chunk-B", 0.80),
            _scored_point("chunk-ROGUE", 0.99),
        ]
    )

    result = await expand_summary_node(
        node=node,
        query_dense=_DENSE,
        query_sparse=_SPARSE,
        query_bm25=_BM25,
        query_colbert=_COLBERT,
        m=10,
        qdrant=qdrant,  # type: ignore[arg-type]
    )

    returned_ids = {c.chunk_id for c in result}
    assert returned_ids <= set(node.source_chunk_ids), (
        "expand_summary_node must filter to node.source_chunk_ids before returning."
    )
    assert "chunk-ROGUE" not in returned_ids
