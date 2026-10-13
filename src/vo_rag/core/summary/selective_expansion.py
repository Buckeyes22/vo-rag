"""Selective expansion contract for Layer 1A summary nodes.

Retrieved summary nodes expand to top-m query-relevant chunks within their
``source_chunk_ids`` set, NOT to all constituent chunks. This is a hard
requirement: expansion never leaves a node's own provenance.

The raw collection name is the vo-rag canonical name:
``vo_rag_books_raw_v1`` (matches
``vo_rag.core.storage.qdrant.COLLECTION_NAMES``).
"""

from __future__ import annotations

from typing import Any

from qdrant_client.models import ScoredPoint

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.chunking.models import hit_payload
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.core.summary.schemas import NodeLevel, SummaryNode

DEFAULT_M_BY_LEVEL: dict[NodeLevel, int] = {
    NodeLevel.SECTION: 8,
    NodeLevel.CHAPTER: 15,
    NodeLevel.BOOK: 25,
}


def default_m_for(level: NodeLevel) -> int:
    """Return the default expansion budget m for a given node level.

    These engineering defaults are fixed before holdout evaluation so the
    evaluation does not tune them.
    """
    return DEFAULT_M_BY_LEVEL.get(level, 8)


async def expand_summary_node(
    *,
    node: SummaryNode,
    query_dense: list[float],
    query_sparse: dict[int, float],
    query_bm25: dict[int, float],
    query_colbert: list[list[float]],
    m: int,
    qdrant: QdrantStore,
    raw_collection: str = "vo_rag_books_raw_v1",
) -> list[RetrievalChunk]:
    """Return the top-m chunks from node.source_chunk_ids most relevant to query.

    Scores candidate chunks within source_chunk_ids against the query using
    hybrid (dense + sparse + BM25 + ColBERT) retrieval and returns top-m.

    Hard requirement: the candidate set is exclusively
    node.source_chunk_ids. Never expand to other books or other sections.
    """
    if not node.source_chunk_ids:
        return []

    chunk_ids_list = list(node.source_chunk_ids)

    scored = await qdrant.search_hybrid(
        collection=raw_collection,
        query_dense=query_dense,
        query_sparse=query_sparse,
        query_bm25=query_bm25,
        query_colbert=query_colbert,
        top_k=m,
        filters={"chunk_id": chunk_ids_list},
        prefetch_depth=min(len(chunk_ids_list), 200),
    )

    return [_point_to_chunk(point) for point in scored]


def _point_to_chunk(point: ScoredPoint) -> RetrievalChunk:
    # A raw figure's embedding-only window never reaches answer context (VOR-1452).
    payload: dict[str, Any] = hit_payload(point.payload)
    return RetrievalChunk(
        chunk_id=payload.get("chunk_id", ""),
        chunk_type=payload.get("chunk_type", "text"),
        document_id=payload.get("document_id", ""),
        source_id=payload.get("source_id", ""),
        page_number=payload.get("page_number"),
        page_range=payload.get("page_range"),
        section_path=payload.get("section_path", ""),
        section_title=payload.get("section_title", ""),
        content=payload.get("content", ""),
        embedding_text=payload.get("embedding_text", ""),
        token_count=payload.get("token_count", 0),
        has_image=payload.get("has_image", False),
        image_id=payload.get("image_id"),
        image_ref=payload.get("image_ref"),
        vlm_description=payload.get("vlm_description"),
        metadata=payload.get("metadata", {}),
    )
