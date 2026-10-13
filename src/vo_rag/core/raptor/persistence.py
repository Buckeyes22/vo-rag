"""Qdrant persistence for Layer 1B RAPTOR semantic cluster nodes.

Collection: vo_rag_books_raptor_v1
Vector config: BGE-M3 dense 1024d + sparse + ColBERT (1024d multivector).
Payload indexes: book_id (keyword), level (keyword), parent_node_id (keyword),
gmm_cluster_selection_criterion (keyword).
"""

from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    Modifier,
    MultiVectorComparator,
    MultiVectorConfig,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    SparseVectorParams,
    VectorParams,
)

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.chunking.models import hit_payload
from vo_rag.core.embedding.point_vectors import sparse_payload_to_int_map
from vo_rag.core.raptor.schemas import ClusterLevel, ClusterNode
from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse
from vo_rag.core.storage.logical_points import retrieve_logical_points
from vo_rag.core.storage.passage_contract import model_payload, passage_payload
from vo_rag.core.storage.qdrant import QdrantStore, bge_sparse_to_qdrant

RAPTOR_COLLECTION = "vo_rag_books_raptor_v1"

DENSE_DIM = 1024
COLBERT_DIM = 1024

_RAPTOR_INDEXES: dict[str, PayloadSchemaType] = {
    "book_id": PayloadSchemaType.KEYWORD,
    "level": PayloadSchemaType.KEYWORD,
    "parent_node_id": PayloadSchemaType.KEYWORD,
    "gmm_cluster_selection_criterion": PayloadSchemaType.KEYWORD,
}


def _vector_config() -> dict[str, VectorParams]:
    return {
        "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        "colbert": VectorParams(
            size=COLBERT_DIM,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
            hnsw_config=HnswConfigDiff(m=0),
        ),
    }


def _sparse_vector_config() -> dict[str, SparseVectorParams]:
    return {
        "sparse": SparseVectorParams(modifier=Modifier.IDF),
        "bm25": SparseVectorParams(modifier=Modifier.IDF),
    }


async def ensure_raptor_collection(
    client: AsyncQdrantClient,
    *,
    collection: str = RAPTOR_COLLECTION,
) -> bool:
    """Idempotent collection setup. Returns True if newly created.

    Same vector config as vo_rag_books_summary_tree_v1 (BGE-M3 dense 1024 + sparse + ColBERT).
    Payload indexes: book_id (keyword), level (keyword), parent_node_id (keyword),
    gmm_cluster_selection_criterion (keyword).
    """
    existing = {c.name for c in (await client.get_collections()).collections}
    if collection in existing:
        return False

    await client.create_collection(
        collection_name=collection,
        vectors_config=_vector_config(),
        sparse_vectors_config=_sparse_vector_config(),
    )

    for field, schema in _RAPTOR_INDEXES.items():
        await client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema,
        )

    return True


async def upsert_cluster_node(
    node: ClusterNode,
    *,
    dense_embedding: list[float],
    sparse_embedding: dict[int, float],
    colbert_embedding: list[list[float]],
    store: QdrantStore,
    collection: str = RAPTOR_COLLECTION,
) -> None:
    """Upsert a cluster node into the collection with all three vector
    representations. Point ID is node.node_id (already deterministic).
    Payload is node.model_dump() with non-vector fields.
    """
    point = cluster_node_point(
        node,
        dense_embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        colbert_embedding=colbert_embedding,
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            await store._client.upsert(
                collection_name=collection,
                points=[point],
                wait=True,
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                import asyncio

                await asyncio.sleep(2 * (attempt + 1))
    if last_error is None:
        raise RuntimeError("upsert_cluster_node retry loop exited with no error and no success")
    raise last_error


def cluster_node_point(
    node: ClusterNode,
    *,
    dense_embedding: list[float],
    sparse_embedding: dict[int, float],
    colbert_embedding: list[list[float]],
) -> PointStruct:
    """Build one complete RAPTOR representation without writing it."""
    lexical = tokens_to_bm25_sparse(tokenize(node.summary_text))
    vectors: dict[str, Any] = {
        "dense": dense_embedding,
    }

    if colbert_embedding:
        vectors["colbert"] = colbert_embedding
    if sparse_embedding:
        vectors["sparse"] = bge_sparse_to_qdrant(sparse_embedding)
    if lexical:
        vectors["bm25"] = bge_sparse_to_qdrant(lexical)
    payload = passage_payload(
        node.model_dump(mode="json"),
        point_id=node.node_id,
        source_id=node.book_id if hasattr(node, "book_id") else node.node_id,
        dense=dense_embedding,
        colbert=colbert_embedding,
        bm25=lexical,
    )

    return PointStruct(
        id=node.node_id,
        vector=vectors,
        payload=payload,
    )


async def get_cluster_node(
    node_id: str,
    *,
    store: QdrantStore,
    collection: str = RAPTOR_COLLECTION,
) -> ClusterNode | None:
    """Fetch a single cluster node by id; None if absent."""
    results = await retrieve_logical_points(store._client, collection=collection, ids=[node_id])
    if not results:
        return None
    return ClusterNode.model_validate(model_payload(results[0].payload, ClusterNode.model_fields))


async def get_children(
    parent_node_id: str | None,
    *,
    store: QdrantStore,
    collection: str = RAPTOR_COLLECTION,
) -> list[ClusterNode]:
    """Fetch all immediate children of a parent node. parent_node_id=None
    returns top-level (LEVEL_2) nodes per the depth-3 cap.
    """
    if parent_node_id is None:
        filter_cond = Filter(
            must=[
                FieldCondition(key="level", match=MatchValue(value=ClusterLevel.LEVEL_2.value)),
            ]
        )
    else:
        filter_cond = Filter(
            must=[
                FieldCondition(key="parent_node_id", match=MatchValue(value=parent_node_id)),
            ]
        )

    results = await store._client.scroll(
        collection_name=collection,
        scroll_filter=filter_cond,
        limit=1000,
        with_payload=True,
    )

    return [
        ClusterNode.model_validate(model_payload(point.payload, ClusterNode.model_fields))
        for point in results[0]
    ]


async def get_book_tree(
    book_id: str,
    *,
    store: QdrantStore,
    collection: str = RAPTOR_COLLECTION,
) -> list[ClusterNode]:
    """Fetch all RAPTOR nodes for a given book_id."""
    results = await store._client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="book_id", match=MatchValue(value=book_id)),
            ]
        ),
        limit=10000,
        with_payload=True,
    )

    return [
        ClusterNode.model_validate(model_payload(point.payload, ClusterNode.model_fields))
        for point in results[0]
    ]


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


async def expand_cluster_node(
    *,
    node: ClusterNode,
    query: str,
    m: int,
    qdrant: QdrantStore,
    raw_collection: str = "vo_rag_books_raw_v1",
) -> list[RetrievalChunk]:
    """Return the top-m chunks from node.source_chunk_ids most relevant to query.

    Hybrid retrieval over raw_collection with filter chunk_id IN node.source_chunk_ids.

    Invariant 2 (hard requirement): the candidate set is exclusively
    node.source_chunk_ids. Never expand to other books or other sections.
    If hybrid retrieval returns results outside the set, they are filtered out.
    """
    if not node.source_chunk_ids:
        return []

    from vo_rag.core.config.settings import get_settings
    from vo_rag.core.embedding.client import EmbeddingClient

    settings = get_settings()
    embed_client = EmbeddingClient(
        settings.embedding.query_url,
        timeout_s=settings.embedding.timeout_s,
    )
    embeds = await embed_client.embed(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert=True,
    )
    await embed_client.aclose()

    query_dense: list[float] = embeds["dense"][0]
    sparse_vec = embeds["sparse"][0]
    query_sparse = sparse_payload_to_int_map(sparse_vec)
    query_bm25 = tokens_to_bm25_sparse(tokenize(query))
    query_colbert: list[list[float]] = embeds["colbert"][0]

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

    source_set = set(node.source_chunk_ids)
    return [
        _point_to_chunk(point)
        for point in scored
        if point.payload and point.payload.get("chunk_id") in source_set
    ]


def default_m_for(level: ClusterLevel) -> int:
    """Return the default expansion budget m for a given cluster level.

    LEVEL_0 -> 8
    LEVEL_1 -> 15
    LEVEL_2 -> 25
    """
    defaults: dict[ClusterLevel, int] = {
        ClusterLevel.LEVEL_0: 8,
        ClusterLevel.LEVEL_1: 15,
        ClusterLevel.LEVEL_2: 25,
    }
    return defaults.get(level, 8)


__all__ = [
    "ensure_raptor_collection",
    "upsert_cluster_node",
    "get_cluster_node",
    "get_children",
    "get_book_tree",
    "expand_cluster_node",
    "default_m_for",
]
