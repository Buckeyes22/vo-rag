"""Qdrant persistence for Layer 1A summary tree nodes.

The collection name matches the canonical list in
``vo_rag.core.storage.qdrant.COLLECTION_NAMES``.

Collection: ``vo_rag_books_summary_tree_v1``
Vector config: BGE-M3 dense (1024d) + sparse + ColBERT (1024d multivector).
Payload indexes: book_id, level, parent_node_id, canonical_path, plus the
shared index set declared in ``SUMMARY_TREE_EXTRA_INDEXES``.
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
    SparseVectorParams,
    VectorParams,
)

from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse
from vo_rag.core.storage.logical_points import retrieve_logical_points
from vo_rag.core.storage.passage_contract import model_payload, passage_payload
from vo_rag.core.storage.qdrant import QdrantStore, bge_sparse_to_qdrant
from vo_rag.core.summary.schemas import SummaryNode

SUMMARY_TREE_COLLECTION = "vo_rag_books_summary_tree_v1"

DENSE_DIM = 1024
COLBERT_DIM = 1024

_SUMMARY_TREE_INDEXES: dict[str, PayloadSchemaType] = {
    "book_id": PayloadSchemaType.KEYWORD,
    "level": PayloadSchemaType.KEYWORD,
    "parent_node_id": PayloadSchemaType.KEYWORD,
    "canonical_path": PayloadSchemaType.KEYWORD,
    "node_type": PayloadSchemaType.KEYWORD,
    "source_collection_version": PayloadSchemaType.KEYWORD,
    "prompt_hash": PayloadSchemaType.KEYWORD,
    "summary_model": PayloadSchemaType.KEYWORD,
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


async def ensure_summary_tree_collection(
    client: AsyncQdrantClient,
    *,
    collection: str = SUMMARY_TREE_COLLECTION,
) -> bool:
    """Idempotent collection setup. Returns True if newly created.

    Same vector config as vo_rag_books_raw_v1 (BGE-M3 dense 1024 + sparse + ColBERT).
    Payload indexes: book_id (keyword), level (keyword), parent_node_id (keyword),
    canonical_path (keyword).
    """
    existing = {c.name for c in (await client.get_collections()).collections}
    if collection in existing:
        return False

    await client.create_collection(
        collection_name=collection,
        vectors_config=_vector_config(),
        sparse_vectors_config=_sparse_vector_config(),
    )

    for field, schema in _SUMMARY_TREE_INDEXES.items():
        await client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema,
        )

    return True


async def upsert_summary_node(
    node: SummaryNode,
    *,
    dense_embedding: list[float],
    sparse_embedding: dict[int, float],
    colbert_embedding: list[list[float]],
    store: QdrantStore,
    collection: str = SUMMARY_TREE_COLLECTION,
) -> None:
    """Upsert a summary node into the collection with all three vector
    representations. Point ID is node.node_id (already deterministic).
    Payload is node.model_dump() with non-vector fields.
    """
    point = summary_node_point(
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
    if last_error is None:  # pragma: no cover - loop body always sets on failure
        raise RuntimeError("qdrant upsert exited retry loop with no error and no success")
    raise last_error


def summary_node_point(
    node: SummaryNode,
    *,
    dense_embedding: list[float],
    sparse_embedding: dict[int, float],
    colbert_embedding: list[list[float]],
) -> PointStruct:
    """Build one complete summary representation without writing it."""
    lexical = tokens_to_bm25_sparse(tokenize(node.summary_text))
    vectors: dict[str, Any] = {"dense": dense_embedding}
    if colbert_embedding:
        vectors["colbert"] = colbert_embedding
    if sparse_embedding:
        vectors["sparse"] = bge_sparse_to_qdrant(sparse_embedding)
    if lexical:
        vectors["bm25"] = bge_sparse_to_qdrant(lexical)
    payload = passage_payload(
        node.model_dump(mode="json"),
        point_id=node.node_id,
        source_id=node.book_id,
        dense=dense_embedding,
        colbert=colbert_embedding,
        bm25=lexical,
    )
    return PointStruct(id=node.node_id, vector=vectors, payload=payload)


async def get_summary_node(
    node_id: str,
    *,
    store: QdrantStore,
    collection: str = SUMMARY_TREE_COLLECTION,
) -> SummaryNode | None:
    """Fetch a single summary node by id; None if absent."""
    results = await retrieve_logical_points(store._client, collection=collection, ids=[node_id])
    if not results:
        return None
    return SummaryNode.model_validate(model_payload(results[0].payload, SummaryNode.model_fields))


async def get_children(
    parent_node_id: str | None,
    *,
    store: QdrantStore,
    collection: str = SUMMARY_TREE_COLLECTION,
) -> list[SummaryNode]:
    """Fetch all immediate children of a parent node. parent_node_id=None
    returns top-level (BOOK) nodes.
    """
    if parent_node_id is None:
        filter_cond = Filter(
            must=[
                FieldCondition(key="level", match=MatchValue(value="book")),
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
        SummaryNode.model_validate(model_payload(point.payload, SummaryNode.model_fields))
        for point in results[0]
    ]


async def get_book_tree(
    book_id: str,
    *,
    store: QdrantStore,
    collection: str = SUMMARY_TREE_COLLECTION,
) -> list[SummaryNode]:
    """Fetch all summary nodes for a given book_id."""
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
        SummaryNode.model_validate(model_payload(point.payload, SummaryNode.model_fields))
        for point in results[0]
    ]
