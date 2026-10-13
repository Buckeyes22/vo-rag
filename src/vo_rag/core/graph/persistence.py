"""Qdrant persistence for Layer 3 Evidence-linked entity graph.

Collections:
  - vo_rag_books_graph_nodes_v1 — BGE-M3 dense 1024d + sparse + ColBERT (node label + aliases)
  - vo_rag_books_graph_edges_v1 — dense-only 1024d (edge_type + canonical evidence span)

Collections are GATED: only persisted after the graph smoke test passes.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Protocol

import structlog
from qdrant_client.models import (
    Condition,
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

from vo_rag.core.graph.schemas import GraphEdge, GraphEdgeType, GraphNode
from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse
from vo_rag.core.storage.logical_points import retrieve_logical_points
from vo_rag.core.storage.passage_contract import model_payload, passage_payload
from vo_rag.core.storage.qdrant import bge_sparse_to_qdrant

log = structlog.get_logger(__name__)

GRAPH_NODES_COLLECTION = "vo_rag_books_graph_nodes_v1"
GRAPH_EDGES_COLLECTION = "vo_rag_books_graph_edges_v1"

DENSE_DIM = 1024
COLBERT_DIM = 1024


class GraphQdrantStore(Protocol):
    _client: Any


class GraphQdrantClient(Protocol):
    async def get_collections(self) -> Any: ...

    async def create_collection(self, *, collection_name: str, **kwargs: Any) -> None: ...

    async def create_payload_index(
        self,
        *,
        collection_name: str,
        field_name: str,
        field_schema: Any,
    ) -> None: ...


_NODE_INDEXES: dict[str, PayloadSchemaType] = {
    "node_type": PayloadSchemaType.KEYWORD,
    "book_ids": PayloadSchemaType.KEYWORD,
    "label": PayloadSchemaType.KEYWORD,
}

_EDGE_INDEXES: dict[str, PayloadSchemaType] = {
    "edge_type": PayloadSchemaType.KEYWORD,
    "source_node_id": PayloadSchemaType.KEYWORD,
    "target_node_id": PayloadSchemaType.KEYWORD,
}


def _node_vector_config() -> dict[str, VectorParams]:
    """BGE-M3 dense 1024 + ColBERT for nodes (vectorize node label + aliases)."""
    return {
        "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        "colbert": VectorParams(
            size=COLBERT_DIM,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
            hnsw_config=HnswConfigDiff(m=0),
        ),
    }


def _node_sparse_vector_config() -> dict[str, SparseVectorParams]:
    """BGE-M3 sparse + BM25 for nodes."""
    return {
        "sparse": SparseVectorParams(modifier=Modifier.IDF),
        "bm25": SparseVectorParams(modifier=Modifier.IDF),
    }


def _edge_vector_config() -> dict[str, VectorParams]:
    """Dense-only 1024d for edges (edge_type + canonical evidence span).

    No sparse/ColBERT: edges are short strings and the extra indexes
    are not justified at pilot scale.
    """
    return {
        "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
    }


async def ensure_graph_collections(
    client: GraphQdrantClient,
    *,
    nodes_collection: str = GRAPH_NODES_COLLECTION,
    edges_collection: str = GRAPH_EDGES_COLLECTION,
) -> tuple[bool, bool]:
    """Idempotent graph collection setup.

    Returns (nodes_newly_created, edges_newly_created).

    Collections are GATED — call this only after the graph smoke test passes.
    Nodes use BGE-M3 dense 1024 + sparse + ColBERT (vectorize node label + aliases).
    Edges use dense-only 1024d (vectorize edge_type + canonical evidence span).
    """
    existing = {c.name for c in (await client.get_collections()).collections}

    nodes_created = False
    if nodes_collection not in existing:
        await client.create_collection(
            collection_name=nodes_collection,
            vectors_config=_node_vector_config(),
            sparse_vectors_config=_node_sparse_vector_config(),
        )
        for field, schema in _NODE_INDEXES.items():
            await client.create_payload_index(
                collection_name=nodes_collection,
                field_name=field,
                field_schema=schema,
            )
        nodes_created = True
        log.info("graph.nodes_collection_created", collection=nodes_collection)

    edges_created = False
    if edges_collection not in existing:
        await client.create_collection(
            collection_name=edges_collection,
            vectors_config=_edge_vector_config(),
        )
        for field, schema in _EDGE_INDEXES.items():
            await client.create_payload_index(
                collection_name=edges_collection,
                field_name=field,
                field_schema=schema,
            )
        edges_created = True
        log.info("graph.edges_collection_created", collection=edges_collection)

    return nodes_created, edges_created


async def upsert_node(
    *,
    node: GraphNode,
    dense_embedding: list[float],
    sparse_embedding: dict[int, float],
    colbert_embedding: list[list[float]],
    store: GraphQdrantStore,
    collection: str = GRAPH_NODES_COLLECTION,
) -> None:
    """Upsert a graph node with BGE-M3 dense + sparse + ColBERT vectors.

    Point ID is node.node_id (deterministic via node_id_for).
    Payload is node.model_dump(mode="json").
    """
    lexical = tokens_to_bm25_sparse(tokenize(node.label))
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
        source_id=node.node_id,
        dense=dense_embedding,
        colbert=colbert_embedding,
        bm25=lexical,
    )
    point = PointStruct(id=node.node_id, vector=vectors, payload=payload)

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
                await asyncio.sleep(2 * (attempt + 1))
    if last_error is None:  # pragma: no cover - retry loop always sets on failure
        raise RuntimeError("graph node upsert exited retry loop without an error")
    raise last_error


async def upsert_edge(
    *,
    edge: GraphEdge,
    dense_embedding: list[float],
    store: GraphQdrantStore,
    collection: str = GRAPH_EDGES_COLLECTION,
) -> None:
    """Upsert a graph edge with a single dense embedding.

    Point ID is edge.edge_id (deterministic via edge_id_for).
    Payload is edge.model_dump(mode="json").
    Provenance is preserved via evidence_chunk_ids on GraphEdge — not invented here.
    """
    vectors: dict[str, Any] = {
        "dense": dense_embedding,
    }
    payload = passage_payload(
        edge.model_dump(mode="json"),
        point_id=edge.edge_id,
        source_id=edge.edge_id,
        dense=dense_embedding,
        colbert=[],
    )
    payload["bm25_representation"] = "absent"
    point = PointStruct(id=edge.edge_id, vector=vectors, payload=payload)

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
                await asyncio.sleep(2 * (attempt + 1))
    if last_error is None:  # pragma: no cover - retry loop always sets on failure
        raise RuntimeError("graph edge upsert exited retry loop without an error")
    raise last_error


async def get_node(
    node_id: str,
    *,
    store: GraphQdrantStore,
    collection: str = GRAPH_NODES_COLLECTION,
) -> GraphNode | None:
    """Fetch a single graph node by ID. Returns None if absent."""
    results = await retrieve_logical_points(store._client, collection=collection, ids=[node_id])
    if not results:
        return None
    return GraphNode.model_validate(model_payload(results[0].payload, GraphNode.model_fields))


async def get_edges_from(
    source_node_id: str,
    *,
    store: GraphQdrantStore,
    edge_type: GraphEdgeType | None = None,
    collection: str = GRAPH_EDGES_COLLECTION,
) -> list[GraphEdge]:
    """Fetch all outgoing edges from source_node_id.

    Optionally filtered by edge_type. Uses scroll pagination.
    """
    conditions: list[Condition] = [
        FieldCondition(key="source_node_id", match=MatchValue(value=source_node_id)),
    ]
    if edge_type is not None:
        conditions.append(
            FieldCondition(key="edge_type", match=MatchValue(value=edge_type.value)),
        )

    scroll_filter = Filter(must=conditions)
    edges: list[GraphEdge] = []
    next_offset: Any = None

    while True:
        results, next_offset = await store._client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=1000,
            with_payload=True,
            offset=next_offset,
        )
        edges.extend(
            GraphEdge.model_validate(model_payload(point.payload, GraphEdge.model_fields))
            for point in results
        )
        if next_offset is None:
            break

    return edges


async def get_edges_to(
    target_node_id: str,
    *,
    store: GraphQdrantStore,
    edge_type: GraphEdgeType | None = None,
    collection: str = GRAPH_EDGES_COLLECTION,
) -> list[GraphEdge]:
    """Fetch all incoming edges to target_node_id.

    Optionally filtered by edge_type. Uses scroll pagination.
    """
    conditions: list[Condition] = [
        FieldCondition(key="target_node_id", match=MatchValue(value=target_node_id)),
    ]
    if edge_type is not None:
        conditions.append(
            FieldCondition(key="edge_type", match=MatchValue(value=edge_type.value)),
        )

    scroll_filter = Filter(must=conditions)
    edges: list[GraphEdge] = []
    next_offset: Any = None

    while True:
        results, next_offset = await store._client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=1000,
            with_payload=True,
            offset=next_offset,
        )
        edges.extend(
            GraphEdge.model_validate(model_payload(point.payload, GraphEdge.model_fields))
            for point in results
        )
        if next_offset is None:
            break

    return edges


async def get_neighbors(
    node_id: str,
    *,
    store: GraphQdrantStore,
    direction: Literal["in", "out", "both"] = "both",
    nodes_collection: str = GRAPH_NODES_COLLECTION,
    edges_collection: str = GRAPH_EDGES_COLLECTION,
) -> list[tuple[GraphNode, GraphEdge]]:
    """Return (neighbor_node, edge) pairs for the given node_id.

    direction="out" — outgoing edges (node is source)
    direction="in"  — incoming edges (node is target)
    direction="both" — union of in and out

    Each pair is (neighbor_node, edge_connecting_to_neighbor).
    Missing neighbor nodes are silently skipped.
    """
    out_edges: list[GraphEdge] = []
    in_edges: list[GraphEdge] = []

    if direction in ("out", "both"):
        out_edges = await get_edges_from(node_id, store=store, collection=edges_collection)
    if direction in ("in", "both"):
        in_edges = await get_edges_to(node_id, store=store, collection=edges_collection)

    neighbor_ids_out = [e.target_node_id for e in out_edges]
    neighbor_ids_in = [e.source_node_id for e in in_edges]
    all_neighbor_ids = list(dict.fromkeys(neighbor_ids_out + neighbor_ids_in))

    if not all_neighbor_ids:
        return []

    raw_points = await retrieve_logical_points(
        store._client, collection=nodes_collection, ids=all_neighbor_ids
    )
    # Qdrant returns point ids in canonical hyphenated UUID form, but
    # GraphEdge.source_node_id/target_node_id store the node's own
    # unhyphenated `node_id` (see schemas.py: node_id_for). Keying on
    # str(point.id) here made every lookup below miss in production. Key on
    # node.node_id instead -- the exact string edges reference.
    neighbor_map: dict[str, GraphNode] = {}
    for point in raw_points:
        if point.payload:
            node = GraphNode.model_validate(model_payload(point.payload, GraphNode.model_fields))
            neighbor_map[node.node_id] = node

    pairs: list[tuple[GraphNode, GraphEdge]] = []
    for edge in out_edges:
        neighbor = neighbor_map.get(edge.target_node_id)
        if neighbor is not None:
            pairs.append((neighbor, edge))
    for edge in in_edges:
        neighbor = neighbor_map.get(edge.source_node_id)
        if neighbor is not None:
            pairs.append((neighbor, edge))

    return pairs


async def count_nodes_and_edges(
    *,
    store: GraphQdrantStore,
    nodes_collection: str = GRAPH_NODES_COLLECTION,
    edges_collection: str = GRAPH_EDGES_COLLECTION,
) -> tuple[int, int]:
    """Return (node_count, edge_count) for the graph collections.

    Useful for smoke test threshold checks (min_node_count_pilot, min_edge_count_pilot).
    """
    nodes_info = await store._client.count(
        collection_name=nodes_collection,
        exact=True,
    )
    edges_info = await store._client.count(
        collection_name=edges_collection,
        exact=True,
    )
    return nodes_info.count, edges_info.count


__all__ = [
    "GraphQdrantStore",
    "GraphQdrantClient",
    "ensure_graph_collections",
    "upsert_node",
    "upsert_edge",
    "get_node",
    "get_edges_from",
    "get_edges_to",
    "get_neighbors",
    "count_nodes_and_edges",
]
