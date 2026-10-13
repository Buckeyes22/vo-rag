from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from vo_rag.core.graph.persistence import (
    GRAPH_EDGES_COLLECTION,
    GRAPH_NODES_COLLECTION,
    count_nodes_and_edges,
    ensure_graph_collections,
    get_edges_from,
    get_edges_to,
    get_neighbors,
    get_node,
    upsert_edge,
    upsert_node,
)
from vo_rag.core.graph.schemas import (
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    edge_id_for,
    node_id_for,
)

# ---------------------------------------------------------------------------
# In-memory Qdrant test double
# ---------------------------------------------------------------------------


@dataclass
class _FakeCollectionInfo:
    name: str


@dataclass
class _FakeCollectionsResult:
    collections: list[_FakeCollectionInfo]


@dataclass
class _FakePoint:
    id: str
    payload: dict[str, Any]
    vector: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeCountResult:
    count: int


class _FakeQdrantClient:
    """Minimal in-memory dict-backed stand-in for AsyncQdrantClient.

    Implements only the methods used by the graph persistence layer:
      get_collections, create_collection, create_payload_index,
      upsert, retrieve, scroll, count.
    """

    def __init__(self, *, initial_collections: set[str] | None = None) -> None:
        self._collections: set[str] = initial_collections or set()
        # {collection_name: {point_id: _FakePoint}}
        self._store: dict[str, dict[str, _FakePoint]] = {}

    # -- collection management -----------------------------------------------

    async def get_collections(self) -> _FakeCollectionsResult:
        return _FakeCollectionsResult(
            collections=[_FakeCollectionInfo(name=n) for n in self._collections]
        )

    async def create_collection(self, *, collection_name: str, **_kwargs: Any) -> None:
        self._collections.add(collection_name)
        self._store.setdefault(collection_name, {})

    async def create_payload_index(
        self,
        *,
        collection_name: str,
        field_name: str,
        field_schema: Any,
    ) -> None:
        # Index creation is a no-op in the test double.
        pass

    # -- point operations ----------------------------------------------------

    async def upsert(
        self,
        *,
        collection_name: str,
        points: list[Any],
        wait: bool = True,
    ) -> None:
        bucket = self._store.setdefault(collection_name, {})
        for point in points:
            bucket[str(point.id)] = _FakePoint(
                id=str(point.id),
                payload=dict(point.payload or {}),
                vector=dict(point.vector or {}),
            )

    async def retrieve(
        self,
        *,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
    ) -> list[_FakePoint]:
        bucket = self._store.get(collection_name, {})
        return [bucket[str(i)] for i in ids if str(i) in bucket]

    async def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 1000,
        with_payload: bool = True,
        offset: Any = None,
    ) -> tuple[list[_FakePoint], None]:
        bucket = self._store.get(collection_name, {})
        results: list[_FakePoint] = []
        for point in bucket.values():
            if scroll_filter is None or _matches_filter(point, scroll_filter):
                results.append(point)
        # Always returns next_offset=None (single-page, no pagination needed for tests).
        return results, None

    async def count(
        self,
        *,
        collection_name: str,
        exact: bool = True,
    ) -> _FakeCountResult:
        bucket = self._store.get(collection_name, {})
        return _FakeCountResult(count=len(bucket))


def _matches_filter(point: _FakePoint, scroll_filter: Any) -> bool:
    """Evaluate a qdrant_client Filter against a _FakePoint payload.

    Only handles the subset of filters used by persistence.py:
      Filter(must=[FieldCondition(key=..., match=MatchValue(value=...))]).
    """
    for condition in scroll_filter.must:
        key: str = condition.key
        expected: Any = condition.match.value
        actual = point.payload.get(key)
        if actual != expected:
            return False
    return True


# ---------------------------------------------------------------------------
# Fake QdrantStore wrapper
# ---------------------------------------------------------------------------


class _FakeStore:
    """Minimal QdrantStore wrapper exposing only _client, as persistence.py uses."""

    def __init__(self, client: _FakeQdrantClient) -> None:
        self._client = client


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_node(
    label: str = "understeer",
    node_type: GraphNodeType = GraphNodeType.CONCEPT,
    book_ids: list[str] | None = None,
) -> GraphNode:
    return GraphNode(
        node_id=node_id_for(node_type=node_type, canonical_label=label),
        node_type=node_type,
        label=label,
        aliases=[],
        book_ids=book_ids or ["book-001"],
        chunk_ids=["c1"],
        attributes={},
        created_at=datetime.now(UTC),
    )


def _make_edge(
    source_node_id: str,
    target_node_id: str,
    edge_type: GraphEdgeType = GraphEdgeType.RELATES_TO,
) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id_for(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type=edge_type,
        ),
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        book_ids=["book-1"],
        weight=1.0,
        evidence_chunk_ids=["c1"],
        created_at=datetime.now(UTC),
    )


_DENSE: list[float] = [0.1] * 1024
_SPARSE: dict[int, float] = {0: 0.5, 10: 0.3}
_COLBERT: list[list[float]] = [[0.1] * 1024]


# ---------------------------------------------------------------------------
# ensure_graph_collections
# ---------------------------------------------------------------------------


class TestEnsureGraphCollections:
    @pytest.mark.asyncio
    async def test_ensure_graph_collections_creates_both_when_absent(self) -> None:
        client = _FakeQdrantClient(initial_collections=set())
        nodes_created, edges_created = await ensure_graph_collections(client)
        assert nodes_created is True
        assert edges_created is True
        existing = {c.name for c in (await client.get_collections()).collections}
        assert GRAPH_NODES_COLLECTION in existing
        assert GRAPH_EDGES_COLLECTION in existing

    @pytest.mark.asyncio
    async def test_ensure_graph_collections_idempotent(self) -> None:
        client = _FakeQdrantClient(
            initial_collections={GRAPH_NODES_COLLECTION, GRAPH_EDGES_COLLECTION}
        )
        nodes_created, edges_created = await ensure_graph_collections(client)
        assert nodes_created is False
        assert edges_created is False

    @pytest.mark.asyncio
    async def test_ensure_graph_collections_partial_exists_nodes_missing(self) -> None:
        client = _FakeQdrantClient(initial_collections={GRAPH_EDGES_COLLECTION})
        nodes_created, edges_created = await ensure_graph_collections(client)
        assert nodes_created is True
        assert edges_created is False

    @pytest.mark.asyncio
    async def test_ensure_graph_collections_partial_exists_edges_missing(self) -> None:
        client = _FakeQdrantClient(initial_collections={GRAPH_NODES_COLLECTION})
        nodes_created, edges_created = await ensure_graph_collections(client)
        assert nodes_created is False
        assert edges_created is True


# ---------------------------------------------------------------------------
# upsert_node
# ---------------------------------------------------------------------------


class TestUpsertNode:
    @pytest.mark.asyncio
    async def test_upsert_node_uses_node_id_as_point_id(self) -> None:
        client = _FakeQdrantClient(initial_collections={GRAPH_NODES_COLLECTION})
        client._store[GRAPH_NODES_COLLECTION] = {}
        store = _FakeStore(client)
        node = _make_node("camber angle")

        await upsert_node(
            node=node,
            dense_embedding=_DENSE,
            sparse_embedding=_SPARSE,
            colbert_embedding=_COLBERT,
            store=store,
        )

        bucket = client._store[GRAPH_NODES_COLLECTION]
        assert node.node_id in bucket
        assert bucket[node.node_id].id == node.node_id

    @pytest.mark.asyncio
    async def test_upsert_node_includes_dense_sparse_colbert_vectors_in_payload(
        self,
    ) -> None:
        client = _FakeQdrantClient(initial_collections={GRAPH_NODES_COLLECTION})
        client._store[GRAPH_NODES_COLLECTION] = {}
        store = _FakeStore(client)
        node = _make_node("slip angle")

        await upsert_node(
            node=node,
            dense_embedding=_DENSE,
            sparse_embedding=_SPARSE,
            colbert_embedding=_COLBERT,
            store=store,
        )

        point = client._store[GRAPH_NODES_COLLECTION][node.node_id]
        assert "dense" in point.vector
        assert "sparse" in point.vector
        assert "colbert" in point.vector
        # Payload holds node model_dump
        assert point.payload["label"] == "slip angle"
        assert point.payload["node_type"] == GraphNodeType.CONCEPT


# ---------------------------------------------------------------------------
# upsert_edge
# ---------------------------------------------------------------------------


class TestUpsertEdge:
    @pytest.mark.asyncio
    async def test_upsert_edge_uses_edge_id_as_point_id(self) -> None:
        client = _FakeQdrantClient(initial_collections={GRAPH_EDGES_COLLECTION})
        client._store[GRAPH_EDGES_COLLECTION] = {}
        store = _FakeStore(client)

        node_a = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="braking")
        node_b = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="abs")
        edge = _make_edge(node_a, node_b)

        await upsert_edge(
            edge=edge,
            dense_embedding=_DENSE,
            store=store,
        )

        bucket = client._store[GRAPH_EDGES_COLLECTION]
        assert edge.edge_id in bucket
        assert bucket[edge.edge_id].id == edge.edge_id

    @pytest.mark.asyncio
    async def test_upsert_edge_dense_only(self) -> None:
        """Edges collection stores only dense vector — no sparse or ColBERT."""
        client = _FakeQdrantClient(initial_collections={GRAPH_EDGES_COLLECTION})
        client._store[GRAPH_EDGES_COLLECTION] = {}
        store = _FakeStore(client)

        node_a = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="traction")
        node_b = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="grip")
        edge = _make_edge(node_a, node_b)

        await upsert_edge(
            edge=edge,
            dense_embedding=_DENSE,
            store=store,
        )

        point = client._store[GRAPH_EDGES_COLLECTION][edge.edge_id]
        assert "dense" in point.vector
        assert "sparse" not in point.vector
        assert "colbert" not in point.vector


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------


class TestGetNode:
    @pytest.mark.asyncio
    async def test_get_node_returns_none_when_absent(self) -> None:
        client = _FakeQdrantClient(initial_collections={GRAPH_NODES_COLLECTION})
        client._store[GRAPH_NODES_COLLECTION] = {}
        store = _FakeStore(client)

        result = await get_node(
            "nonexistent-id",
            store=store,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_node_parses_graph_node_payload(self) -> None:
        client = _FakeQdrantClient(initial_collections={GRAPH_NODES_COLLECTION})
        client._store[GRAPH_NODES_COLLECTION] = {}
        store = _FakeStore(client)
        node = _make_node("oversteer")

        await upsert_node(
            node=node,
            dense_embedding=_DENSE,
            sparse_embedding=_SPARSE,
            colbert_embedding=_COLBERT,
            store=store,
        )

        retrieved = await get_node(node.node_id, store=store)
        assert retrieved is not None
        assert isinstance(retrieved, GraphNode)
        assert retrieved.node_id == node.node_id
        assert retrieved.label == "oversteer"
        assert retrieved.node_type == GraphNodeType.CONCEPT


# ---------------------------------------------------------------------------
# get_edges_from
# ---------------------------------------------------------------------------


class TestGetEdgesFrom:
    @pytest.mark.asyncio
    async def test_get_edges_from_filters_by_source_node_id(self) -> None:
        client = _FakeQdrantClient(initial_collections={GRAPH_EDGES_COLLECTION})
        client._store[GRAPH_EDGES_COLLECTION] = {}
        store = _FakeStore(client)

        src = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="downforce")
        tgt1 = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="drag")
        tgt2 = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="lift")
        other_src = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="other")

        edge_a = _make_edge(src, tgt1)
        edge_b = _make_edge(src, tgt2)
        edge_other = _make_edge(other_src, tgt1)

        for edge in (edge_a, edge_b, edge_other):
            await upsert_edge(
                edge=edge,
                dense_embedding=_DENSE,
                store=store,
            )

        results = await get_edges_from(src, store=store)
        result_ids = {e.edge_id for e in results}
        assert edge_a.edge_id in result_ids
        assert edge_b.edge_id in result_ids
        assert edge_other.edge_id not in result_ids

    @pytest.mark.asyncio
    async def test_get_edges_from_optional_edge_type_filter(self) -> None:
        client = _FakeQdrantClient(initial_collections={GRAPH_EDGES_COLLECTION})
        client._store[GRAPH_EDGES_COLLECTION] = {}
        store = _FakeStore(client)

        src = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="aero")
        tgt_a = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="x")
        tgt_b = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="y")

        edge_relates = _make_edge(src, tgt_a, GraphEdgeType.RELATES_TO)
        edge_part_of = _make_edge(src, tgt_b, GraphEdgeType.PART_OF)

        for edge in (edge_relates, edge_part_of):
            await upsert_edge(
                edge=edge,
                dense_embedding=_DENSE,
                store=store,
            )

        results = await get_edges_from(
            src,
            store=store,
            edge_type=GraphEdgeType.RELATES_TO,
        )
        assert len(results) == 1
        assert results[0].edge_id == edge_relates.edge_id


# ---------------------------------------------------------------------------
# get_edges_to
# ---------------------------------------------------------------------------


class TestGetEdgesTo:
    @pytest.mark.asyncio
    async def test_get_edges_to_filters_by_target_node_id(self) -> None:
        client = _FakeQdrantClient(initial_collections={GRAPH_EDGES_COLLECTION})
        client._store[GRAPH_EDGES_COLLECTION] = {}
        store = _FakeStore(client)

        tgt = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="apex")
        src1 = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="braking")
        src2 = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="acceleration")
        other_tgt = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="other-tgt")

        edge_a = _make_edge(src1, tgt)
        edge_b = _make_edge(src2, tgt)
        edge_other = _make_edge(src1, other_tgt)

        for edge in (edge_a, edge_b, edge_other):
            await upsert_edge(
                edge=edge,
                dense_embedding=_DENSE,
                store=store,
            )

        results = await get_edges_to(tgt, store=store)
        result_ids = {e.edge_id for e in results}
        assert edge_a.edge_id in result_ids
        assert edge_b.edge_id in result_ids
        assert edge_other.edge_id not in result_ids


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------


class TestGetNeighbors:
    @pytest.mark.asyncio
    async def test_get_neighbors_both_direction_returns_in_and_out(self) -> None:
        client = _FakeQdrantClient(
            initial_collections={GRAPH_NODES_COLLECTION, GRAPH_EDGES_COLLECTION}
        )
        client._store[GRAPH_NODES_COLLECTION] = {}
        client._store[GRAPH_EDGES_COLLECTION] = {}
        store = _FakeStore(client)

        node_center = _make_node("center")
        node_out_target = _make_node("out-target")
        node_in_source = _make_node("in-source")

        for node in (node_center, node_out_target, node_in_source):
            await upsert_node(
                node=node,
                dense_embedding=_DENSE,
                sparse_embedding=_SPARSE,
                colbert_embedding=_COLBERT,
                store=store,
            )

        # center → out_target  (outgoing)
        edge_out = _make_edge(node_center.node_id, node_out_target.node_id)
        # in_source → center   (incoming)
        edge_in = _make_edge(node_in_source.node_id, node_center.node_id)

        for edge in (edge_out, edge_in):
            await upsert_edge(
                edge=edge,
                dense_embedding=_DENSE,
                store=store,
            )

        pairs = await get_neighbors(
            node_center.node_id,
            store=store,
            direction="both",
        )
        neighbor_node_ids = {n.node_id for n, _ in pairs}
        assert node_out_target.node_id in neighbor_node_ids
        assert node_in_source.node_id in neighbor_node_ids
        assert len(pairs) == 2


# ---------------------------------------------------------------------------
# count_nodes_and_edges
# ---------------------------------------------------------------------------


class TestCountNodesAndEdges:
    @pytest.mark.asyncio
    async def test_count_nodes_and_edges_returns_collection_counts(self) -> None:
        client = _FakeQdrantClient(
            initial_collections={GRAPH_NODES_COLLECTION, GRAPH_EDGES_COLLECTION}
        )
        client._store[GRAPH_NODES_COLLECTION] = {}
        client._store[GRAPH_EDGES_COLLECTION] = {}
        store = _FakeStore(client)

        node_a = _make_node("node-a")
        node_b = _make_node("node-b")
        for node in (node_a, node_b):
            await upsert_node(
                node=node,
                dense_embedding=_DENSE,
                sparse_embedding=_SPARSE,
                colbert_embedding=_COLBERT,
                store=store,
            )

        edge = _make_edge(node_a.node_id, node_b.node_id)
        await upsert_edge(
            edge=edge,
            dense_embedding=_DENSE,
            store=store,
        )

        node_count, edge_count = await count_nodes_and_edges(store=store)
        assert node_count == 2
        assert edge_count == 1
