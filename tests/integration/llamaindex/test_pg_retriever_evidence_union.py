"""Evidence-union provenance test for VoRagPGRetriever (VOR-216).

Asserts invariant 2: the chunks ``custom_retrieve`` returns are EXACTLY the
union of the evidence referenced by the edges traversed and the nodes visited —
no invented provenance.

VoRagPGRetriever resolves seeds from the query string (``entity_labels=[query_str]``);
the core ``graph_query_to_evidence_chunks`` test verifies the same invariant directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.graph.retrieval import graph_query_to_evidence_chunks
from vo_rag.core.graph.schemas import GraphEdge, GraphEdgeType, GraphNode, GraphNodeType
from vo_rag.integrations.llamaindex.pg_retriever import VoRagPGRetriever


class FakePoint:
    def __init__(self, point_id: str, payload: dict[str, Any]) -> None:
        self.id = point_id
        self.payload = payload


class FakeQdrantClient:
    def __init__(
        self,
        nodes: list[FakePoint],
        edges: list[FakePoint],
        raw_chunks: list[FakePoint],
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._raw_chunks = raw_chunks

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        with_payload: bool = True,
        offset: Any = None,
        scroll_filter: Any = None,
    ) -> tuple[list[FakePoint], Any]:
        if collection_name == "vo_rag_books_graph_nodes_v1":
            data = self._nodes
        elif collection_name == "vo_rag_books_graph_edges_v1":
            data = self._edges
        elif collection_name == "vo_rag_books_raw_v1":
            data = self._raw_chunks
        else:
            data = []

        if scroll_filter is not None:
            data = [p for p in data if self._matches_filter(p, scroll_filter)]

        start = int(offset) if offset is not None else 0
        page = data[start : start + limit]
        next_offset = start + limit if start + limit < len(data) else None
        return page, next_offset

    async def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
    ) -> list[FakePoint]:
        if collection_name == "vo_rag_books_graph_nodes_v1":
            by_id = {p.id: p for p in self._nodes}
            return [by_id[i] for i in ids if i in by_id]
        return []

    def _matches_filter(self, point: FakePoint, scroll_filter: Any) -> bool:
        must = getattr(scroll_filter, "must", None)
        if not must:
            return True
        return all(self._condition_matches(point, condition) for condition in must)

    def _condition_matches(self, point: FakePoint, condition: Any) -> bool:
        key = getattr(condition, "key", None)
        match = getattr(condition, "match", None)
        if key is None or match is None:
            return True
        value = getattr(match, "value", None)
        if value is None:
            return True
        return bool(point.payload.get(key) == value)


class FakeGraphQdrantStore:
    def __init__(self, client: FakeQdrantClient) -> None:
        self._client = client


def _make_graph_fixture() -> tuple[
    GraphNode,
    GraphNode,
    GraphNode,
    GraphNode,
    GraphEdge,
    GraphEdge,
    FakeQdrantClient,
]:
    now = datetime.now(UTC)

    node_a = GraphNode(
        node_id="node-a",
        node_type=GraphNodeType.CONCEPT,
        label="Downforce",
        aliases=[],
        book_ids=["book-1"],
        chunk_ids=["chunk-a1"],
        created_at=now,
    )
    node_b = GraphNode(
        node_id="node-b",
        node_type=GraphNodeType.CONCEPT,
        label="Aerodynamics",
        aliases=[],
        book_ids=["book-1"],
        chunk_ids=["chunk-b1"],
        created_at=now,
    )
    node_c = GraphNode(
        node_id="node-c",
        node_type=GraphNodeType.CONCEPT,
        label="Speed",
        aliases=[],
        book_ids=["book-1"],
        chunk_ids=["chunk-c1"],
        created_at=now,
    )
    node_d = GraphNode(
        node_id="node-d",
        node_type=GraphNodeType.CONCEPT,
        label="Cooking",
        aliases=[],
        book_ids=["book-1"],
        chunk_ids=["chunk-d1"],
        created_at=now,
    )

    edge_ab = GraphEdge(
        edge_id="edge-ab",
        source_node_id="node-a",
        target_node_id="node-b",
        edge_type=GraphEdgeType.RELATES_TO,
        book_ids=["book-1"],
        weight=1.0,
        evidence_chunk_ids=["edge-ab-1"],
        created_at=now,
    )
    edge_bc = GraphEdge(
        edge_id="edge-bc",
        source_node_id="node-b",
        target_node_id="node-c",
        edge_type=GraphEdgeType.RELATES_TO,
        book_ids=["book-1"],
        weight=1.0,
        evidence_chunk_ids=["edge-bc-1"],
        created_at=now,
    )

    nodes = [
        FakePoint("node-a", node_a.model_dump(mode="json")),
        FakePoint("node-b", node_b.model_dump(mode="json")),
        FakePoint("node-c", node_c.model_dump(mode="json")),
        FakePoint("node-d", node_d.model_dump(mode="json")),
    ]
    edges = [
        FakePoint("edge-ab", edge_ab.model_dump(mode="json")),
        FakePoint("edge-bc", edge_bc.model_dump(mode="json")),
    ]

    raw_chunks: list[FakePoint] = []
    for cid in [
        "chunk-a1",
        "chunk-b1",
        "chunk-c1",
        "edge-ab-1",
        "edge-bc-1",
        "chunk-d1",
        "chunk-outside-1",
    ]:
        chunk = RetrievalChunk(
            chunk_id=cid,
            chunk_type="text",
            document_id="doc-1",
            source_id="source-1",
            section_path="Test",
            section_title="Test",
            content=f"Content for {cid}",
            embedding_text=f"Test\n\nContent for {cid}",
            token_count=5,
        )
        raw_chunks.append(FakePoint(f"point-{cid}", chunk.model_dump(mode="json")))

    client = FakeQdrantClient(nodes, edges, raw_chunks)
    return node_a, node_b, node_c, node_d, edge_ab, edge_bc, client


def test_custom_retrieve_only_returns_traversal_evidence() -> None:
    """Retrieved node.id_ set must equal the union of traversed edge/node evidence."""
    _node_a, _node_b, _node_c, _node_d, _edge_ab, _edge_bc, client = _make_graph_fixture()
    store = FakeGraphQdrantStore(client)

    # CustomPGRetriever requires a graph_store arg even though we don't use it.
    graph_store = MagicMock()
    retriever = VoRagPGRetriever(graph_store=graph_store, store=store, top_k=10)

    nodes = retriever.custom_retrieve("downforce")

    ids = {n.node.id_ for n in nodes}

    # Compute expected union from the fixture graph structure.
    # BFS from node-a (seed) -> edge_ab -> node-b -> edge_bc -> node-c.
    expected = set(_node_a.chunk_ids)
    expected.update(_edge_ab.evidence_chunk_ids)
    expected.update(_node_b.chunk_ids)
    expected.update(_edge_bc.evidence_chunk_ids)
    expected.update(_node_c.chunk_ids)

    # Subset invariant — no invented chunks.
    assert ids <= expected
    # Exact match for the bounded fixture.
    assert ids == expected
    # Negative: chunks present in raw collection but NOT in the union must not leak in.
    assert "chunk-d1" not in ids
    assert "chunk-outside-1" not in ids


async def test_graph_query_to_evidence_chunks_union_invariant() -> None:
    """Direct core test: evidence set is exactly the traversal union."""
    _node_a, _node_b, _node_c, _node_d, _edge_ab, _edge_bc, client = _make_graph_fixture()
    store = FakeGraphQdrantStore(client)

    chunks = await graph_query_to_evidence_chunks(
        query_text="downforce",
        entity_labels=["downforce"],
        qdrant=store,
        top_k=10,
    )

    ids = {c.chunk_id for c in chunks}

    expected = set(_node_a.chunk_ids)
    expected.update(_edge_ab.evidence_chunk_ids)
    expected.update(_node_b.chunk_ids)
    expected.update(_edge_bc.evidence_chunk_ids)
    expected.update(_node_c.chunk_ids)

    assert ids <= expected
    assert ids == expected
    assert "chunk-d1" not in ids
    assert "chunk-outside-1" not in ids
