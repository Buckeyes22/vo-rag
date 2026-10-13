from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.graph.retrieval import (
    GraphRetrievalConfig,
    GraphTraversalResult,
    expand_neighborhood,
    graph_query_to_evidence_chunks,
    resolve_seed_nodes,
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
# Helpers
# ---------------------------------------------------------------------------


def _make_node(label: str = "Understeer") -> GraphNode:
    return GraphNode(
        node_id=node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label=label),
        node_type=GraphNodeType.CONCEPT,
        label=label,
        book_ids=["book-001"],
        chunk_ids=["c1"],
        created_at=datetime.now(UTC),
    )


def _make_edge(source_id: str, target_id: str) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id_for(
            source_node_id=source_id,
            target_node_id=target_id,
            edge_type=GraphEdgeType.RELATES_TO,
        ),
        source_node_id=source_id,
        target_node_id=target_id,
        edge_type=GraphEdgeType.RELATES_TO,
        book_ids=["book-1"],
        evidence_chunk_ids=["c1"],
        created_at=datetime.now(UTC),
    )


def _make_chunk(chunk_id: str, content: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type="text",
        document_id="doc-001",
        source_id="source-001",
        page_number=None,
        page_range=None,
        section_path="[Book > Chapter > Section]",
        section_title="Section",
        content=content,
        embedding_text=content,
        token_count=max(1, len(content.split())),
        has_image=False,
        metadata={},
    )


@dataclass
class _FakePoint:
    id: str
    payload: dict[str, Any]


@dataclass
class _FakeGraphClient:
    nodes: dict[str, GraphNode]
    edges: dict[str, GraphEdge]
    chunks: dict[str, RetrievalChunk]
    scroll_calls: list[str] = field(default_factory=list)

    async def retrieve(
        self,
        *,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
    ) -> list[_FakePoint]:
        if "graph_nodes" in collection_name:
            return [
                _FakePoint(id=node_id, payload=self.nodes[node_id].model_dump(mode="json"))
                for node_id in ids
                if node_id in self.nodes
            ]
        return []

    async def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 1000,
        with_payload: bool = True,
        offset: Any = None,
    ) -> tuple[list[_FakePoint], None]:
        self.scroll_calls.append(collection_name)
        if "graph_nodes" in collection_name:
            return (
                [
                    _FakePoint(id=node.node_id, payload=node.model_dump(mode="json"))
                    for node in self.nodes.values()
                ],
                None,
            )
        if "graph_edges" in collection_name:
            points = [
                _FakePoint(id=edge.edge_id, payload=edge.model_dump(mode="json"))
                for edge in self.edges.values()
            ]
            if scroll_filter is not None:
                points = [point for point in points if _matches_filter(point, scroll_filter)]
            return points, None
        return (
            [
                _FakePoint(id=chunk.chunk_id, payload=chunk.model_dump(mode="json"))
                for chunk in self.chunks.values()
            ],
            None,
        )


class _FakeGraphStore:
    def __init__(self, client: _FakeGraphClient) -> None:
        self._client = client


def _matches_filter(point: _FakePoint, scroll_filter: Any) -> bool:
    for condition in scroll_filter.must:
        actual = point.payload.get(condition.key)
        expected = condition.match.value
        if actual != expected:
            return False
    return True


def _fake_graph_store_with_candidates(chunk_ids: list[str]) -> _FakeGraphStore:
    seed = _make_node("anti-dive")
    neighbor = _make_node("dive geometry")
    edge = _make_edge(seed.node_id, neighbor.node_id)
    edge = edge.model_copy(update={"evidence_chunk_ids": chunk_ids})
    chunks = {
        chunk_id: _make_chunk(
            chunk_id,
            "anti dive relationship" if chunk_id == "c1" else "geometry linkage",
        )
        for chunk_id in chunk_ids
    }
    return _FakeGraphStore(
        _FakeGraphClient(
            nodes={seed.node_id: seed, neighbor.node_id: neighbor},
            edges={edge.edge_id: edge},
            chunks=chunks,
        )
    )


# ---------------------------------------------------------------------------
# GraphRetrievalConfig defaults
# ---------------------------------------------------------------------------


class TestGraphRetrievalConfigDefaults:
    def test_graph_retrieval_config_defaults(self) -> None:
        cfg = GraphRetrievalConfig()
        assert cfg.max_hops == 2
        assert cfg.max_neighbors_per_hop == 10
        assert cfg.edge_type_whitelist is None
        assert cfg.min_edge_weight == 0.0

    def test_graph_retrieval_config_custom_values(self) -> None:
        cfg = GraphRetrievalConfig(
            max_hops=3,
            max_neighbors_per_hop=5,
            edge_type_whitelist=[GraphEdgeType.RELATES_TO],
            min_edge_weight=1.5,
        )
        assert cfg.max_hops == 3
        assert cfg.max_neighbors_per_hop == 5
        assert cfg.edge_type_whitelist == [GraphEdgeType.RELATES_TO]
        assert cfg.min_edge_weight == 1.5


# ---------------------------------------------------------------------------
# GraphTraversalResult schema
# ---------------------------------------------------------------------------


class TestGraphTraversalResultSchema:
    def test_graph_traversal_result_schema_accepts_empty_visits(self) -> None:
        result = GraphTraversalResult(
            seed_node_ids=[],
            nodes_visited=[],
            edges_traversed=[],
            evidence_chunk_ids=[],
            hop_depth_per_node={},
        )
        assert result.seed_node_ids == []
        assert result.nodes_visited == []
        assert result.edges_traversed == []
        assert result.evidence_chunk_ids == []
        assert result.hop_depth_per_node == {}

    def test_graph_traversal_result_hop_depth_includes_seeds_at_depth_0(self) -> None:
        node = _make_node("Oversteer")
        seed_id = node.node_id
        result = GraphTraversalResult(
            seed_node_ids=[seed_id],
            nodes_visited=[node],
            edges_traversed=[],
            evidence_chunk_ids=["c1"],
            hop_depth_per_node={seed_id: 0},
        )
        assert result.hop_depth_per_node[seed_id] == 0

    def test_graph_traversal_result_evidence_chunk_ids_sorted(self) -> None:
        node_a = _make_node("Downforce")
        node_b = _make_node("Drag")
        edge = _make_edge(node_a.node_id, node_b.node_id)
        evidence = sorted(["c3", "c1", "c2"])
        result = GraphTraversalResult(
            seed_node_ids=[node_a.node_id],
            nodes_visited=[node_a, node_b],
            edges_traversed=[edge],
            evidence_chunk_ids=evidence,
            hop_depth_per_node={node_a.node_id: 0, node_b.node_id: 1},
        )
        assert result.evidence_chunk_ids == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# Callable signature checks (no live Qdrant)
# ---------------------------------------------------------------------------


class TestResolveSeedNodesSignature:
    def test_resolve_seed_nodes_signature(self) -> None:
        assert callable(resolve_seed_nodes)
        sig = inspect.signature(resolve_seed_nodes)
        assert "entity_labels" in sig.parameters
        assert "qdrant" in sig.parameters
        # Return annotation is a coroutine (async def)
        assert inspect.iscoroutinefunction(resolve_seed_nodes)


class TestExpandNeighborhoodSignature:
    def test_expand_neighborhood_signature(self) -> None:
        assert callable(expand_neighborhood)
        sig = inspect.signature(expand_neighborhood)
        assert "seed_node_ids" in sig.parameters
        assert "config" in sig.parameters
        assert "qdrant" in sig.parameters
        assert inspect.iscoroutinefunction(expand_neighborhood)


class TestGraphQueryToEvidenceChunks:
    @pytest.mark.asyncio
    async def test_graph_query_uses_supplied_chunk_ranker_for_final_evidence_order(self) -> None:
        calls: list[list[str]] = []

        async def rank_chunks(
            query_text: str,
            candidate_chunk_ids: list[str],
            top_k: int,
        ) -> list[str]:
            calls.append(candidate_chunk_ids)
            return list(reversed(candidate_chunk_ids))[:top_k]

        result = await graph_query_to_evidence_chunks(
            query_text="anti dive relationship",
            seed_labels=["anti-dive"],
            store=_fake_graph_store_with_candidates(["c1", "c2"]),
            rank_chunks=rank_chunks,
            top_k=2,
        )

        assert calls == [["c1", "c2"]]
        assert [chunk.chunk_id for chunk in result] == ["c2", "c1"]
        assert {chunk.metadata["graph_final_ranking_mode"] for chunk in result} == {
            "supplied_ranker"
        }

    @pytest.mark.asyncio
    async def test_graph_query_marks_token_overlap_fallback_when_no_ranker_supplied(self) -> None:
        result = await graph_query_to_evidence_chunks(
            query_text="anti dive relationship",
            seed_labels=["anti-dive"],
            store=_fake_graph_store_with_candidates(["c1", "c2"]),
            top_k=2,
        )

        assert [chunk.chunk_id for chunk in result] == ["c1", "c2"]
        assert {chunk.metadata["graph_final_ranking_mode"] for chunk in result} == {
            "token_overlap_fallback"
        }
        assert {chunk.metadata["graph_seed_resolution_mode"] for chunk in result} == {
            "keyword_scroll_fallback"
        }


async def test_graph_evidence_reads_actual_raw_writer_shape_without_embedding_duplication() -> None:
    from types import SimpleNamespace
    from typing import cast

    from vo_rag.core.graph.retrieval import _load_raw_chunks

    original = _make_chunk("raw-chunk", "own substantive evidence")
    payload = original.model_dump(mode="json")
    payload.pop("embedding_text")
    payload.update(
        {
            "stage": "raw",
            "trust_level": "internal",
            "generation_id": "generation",
            "ingest_state": "active",
            "colbert_fallback": True,
            "writer_contract_version": 1,
            "version_hash": "writer-hash",
        }
    )

    class Client:
        async def scroll(self, **kwargs: Any) -> Any:
            return [SimpleNamespace(payload=payload)], None

    store = SimpleNamespace(_client=Client())
    result = await _load_raw_chunks(
        store=cast(Any, store), raw_collection="raw", evidence_set={"raw-chunk"}
    )
    assert len(result) == 1
    assert result[0].content == original.content
    assert result[0].embedding_text == original.content
    assert result[0].source_id == original.source_id
