from __future__ import annotations

from datetime import UTC, datetime

import pytest

from vo_rag.core.graph.schemas import (
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    GraphSmokeTestConfig,
    edge_id_for,
    node_id_for,
)


def _make_node(
    *,
    node_type: GraphNodeType = GraphNodeType.CONCEPT,
    label: str = "understeer",
    book_ids: list[str] | None = None,
    chunk_ids: list[str] | None = None,
) -> GraphNode:
    return GraphNode(
        node_id=node_id_for(node_type=node_type, canonical_label=label),
        node_type=node_type,
        label=label,
        book_ids=book_ids or ["book-001"],
        chunk_ids=chunk_ids or [],
        created_at=datetime.now(UTC),
    )


def _make_edge(
    *,
    source_node_id: str = "aaa",
    target_node_id: str = "bbb",
    edge_type: GraphEdgeType = GraphEdgeType.RELATES_TO,
    evidence_chunk_ids: list[str] | None = None,
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
        evidence_chunk_ids=evidence_chunk_ids or [],
        created_at=datetime.now(UTC),
    )


class TestNodeIdFor:
    def test_node_id_for_is_stable_across_repeated_calls(self) -> None:
        calls = [
            node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="Brake Bias")
            for _ in range(5)
        ]
        assert calls == [calls[0]] * 5

    def test_node_id_for_is_deterministic_per_label_and_type(self) -> None:
        id_a = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="oversteer")
        id_b = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="oversteer")
        assert id_a == id_b

    def test_node_id_for_distinguishes_types_with_same_label(self) -> None:
        id_concept = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="understeer")
        id_method = node_id_for(node_type=GraphNodeType.METHOD, canonical_label="understeer")
        assert id_concept != id_method

    def test_node_id_for_is_32_hex_chars(self) -> None:
        nid = node_id_for(node_type=GraphNodeType.PERSON, canonical_label="Ayrton Senna")
        assert len(nid) == 32
        assert all(c in "0123456789abcdef" for c in nid)

    def test_node_id_for_case_insensitive_normalisation(self) -> None:
        # label is lowercased inside node_id_for
        id_upper = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="Downforce")
        id_lower = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="downforce")
        assert id_upper == id_lower


class TestEdgeIdFor:
    def test_edge_id_for_is_stable_across_repeated_calls(self) -> None:
        calls = [
            edge_id_for(
                source_node_id="src-001",
                target_node_id="tgt-002",
                edge_type=GraphEdgeType.RELATES_TO,
            )
            for _ in range(5)
        ]
        assert calls == [calls[0]] * 5

    def test_edge_id_for_is_deterministic(self) -> None:
        eid_a = edge_id_for(
            source_node_id="src-001",
            target_node_id="tgt-002",
            edge_type=GraphEdgeType.RELATES_TO,
        )
        eid_b = edge_id_for(
            source_node_id="src-001",
            target_node_id="tgt-002",
            edge_type=GraphEdgeType.RELATES_TO,
        )
        assert eid_a == eid_b

    def test_edge_id_for_directed_a_to_b_differs_from_b_to_a(self) -> None:
        eid_ab = edge_id_for(
            source_node_id="src-001",
            target_node_id="tgt-002",
            edge_type=GraphEdgeType.RELATES_TO,
        )
        eid_ba = edge_id_for(
            source_node_id="tgt-002",
            target_node_id="src-001",
            edge_type=GraphEdgeType.RELATES_TO,
        )
        assert eid_ab != eid_ba

    def test_edge_id_for_is_32_hex_chars(self) -> None:
        eid = edge_id_for(
            source_node_id="src-001",
            target_node_id="tgt-002",
            edge_type=GraphEdgeType.PART_OF,
        )
        assert len(eid) == 32
        assert all(c in "0123456789abcdef" for c in eid)


class TestGraphEdgeType:
    def test_is_a_edge_type_is_available_for_taxonomy_edges(self) -> None:
        assert GraphEdgeType.IS_A.value == "is_a"
        assert "is_a" in [t.value for t in GraphEdgeType]


class TestGraphNodeValidation:
    def test_graph_node_validates_with_empty_chunk_ids(self) -> None:
        node = _make_node(chunk_ids=[])
        assert node.chunk_ids == []

    def test_graph_node_validates_with_populated_chunk_ids(self) -> None:
        node = _make_node(chunk_ids=["c1", "c2", "c3"])
        assert node.chunk_ids == ["c1", "c2", "c3"]

    def test_graph_node_aliases_defaults_to_empty_list(self) -> None:
        node = _make_node()
        assert node.aliases == []

    def test_graph_node_attributes_defaults_to_empty_dict(self) -> None:
        node = _make_node()
        assert node.attributes == {}


class TestGraphEdgeDefaults:
    def test_graph_edge_default_weight_is_1(self) -> None:
        edge = _make_edge()
        assert edge.weight == 1.0

    def test_graph_edge_prompt_hash_defaults_to_none(self) -> None:
        edge = _make_edge()
        assert edge.prompt_hash is None


class TestGraphSmokeTestConfigDefaults:
    def test_graph_smoke_test_config_defaults(self) -> None:
        cfg = GraphSmokeTestConfig()
        assert cfg.min_node_count_pilot == 100
        assert cfg.min_edge_count_pilot == 200
        assert cfg.judge_min_improvement_pp == pytest.approx(1.0)
        assert "recall_at_10" in cfg.metrics_to_test
        assert "ndcg_at_10" in cfg.metrics_to_test
        assert "answer_quality_judge" in cfg.metrics_to_test
