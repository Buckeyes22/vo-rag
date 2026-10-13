"""Contract tests for exported graph converters."""

from __future__ import annotations

from datetime import UTC, datetime

from llama_index.core.graph_stores.types import EntityNode, Relation

from vo_rag.core.graph.schemas import (
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    node_id_for,
)
from vo_rag.integrations.llamaindex import convert as convert_module
from vo_rag.integrations.llamaindex.convert import (
    graph_edge_to_relation,
    graph_node_to_entity_node,
)

_NOW = datetime.now(UTC)


def _make_graph_node() -> GraphNode:
    return GraphNode(
        node_id=node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="Slip Angle"),
        node_type=GraphNodeType.CONCEPT,
        label="Slip Angle",
        aliases=["sideslip"],
        book_ids=["book-1"],
        chunk_ids=["chunk-1"],
        attributes={"domain": "vehicle_dynamics"},
        created_at=_NOW,
    )


def _make_graph_edge() -> GraphEdge:
    source_id = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="Slip Angle")
    target_id = node_id_for(node_type=GraphNodeType.COMPONENT, canonical_label="Tire Model")
    return GraphEdge(
        edge_id=f"{source_id}-part-of-{target_id}",
        source_node_id=source_id,
        target_node_id=target_id,
        edge_type=GraphEdgeType.PART_OF,
        book_ids=["book-1"],
        weight=0.75,
        evidence_chunk_ids=["chunk-1", "chunk-2"],
        created_at=_NOW,
    )


def test_graph_node_to_entity_node_maps_working_graph_store_contract() -> None:
    node = _make_graph_node()

    entity = graph_node_to_entity_node(node)

    assert isinstance(entity, EntityNode)
    assert entity.name == node.label
    assert entity.label == node.node_type.value
    assert entity.properties["node_id"] == node.node_id
    assert entity.properties["aliases"] == node.aliases
    assert entity.properties["book_ids"] == node.book_ids
    assert entity.properties["chunk_ids"] == node.chunk_ids
    assert entity.properties["domain"] == "vehicle_dynamics"


def test_graph_edge_to_relation_maps_working_graph_store_contract() -> None:
    edge = _make_graph_edge()

    relation = graph_edge_to_relation(edge)

    assert isinstance(relation, Relation)
    assert relation.label == edge.edge_type.value
    assert relation.source_id == edge.source_node_id
    assert relation.target_id == edge.target_node_id
    assert relation.properties["edge_id"] == edge.edge_id
    assert relation.properties["weight"] == edge.weight
    assert relation.properties["evidence_chunk_ids"] == edge.evidence_chunk_ids
    assert relation.properties["book_ids"] == edge.book_ids


def test_graph_converters_are_in_module_all() -> None:
    assert "graph_node_to_entity_node" in convert_module.__all__
    assert "graph_edge_to_relation" in convert_module.__all__
