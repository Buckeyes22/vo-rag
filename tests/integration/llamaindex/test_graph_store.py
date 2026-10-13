"""Regression tests for VoRagPropertyGraphStore relation-map direction."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from llama_index.core.graph_stores.types import EntityNode, Triplet
from llama_index.core.llms import CompletionResponse
from llama_index.core.llms.mock import MockLLM

from vo_rag.core.graph.persistence import GRAPH_EDGES_COLLECTION, GRAPH_NODES_COLLECTION
from vo_rag.core.graph.schemas import (
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    node_id_for,
)
from vo_rag.integrations.llamaindex.graph_store import (
    VoRagPropertyGraphStore,
    _entity_to_graphnode,
    _graphnode_to_entity,
)
from vo_rag.integrations.llamaindex.property_graph_index import (
    build_vo_rag_property_graph_index,
)

_NOW = datetime.now(UTC)


class _Point:
    def __init__(self, point_id: str, payload: dict[str, Any]) -> None:
        self.id = point_id
        self.payload = payload


class _MemoryGraphClient:
    def __init__(self, *, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
        self._nodes = {node.node_id: node for node in nodes}
        self._edges = edges

    async def retrieve(
        self,
        *,
        collection_name: str,
        ids: list[str],
        with_payload: bool,
    ) -> list[_Point]:
        assert collection_name == GRAPH_NODES_COLLECTION
        assert with_payload is True
        return [
            _Point(node.node_id, node.model_dump(mode="json"))
            for node_id in ids
            if (node := self._nodes.get(node_id)) is not None
        ]

    async def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: Any,
        limit: int,
        with_payload: bool,
        offset: Any,
    ) -> tuple[list[_Point], None]:
        assert collection_name == GRAPH_EDGES_COLLECTION
        assert limit == 1000
        assert with_payload is True
        assert offset is None
        edges = [
            edge
            for edge in self._edges
            if all(
                getattr(edge, condition.key) == condition.match.value
                for condition in scroll_filter.must
            )
        ]
        return (
            [_Point(edge.edge_id, edge.model_dump(mode="json")) for edge in edges],
            None,
        )


class _MemoryGraphStore:
    def __init__(self, *, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
        self._client = _MemoryGraphClient(nodes=nodes, edges=edges)


class _SynonymLLM(MockLLM):
    response: str = "parent"

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self.response)


def make_graph_node(node_id: str, *, label: str) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        node_type=GraphNodeType.CONCEPT,
        label=label,
        book_ids=["book-1"],
        chunk_ids=["chunk-1"],
        created_at=_NOW,
    )


def make_graph_edge(
    *,
    source_node_id: str,
    target_node_id: str,
    edge_type: GraphEdgeType = GraphEdgeType.PART_OF,
) -> GraphEdge:
    return GraphEdge(
        edge_id=f"{source_node_id}-{edge_type.value}-{target_node_id}",
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        book_ids=["book-1"],
        evidence_chunk_ids=["chunk-1"],
        created_at=_NOW,
    )


def persist_graph(
    graph_store: VoRagPropertyGraphStore,
    *,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> None:
    graph_store._store = _MemoryGraphStore(nodes=nodes, edges=edges)


def _node_id_pairs(triplets: list[Triplet]) -> list[tuple[str, str]]:
    return [
        (triplet[0].properties["node_id"], triplet[2].properties["node_id"]) for triplet in triplets
    ]


def _relation_id_pairs(triplets: list[Triplet]) -> list[tuple[str, str]]:
    return [(triplet[1].source_id, triplet[1].target_id) for triplet in triplets]


def test_get_rel_map_preserves_incoming_edge_direction() -> None:
    graph_store = VoRagPropertyGraphStore(store=_MemoryGraphStore(nodes=[], edges=[]))
    a = make_graph_node("A", label="Parent")
    b = make_graph_node("B", label="Child")
    edge = make_graph_edge(source_node_id="A", target_node_id="B")
    persist_graph(graph_store, nodes=[a, b], edges=[edge])

    triplets = graph_store.get_rel_map([_graphnode_to_entity(b)], depth=1)

    assert _node_id_pairs(triplets) == [("A", "B")]
    assert _relation_id_pairs(triplets) == [("A", "B")]


def test_get_rel_map_preserves_outgoing_edge_direction() -> None:
    graph_store = VoRagPropertyGraphStore(store=_MemoryGraphStore(nodes=[], edges=[]))
    a = make_graph_node("A", label="Parent")
    b = make_graph_node("B", label="Child")
    edge = make_graph_edge(source_node_id="A", target_node_id="B")
    persist_graph(graph_store, nodes=[a, b], edges=[edge])

    triplets = graph_store.get_rel_map([_graphnode_to_entity(a)], depth=1)

    assert _node_id_pairs(triplets) == [("A", "B")]
    assert _relation_id_pairs(triplets) == [("A", "B")]


def test_get_triplets_preserves_incoming_edge_direction() -> None:
    # get_triplets has the same seed-vs-endpoint hazard as get_rel_map: seeding
    # from the target of A -> B must still return A -> B, not B -> A.
    graph_store = VoRagPropertyGraphStore(store=_MemoryGraphStore(nodes=[], edges=[]))
    a = make_graph_node("A", label="Parent")
    b = make_graph_node("B", label="Child")
    edge = make_graph_edge(source_node_id="A", target_node_id="B")
    persist_graph(graph_store, nodes=[a, b], edges=[edge])

    triplets = graph_store.get_triplets(ids=["B"])

    assert _node_id_pairs(triplets) == [("A", "B")]
    assert _relation_id_pairs(triplets) == [("A", "B")]


def test_get_triplets_preserves_outgoing_edge_direction() -> None:
    graph_store = VoRagPropertyGraphStore(store=_MemoryGraphStore(nodes=[], edges=[]))
    a = make_graph_node("A", label="Parent")
    b = make_graph_node("B", label="Child")
    edge = make_graph_edge(source_node_id="A", target_node_id="B")
    persist_graph(graph_store, nodes=[a, b], edges=[edge])

    triplets = graph_store.get_triplets(ids=["A"])

    assert _node_id_pairs(triplets) == [("A", "B")]
    assert _relation_id_pairs(triplets) == [("A", "B")]


def test_get_rejects_unsupported_properties_filter() -> None:
    graph_store = VoRagPropertyGraphStore(store=_MemoryGraphStore(nodes=[], edges=[]))

    with pytest.raises(NotImplementedError, match="properties"):
        graph_store.get(properties={"label": "Parent"})


def test_get_resolves_framework_entity_names_to_vo_rag_node_ids() -> None:
    graph_store = VoRagPropertyGraphStore(store=_MemoryGraphStore(nodes=[], edges=[]))
    parent_id = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="Parent")
    parent = make_graph_node(parent_id, label="Parent")
    persist_graph(graph_store, nodes=[parent], edges=[])

    nodes = graph_store.get(ids=["Parent"])

    assert [node.properties["node_id"] for node in nodes] == [parent_id]


def test_get_triplets_rejects_unsupported_entity_name_filter() -> None:
    graph_store = VoRagPropertyGraphStore(store=_MemoryGraphStore(nodes=[], edges=[]))

    with pytest.raises(NotImplementedError, match="entity_names"):
        graph_store.get_triplets(entity_names=["Parent"])


def test_get_triplets_rejects_unsupported_properties_filter() -> None:
    graph_store = VoRagPropertyGraphStore(store=_MemoryGraphStore(nodes=[], edges=[]))

    with pytest.raises(NotImplementedError, match="properties"):
        graph_store.get_triplets(properties={"label": "Parent"})


def test_get_rel_map_rejects_seed_nodes_without_vo_rag_node_ids() -> None:
    graph_store = VoRagPropertyGraphStore(store=_MemoryGraphStore(nodes=[], edges=[]))
    framework_node = EntityNode(name="Parent", label=GraphNodeType.CONCEPT.value)

    with pytest.raises(NotImplementedError, match="node_id"):
        graph_store.get_rel_map([framework_node], depth=1)


def test_default_framework_retriever_resolves_entity_names_to_vo_rag_node_ids() -> None:
    parent_id = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="Parent")
    child_id = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="Child")
    assert parent_id != "Parent"

    parent = make_graph_node(parent_id, label="Parent")
    child = make_graph_node(child_id, label="Child")
    edge = make_graph_edge(source_node_id=parent_id, target_node_id=child_id)
    store = _MemoryGraphStore(nodes=[parent, child], edges=[edge])
    index = build_vo_rag_property_graph_index(
        store=store,
        kg_extractors=[],
        embed_kg_nodes=False,
        llm=_SynonymLLM(max_tokens=None),
        use_async=False,
    )

    retriever = index.as_retriever(include_text=False, max_keywords=1, path_depth=1)
    results = retriever.retrieve("parent")

    assert [result.node.get_content() for result in results] == [
        f"Parent -> {GraphEdgeType.PART_OF.value} -> Child"
    ]


def test_entity_to_graphnode_preserves_stored_node_id_without_attribute_leak() -> None:
    entity = EntityNode(
        name="Parent",
        label=GraphNodeType.CONCEPT.value,
        properties={
            "node_id": "legacy-node-id",
            "aliases": ["P"],
            "book_ids": ["book-1"],
            "chunk_ids": ["chunk-1"],
            "custom": "kept",
        },
    )

    node = _entity_to_graphnode(entity)

    assert node.node_id == "legacy-node-id"
    assert node.attributes == {"custom": "kept"}
