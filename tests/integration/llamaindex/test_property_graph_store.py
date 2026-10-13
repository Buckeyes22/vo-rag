"""Contract tests for VoRagPropertyGraphStore (VOR-211).

This module tests the read-path converters and the
PropertyGraphStore wrapper over a fake Qdrant store.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.graph_stores.types import EntityNode

from vo_rag.core.graph.schemas import (
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    node_id_for,
)
from vo_rag.integrations.llamaindex.graph_store import (
    VoRagPropertyGraphStore,
    _edge_to_relation,
    _entity_to_graphnode,
    _graphnode_to_entity,
)

_NOW = datetime.now(UTC)


def _make_graph_node(**overrides: Any) -> GraphNode:
    defaults = {
        "node_id": node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label="Slip Angle"),
        "node_type": GraphNodeType.CONCEPT,
        "label": "Slip Angle",
        "aliases": ["sideslip"],
        "book_ids": ["b1"],
        "chunk_ids": ["c1"],
        "attributes": {"domain": "vehicle_dynamics"},
        "created_at": _NOW,
    }
    defaults.update(overrides)
    return GraphNode.model_validate(defaults)


def _make_graph_edge(**overrides: Any) -> GraphEdge:
    defaults = {
        "edge_id": "edge-123",
        "source_node_id": node_id_for(
            node_type=GraphNodeType.CONCEPT, canonical_label="Slip Angle"
        ),
        "target_node_id": node_id_for(
            node_type=GraphNodeType.COMPONENT, canonical_label="Tire Model"
        ),
        "edge_type": GraphEdgeType.PART_OF,
        "book_ids": ["b1"],
        "weight": 1.0,
        "evidence_chunk_ids": ["c1"],
        "created_at": _NOW,
    }
    defaults.update(overrides)
    return GraphEdge.model_validate(defaults)


def _fake_store() -> Any:
    """Return a fake GraphQdrantStore whose _client is an AsyncMock."""
    store = MagicMock()
    store._client = AsyncMock()
    return store


def _retrieve_by_id(*nodes: GraphNode) -> Any:
    """Build a ``retrieve`` side-effect that resolves points by requested id.

    The relation-map direction fix loads both stored edge endpoints per edge,
    so the number of ``get_node`` calls is no longer fixed. Resolving by id
    keeps these tests robust to that call count instead of a brittle list keyed
    on exact call order.
    """
    by_id = {node.node_id: node for node in nodes}

    def _retrieve(*, collection_name: str, ids: list[str], with_payload: bool) -> list[Any]:
        points: list[Any] = []
        for node_id in ids:
            node = by_id.get(node_id)
            if node is None:
                continue
            point = MagicMock()
            point.id = node.node_id
            point.payload = node.model_dump(mode="json")
            points.append(point)
        return points

    return _retrieve


def _scroll_edges(*edges: GraphEdge) -> Any:
    """Build a filter-aware ``scroll`` side-effect over the supplied edges.

    ``get_neighbors(direction="both")`` scrolls the source side and the target
    side separately; a static ``return_value`` would hand back every edge for
    both passes, double-counting neighbors. Honor ``scroll_filter.must`` so each
    pass sees only its matching edges, exactly as the real store does.
    """

    def _scroll(
        *,
        collection_name: str,
        scroll_filter: Any,
        limit: int,
        with_payload: bool,
        offset: Any,
    ) -> tuple[list[Any], None]:
        points: list[Any] = []
        for edge in edges:
            if all(
                getattr(edge, condition.key) == condition.match.value
                for condition in scroll_filter.must
            ):
                point = MagicMock()
                point.id = edge.edge_id
                point.payload = edge.model_dump(mode="json")
                points.append(point)
        return (points, None)

    return _scroll


class TestConverters:
    def test_graphnode_to_entity_label_and_name(self) -> None:
        n = _make_graph_node()
        e = _graphnode_to_entity(n)
        assert e.label == n.node_type.value
        assert e.name == n.label

    def test_entity_roundtrip_preserves_node_id(self) -> None:
        n = _make_graph_node()
        e = _graphnode_to_entity(n)
        n2 = _entity_to_graphnode(e)
        assert n2.node_id == n.node_id

    def test_entity_to_graphnode_unknown_label_raises(self) -> None:
        e = EntityNode(name="Foo", label="UNKNOWN_TYPE")
        with pytest.raises(ValueError):
            _entity_to_graphnode(e)

    def test_edge_to_relation_fields(self) -> None:
        edge = _make_graph_edge()
        r = _edge_to_relation(edge)
        assert r.label == edge.edge_type.value
        assert r.source_id == edge.source_node_id
        assert r.target_id == edge.target_node_id
        assert r.properties["edge_id"] == edge.edge_id
        assert r.properties["weight"] == edge.weight


class TestGet:
    def test_get_returns_matching_entities(self) -> None:
        store = _fake_store()
        n = _make_graph_node()
        # Fake _client.retrieve returning a Qdrant point with the node payload
        fake_point = MagicMock()
        fake_point.payload = n.model_dump(mode="json")
        store._client.retrieve.return_value = [fake_point]

        pg = VoRagPropertyGraphStore(store=store)
        results = pg.get(ids=[n.node_id])

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, EntityNode)
        assert result.name == n.label
        assert result.label == n.node_type.value
        store._client.retrieve.assert_awaited_once_with(
            collection_name=pg._nodes_c,
            ids=[n.node_id],
            with_payload=True,
        )

    def test_get_missing_node_returns_empty(self) -> None:
        store = _fake_store()
        store._client.retrieve.return_value = []

        pg = VoRagPropertyGraphStore(store=store)
        results = pg.get(ids=["nonexistent"])
        assert results == []


class TestGetRelMap:
    def test_get_rel_map_depth_one(self) -> None:
        store = _fake_store()
        src = _make_graph_node(label="Slip Angle")
        neighbor = _make_graph_node(
            node_type=GraphNodeType.COMPONENT,
            label="Tire Model",
            node_id=node_id_for(node_type=GraphNodeType.COMPONENT, canonical_label="Tire Model"),
        )
        edge = _make_graph_edge(
            source_node_id=src.node_id,
            target_node_id=neighbor.node_id,
        )

        store._client.retrieve.side_effect = _retrieve_by_id(src, neighbor)

        # get_edges_from / get_edges_to use scroll to find edges
        store._client.scroll.side_effect = _scroll_edges(edge)

        pg = VoRagPropertyGraphStore(store=store)
        src_entity = _graphnode_to_entity(src)
        triplets = pg.get_rel_map([src_entity], depth=1)

        assert len(triplets) == 1
        t_src, t_rel, t_neighbor = triplets[0]
        assert isinstance(t_src, EntityNode)
        assert isinstance(t_neighbor, EntityNode)
        assert t_src.name == src.label
        assert t_neighbor.name == neighbor.label
        assert t_rel.label == edge.edge_type.value

    def test_get_rel_map_ignore_rels(self) -> None:
        store = _fake_store()
        src = _make_graph_node(label="Slip Angle")
        neighbor = _make_graph_node(
            node_type=GraphNodeType.COMPONENT,
            label="Tire Model",
            node_id=node_id_for(node_type=GraphNodeType.COMPONENT, canonical_label="Tire Model"),
        )
        edge = _make_graph_edge(
            source_node_id=src.node_id,
            target_node_id=neighbor.node_id,
            edge_type=GraphEdgeType.PART_OF,
        )

        store._client.retrieve.side_effect = _retrieve_by_id(src, neighbor)
        store._client.scroll.side_effect = _scroll_edges(edge)

        pg = VoRagPropertyGraphStore(store=store)
        src_entity = _graphnode_to_entity(src)
        triplets = pg.get_rel_map([src_entity], depth=1, ignore_rels=[GraphEdgeType.PART_OF.value])
        assert triplets == []

    def test_get_rel_map_respects_limit(self) -> None:
        store = _fake_store()
        src = _make_graph_node(label="Slip Angle")
        neighbor = _make_graph_node(
            node_type=GraphNodeType.COMPONENT,
            label="Tire Model",
            node_id=node_id_for(node_type=GraphNodeType.COMPONENT, canonical_label="Tire Model"),
        )
        edge = _make_graph_edge(
            source_node_id=src.node_id,
            target_node_id=neighbor.node_id,
        )

        store._client.retrieve.side_effect = _retrieve_by_id(src, neighbor)
        store._client.scroll.side_effect = _scroll_edges(edge)

        pg = VoRagPropertyGraphStore(store=store)
        src_entity = _graphnode_to_entity(src)
        triplets = pg.get_rel_map([src_entity], depth=1, limit=0)
        assert triplets == []


class TestWriteStubs:
    def test_upsert_nodes_raises(self) -> None:
        pg = VoRagPropertyGraphStore(store=_fake_store())
        with pytest.raises(NotImplementedError):
            pg.upsert_nodes([])

    def test_upsert_relations_raises(self) -> None:
        pg = VoRagPropertyGraphStore(store=_fake_store())
        with pytest.raises(NotImplementedError):
            pg.upsert_relations([])
