"""LlamaIndex PropertyGraphStore over vo-rag's GraphQdrantStore.

This module maps vo-rag's typed graph primitives (GraphNode, GraphEdge)
onto LlamaIndex's labelled-property-graph interfaces so that
PropertyGraphIndex can be built ``from_existing`` over the live Qdrant
collections.

Scope
-----
This adapter delivers the **read path** and converters for existing graph data.
Direct writes are intentionally unsupported: graph writes must go through
``VoRagPathExtractor`` and the graph build pipeline so extraction, orientation,
embedding, and provenance stay coupled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llama_index.core.graph_stores.types import (
    EntityNode,
    LabelledNode,
    PropertyGraphStore,
    Relation,
    Triplet,
)
from llama_index.core.vector_stores.types import VectorStoreQuery

from vo_rag.core.graph import persistence as _P
from vo_rag.core.graph.schemas import (
    GraphEdge,
    GraphNode,
    GraphNodeType,
    node_id_for,
)
from vo_rag.integrations.llamaindex._async import run_coro_sync


def _entity_to_graphnode(e: EntityNode) -> GraphNode:
    """Convert an LlamaIndex EntityNode to a vo-rag GraphNode.

    ``EntityNode.label`` carries the vo-rag ``node_type``;
    ``EntityNode.name`` is the human ``label``.
    The resulting ``node_id`` is deterministic via ``node_id_for``.
    """
    from datetime import UTC, datetime

    ntype = GraphNodeType(e.label.lower())
    props = dict(e.properties or {})
    stored_node_id = props.pop("node_id", None)
    nid = (
        stored_node_id
        if isinstance(stored_node_id, str)
        else node_id_for(node_type=ntype, canonical_label=e.name)
    )
    return GraphNode(
        node_id=nid,
        node_type=ntype,
        label=e.name,
        aliases=list(props.get("aliases", [])),
        book_ids=list(props.get("book_ids", [])),
        chunk_ids=list(props.get("chunk_ids", [])),
        attributes={
            k: str(v) for k, v in props.items() if k not in {"aliases", "book_ids", "chunk_ids"}
        },
        created_at=datetime.now(UTC),
    )


def _graphnode_to_entity(n: GraphNode) -> EntityNode:
    """Convert a vo-rag GraphNode to an LlamaIndex EntityNode.

    Round-trips through ``node_id_for`` so that
    ``_entity_to_graphnode(_graphnode_to_entity(n)).node_id == n.node_id``.
    """
    return EntityNode(
        name=n.label,
        label=n.node_type.value,
        properties={
            "node_id": n.node_id,
            "aliases": n.aliases,
            "book_ids": n.book_ids,
            "chunk_ids": n.chunk_ids,
            **n.attributes,
        },
    )


def _edge_to_relation(edge: GraphEdge) -> Relation:
    """Convert a vo-rag GraphEdge to an LlamaIndex Relation."""
    return Relation(
        label=edge.edge_type.value,
        source_id=edge.source_node_id,
        target_id=edge.target_node_id,
        properties={
            "edge_id": edge.edge_id,
            "weight": edge.weight,
            "evidence_chunk_ids": edge.evidence_chunk_ids,
            "book_ids": edge.book_ids,
        },
    )


class VoRagPropertyGraphStore(PropertyGraphStore):
    """LlamaIndex PropertyGraphStore over vo-rag's GraphQdrantStore."""

    supports_structured_queries: bool = False
    supports_vector_queries: bool = False

    def __init__(
        self,
        store: _P.GraphQdrantStore,
        nodes_collection: str = _P.GRAPH_NODES_COLLECTION,
        edges_collection: str = _P.GRAPH_EDGES_COLLECTION,
    ) -> None:
        self._store = store
        self._nodes_c = nodes_collection
        self._edges_c = edges_collection

    # ---- read path ----

    def get(
        self,
        properties: dict[str, Any] | None = None,
        ids: list[str] | None = None,
    ) -> list[LabelledNode]:
        """Get nodes by ID (properties filtering is not yet implemented)."""
        if properties is not None:
            raise NotImplementedError(
                "VoRagPropertyGraphStore.get(properties=...) is not supported; "
                "use VoRagPGRetriever for label/property lookup."
            )
        return run_coro_sync(
            lambda: self._aget(ids or []),
            component="VoRagPropertyGraphStore.get",
        )

    async def _aget(self, ids: list[str]) -> list[LabelledNode]:
        out: list[LabelledNode] = []
        seen: set[str] = set()
        for nid_or_name in ids:
            for candidate_id in self._node_id_candidates(nid_or_name):
                if candidate_id in seen:
                    continue
                n = await _P.get_node(candidate_id, store=self._store, collection=self._nodes_c)
                if n is not None:
                    seen.add(candidate_id)
                    out.append(_graphnode_to_entity(n))
                    break
        return out

    @staticmethod
    def _node_id_candidates(value: str) -> list[str]:
        return [
            value,
            *(
                node_id_for(node_type=node_type, canonical_label=value)
                for node_type in GraphNodeType
            ),
        ]

    def get_rel_map(
        self,
        graph_nodes: list[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: list[str] | None = None,
    ) -> list[Triplet]:
        return run_coro_sync(
            lambda: self._aget_rel_map(graph_nodes, depth, limit, ignore_rels),
            component="VoRagPropertyGraphStore.get_rel_map",
        )

    async def _aget_rel_map(
        self,
        graph_nodes: list[LabelledNode],
        depth: int,
        limit: int,
        ignore_rels: list[str] | None,
    ) -> list[Triplet]:
        triplets: list[Triplet] = []
        seen: set[str] = set()
        frontier: list[str] = []
        for g in graph_nodes:
            nid = (g.properties or {}).get("node_id")
            if isinstance(nid, str):
                frontier.append(nid)
        if graph_nodes and not frontier:
            raise NotImplementedError(
                "VoRagPropertyGraphStore.get_rel_map requires nodes with vo-rag node_id "
                "metadata; use VoRagPGRetriever for label lookup."
            )
        for _ in range(max(1, depth)):
            nxt: list[str] = []
            for nid in frontier:
                if nid in seen:
                    continue
                seen.add(nid)
                current = await _P.get_node(nid, store=self._store, collection=self._nodes_c)
                if current is None:
                    continue
                pairs = await _P.get_neighbors(
                    nid,
                    store=self._store,
                    direction="both",
                    nodes_collection=self._nodes_c,
                    edges_collection=self._edges_c,
                )
                for neighbor, edge in pairs:
                    source = await _P.get_node(
                        edge.source_node_id,
                        store=self._store,
                        collection=self._nodes_c,
                    )
                    target = await _P.get_node(
                        edge.target_node_id,
                        store=self._store,
                        collection=self._nodes_c,
                    )
                    if source is None or target is None:
                        continue
                    triplets.append(
                        (
                            _graphnode_to_entity(source),
                            _edge_to_relation(edge),
                            _graphnode_to_entity(target),
                        )
                    )
                    nxt.append(neighbor.node_id)
            frontier = nxt
        if ignore_rels:
            triplets = [t for t in triplets if t[1].label not in ignore_rels]
        return triplets[:limit]

    def get_triplets(
        self,
        entity_names: list[str] | None = None,
        relation_names: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        ids: list[str] | None = None,
    ) -> list[Triplet]:
        if not ids and not properties and not entity_names and not relation_names:
            return []
        if properties is not None:
            raise NotImplementedError(
                "VoRagPropertyGraphStore.get_triplets(properties=...) is not supported; "
                "use VoRagPGRetriever for property lookup."
            )
        if entity_names:
            raise NotImplementedError(
                "VoRagPropertyGraphStore.get_triplets(entity_names=...) is not supported; "
                "use VoRagPGRetriever for label lookup."
            )
        return run_coro_sync(
            lambda: self._aget_triplets(ids, relation_names),
            component="VoRagPropertyGraphStore.get_triplets",
        )

    async def _aget_triplets(
        self,
        ids: list[str] | None,
        relation_names: list[str] | None,
    ) -> list[Triplet]:
        triplets: list[Triplet] = []
        if not ids:
            return triplets
        for nid in ids:
            current = await _P.get_node(nid, store=self._store, collection=self._nodes_c)
            if current is None:
                continue
            pairs = await _P.get_neighbors(
                nid,
                store=self._store,
                direction="both",
                nodes_collection=self._nodes_c,
                edges_collection=self._edges_c,
            )
            for _neighbor, edge in pairs:
                if relation_names and edge.edge_type.value not in relation_names:
                    continue
                # Build the triplet from the stored edge endpoints, not the
                # traversal seed (`nid`): for an incoming edge the seed is the
                # target, so (src, edge, neighbor) would invert direction.
                source = await _P.get_node(
                    edge.source_node_id,
                    store=self._store,
                    collection=self._nodes_c,
                )
                target = await _P.get_node(
                    edge.target_node_id,
                    store=self._store,
                    collection=self._nodes_c,
                )
                if source is None or target is None:
                    continue
                triplets.append(
                    (
                        _graphnode_to_entity(source),
                        _edge_to_relation(edge),
                        _graphnode_to_entity(target),
                    )
                )
        return triplets

    # ---- unsupported direct writes ----

    def upsert_nodes(self, nodes: Sequence[LabelledNode]) -> None:
        raise NotImplementedError(
            "Direct node upsert bypasses embedding + provenance; "
            "use VoRagPathExtractor + the graph build pipeline instead (see notes)."
        )

    def upsert_relations(self, relations: list[Relation]) -> None:
        raise NotImplementedError("See upsert_nodes — graph writes go through the build pipeline.")

    # ---- unsupported operations (outside VOR-211 scope) ----

    def delete(
        self,
        entity_names: list[str] | None = None,
        relation_names: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        raise NotImplementedError(
            "Delete is not supported on VoRagPropertyGraphStore; "
            "graph mutations go through the graph build pipeline."
        )

    def structured_query(self, query: str, param_map: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError(
            "Structured (Cypher) query is not supported on VoRagPropertyGraphStore; "
            "use the BFS traversal exposed by get_rel_map / VoRagPGRetriever."
        )

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> tuple[list[LabelledNode], list[float]]:
        raise NotImplementedError(
            "Vector query is not supported on VoRagPropertyGraphStore; "
            "use VoRagPGRetriever for graph traversal, the retrieval layer for vector search, "
            "or the graph build pipeline for writes."
        )


__all__ = [
    "_entity_to_graphnode",
    "_graphnode_to_entity",
    "_edge_to_relation",
    "VoRagPropertyGraphStore",
]
