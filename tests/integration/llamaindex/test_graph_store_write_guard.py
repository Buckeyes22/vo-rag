"""Ingest-write guard contract (VOR-212).

The query path is READ-ONLY: every write entry point on the
``PropertyGraphStore`` surface must raise ``NotImplementedError`` so a caller
never believes a partial upsert happened (which would skip BGE-M3 embedding
+ provenance population and silently corrupt recall).

The ``upsert_nodes``/``upsert_relations`` raise-block ("Writes intentionally
raise") is the behavior pinned here.

This test is the unit of truth for VOR-212; the read-path coverage
lives in ``test_property_graph_store.py`` (VOR-211).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.graph_stores.types import (
    EntityNode,
    LabelledNode,
    PropertyGraphStore,
    Relation,
)
from llama_index.core.vector_stores.types import VectorStoreQuery

from vo_rag.integrations.llamaindex.graph_store import VoRagPropertyGraphStore


def _fake_store() -> Any:
    """Return a fake GraphQdrantStore whose _client is an AsyncMock.

    The guard must fire before any I/O — these methods MUST NOT touch the
    client, but the constructor needs a duck-typed ``_client`` attribute
    so we use the same shape as the VOR-211 read-path tests.
    """
    store = MagicMock()
    store._client = AsyncMock()
    return store


def _entity(name: str = "Slip Angle", label: str = "concept") -> EntityNode:
    return EntityNode(name=name, label=label, properties={"node_id": f"id-{name}"})


def _relation(source_id: str = "a", target_id: str = "b") -> Relation:
    return Relation(label="PART_OF", source_id=source_id, target_id=target_id, properties={})


class TestInstantiability:
    """The guard must make the class concrete, not broken.

    If any abstract method on the installed ``PropertyGraphStore`` ABC is
    left unimplemented, instantiation raises ``TypeError`` (Python's ABC
    machinery). The guard is only doing its job if the class is both
    instantiable AND unable to perform a silent write.
    """

    def test_store_is_instantiable_with_fake_qdrant(self) -> None:
        pg = VoRagPropertyGraphStore(store=_fake_store())
        assert isinstance(pg, PropertyGraphStore)

    def test_store_has_no_unfulfilled_abstract_methods(self) -> None:
        pg = VoRagPropertyGraphStore(store=_fake_store())
        # If any ABC method is missing, Python populates __abstractmethods__.
        # The guard's job is to keep that set empty so ``PropertyGraphIndex``
        # can wrap the store without an ABC ``TypeError``.
        assert not getattr(pg, "__abstractmethods__", set())


class TestWriteGuard:
    """``upsert_nodes`` / ``upsert_relations`` MUST raise.

    The error message must name the supported write path
    (``VoRagPathExtractor`` / ``build pipeline``) so a future maintainer is
    routed to the supported surface, not left to invent one.
    """

    def test_upsert_nodes_raises_not_implemented(self) -> None:
        pg = VoRagPropertyGraphStore(store=_fake_store())
        with pytest.raises(NotImplementedError) as excinfo:
            pg.upsert_nodes([_entity()])
        msg = str(excinfo.value)
        assert "VoRagPathExtractor" in msg
        assert "build pipeline" in msg

    def test_upsert_nodes_raises_with_empty_list(self) -> None:
        """An empty list is still a write entry point — must still raise."""
        pg = VoRagPropertyGraphStore(store=_fake_store())
        with pytest.raises(NotImplementedError):
            pg.upsert_nodes([])

    def test_upsert_nodes_raises_with_multiple_entities(self) -> None:
        pg = VoRagPropertyGraphStore(store=_fake_store())
        nodes: Sequence[LabelledNode] = [
            _entity("Slip Angle"),
            _entity("Tire Model", "component"),
        ]
        with pytest.raises(NotImplementedError):
            pg.upsert_nodes(nodes)

    def test_upsert_nodes_does_not_touch_qdrant(self) -> None:
        """The guard must fire BEFORE any client call — a no-op would be a
        silent write. If the client's ``upsert`` is called, recall is at risk."""
        store = _fake_store()
        pg = VoRagPropertyGraphStore(store=store)
        with pytest.raises(NotImplementedError):
            pg.upsert_nodes([_entity()])
        store._client.upsert.assert_not_called()
        store._client.scroll.assert_not_called()

    def test_upsert_relations_raises_not_implemented(self) -> None:
        pg = VoRagPropertyGraphStore(store=_fake_store())
        with pytest.raises(NotImplementedError) as excinfo:
            pg.upsert_relations([_relation()])
        # Message references the supported write path (the build pipeline)
        # so the maintainer is routed to the same place as upsert_nodes.
        assert "build pipeline" in str(excinfo.value)

    def test_upsert_relations_raises_with_empty_list(self) -> None:
        pg = VoRagPropertyGraphStore(store=_fake_store())
        with pytest.raises(NotImplementedError):
            pg.upsert_relations([])

    def test_upsert_relations_does_not_touch_qdrant(self) -> None:
        store = _fake_store()
        pg = VoRagPropertyGraphStore(store=store)
        with pytest.raises(NotImplementedError):
            pg.upsert_relations([_relation()])
        store._client.upsert.assert_not_called()
        store._client.scroll.assert_not_called()


class TestOtherAbstractMethods:
    """Other ABC methods that would otherwise leave the class un-instantiable.

    The installed ``llama_index.core.graph_stores.types.PropertyGraphStore``
    ABC also declares ``delete``, ``structured_query``, and ``vector_query``
    as abstract. Each must raise ``NotImplementedError`` so a caller cannot
    mistake a no-op for success, and each message must name the supported
    surface so a future maintainer is routed correctly.
    """

    def test_delete_raises_and_names_supported_path(self) -> None:
        store = _fake_store()
        pg = VoRagPropertyGraphStore(store=store)
        with pytest.raises(NotImplementedError) as excinfo:
            pg.delete(ids=["id-1"])
        assert "build pipeline" in str(excinfo.value)
        store._client.delete.assert_not_called()

    def test_structured_query_raises_and_routes_to_bfs(self) -> None:
        store = _fake_store()
        pg = VoRagPropertyGraphStore(store=store)
        with pytest.raises(NotImplementedError) as excinfo:
            pg.structured_query("MATCH (n) RETURN n")
        msg = str(excinfo.value)
        # No Cypher — the supported query surface is BFS via get_rel_map /
        # VoRagPGRetriever. Either name counts; we accept either to keep the
        # test stable if the message is reworded.
        assert "get_rel_map" in msg or "VoRagPGRetriever" in msg

    def test_vector_query_raises_and_names_supported_path(self) -> None:
        store = _fake_store()
        pg = VoRagPropertyGraphStore(store=store)
        query = MagicMock(spec=VectorStoreQuery)
        with pytest.raises(NotImplementedError) as excinfo:
            pg.vector_query(query)
        assert "build pipeline" in str(excinfo.value)

    def test_store_does_not_advertise_vector_queries(self) -> None:
        pg = VoRagPropertyGraphStore(store=_fake_store())
        assert pg.supports_vector_queries is False


class TestPublicSurface:
    """``VoRagPropertyGraphStore`` stays exported in ``__all__``."""

    def test_store_in_all(self) -> None:
        import vo_rag.integrations.llamaindex.graph_store as mod

        assert "VoRagPropertyGraphStore" in mod.__all__
