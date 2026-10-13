"""Contract tests for the PropertyGraphIndex assembly factory (VOR-217).

Wires ``VoRagPropertyGraphStore`` + ``VoRagPGRetriever`` into a live
``PropertyGraphIndex`` via ``from_existing`` without triggering extraction
or writes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from llama_index.core import PropertyGraphIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms.mock import MockLLM

from vo_rag.integrations.llamaindex import property_graph_index as pgi_mod
from vo_rag.integrations.llamaindex.property_graph_index import (
    build_vo_rag_pg_retriever,
    build_vo_rag_property_graph_index,
)


def _fake_store() -> Any:
    """Return a fake GraphQdrantStore whose _client is an AsyncMock."""
    store = MagicMock()
    store._client = AsyncMock()
    return store


def _dummy_extractor() -> Any:
    """Return a truthy stand-in for a TransformComponent."""
    return MagicMock()


class TestBuildVoRagPropertyGraphIndex:
    def test_returns_property_graph_index(self) -> None:
        store = _fake_store()
        dummy = _dummy_extractor()
        index = build_vo_rag_property_graph_index(
            store=store,
            kg_extractors=[dummy],
        )
        assert isinstance(index, PropertyGraphIndex)

    def test_property_graph_store_is_vo_rag_wrapper(self) -> None:
        store = _fake_store()
        dummy = _dummy_extractor()
        index = build_vo_rag_property_graph_index(
            store=store,
            kg_extractors=[dummy],
        )
        pg_store = index.property_graph_store
        from vo_rag.integrations.llamaindex.graph_store import VoRagPropertyGraphStore

        assert isinstance(pg_store, VoRagPropertyGraphStore)
        assert pg_store._store is store

    def test_assembly_does_not_upsert(self) -> None:
        store = _fake_store()
        dummy = _dummy_extractor()
        build_vo_rag_property_graph_index(
            store=store,
            kg_extractors=[dummy],
        )
        store._client.upsert.assert_not_called()
        store._client.scroll.assert_not_called()

    def test_factory_calls_require_llamaindex(self, monkeypatch: Any) -> None:
        calls: list[None] = []

        def _mock_guard() -> None:
            calls.append(None)

        monkeypatch.setattr(pgi_mod, "require_llamaindex", _mock_guard)
        store = _fake_store()
        dummy = _dummy_extractor()
        build_vo_rag_property_graph_index(
            store=store,
            kg_extractors=[dummy],
        )
        assert len(calls) == 1

    def test_default_retriever_does_not_call_unimplemented_vector_query(
        self, monkeypatch: Any
    ) -> None:
        index = build_vo_rag_property_graph_index(
            store=_fake_store(),
            kg_extractors=[_dummy_extractor()],
            embed_kg_nodes=False,
            llm=MockLLM(max_tokens=1),
            use_async=False,
        )
        pg_store = index.property_graph_store

        def fail_vector_query(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("default retriever must not call vector_query")

        monkeypatch.setattr(pg_store, "vector_query", fail_vector_query)
        monkeypatch.setattr(pg_store, "get", lambda *args, **kwargs: [])
        monkeypatch.setattr(pg_store, "get_rel_map", lambda *args, **kwargs: [])

        retriever = index.as_retriever()
        result = retriever.retrieve("Known Entity")

        assert isinstance(result, list)

    def test_embed_model_does_not_route_default_retriever_to_graph_vector_query(
        self, monkeypatch: Any
    ) -> None:
        index = build_vo_rag_property_graph_index(
            store=_fake_store(),
            kg_extractors=[_dummy_extractor()],
            embed_kg_nodes=True,
            embed_model=MockEmbedding(embed_dim=2),
            llm=MockLLM(max_tokens=1),
            use_async=False,
        )
        pg_store = index.property_graph_store

        def fail_vector_query(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("vector_query is not implemented")

        monkeypatch.setattr(pg_store, "vector_query", fail_vector_query)
        monkeypatch.setattr(pg_store, "get", lambda *args, **kwargs: [])
        monkeypatch.setattr(pg_store, "get_rel_map", lambda *args, **kwargs: [])

        retriever = index.as_retriever()
        result = retriever.retrieve("Known Entity")

        assert isinstance(result, list)


class TestBuildVoRagPgRetriever:
    def test_returns_vo_rag_pg_retriever(self) -> None:
        store = _fake_store()
        retriever = build_vo_rag_pg_retriever(store=store, top_k=5)
        from vo_rag.integrations.llamaindex.pg_retriever import VoRagPGRetriever

        assert isinstance(retriever, VoRagPGRetriever)

    def test_retriever_wired_to_same_store(self) -> None:
        store = _fake_store()
        retriever = build_vo_rag_pg_retriever(store=store, top_k=5)
        assert retriever._store is store
        assert retriever._top_k == 5

    def test_default_top_k(self) -> None:
        store = _fake_store()
        retriever = build_vo_rag_pg_retriever(store=store)
        assert retriever._top_k == 10

    def test_graph_search_public_path_uses_read_only_vo_rag_retriever(self) -> None:
        store = _fake_store()
        retriever = build_vo_rag_pg_retriever(store=store, top_k=10)
        assert retriever._store is store
        assert retriever._graph_store.supports_vector_queries is False


class TestPublicSurface:
    def test_symbols_in_all(self) -> None:
        assert "build_vo_rag_property_graph_index" in pgi_mod.__all__
        assert "build_vo_rag_pg_retriever" in pgi_mod.__all__
