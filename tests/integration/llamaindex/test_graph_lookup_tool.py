"""Tests for the graph_lookup QueryEngineTool factory."""

from __future__ import annotations

import inspect
from typing import Any, cast, get_type_hints
from unittest.mock import MagicMock

import pytest
from llama_index.core.base.response.schema import Response
from llama_index.core.llms.mock import MockLLM
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer, get_response_synthesizer
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.settings import Settings
from llama_index.core.tools import QueryEngineTool

from vo_rag.core.llm.client import LLMClient
from vo_rag.integrations.llamaindex.agents.tools.graph_lookup import (
    build_graph_lookup_tool,
)
from vo_rag.integrations.llamaindex.llm import VoRagLLM
from vo_rag.integrations.llamaindex.pg_retriever import VoRagPGRetriever


def _fake_graph_store() -> Any:
    """Minimal PropertyGraphStore stand-in for CustomPGRetriever construction."""
    store = MagicMock()
    store._client = MagicMock()
    return store


def _fake_qdrant_store() -> Any:
    """Minimal GraphQdrantStore stand-in for the store argument."""
    store = MagicMock()
    store._client = MagicMock()
    return store


class _FakeProvider:
    name = "test_provider"
    model = "test-model"


class _FakeLLMClient:
    provider = _FakeProvider()

    async def complete(self, **kwargs: Any) -> object:
        raise AssertionError("graph lookup tests must not call a live LLM")


def _fake_vorag_llm() -> VoRagLLM:
    """Return a real ``VoRagLLM`` adapter over a non-network fake client."""
    return VoRagLLM(client=cast(LLMClient, _FakeLLMClient()))


def _fake_synthesizer(llm: VoRagLLM) -> BaseSynthesizer:
    """Return a real LlamaIndex synthesizer bound to the supplied vo-rag LLM."""
    return get_response_synthesizer(llm=llm)


def _build_graph_lookup_tool(
    *,
    graph_store: Any | None = None,
    store: Any | None = None,
    top_k: int = 10,
) -> QueryEngineTool:
    llm = _fake_vorag_llm()
    return build_graph_lookup_tool(
        graph_store=_fake_graph_store() if graph_store is None else graph_store,
        store=_fake_qdrant_store() if store is None else store,
        top_k=top_k,
        llm=llm,
        response_synthesizer=_fake_synthesizer(llm),
    )


@pytest.fixture(autouse=True)
def _patch_settings_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent RetrieverQueryEngine from trying to resolve a default OpenAI LLM."""
    monkeypatch.setattr(Settings, "_llm", MockLLM(), raising=False)


def _unwrap_engine(tool: Any) -> RetrieverQueryEngine:
    """Cast the wrapped query engine for internal introspection."""
    return cast(RetrieverQueryEngine, tool.query_engine)


def test_graph_lookup_tool_requires_llm_and_synthesizer_in_signature() -> None:
    signature = inspect.signature(build_graph_lookup_tool)
    type_hints = get_type_hints(build_graph_lookup_tool)

    for parameter_name in ("llm", "response_synthesizer"):
        parameter = signature.parameters[parameter_name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty

    assert type_hints["llm"] is VoRagLLM
    assert type_hints["response_synthesizer"] is BaseSynthesizer


def test_graph_lookup_tool_uses_supplied_vorag_llm_and_synthesizer() -> None:
    llm = _fake_vorag_llm()
    synthesizer = _fake_synthesizer(llm)
    tool = build_graph_lookup_tool(
        graph_store=_fake_graph_store(),
        store=_fake_qdrant_store(),
        llm=llm,
        response_synthesizer=synthesizer,
    )

    engine = _unwrap_engine(tool)

    assert engine._response_synthesizer is synthesizer
    assert engine._response_synthesizer._llm is llm
    assert engine._response_synthesizer._llm is not Settings.llm


def test_graph_lookup_tool_requires_response_synthesizer_keyword() -> None:
    with pytest.raises(TypeError, match="response_synthesizer"):
        cast(Any, build_graph_lookup_tool)(
            graph_store=_fake_graph_store(),
            store=_fake_qdrant_store(),
            llm=_fake_vorag_llm(),
        )


def test_graph_lookup_tool_rejects_none_response_synthesizer_at_runtime() -> None:
    with pytest.raises(ValueError, match="graph_lookup requires a vo-rag response_synthesizer"):
        build_graph_lookup_tool(
            graph_store=_fake_graph_store(),
            store=_fake_qdrant_store(),
            llm=_fake_vorag_llm(),
            response_synthesizer=cast(BaseSynthesizer, None),
        )


def test_graph_lookup_tool_rejects_none_llm_at_runtime() -> None:
    with pytest.raises(ValueError, match="graph_lookup requires a vo-rag llm"):
        build_graph_lookup_tool(
            graph_store=_fake_graph_store(),
            store=_fake_qdrant_store(),
            llm=cast(VoRagLLM, None),
            response_synthesizer=_fake_synthesizer(_fake_vorag_llm()),
        )


def test_graph_lookup_tool_rejects_mismatched_synthesizer_llm() -> None:
    llm = _fake_vorag_llm()
    synthesizer = _fake_synthesizer(_fake_vorag_llm())

    with pytest.raises(ValueError, match="response_synthesizer llm must be the supplied llm"):
        build_graph_lookup_tool(
            graph_store=_fake_graph_store(),
            store=_fake_qdrant_store(),
            llm=llm,
            response_synthesizer=synthesizer,
        )


def test_graph_lookup_tool_name_and_description() -> None:
    tool = _build_graph_lookup_tool()
    assert tool.metadata.name == "graph_lookup"
    desc = tool.metadata.description
    assert desc
    assert "entities" in desc
    assert "relationships" in desc
    assert "evidence chunks" in desc


def test_graph_lookup_tool_return_direct_is_false() -> None:
    tool = _build_graph_lookup_tool()
    assert tool.metadata.return_direct is False


def test_graph_lookup_tool_dispatches_to_stubbed_query_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _build_graph_lookup_tool()
    node = TextNode(
        text="connected entity chunk",
        id_="chunk-1",
        metadata={
            "source_id": "src-1",
            "section_path": "Ch 1 > A",
        },
    )
    stub_response = Response(
        response="connected entities and relationships",
        source_nodes=[NodeWithScore(node=node, score=1.0)],
    )
    monkeypatch.setattr(tool.query_engine, "query", MagicMock(return_value=stub_response))

    output = tool.call(query="active suspension")

    assert "connected entities and relationships" in output.content
    cast(MagicMock, tool.query_engine.query).assert_called_once()


def test_graph_lookup_tool_does_not_synthesize_through_global_settings_llm() -> None:
    """The factory must not let RetrieverQueryEngine synthesize through Settings.llm."""
    llm = _fake_vorag_llm()
    synthesizer = _fake_synthesizer(llm)
    tool = build_graph_lookup_tool(
        graph_store=_fake_graph_store(),
        store=_fake_qdrant_store(),
        llm=llm,
        response_synthesizer=synthesizer,
    )
    engine = _unwrap_engine(tool)
    assert engine._response_synthesizer is synthesizer
    assert engine._response_synthesizer._llm is llm
    assert engine._response_synthesizer._llm is not Settings.llm


def test_graph_lookup_tool_reads_internal_graph_only() -> None:
    """The retriever must be wired to the provided internal store; the factory
    does not default to an untrusted scratch collection."""
    graph_store = _fake_graph_store()
    store = _fake_qdrant_store()
    tool = _build_graph_lookup_tool(
        graph_store=graph_store,
        store=store,
    )
    retriever = cast(VoRagPGRetriever, _unwrap_engine(tool)._retriever)
    assert retriever._store is store
    # No mode or collection parameter exists that could redirect to scratch.
    assert not hasattr(retriever, "_mode")


def test_graph_lookup_tool_accepts_top_k() -> None:
    graph_store = _fake_graph_store()
    store = _fake_qdrant_store()
    tool = _build_graph_lookup_tool(
        graph_store=graph_store,
        store=store,
        top_k=5,
    )
    retriever = cast(VoRagPGRetriever, _unwrap_engine(tool)._retriever)
    assert retriever._top_k == 5
