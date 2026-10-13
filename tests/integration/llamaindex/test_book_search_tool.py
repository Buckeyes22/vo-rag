"""Tests for the book_search QueryEngineTool factory."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.base.response.schema import Response
from llama_index.core.llms.mock import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.settings import Settings

from vo_rag.core.answer.synthesizer import AnswerSynthesizer
from vo_rag.core.config.settings import AnswerSettings
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import HybridRetriever
from vo_rag.integrations.llamaindex.agents.tools.book_search import build_book_search_tool
from vo_rag.integrations.llamaindex.retriever import VoRagRetriever
from vo_rag.integrations.llamaindex.synthesizer import VoRagSynthesizer


def _stub_hybrid_retriever() -> HybridRetriever:
    """Return a minimal HybridRetriever test double."""
    retriever = MagicMock(spec=HybridRetriever)
    retriever.cross_collection_search = AsyncMock(return_value=[])
    return retriever


def _stub_answer_synthesizer() -> AnswerSynthesizer:
    """Return a minimal AnswerSynthesizer test double."""
    return AnswerSynthesizer(
        AnswerSettings(
            allow_hosted=True,
            anthropic_api_key="test-key",  # pragma: allowlist secret
            max_context_chunks=4,
            max_output_tokens=64,
        )
    )


@pytest.fixture(autouse=True)
def _patch_settings_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent VoRagSynthesizer from trying to resolve a default OpenAI LLM."""
    monkeypatch.setattr(Settings, "_llm", MockLLM(), raising=False)


def _unwrap_engine(tool: Any) -> RetrieverQueryEngine:
    """Cast the wrapped query engine for internal introspection."""
    return cast(RetrieverQueryEngine, tool.query_engine)


def test_book_search_tool_name_and_description() -> None:
    tool = build_book_search_tool(
        retriever=_stub_hybrid_retriever(),
        answer=_stub_answer_synthesizer(),
    )
    assert tool.metadata.name == "book_search"
    desc = tool.metadata.description
    assert desc
    assert "source ids" in desc
    assert "page refs" in desc


def test_book_search_tool_uses_internal_mode() -> None:
    tool = build_book_search_tool(
        retriever=_stub_hybrid_retriever(),
        answer=_stub_answer_synthesizer(),
    )
    retriever = cast(VoRagRetriever, _unwrap_engine(tool).retriever)
    assert retriever._mode is QueryMode.INTERNAL
    assert retriever._requested_collections is None
    # When collections are resolved internally, the default includes raw books.
    from vo_rag.core.retrieval.query_mode import resolve_collections

    collections = resolve_collections(retriever._mode, retriever._requested_collections)
    assert "vo_rag_books_raw_v1" in collections
    # Scratch / untrusted collections must not appear.
    assert not any("scratch" in c for c in collections)


def test_book_search_tool_return_direct_is_false() -> None:
    tool = build_book_search_tool(
        retriever=_stub_hybrid_retriever(),
        answer=_stub_answer_synthesizer(),
    )
    assert tool.metadata.return_direct is False


def test_book_search_tool_dispatches_to_stubbed_query_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = build_book_search_tool(
        retriever=_stub_hybrid_retriever(),
        answer=_stub_answer_synthesizer(),
    )
    node = TextNode(
        text="ranked excerpt",
        id_="chunk-1",
        metadata={
            "chunk_id": "chunk-1",
            "source_id": "src-1",
            "source_title": "Test Source",
            "page_number": 42,
            "collection": "vo_rag_books_raw_v1",
        },
    )
    stub_response = Response(
        response="defended answer",
        source_nodes=[NodeWithScore(node=node, score=0.95)],
    )
    monkeypatch.setattr(tool.query_engine, "query", MagicMock(return_value=stub_response))

    output = tool.call(query="test query")

    assert "defended answer" in output.content
    cast(MagicMock, tool.query_engine.query).assert_called_once()


def test_book_search_tool_no_second_llm_handle() -> None:
    """The factory must not construct an additional LLM handle beyond the
    AnswerSynthesizer it is handed."""
    answer = _stub_answer_synthesizer()
    tool = build_book_search_tool(
        retriever=_stub_hybrid_retriever(),
        answer=answer,
    )
    synthesizer = cast(VoRagSynthesizer, _unwrap_engine(tool)._response_synthesizer)
    # The synthesis path delegates to the AnswerSynthesizer (vo-rag substrate),
    # not to a separately constructed bare LLM handle.
    assert synthesizer._answer is answer


def test_book_search_tool_accepts_optional_parameters() -> None:
    retriever = _stub_hybrid_retriever()
    answer = _stub_answer_synthesizer()
    tool = build_book_search_tool(
        retriever=retriever,
        answer=answer,
        top_k=10,
        requested_collections=["vo_rag_books_raw_v1"],
        filters={"license": "cc-by"},
        visual_mode="fast",
    )
    vo_retriever = cast(VoRagRetriever, _unwrap_engine(tool).retriever)
    assert vo_retriever._top_k == 10
    assert vo_retriever._requested_collections == ["vo_rag_books_raw_v1"]
    assert vo_retriever._filters == {"license": "cc-by"}
    assert vo_retriever._visual_mode == "fast"
