"""VOR-196 contract tests — [S{i}] citation fidelity through VoRagSynthesizer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from vo_rag.core.answer.context import format_answer_context
from vo_rag.core.answer.synthesizer import (
    AnswerDraft,
    AnswerProfile,
    AnswerSynthesis,
)
from vo_rag.core.llm.output_validator import ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.integrations.llamaindex.synthesizer import (
    VO_RAG_RESPONSE_METADATA_KEY,
    VoRagSynthesizer,
    _node_to_search_result,
)


def _node(
    *,
    text: str = "alpha",
    chunk_id: str = "c1",
    source_id: str = "s1",
    title: str = "Source One",
    page: int = 7,
    use_section_title: bool = False,
) -> NodeWithScore:
    """Build a NodeWithScore simulating retriever-adapter output.

    When *use_section_title* is True the title rides under the converter's
    canonical key (``section_title``); otherwise it uses the direct
    ``source_title`` key that ``format_answer_context`` reads.
    """
    metadata: dict[str, object] = {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "page_number": page,
        "collection": "vo_rag_books_sterile_v1",
        "qdrant_point_id": f"point-{chunk_id}",
    }
    if use_section_title:
        metadata["section_title"] = title
    else:
        metadata["source_title"] = title
    return NodeWithScore(
        node=TextNode(text=text, id_=chunk_id, metadata=metadata),
        score=0.9,
    )


def _answer_synth(*, cited_chunk_ids: list[str] | None = None) -> MagicMock:
    answer = MagicMock()
    answer.synthesize = AsyncMock(
        return_value=AnswerSynthesis(
            draft=AnswerDraft(
                answer="grounded [S1]",
                confidence=0.8,
                cited_chunk_ids=cited_chunk_ids or ["c1"],
            ),
            llm_response=LLMResponse(text="grounded [S1]", model="claude-x"),
            validation=ValidationResult(ok=True),
            should_quarantine=False,
            model="claude-x",
        )
    )
    return answer


class TestNodeToSearchResult:
    """The lowering must preserve every field ``format_answer_context`` reads."""

    def test_preserves_chunk_id(self) -> None:
        node = _node(chunk_id="chunk-42")
        result = _node_to_search_result(node)
        assert result.payload["chunk_id"] == "chunk-42"

    def test_preserves_source_id(self) -> None:
        node = _node(source_id="book-99")
        result = _node_to_search_result(node)
        assert result.payload["source_id"] == "book-99"

    @pytest.mark.parametrize("use_section_title", [False, True])
    def test_preserves_title(self, use_section_title: bool) -> None:
        node = _node(title="The Title", use_section_title=use_section_title)
        result = _node_to_search_result(node)
        payload = result.payload
        assert payload.get("source_title") == "The Title" or payload.get("title") == "The Title"

    def test_section_title_mapped_to_source_title(self) -> None:
        """When metadata only has ``section_title``, lowering normalises it."""
        node = _node(title="Section A", use_section_title=True)
        result = _node_to_search_result(node)
        assert result.payload.get("source_title") == "Section A"

    def test_preserves_page_number(self) -> None:
        node = _node(page=42)
        result = _node_to_search_result(node)
        assert result.payload["page_number"] == 42

    def test_preserves_content(self) -> None:
        node = _node(text="body text")
        result = _node_to_search_result(node)
        assert result.payload["content"] == "body text"

    def test_preserves_qdrant_point_id(self) -> None:
        node = _node(chunk_id="c1")
        result = _node_to_search_result(node)
        assert result.id == "point-c1"


class TestFormatAnswerContext:
    """Rendered [S{i}] blocks must carry all citation fields."""

    def test_renders_complete_si_block(self) -> None:
        nodes = [
            _node(text="first chunk", chunk_id="c1", source_id="s1", title="Book One", page=3),
            _node(
                text="second chunk",
                chunk_id="c2",
                source_id="s2",
                title="Book Two",
                page=5,
                use_section_title=True,
            ),
        ]
        results = [_node_to_search_result(n) for n in nodes]
        context = format_answer_context(results)

        assert "[S1]" in context
        assert "chunk_id: c1" in context
        assert "source_id: s1" in context
        assert "title: Book One" in context
        assert "page: 3" in context
        assert "first chunk" in context

        assert "[S2]" in context
        assert "chunk_id: c2" in context
        assert "source_id: s2" in context
        assert "title: Book Two" in context
        assert "page: 5" in context
        assert "second chunk" in context

    def test_section_title_round_trips_through_block(self) -> None:
        """A node built with ``section_title`` produces a block that cites the title."""
        node = _node(
            text="content",
            chunk_id="c1",
            source_id="s1",
            title="My Section",
            page=10,
            use_section_title=True,
        )
        result = _node_to_search_result(node)
        context = format_answer_context([result])
        assert "title: My Section" in context


class TestCitedChunkIds:
    """``cited_chunk_ids`` returned by the defended answer must surface unchanged."""

    def test_round_trip_through_response_metadata(self) -> None:
        answer = _answer_synth(cited_chunk_ids=["chunk-a", "chunk-b"])
        synthesizer = VoRagSynthesizer(
            answer=answer,
            profile=AnswerProfile.QUALITY,
            llm=MagicMock(),
        )
        response = synthesizer.synthesize(
            "q", [_node(chunk_id="chunk-a"), _node(chunk_id="chunk-b")]
        )

        assert response.metadata is not None
        vo_rag_meta = response.metadata["vo_rag"]
        assert vo_rag_meta["cited_chunk_ids"] == ["chunk-a", "chunk-b"]

    def test_round_trip_with_request_scoped_nodes(self) -> None:
        answer = _answer_synth(cited_chunk_ids=["c1", "c2"])
        synthesizer = VoRagSynthesizer(
            answer=answer,
            profile=AnswerProfile.QUALITY,
            llm=MagicMock(),
        )
        nodes = [_node(chunk_id="c1"), _node(chunk_id="c2")]

        response = synthesizer.synthesize("q", nodes)

        assert response.metadata is not None
        assert response.metadata[VO_RAG_RESPONSE_METADATA_KEY]["cited_chunk_ids"] == ["c1", "c2"]


def test_no_citation_query_engine_import() -> None:
    """The adapter keeps vo-rag's formatter; it does not adopt CitationQueryEngine."""
    import vo_rag.integrations.llamaindex.synthesizer as synth_module

    assert not hasattr(synth_module, "CitationQueryEngine")
    # Also assert the literal string does not appear in the module source
    import inspect

    source = inspect.getsource(synth_module)
    assert "CitationQueryEngine" not in source
