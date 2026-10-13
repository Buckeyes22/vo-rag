from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.graph.extraction import (
    GraphExtractionConfig,
    GraphExtractionResult,
    _extract_json_object,
    build_extraction_prompt,
    extract_graph_from_book,
    graph_response_json_schema,
    parse_graph_response,
)
from vo_rag.core.graph.prompts import GRAPH_EXTRACTION_PROMPT_HASH, GRAPH_EXTRACTION_SYSTEM_PROMPT
from vo_rag.core.graph.schemas import GraphEdgeType, GraphNodeType, node_id_for
from vo_rag.core.llm.response import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    chunk_id: str,
    content: str,
    section_path: str = "[Book > Chapter > Section]",
    section_title: str = "Section",
) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        chunk_type="text",
        document_id="doc-001",
        source_id="source-001",
        page_number=None,
        page_range=None,
        section_path=section_path,
        section_title=section_title,
        content=content,
        embedding_text=f"{section_path}\n\n{content}",
        token_count=max(1, len(content.split())),
        has_image=False,
        metadata={},
    )


def _make_config(prompt_hash: str = GRAPH_EXTRACTION_PROMPT_HASH) -> GraphExtractionConfig:
    return GraphExtractionConfig(prompt_hash=prompt_hash)


def _valid_llm_json(
    *,
    chunk_ids: list[str],
    node_label: str = "Understeer",
    node_type: str = "concept",
) -> str:
    return json.dumps(
        {
            "nodes": [
                {
                    "node_type": node_type,
                    "label": node_label,
                    "aliases": [],
                    "chunk_ids": chunk_ids[:1],
                }
            ],
            "edges": [],
        }
    )


# ---------------------------------------------------------------------------
# build_extraction_prompt
# ---------------------------------------------------------------------------


class TestBuildExtractionPrompt:
    def test_build_extraction_prompt_includes_book_metadata(self) -> None:
        chunks = [_make_chunk(chunk_id="c1", content="Understeer is a handling balance condition.")]
        system, user, _ = build_extraction_prompt(
            book_id="book-001",
            title="The Circuit Driver",
            author="Sam Ellis",
            chunks=chunks,
        )
        assert "The Circuit Driver" in user
        assert "Sam Ellis" in user

    def test_build_extraction_prompt_formats_chunk_block_with_provenance(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="c42",
                content="Oversteer occurs when the rear tyres lose grip before the front.",
                section_path="[Book > Chapter 3 > Handling]",
            )
        ]
        system, user, chunk_index_map = build_extraction_prompt(
            book_id="book-001",
            title="Any Title",
            author="Any Author",
            chunks=chunks,
        )
        # Chunks are shown by compact label, not raw UUID, to shrink model output.
        assert "[c0]" in user
        assert "c42" not in user
        assert "Oversteer occurs" in user
        # The returned map translates the label back to the real chunk_id.
        assert chunk_index_map == {"c0": "c42"}

    def test_build_extraction_prompt_system_is_standard_prompt(self) -> None:
        chunks = [_make_chunk(chunk_id="c1", content="Sample content.")]
        system, _, _ = build_extraction_prompt(
            book_id="book-001",
            title="T",
            author="A",
            chunks=chunks,
        )
        assert system == GRAPH_EXTRACTION_SYSTEM_PROMPT

    def test_build_extraction_prompt_returns_prompts_and_index_map(self) -> None:
        chunks = [_make_chunk(chunk_id="c1", content="Content.")]
        result = build_extraction_prompt(
            book_id="b1",
            title="T",
            author="A",
            chunks=chunks,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        system, user, chunk_index_map = result
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert chunk_index_map == {"c0": "c1"}


# ---------------------------------------------------------------------------
# parse_graph_response
# ---------------------------------------------------------------------------


class TestParseGraphResponse:
    def test_parse_graph_response_valid_builds_nodes_and_edges(self) -> None:
        index_map = {"c1": "chunk-1", "c2": "chunk-2"}
        payload = json.dumps(
            {
                "nodes": [
                    {
                        "node_type": "concept",
                        "label": "Traction Circle",
                        "aliases": ["friction circle"],
                        "chunk_ids": ["c1"],
                    },
                    {
                        "node_type": "method",
                        "label": "Trail Braking",
                        "aliases": [],
                        "chunk_ids": ["c2"],
                    },
                ],
                "edges": [
                    {
                        "source_label": "Traction Circle",
                        "source_type": "concept",
                        "target_label": "Trail Braking",
                        "target_type": "method",
                        "edge_type": "relates_to",
                        "evidence_chunk_ids": ["c1", "c2"],
                    }
                ],
            }
        )
        result = parse_graph_response(
            book_id="b1",
            llm_response_json=payload,
            chunk_index_map=index_map,
            prompt_hash="hash-001",
        )
        assert len(result.nodes) == 2
        assert len(result.edges) == 1
        assert result.validation_errors == []
        # compact labels are translated back to real chunk_ids on the way out
        assert {tuple(n.chunk_ids) for n in result.nodes} == {("chunk-1",), ("chunk-2",)}
        assert result.edges[0].evidence_chunk_ids == ["chunk-1", "chunk-2"]
        # edges carry book_ids (VOR: edges must be book-filterable in Qdrant)
        assert all(e.book_ids == ["b1"] for e in result.edges)

    def test_parse_graph_response_handles_fenced_json(self) -> None:
        """qwen3-6-27b often wraps JSON in markdown fences / preamble despite the
        prompt; the parser must still recover it (was: raw json.loads lost the
        whole batch -> the high validation-failure rate)."""
        index_map = {"c1": "chunk-1"}
        inner = {
            "nodes": [{"node_type": "concept", "label": "Yaw", "aliases": [], "chunk_ids": ["c1"]}],
            "edges": [],
        }
        fenced = "Here is the graph:\n```json\n" + json.dumps(inner) + "\n```\nDone."
        result = parse_graph_response(
            book_id="b1",
            llm_response_json=fenced,
            chunk_index_map=index_map,
            prompt_hash="hash-001",
        )
        assert result.validation_errors == []
        assert len(result.nodes) == 1
        assert result.nodes[0].label == "Yaw"

    def test_parse_graph_response_unknown_chunk_in_node_drops_node(self) -> None:
        """Unknown chunk_ids cause node rejection."""
        index_map = {"c1": "chunk-1"}
        payload = json.dumps(
            {
                "nodes": [
                    {
                        "node_type": "concept",
                        "label": "Oversteer",
                        "aliases": [],
                        "chunk_ids": ["c-unknown"],
                    }
                ],
                "edges": [],
            }
        )
        result = parse_graph_response(
            book_id="b1",
            llm_response_json=payload,
            chunk_index_map=index_map,
            prompt_hash="hash-001",
        )
        assert len(result.nodes) == 0
        assert any("unknown chunk_ids" in e for e in result.validation_errors)

    def test_parse_graph_response_unknown_chunk_in_edge_drops_edge(self) -> None:
        """Unknown evidence_chunk_ids cause edge rejection."""
        index_map = {"c1": "chunk-1", "c2": "chunk-2"}
        payload = json.dumps(
            {
                "nodes": [
                    {
                        "node_type": "concept",
                        "label": "Understeer",
                        "aliases": [],
                        "chunk_ids": ["c1"],
                    },
                    {
                        "node_type": "method",
                        "label": "Weight Transfer",
                        "aliases": [],
                        "chunk_ids": ["c2"],
                    },
                ],
                "edges": [
                    {
                        "source_label": "Understeer",
                        "source_type": "concept",
                        "target_label": "Weight Transfer",
                        "target_type": "method",
                        "edge_type": "relates_to",
                        "evidence_chunk_ids": ["c-ghost"],
                    }
                ],
            }
        )
        result = parse_graph_response(
            book_id="b1",
            llm_response_json=payload,
            chunk_index_map=index_map,
            prompt_hash="hash-001",
        )
        # Nodes pass; edge with unknown chunk is dropped
        assert len(result.nodes) == 2
        assert len(result.edges) == 0
        assert any("unknown evidence_chunk_ids" in e for e in result.validation_errors)

    def test_parse_graph_response_orphan_edge_dropped(self) -> None:
        """Edge referencing a node label not present in nodes is dropped."""
        index_map = {"c1": "chunk-1"}
        payload = json.dumps(
            {
                "nodes": [
                    {
                        "node_type": "concept",
                        "label": "Aero Balance",
                        "aliases": [],
                        "chunk_ids": ["c1"],
                    }
                ],
                "edges": [
                    {
                        "source_label": "Aero Balance",
                        "source_type": "concept",
                        "target_label": "Ghost Node",
                        "target_type": "concept",
                        "edge_type": "relates_to",
                        "evidence_chunk_ids": ["c1"],
                    }
                ],
            }
        )
        result = parse_graph_response(
            book_id="b1",
            llm_response_json=payload,
            chunk_index_map=index_map,
            prompt_hash="hash-001",
        )
        assert len(result.edges) == 0
        assert any("not in extracted nodes" in e for e in result.validation_errors)

    def test_graph_extraction_result_drops_edges_not_in_extracted_node_set(self) -> None:
        index_map = {"c1": "chunk-1", "c2": "chunk-2"}
        payload = json.dumps(
            {
                "nodes": [
                    {
                        "node_type": "concept",
                        "label": "Anti Dive",
                        "aliases": [],
                        "chunk_ids": ["c1"],
                    }
                ],
                "edges": [
                    {
                        "source_label": "Anti Dive",
                        "source_type": "concept",
                        "target_label": "Unsprung Weight",
                        "target_type": "concept",
                        "edge_type": "relates_to",
                        "evidence_chunk_ids": ["c1", "c2"],
                    }
                ],
            }
        )

        result = parse_graph_response(
            book_id="b1",
            llm_response_json=payload,
            chunk_index_map=index_map,
            prompt_hash="hash-001",
        )

        assert isinstance(result, GraphExtractionResult)
        assert [node.label for node in result.nodes] == ["Anti Dive"]
        assert result.edges == []
        assert any(
            "target_label 'Unsprung Weight' not in extracted nodes" in e
            for e in result.validation_errors
        )

    def test_node_ids_built_via_deterministic_helper(self) -> None:
        index_map = {"c1": "chunk-1"}
        label = "Camber Thrust"
        node_type_str = "concept"
        payload = json.dumps(
            {
                "nodes": [
                    {
                        "node_type": node_type_str,
                        "label": label,
                        "aliases": [],
                        "chunk_ids": ["c1"],
                    }
                ],
                "edges": [],
            }
        )
        result = parse_graph_response(
            book_id="b1",
            llm_response_json=payload,
            chunk_index_map=index_map,
            prompt_hash="hash-001",
        )
        assert len(result.nodes) == 1
        expected_id = node_id_for(node_type=GraphNodeType.CONCEPT, canonical_label=label)
        assert result.nodes[0].node_id == expected_id

    def test_parse_graph_response_accepts_is_a_edge_type(self) -> None:
        index_map = {"c1": "chunk-1", "c2": "chunk-2"}
        payload = json.dumps(
            {
                "nodes": [
                    {
                        "node_type": "component",
                        "label": "Honda EU2000i Generator",
                        "aliases": [],
                        "chunk_ids": ["c1"],
                    },
                    {
                        "node_type": "component",
                        "label": "Generator",
                        "aliases": [],
                        "chunk_ids": ["c2"],
                    },
                ],
                "edges": [
                    {
                        "source_label": "Honda EU2000i Generator",
                        "source_type": "component",
                        "target_label": "Generator",
                        "target_type": "component",
                        "edge_type": "is_a",
                        "evidence_chunk_ids": ["c1", "c2"],
                    }
                ],
            }
        )

        result = parse_graph_response(
            book_id="b1",
            llm_response_json=payload,
            chunk_index_map=index_map,
            prompt_hash="hash-001",
        )

        assert result.validation_errors == []
        assert len(result.edges) == 1
        assert result.edges[0].edge_type == GraphEdgeType.IS_A


# ---------------------------------------------------------------------------
# extract_graph_from_book (async)
# ---------------------------------------------------------------------------


class TestExtractGraphFromBook:
    @pytest.mark.asyncio
    async def test_extract_graph_from_book_invokes_llm_once(self) -> None:
        chunks = [
            _make_chunk(chunk_id="c1", content="Downforce acts perpendicular to the road surface."),
        ]
        llm_payload = _valid_llm_json(chunk_ids=["c0"])
        mock_llm: Any = AsyncMock(
            return_value=LLMResponse(
                text=llm_payload,
                model="qwen3.6-27b-fp8",
                input_tokens=100,
                output_tokens=50,
            )
        )
        config = _make_config()

        await extract_graph_from_book(
            book_id="book-001",
            title="Race Car Design",
            author="Jordan Avery",
            chunks=chunks,
            config=config,
            llm_call=mock_llm,
        )

        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_graph_from_book_records_token_accounting(self) -> None:
        chunks = [
            _make_chunk(chunk_id="c1", content="The wing generates downforce via angle of attack."),
        ]
        llm_payload = _valid_llm_json(chunk_ids=["c0"])
        mock_llm: Any = AsyncMock(
            return_value=LLMResponse(
                text=llm_payload,
                model="qwen3.6-27b-fp8",
                input_tokens=123,
                output_tokens=77,
            )
        )
        config = _make_config()

        result = await extract_graph_from_book(
            book_id="book-001",
            title="Race Car Design",
            author="Jordan Avery",
            chunks=chunks,
            config=config,
            llm_call=mock_llm,
        )

        assert result.input_tokens == 123
        assert result.output_tokens == 77
        assert result.wall_time_ms >= 0

    @pytest.mark.asyncio
    async def test_extract_graph_from_book_returns_extraction_result(self) -> None:
        chunks = [
            _make_chunk(
                chunk_id="c1", content="Grip is limited by the tyre's friction coefficient."
            ),
        ]
        llm_payload = _valid_llm_json(chunk_ids=["c0"])
        mock_llm: Any = AsyncMock(
            return_value=LLMResponse(
                text=llm_payload,
                model="qwen3.6-27b-fp8",
                input_tokens=10,
                output_tokens=10,
            )
        )

        result = await extract_graph_from_book(
            book_id="book-001",
            title="T",
            author="A",
            chunks=chunks,
            config=_make_config(),
            llm_call=mock_llm,
        )

        assert isinstance(result, GraphExtractionResult)
        assert result.book_id == "book-001"
        assert result.prompt_hash == GRAPH_EXTRACTION_PROMPT_HASH


class TestGraphResponseJsonSchema:
    def test_top_level_shape(self) -> None:
        s = graph_response_json_schema()
        assert s["type"] == "object"
        assert set(s["required"]) == {"nodes", "edges"}
        assert s["properties"]["nodes"]["type"] == "array"
        assert s["properties"]["edges"]["type"] == "array"

    def test_node_type_enum_excludes_book(self) -> None:
        s = graph_response_json_schema()
        node_enum = s["properties"]["nodes"]["items"]["properties"]["node_type"]["enum"]
        # The book node is synthesized by the pipeline, not extracted — the model
        # must not be allowed to mint it.
        assert "book" not in node_enum
        assert set(node_enum) == {"concept", "method", "equation", "person", "component"}

    def test_edge_type_enum_matches_all_edge_types(self) -> None:
        s = graph_response_json_schema()
        edge_enum = s["properties"]["edges"]["items"]["properties"]["edge_type"]["enum"]
        assert set(edge_enum) == {t.value for t in GraphEdgeType}
        assert "is_a" in edge_enum

    def test_chunk_refs_constrained_to_compact_label_pattern(self) -> None:
        s = graph_response_json_schema()
        node_refs = s["properties"]["nodes"]["items"]["properties"]["chunk_ids"]["items"]
        edge_refs = s["properties"]["edges"]["items"]["properties"]["evidence_chunk_ids"]["items"]
        assert node_refs["pattern"] == "^c[0-9]+$"
        assert edge_refs["pattern"] == "^c[0-9]+$"


class TestExtractJsonObject:
    def test_strips_markdown_fences(self) -> None:
        assert _extract_json_object('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert _extract_json_object('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_surrounding_prose(self) -> None:
        assert _extract_json_object('Sure! {"a": 1} hope that helps') == '{"a": 1}'

    def test_plain_json_unchanged(self) -> None:
        assert _extract_json_object('{"a": 1}') == '{"a": 1}'

    def test_no_object_returns_stripped(self) -> None:
        assert _extract_json_object("  no json here  ") == "no json here"
