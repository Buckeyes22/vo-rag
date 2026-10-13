"""Coverage-gate tests for the vo-rag PropertyGraphIndex path extractor."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from llama_index.core.schema import BaseNode, TextNode

from vo_rag.core.graph.extraction import GraphExtractionConfig, GraphExtractionResult
from vo_rag.core.graph.schemas import (
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    edge_id_for,
    node_id_for,
)
from vo_rag.integrations.llamaindex import pg_extractor
from vo_rag.integrations.llamaindex.pg_extractor import (
    DEFAULT_DIRECTION_COVERAGE_FLOOR,
    VoRagPathExtractor,
)

LLMCall = Callable[[str, str, int, float], Awaitable[Any]]


async def _unused_llm_call(
    _system: str,
    _user: str,
    _max_tokens: int,
    _temperature: float,
) -> Any:
    raise AssertionError("extract_graph_from_book is monkeypatched in this test")


def _text_node() -> TextNode:
    return TextNode(
        text="The graph edge direction fixture lives here.",
        id_="chunk-1",
        metadata={
            "chunk_type": "text",
            "document_id": "doc-1",
            "source_id": "source-1",
            "section_path": "Chapter 1 > Graphs",
            "section_title": "Graphs",
            "chapter_path": "Chapter 1",
            "token_count": 7,
        },
    )


def _graph_node(
    *,
    label: str,
    node_type: GraphNodeType = GraphNodeType.CONCEPT,
) -> GraphNode:
    return GraphNode(
        node_id=node_id_for(node_type=node_type, canonical_label=label),
        node_type=node_type,
        label=label,
        aliases=[],
        book_ids=["book-1"],
        chunk_ids=["chunk-1"],
        created_at=datetime(2026, 6, 8, tzinfo=UTC),
        prompt_hash="prompt-hash",
    )


def _graph_edge(
    *,
    source: GraphNode,
    target: GraphNode,
    edge_type: GraphEdgeType = GraphEdgeType.PART_OF,
) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id_for(
            source_node_id=source.node_id,
            target_node_id=target.node_id,
            edge_type=edge_type,
        ),
        source_node_id=source.node_id,
        target_node_id=target.node_id,
        edge_type=edge_type,
        book_ids=["book-1"],
        evidence_chunk_ids=["chunk-1"],
        created_at=datetime(2026, 6, 8, tzinfo=UTC),
        prompt_hash="prompt-hash",
    )


def _low_coverage_graph() -> tuple[list[GraphNode], list[GraphEdge]]:
    alpha = _graph_node(label="Hydraulic bias")
    beta = _graph_node(label="Thermal offset")
    return [alpha, beta], [
        _graph_edge(source=alpha, target=beta),
        _graph_edge(source=beta, target=alpha),
    ]


def _high_coverage_graph() -> tuple[list[GraphNode], list[GraphEdge]]:
    system = _graph_node(label="Brake system", node_type=GraphNodeType.COMPONENT)
    caliper = _graph_node(label="Brake caliper", node_type=GraphNodeType.COMPONENT)
    return [system, caliper], [_graph_edge(source=system, target=caliper)]


def _extraction_result(
    *,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> GraphExtractionResult:
    return GraphExtractionResult(
        book_id="book-1",
        nodes=nodes,
        edges=edges,
        validation_errors=[],
        prompt_hash="prompt-hash",
        input_tokens=0,
        output_tokens=0,
        wall_time_ms=0,
    )


async def _run_extractor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    fallback_llm_call: LLMCall | None = None,
) -> Sequence[BaseNode]:
    async def fake_extract_graph_from_book(**_kwargs: Any) -> GraphExtractionResult:
        return _extraction_result(nodes=nodes, edges=edges)

    monkeypatch.setattr(
        pg_extractor,
        "extract_graph_from_book",
        fake_extract_graph_from_book,
    )
    extractor = VoRagPathExtractor(
        book_id="book-1",
        title="Race Car Systems",
        author="Ada",
        config=GraphExtractionConfig(prompt_hash="prompt-hash"),
        llm_call=_unused_llm_call,
        fallback_llm_call=fallback_llm_call,
    )
    return await extractor.acall([_text_node()])


def test_default_direction_coverage_floor_matches_documented_gate() -> None:
    assert DEFAULT_DIRECTION_COVERAGE_FLOOR == 0.85


@pytest.mark.asyncio
async def test_low_coverage_with_fallback_reorients_only_low_confidence_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes, edges = _low_coverage_graph()
    expected_low_confidence_ids = [edge.edge_id for edge in edges]
    fallback_calls: list[tuple[str, str, int, float]] = []
    orient_calls: list[list[str]] = []

    async def fallback_llm_call(
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> Any:
        fallback_calls.append((system, user, max_tokens, temperature))
        return object()

    async def fake_orient_low_confidence_via_llm(
        call_nodes: list[GraphNode],
        call_edges: list[GraphEdge],
        low_confidence_edge_ids: list[str],
        llm_call: LLMCall,
    ) -> list[GraphEdge]:
        assert call_nodes == nodes
        assert [edge.edge_id for edge in call_edges] == expected_low_confidence_ids
        assert llm_call is fallback_llm_call
        orient_calls.append(low_confidence_edge_ids)
        await llm_call("system", "user", 128, 0.0)
        return [
            call_edges[0].model_copy(
                update={
                    "source_node_id": call_edges[0].target_node_id,
                    "target_node_id": call_edges[0].source_node_id,
                }
            ),
            call_edges[1],
        ]

    monkeypatch.setattr(
        pg_extractor,
        "orient_low_confidence_via_llm",
        fake_orient_low_confidence_via_llm,
    )

    out = await _run_extractor(
        monkeypatch,
        nodes=nodes,
        edges=edges,
        fallback_llm_call=fallback_llm_call,
    )

    metadata = out[0].metadata
    assert orient_calls == [expected_low_confidence_ids]
    assert fallback_calls == [("system", "user", 128, 0.0)]
    assert metadata["vo_graph_coverage"] == 0.0
    assert metadata["vo_graph_low_confidence_edge_ids"] == expected_low_confidence_ids
    assert metadata["vo_graph_orientation_fallback_ran"] is True
    assert metadata["vo_graph_edges"] == [
        {
            **edges[0].model_dump(mode="json"),
            "source_node_id": edges[0].target_node_id,
            "target_node_id": edges[0].source_node_id,
        },
        edges[1].model_dump(mode="json"),
    ]


@pytest.mark.asyncio
async def test_high_coverage_skips_configured_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes, edges = _high_coverage_graph()

    async def fallback_llm_call(
        _system: str,
        _user: str,
        _max_tokens: int,
        _temperature: float,
    ) -> Any:
        raise AssertionError("fallback llm_call must not run above the coverage floor")

    async def fail_orient_low_confidence_via_llm(
        _call_nodes: list[GraphNode],
        _call_edges: list[GraphEdge],
        _low_confidence_edge_ids: list[str],
        _llm_call: LLMCall,
    ) -> list[GraphEdge]:
        raise AssertionError("orientation fallback must not run above the floor")

    monkeypatch.setattr(
        pg_extractor,
        "orient_low_confidence_via_llm",
        fail_orient_low_confidence_via_llm,
    )

    out = await _run_extractor(
        monkeypatch,
        nodes=nodes,
        edges=edges,
        fallback_llm_call=fallback_llm_call,
    )

    metadata = out[0].metadata
    assert metadata["vo_graph_coverage"] == 1.0
    assert metadata["vo_graph_low_confidence_edge_ids"] == []
    assert metadata["vo_graph_orientation_fallback_ran"] is False
    assert metadata["vo_graph_edges"] == [
        {
            **edges[0].model_dump(mode="json"),
            "source_node_id": edges[0].target_node_id,
            "target_node_id": edges[0].source_node_id,
        }
    ]


@pytest.mark.asyncio
async def test_low_coverage_without_fallback_keeps_deterministic_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes, edges = _low_coverage_graph()
    expected_low_confidence_ids = [edge.edge_id for edge in edges]

    async def fail_orient_low_confidence_via_llm(
        _call_nodes: list[GraphNode],
        _call_edges: list[GraphEdge],
        _low_confidence_edge_ids: list[str],
        _llm_call: LLMCall,
    ) -> list[GraphEdge]:
        raise AssertionError("orientation fallback must not run without a hook")

    monkeypatch.setattr(
        pg_extractor,
        "orient_low_confidence_via_llm",
        fail_orient_low_confidence_via_llm,
    )

    out = await _run_extractor(monkeypatch, nodes=nodes, edges=edges)

    metadata = out[0].metadata
    assert metadata["vo_graph_coverage"] == 0.0
    assert metadata["vo_graph_low_confidence_edge_ids"] == expected_low_confidence_ids
    assert metadata["vo_graph_orientation_fallback_ran"] is False
    assert metadata["vo_graph_edges"] == [
        edges[0].model_dump(mode="json"),
        edges[1].model_dump(mode="json"),
    ]
