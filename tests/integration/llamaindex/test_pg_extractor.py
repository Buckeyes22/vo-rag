"""Contract tests for the vo-rag PropertyGraphIndex path extractor."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from llama_index.core.schema import TextNode
from pydantic import ValidationError

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
from vo_rag.integrations.llamaindex.pg_extractor import VoRagPathExtractor


async def _unused_llm_call(
    _system: str,
    _user: str,
    _max_tokens: int,
    _temperature: float,
) -> Any:
    raise AssertionError("extract_graph_from_book is monkeypatched in this test")


@pytest.mark.asyncio
async def test_acall_stashes_oriented_graph_metadata_and_preserves_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = datetime(2026, 6, 8, tzinfo=UTC)
    system_id = node_id_for(
        node_type=GraphNodeType.COMPONENT,
        canonical_label="Brake system",
    )
    caliper_id = node_id_for(
        node_type=GraphNodeType.COMPONENT,
        canonical_label="Brake caliper",
    )
    nodes = [
        GraphNode(
            node_id=system_id,
            node_type=GraphNodeType.COMPONENT,
            label="Brake system",
            aliases=[],
            book_ids=["book-1"],
            chunk_ids=["chunk-1"],
            created_at=created_at,
            prompt_hash="prompt-hash",
        ),
        GraphNode(
            node_id=caliper_id,
            node_type=GraphNodeType.COMPONENT,
            label="Brake caliper",
            aliases=[],
            book_ids=["book-1"],
            chunk_ids=["chunk-1"],
            created_at=created_at,
            prompt_hash="prompt-hash",
        ),
    ]
    raw_edge_id = edge_id_for(
        source_node_id=system_id,
        target_node_id=caliper_id,
        edge_type=GraphEdgeType.PART_OF,
    )
    raw_edge = GraphEdge(
        edge_id=raw_edge_id,
        source_node_id=system_id,
        target_node_id=caliper_id,
        edge_type=GraphEdgeType.PART_OF,
        book_ids=["book-1"],
        evidence_chunk_ids=["chunk-1"],
        created_at=created_at,
        prompt_hash="prompt-hash",
    )

    async def fake_extract_graph_from_book(**kwargs: Any) -> GraphExtractionResult:
        assert kwargs["book_id"] == "book-1"
        assert kwargs["title"] == "Race Car Systems"
        assert kwargs["author"] == "Ada"
        assert [chunk.chunk_id for chunk in kwargs["chunks"]] == ["chunk-1"]
        return GraphExtractionResult(
            book_id="book-1",
            nodes=nodes,
            edges=[raw_edge],
            validation_errors=[],
            prompt_hash="prompt-hash",
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    monkeypatch.setattr(
        pg_extractor,
        "extract_graph_from_book",
        fake_extract_graph_from_book,
    )

    textnode = TextNode(
        text="The brake caliper is part of the brake system.",
        id_="chunk-1",
        metadata={
            "chunk_type": "text",
            "document_id": "doc-1",
            "source_id": "source-1",
            "section_path": "Chapter 1 > Brakes",
            "section_title": "Brakes",
            "chapter_path": "Chapter 1",
            "token_count": 9,
        },
    )
    extractor = VoRagPathExtractor(
        book_id="book-1",
        title="Race Car Systems",
        author="Ada",
        config=GraphExtractionConfig(prompt_hash="prompt-hash"),
        llm_call=_unused_llm_call,
    )

    out = await extractor.acall([textnode])

    metadata = out[0].metadata
    assert metadata["vo_graph_coverage"] == 1.0
    assert metadata["vo_graph_low_confidence_edge_ids"] == []
    assert [node["node_id"] for node in metadata["vo_graph_nodes"]] == [
        system_id,
        caliper_id,
    ]
    assert metadata["vo_graph_edges"] == [
        {
            **raw_edge.model_dump(mode="json"),
            "source_node_id": caliper_id,
            "target_node_id": system_id,
        }
    ]
    assert metadata["vo_graph_edges"][0]["edge_id"] == raw_edge_id


@pytest.mark.asyncio
async def test_acall_stashes_full_graph_metadata_once_with_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = datetime(2026, 6, 8, tzinfo=UTC)
    system_id = node_id_for(
        node_type=GraphNodeType.COMPONENT,
        canonical_label="Brake system",
    )
    caliper_id = node_id_for(
        node_type=GraphNodeType.COMPONENT,
        canonical_label="Brake caliper",
    )
    nodes = [
        GraphNode(
            node_id=system_id,
            node_type=GraphNodeType.COMPONENT,
            label="Brake system",
            aliases=[],
            book_ids=["book-1"],
            chunk_ids=["chunk-1"],
            created_at=created_at,
            prompt_hash="prompt-hash",
        ),
        GraphNode(
            node_id=caliper_id,
            node_type=GraphNodeType.COMPONENT,
            label="Brake caliper",
            aliases=[],
            book_ids=["book-1"],
            chunk_ids=["chunk-2"],
            created_at=created_at,
            prompt_hash="prompt-hash",
        ),
    ]
    raw_edge = GraphEdge(
        edge_id=edge_id_for(
            source_node_id=system_id,
            target_node_id=caliper_id,
            edge_type=GraphEdgeType.PART_OF,
        ),
        source_node_id=system_id,
        target_node_id=caliper_id,
        edge_type=GraphEdgeType.PART_OF,
        book_ids=["book-1"],
        evidence_chunk_ids=["chunk-1", "chunk-2"],
        created_at=created_at,
        prompt_hash="prompt-hash",
    )

    async def fake_extract_graph_from_book(**kwargs: Any) -> GraphExtractionResult:
        assert [chunk.chunk_id for chunk in kwargs["chunks"]] == ["chunk-1", "chunk-2"]
        return GraphExtractionResult(
            book_id="book-1",
            nodes=nodes,
            edges=[raw_edge],
            validation_errors=[],
            prompt_hash="prompt-hash",
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    monkeypatch.setattr(
        pg_extractor,
        "extract_graph_from_book",
        fake_extract_graph_from_book,
    )
    textnodes = [
        TextNode(
            text="The brake caliper is part of the brake system.",
            id_="chunk-1",
            metadata={
                "chunk_type": "text",
                "document_id": "doc-1",
                "source_id": "source-1",
                "section_path": "Chapter 1 > Brakes",
                "section_title": "Brakes",
                "chapter_path": "Chapter 1",
                "token_count": 9,
            },
        ),
        TextNode(
            text="Brake system details continue on the next page.",
            id_="chunk-2",
            metadata={
                "chunk_type": "text",
                "document_id": "doc-1",
                "source_id": "source-1",
                "section_path": "Chapter 1 > Brakes",
                "section_title": "Brakes",
                "chapter_path": "Chapter 1",
                "token_count": 8,
            },
        ),
    ]
    extractor = VoRagPathExtractor(
        book_id="book-1",
        title="Race Car Systems",
        author="Ada",
        config=GraphExtractionConfig(prompt_hash="prompt-hash"),
        llm_call=_unused_llm_call,
    )

    out = await extractor.acall(textnodes)

    metadatas = [node.metadata for node in out]
    full_graph_metadatas = [
        metadata
        for metadata in metadatas
        if "vo_graph_nodes" in metadata or "vo_graph_edges" in metadata
    ]
    assert len(full_graph_metadatas) == 1
    assert metadatas[0]["vo_graph_payload_role"] == "primary"
    assert metadatas[1]["vo_graph_payload_role"] == "reference"
    assert metadatas[0]["vo_graph_payload_node_id"] == out[0].node_id
    assert metadatas[1]["vo_graph_payload_node_id"] == out[0].node_id
    assert metadatas[0]["vo_graph_node_count"] == 2
    assert metadatas[1]["vo_graph_node_count"] == 2
    assert metadatas[0]["vo_graph_edge_count"] == 1
    assert metadatas[1]["vo_graph_edge_count"] == 1


@pytest.mark.parametrize("coverage_floor", [-0.01, 1.01])
def test_coverage_floor_rejects_values_outside_probability_range(
    coverage_floor: float,
) -> None:
    with pytest.raises(ValidationError):
        VoRagPathExtractor(
            book_id="book-1",
            title="Race Car Systems",
            author="Ada",
            config=GraphExtractionConfig(prompt_hash="prompt-hash"),
            llm_call=_unused_llm_call,
            coverage_floor=coverage_floor,
        )
