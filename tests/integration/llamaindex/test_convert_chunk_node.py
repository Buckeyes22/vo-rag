"""Contract tests for RetrievalChunk <-> TextNode conversion (VOR-177)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llama_index.core.schema import MetadataMode

from vo_rag.integrations.llamaindex.convert import (
    chunk_to_text_node,
    text_node_to_chunk,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vo_rag.core.chunking.models import RetrievalChunk


def test_chunk_to_node_preserves_id_and_embed_view(
    make_chunk: Callable[..., RetrievalChunk],
) -> None:
    chunk = make_chunk(section_path="A > B", content="body text")
    node = chunk_to_text_node(chunk)

    assert node.id_ == chunk.chunk_id

    embed_view = node.get_content(metadata_mode=MetadataMode.EMBED)
    assert "A > B" in embed_view
    assert "body text" in embed_view

    llm_view = node.get_content(metadata_mode=MetadataMode.LLM)
    assert "body text" in llm_view
    assert "A > B" not in llm_view


def test_round_trip_identity(make_chunk: Callable[..., RetrievalChunk]) -> None:
    chunk = make_chunk()
    round_tripped = text_node_to_chunk(chunk_to_text_node(chunk))
    assert round_tripped.chunk_id == chunk.chunk_id


def test_round_trip_preserves_all_retrieval_chunk_fields(
    make_chunk: Callable[..., RetrievalChunk],
) -> None:
    chunk = make_chunk(
        chunk_type="visual",
        document_id="doc-77",
        source_id="source-77",
        page_number=42,
        page_range="42-43",
        section_path="Book > Chapter > Section",
        section_title="Section",
        chapter_path="Book > Chapter",
        content="body text with a figure",
        embedding_text="custom embedding surface",
        token_count=123,
        has_image=True,
        image_id="image-77",
        image_ref="garage/key/77.png",
        vlm_description="Diagram showing load transfer.",
        metadata={
            "front_matter_kind": "preface",
            "source_book": "Vehicle Dynamics",
            "domain_tags": ["vehicle_dynamics", "tires"],
            "nested": {"chapter": 3},
        },
    )

    round_tripped = text_node_to_chunk(chunk_to_text_node(chunk))

    assert round_tripped.model_dump() == chunk.model_dump()


def test_round_trip_preserves_chunk_metadata(
    make_chunk: Callable[..., RetrievalChunk],
) -> None:
    chunk = make_chunk(
        metadata={
            "front_matter_kind": "preface",
            "source_book": "Vehicle Dynamics",
            "domain_tags": ["vehicle_dynamics", "tires"],
        }
    )

    round_tripped = text_node_to_chunk(chunk_to_text_node(chunk))

    assert round_tripped.metadata == chunk.metadata


def test_text_node_to_chunk_rejects_non_mapping_vo_rag_metadata(
    make_chunk: Callable[..., RetrievalChunk],
) -> None:
    node = chunk_to_text_node(make_chunk())
    node.metadata["_vo_rag_chunk_metadata"] = ["not", "a", "mapping"]

    with pytest.raises(ValueError, match="_vo_rag_chunk_metadata"):
        text_node_to_chunk(node)


def test_round_trip_preserves_verbatim_embedding_text_when_custom(
    make_chunk: Callable[..., RetrievalChunk],
) -> None:
    chunk = make_chunk(
        section_path="Chapter 2 > Slip Angle",
        content="body text",
        embedding_text="verbatim embed text with producer-specific normalization",
    )
    assert chunk.embedding_text != f"{chunk.section_path}\n\n{chunk.content}"

    round_tripped = text_node_to_chunk(chunk_to_text_node(chunk))

    assert round_tripped.embedding_text == chunk.embedding_text
