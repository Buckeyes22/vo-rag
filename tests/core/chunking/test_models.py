from __future__ import annotations

from vo_rag.core.chunking.ids import _chunk_id
from vo_rag.core.chunking.models import RetrievalChunk


def test_retrieval_chunk_metadata_defaults_to_empty_dict() -> None:
    chunk = RetrievalChunk(
        chunk_id="c",
        chunk_type="text",
        document_id="d",
        source_id="s",
        section_path="[Book]",
        section_title="Book",
        content="Brake pressure trace.",
        embedding_text="[Book]\n\nBrake pressure trace.",
        token_count=5,
    )

    assert chunk.metadata == {}


def test_chunk_id_is_deterministic_for_source_content_and_stage() -> None:
    first = _chunk_id("source", "same content", "text")
    second = _chunk_id("source", "same content", "text")

    assert first == second
    assert _chunk_id("source", "different content", "text") != first
    assert _chunk_id("source", "same content", "table") != first
