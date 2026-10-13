"""Contract tests for VoRagPGRetriever (VOR-215).

Validates that the CustomPGRetriever adapter:
- calls graph_query_to_evidence_chunks with the right args,
- returns NodeWithScore objects whose node.id_ equals chunk.chunk_id,
- carries text and metadata through faithfully,
- never re-keys or invents chunks beyond the traversal result.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from llama_index.core.schema import TextNode

from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.integrations.llamaindex.pg_retriever import VoRagPGRetriever


def _make_chunk(**overrides: Any) -> RetrievalChunk:
    defaults: dict[str, Any] = {
        "chunk_id": "chunk-default",
        "chunk_type": "text",
        "document_id": "doc-1",
        "source_id": "source-1",
        "page_number": 1,
        "page_range": "1-2",
        "section_path": "Chapter 1 > Section A",
        "section_title": "Section A",
        "chapter_path": "Chapter 1",
        "content": "default content",
        "embedding_text": "default content",
        "token_count": 2,
        "has_image": False,
        "metadata": {},
    }
    defaults.update(overrides)
    return RetrievalChunk.model_validate(defaults)


def _fake_graph_store() -> Any:
    """Minimal PropertyGraphStore stand-in for CustomPGRetriever construction."""
    store = MagicMock()
    store._client = MagicMock()
    return store


def _fake_qdrant_store() -> Any:
    """Minimal GraphQdrantStore stand-in for the qdrant= argument."""
    store = MagicMock()
    store._client = MagicMock()
    return store


def test_custom_retrieve_maps_chunks_to_nodes_with_content_addressed_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned NodeWithScore.id_ must equal RetrievalChunk.chunk_id (no re-keying)."""
    captured_calls: list[dict[str, Any]] = []

    async def fake_graph_query_to_evidence_chunks(**kwargs: Any) -> list[RetrievalChunk]:
        captured_calls.append(kwargs)
        return [
            _make_chunk(
                chunk_id="chunk-alpha",
                content="Alpha content",
                source_id="book-a",
                section_path="Ch 1 > A",
            ),
            _make_chunk(
                chunk_id="chunk-beta",
                content="Beta content",
                source_id="book-b",
                section_path="Ch 2 > B",
            ),
        ]

    monkeypatch.setattr(
        "vo_rag.integrations.llamaindex.pg_retriever.graph_query_to_evidence_chunks",
        fake_graph_query_to_evidence_chunks,
    )

    qdrant_store = _fake_qdrant_store()
    retriever = VoRagPGRetriever(
        graph_store=_fake_graph_store(),
        store=qdrant_store,
        top_k=5,
    )

    nodes = retriever.custom_retrieve("how does downforce scale with speed")

    assert isinstance(nodes, list)
    text_nodes = [cast(TextNode, n.node) for n in nodes]
    assert {n.id_ for n in text_nodes} == {"chunk-alpha", "chunk-beta"}
    assert [n.text for n in text_nodes] == ["Alpha content", "Beta content"]
    assert [n.metadata["source_id"] for n in text_nodes] == ["book-a", "book-b"]
    assert [n.metadata["section_path"] for n in text_nodes] == ["Ch 1 > A", "Ch 2 > B"]
    assert [n.score for n in nodes] == [None, None]

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["query_text"] == "how does downforce scale with speed"
    # The retriever resolves seeds from the query string (entity_labels=None is a
    # no-op in graph_query_to_evidence_chunks — see pg_retriever.py).
    assert call["entity_labels"] == ["how does downforce scale with speed"]
    assert call["qdrant"] is qdrant_store
    assert call["top_k"] == 5


def test_custom_retrieve_returns_empty_list_when_no_evidence_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Traversal returning no evidence chunks yields an empty NodeWithScore list."""

    async def fake_graph_query_to_evidence_chunks(**kwargs: Any) -> list[RetrievalChunk]:
        return []

    monkeypatch.setattr(
        "vo_rag.integrations.llamaindex.pg_retriever.graph_query_to_evidence_chunks",
        fake_graph_query_to_evidence_chunks,
    )

    retriever = VoRagPGRetriever(
        graph_store=_fake_graph_store(),
        store=_fake_qdrant_store(),
        top_k=3,
    )

    assert retriever.custom_retrieve("nonmatching query") == []


def test_custom_retrieve_uses_default_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    """When top_k is omitted, the retriever defaults to 10."""
    captured: list[dict[str, Any]] = []

    async def fake_graph_query_to_evidence_chunks(**kwargs: Any) -> list[RetrievalChunk]:
        captured.append(kwargs)
        return []

    monkeypatch.setattr(
        "vo_rag.integrations.llamaindex.pg_retriever.graph_query_to_evidence_chunks",
        fake_graph_query_to_evidence_chunks,
    )

    retriever = VoRagPGRetriever(
        graph_store=_fake_graph_store(),
        store=_fake_qdrant_store(),
    )

    retriever.custom_retrieve("anything")

    assert captured[0]["top_k"] == 10


def test_init_requires_store_with_clear_value_error() -> None:
    with pytest.raises(ValueError, match="requires store=GraphQdrantStore"):
        VoRagPGRetriever(graph_store=_fake_graph_store())


@pytest.mark.parametrize("top_k", [0, -1, 1.5, "5", True])
def test_init_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        VoRagPGRetriever(
            graph_store=_fake_graph_store(),
            store=_fake_qdrant_store(),
            top_k=top_k,
        )
