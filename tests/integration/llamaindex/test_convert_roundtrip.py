"""Layer-1 round-trip ID-fidelity contract tests for chunk<->TextNode converters (VOR-180).

These tests enforce the single load-bearing invariant of the B-thin strategy:
a TextNode derived from a vo-rag chunk must carry the chunk's logical id as its
node.id_, or LlamaIndex's index/storage layer re-keys the node and vo-rag's
deterministic-write idempotency breaks.

The tests are written to FAIL against a naive-but-plausible converter that lets
LlamaIndex auto-generate node.id_ (random UUID per construction). The control-
TextNode-with-no-id_ assertion proves the converter added no entropy and let no
random UUID survive.

Adjacent converter tests:
- search_result_to_node_with_score / node_with_score_to_search_result: VOR-178
  (see test_convert_identity.py)
- graph_node_to_entity_node / graph_edge_to_relation: VOR-179 (delegate to the
  property-graph adapter's converters; see test_convert_graph_stubs.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import MetadataMode, TextNode

from vo_rag.integrations.llamaindex.convert import (
    chunk_to_text_node,
    text_node_to_chunk,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vo_rag.core.chunking.models import RetrievalChunk


def test_round_trip_identity(make_chunk: Callable[..., RetrievalChunk]) -> None:
    """chunk -> TextNode -> chunk must reproduce the original chunk_id."""
    chunk = make_chunk()
    round_tripped = text_node_to_chunk(chunk_to_text_node(chunk))
    assert round_tripped.chunk_id == chunk.chunk_id


def test_rerun_determinism_no_auto_uuid(
    make_chunk: Callable[..., RetrievalChunk],
) -> None:
    """Two conversions of the same chunk yield identical node.id_ values, and
    that id differs from a control TextNode built with no explicit id_ (whose
    auto-generated UUID changes every run).

    This is the guard against the auto-UUID leak: if the converter forgot to
    set id_=chunk.chunk_id, LlamaIndex would mint a fresh random UUID each time
    and the two nodes would differ.
    """
    chunk = make_chunk()
    node_a = chunk_to_text_node(chunk)
    node_b = chunk_to_text_node(chunk)

    assert node_a.id_ == node_b.id_, "Converter must be deterministic across runs"

    control = TextNode(text=chunk.content)
    assert node_a.id_ != control.id_, (
        "Converter output must differ from auto-generated UUID control"
    )


def test_embed_vs_llm_view(make_chunk: Callable[..., RetrievalChunk]) -> None:
    """EMBED view reproduces embedding_text (section_path + content); LLM view
    contains content but not section_path."""
    chunk = make_chunk(section_path="A > B", content="body text")
    node = chunk_to_text_node(chunk)

    embed_view = node.get_content(metadata_mode=MetadataMode.EMBED)
    assert "A > B" in embed_view
    assert "body text" in embed_view

    llm_view = node.get_content(metadata_mode=MetadataMode.LLM)
    assert "body text" in llm_view
    assert "A > B" not in llm_view
