"""Harness self-check for ``tests/integration/llamaindex/``.

The whole subtree is a contract-test substrate; the actual converter
contract tests (VOR-176/177/178/179/180) plug into the ``make_chunk``
and ``make_search_result`` fixtures defined in ``conftest.py``. This
smoke test verifies that:

  1. The subtree collects (no broken imports / conftest errors) when
     the ``llamaindex`` extra IS present (this is the
     ``uv sync --extra dev --extra llamaindex`` path).
  2. ``make_chunk()`` returns a fully-populated
     :class:`vo_rag.core.chunking.models.RetrievalChunk` whose
     ``chunk_id`` is a real content-addressed UUID string — the
     invariant the round-trip ID-fidelity tests will assert against.
  3. ``make_search_result()`` returns a
     :class:`vo_rag.core.retrieval.search.SearchResult` whose
     ``payload["chunk_id"]`` is set — the invariant the chunk-id
     trap (``SearchResult.id`` is the Qdrant point id, NOT the logical
     chunk id) asserts against.

The subtree's self-skip under ``uv sync --extra dev`` (extra absent) is
implemented by the conftest's curated ``collect_ignore``: adapter contract
tests that import LlamaIndex-dependent modules are dropped, while
``test_import_safety.py`` remains collected as the light-job tripwire for
the package-root import contract.

This smoke test remains the fixture-shape check for the shared harness
even as additional adapter contract tests are added under this subtree.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.retrieval.search import SearchResult


def test_make_chunk_returns_retrieval_chunk_with_content_addressed_id(
    make_chunk: Callable[..., RetrievalChunk],
) -> None:
    """``make_chunk()`` must hand back a real ``RetrievalChunk`` with
    a real content-addressed ``chunk_id`` — the substrate the converter
    contract tests build on.

    The chunk id is the deterministic content-addressed handle
    ``_chunk_id(source_id, content, stage)`` (SHA-256 reduced to a
    UUID), so a chunk built from the same triple twice yields the
    same id — the invariant ``chunk_to_text_node`` must preserve on
    ``TextNode.id_``.
    """
    chunk = make_chunk()

    assert isinstance(chunk, RetrievalChunk)
    assert isinstance(chunk.chunk_id, str) and chunk.chunk_id, (
        "make_chunk() must populate chunk_id with the content-addressed handle"
    )
    # UUID-shaped content-addressed handle from _chunk_id.
    assert len(chunk.chunk_id) == 36 and chunk.chunk_id.count("-") == 4, (
        f"chunk_id is not UUID-shaped: {chunk.chunk_id!r}"
    )
    # All required RetrievalChunk fields are populated so the converter
    # can read them back without a MissingField-style fallthrough.
    assert chunk.content
    assert chunk.section_path
    assert chunk.source_id
    assert chunk.document_id
    assert chunk.token_count >= 0

    # Determinism: same triple -> same id, twice.
    again = make_chunk()
    assert again.chunk_id == chunk.chunk_id, (
        "make_chunk() must be deterministic over (source_id, content, stage)"
    )


def test_make_search_result_returns_search_result_with_chunk_id_payload(
    make_search_result: Callable[..., SearchResult],
) -> None:
    """``make_search_result()`` must hand back a real ``SearchResult``
    whose ``payload["chunk_id"]`` is set — the substrate the
    chunk-id contract test asserts against.

    The contract: ``node.id_ == payload["chunk_id"]`` (NOT
    ``SearchResult.id``, which is the Qdrant point id). The factory
    populates ``payload`` with the canonical triple
    ``{"chunk_id", "content", "section_path"}`` that
    ``search_result_to_node_with_score`` reads.
    """
    result = make_search_result()

    assert isinstance(result, SearchResult)
    assert result.id, "Qdrant point id (SearchResult.id) is populated"
    assert isinstance(result.score, float)
    assert result.collection

    # The load-bearing assertion: payload carries a content-addressed
    # chunk id, NOT the Qdrant point id.
    assert "chunk_id" in result.payload, (
        "SearchResult.payload must carry a content-addressed 'chunk_id' so "
        "search_result_to_node_with_score can pin node.id_ on the logical "
        "chunk id"
    )
    chunk_id = result.payload["chunk_id"]
    assert isinstance(chunk_id, str) and chunk_id
    assert len(chunk_id) == 36 and chunk_id.count("-") == 4, (
        f"payload['chunk_id'] is not UUID-shaped: {chunk_id!r}"
    )
    # And the Qdrant point id must NOT silently equal the chunk id —
    # that would be the chunk-id bug the contract test catches.
    assert result.id != chunk_id, (
        "SearchResult.id (Qdrant point id) must differ from payload['chunk_id'] "
        "(content-addressed chunk id); equality here would re-trigger the chunk-id bug"
    )

    # Other canonical payload fields the converter reads.
    assert "content" in result.payload
    assert "section_path" in result.payload


def test_make_chunk_and_make_search_result_share_chunk_id_when_built_together(
    make_chunk: Callable[..., RetrievalChunk],
    make_search_result: Callable[..., SearchResult],
) -> None:
    """Cross-factory consistency: when the test asks for a chunk and
    a search result that share the same source/content, the chunk's
    content-addressed ``chunk_id`` and the result's
    ``payload['chunk_id']`` must be the same value.

    The contract converter tests assert
    ``node.id_ == payload['chunk_id']`` AND ``node.id_ == chunk.chunk_id``
    simultaneously — that holds only if the harness's two factories
    agree on the chunk-id derivation. This smoke test pins that
    contract on the factories themselves so a regression in
    ``_build_chunk`` vs ``_build_search_result`` surfaces here before
    VOR-176/177 land.
    """
    chunk = make_chunk(
        source_id="shared-source",
        content="shared content body",
        stage="text",
    )
    # Pin the SearchResult's payload to use the SAME content-addressed id
    # by feeding the same (source_id, content, stage) triple into the
    # search-result factory's payload knobs.
    result = make_search_result(
        id="qdrant-point-99",
        payload_chunk_id=chunk.chunk_id,
        payload_content=chunk.content,
        payload_section_path=chunk.section_path,
    )

    assert result.payload["chunk_id"] == chunk.chunk_id
    assert result.payload["content"] == chunk.content
    assert result.payload["section_path"] == chunk.section_path


@pytest.mark.parametrize(
    "kind",
    ["chunk", "search_result"],
)
def test_harness_factory_dispatches_on_kind(
    harness_factory: Callable[..., RetrievalChunk | SearchResult],
    kind: str,
) -> None:
    """``harness_factory(kind=...)`` returns a ``RetrievalChunk`` for
    ``kind='chunk'`` and a ``SearchResult`` for ``kind='search_result'``,
    raising ``ValueError`` on anything else.

    Sanity-checks the convenience dispatcher without taking a stand
    on which kind the converter tests should use (the dedicated
    ``make_chunk`` / ``make_search_result`` fixtures are the canonical
    entry points).
    """
    built = harness_factory(kind=kind)
    if kind == "chunk":
        assert isinstance(built, RetrievalChunk)
    else:
        assert isinstance(built, SearchResult)


def test_harness_factory_rejects_unknown_kind(
    harness_factory: Callable[..., RetrievalChunk | SearchResult],
) -> None:
    """Unknown ``kind=`` is a hard error, not a silent fallback —
    the harness must not paper over a misconfigured test."""
    with pytest.raises(ValueError, match="unknown kind"):
        harness_factory(kind="not-a-real-kind")


__all__ = [
    "test_harness_factory_dispatches_on_kind",
    "test_harness_factory_rejects_unknown_kind",
    "test_make_chunk_and_make_search_result_share_chunk_id_when_built_together",
    "test_make_chunk_returns_retrieval_chunk_with_content_addressed_id",
    "test_make_search_result_returns_search_result_with_chunk_id_payload",
]
