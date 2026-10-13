"""Tests for the retrieval context helpers.

These are pure helpers used by the answer route:

- ``resolve_answer_collections``: maps raw → sterile for STERILE /
  PUBLICATION / PARTNER modes (closes the raw hosted-egress path
  except when ``allow_raw_hosted_egress`` is set, which is enforced
  upstream in the route, not here).
- ``parent_ids_for_results``: unique parent/chunk ids in input order
  (used to fetch sub-chunks for the compliance gate).
- ``source_ids``: sorted unique source ids (used in answer_log rows).
- ``format_answer_context``: the per-chunk textual block the LLM sees
  via the protected completion path.
"""

from __future__ import annotations

from typing import Any

from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import SearchResult


def _result(
    point_id: str = "p1",
    *,
    source_id: str = "src-1",
    chunk_id: str = "chunk-1",
    parent_chunk_id: str | None = None,
    title: str = "Brake Handbook",
    page: int | None = 7,
    content: str = "Brake bias should be adjusted gradually.",
    collection: str = "vo_rag_books_raw_v1",
) -> SearchResult:
    payload: dict[str, Any] = {
        "source_id": source_id,
        "chunk_id": chunk_id,
        "source_title": title,
        "page_number": page,
        "content": content,
    }
    if parent_chunk_id is not None:
        payload["parent_chunk_id"] = parent_chunk_id
    return SearchResult(id=point_id, score=0.9, collection=collection, payload=payload)


# --------------------------------------------------------------------------- #
# resolve_answer_collections
# --------------------------------------------------------------------------- #


def test_resolve_answer_collections_internal_default_is_sterile_under_sterile_scope() -> None:
    """The STERILE context scope is unconditional — even INTERNAL mode
    defaults to the sterile books collection, because STERILE means
    'never show the operator the raw text'. The mode is the load-bearing
    remap trigger; the scope is the load-bearing default-selection
    trigger. Both contribute to keeping raw out of the answer path."""
    from vo_rag.core.answer.context import (
        ContextScope,
        resolve_answer_collections,
    )

    out = resolve_answer_collections(
        mode=QueryMode.INTERNAL,
        requested=None,
        context_scope=ContextScope.STERILE,
    )
    assert "vo_rag_books_sterile_v1" in out
    assert "vo_rag_books_raw_v1" not in out


def test_resolve_answer_collections_sterile_scope_remaps_raw_books() -> None:
    """STERILE scope always remaps raw→sterile, even for INTERNAL mode.
    The contract: STERILE means "never show the operator the raw text"."""
    from vo_rag.core.answer.context import (
        ContextScope,
        resolve_answer_collections,
    )

    out = resolve_answer_collections(
        mode=QueryMode.INTERNAL,
        requested=["vo_rag_books_raw_v1"],
        context_scope=ContextScope.STERILE,
    )
    assert "vo_rag_books_raw_v1" not in out
    assert "vo_rag_books_sterile_v1" in out


def test_resolve_answer_collections_publication_mode_remaps_raw() -> None:
    from vo_rag.core.answer.context import (
        ContextScope,
        resolve_answer_collections,
    )

    out = resolve_answer_collections(
        mode=QueryMode.PUBLICATION,
        requested=["vo_rag_books_raw_v1"],
        context_scope=ContextScope.STERILE,
    )
    assert "vo_rag_books_raw_v1" not in out
    assert "vo_rag_books_sterile_v1" in out


def test_resolve_answer_collections_partner_mode_remaps_raw() -> None:
    from vo_rag.core.answer.context import (
        ContextScope,
        resolve_answer_collections,
    )

    out = resolve_answer_collections(
        mode=QueryMode.PARTNER,
        requested=["vo_rag_books_raw_v1"],
        context_scope=ContextScope.STERILE,
    )
    assert "vo_rag_books_raw_v1" not in out
    assert "vo_rag_books_sterile_v1" in out


def test_resolve_answer_collections_deduplicates_after_remap() -> None:
    """PUBLICATION with both raw and sterile requested → both remap to
    sterile, and the dedup must collapse to a single sterile entry."""
    from vo_rag.core.answer.context import (
        ContextScope,
        resolve_answer_collections,
    )

    out = resolve_answer_collections(
        mode=QueryMode.PUBLICATION,
        requested=["vo_rag_books_raw_v1", "vo_rag_books_sterile_v1"],
        context_scope=ContextScope.STERILE,
    )
    assert out.count("vo_rag_books_sterile_v1") == 1


def test_resolve_answer_collections_raw_scope_keeps_raw_books() -> None:
    """The RAW context scope preserves raw collections (INTERNAL only);
    this is what internal operators use to query pre-sterilization
    text. The 403 gate that blocks raw hosted egress is a separate
    check in the route, not here."""
    from vo_rag.core.answer.context import (
        ContextScope,
        resolve_answer_collections,
    )

    out = resolve_answer_collections(
        mode=QueryMode.INTERNAL,
        requested=["vo_rag_books_raw_v1"],
        context_scope=ContextScope.RAW,
    )
    assert "vo_rag_books_raw_v1" in out


def test_resolve_answer_collections_publication_ignores_raw_scope() -> None:
    """PUBLICATION + RAW is an inconsistent combo: the mode is the
    load-bearing remap trigger, not the context scope. Raw collection
    still gets remapped to sterile under PUBLICATION."""
    from vo_rag.core.answer.context import (
        ContextScope,
        resolve_answer_collections,
    )

    out = resolve_answer_collections(
        mode=QueryMode.PUBLICATION,
        requested=["vo_rag_books_raw_v1"],
        context_scope=ContextScope.RAW,
    )
    assert "vo_rag_books_raw_v1" not in out
    assert "vo_rag_books_sterile_v1" in out


# --------------------------------------------------------------------------- #
# parent_ids_for_results
# --------------------------------------------------------------------------- #


def test_parent_ids_for_results_prefers_parent_chunk_id() -> None:
    from vo_rag.core.answer.context import parent_ids_for_results

    results = [
        _result(point_id="p1", chunk_id="c1", parent_chunk_id="parent-A"),
        _result(point_id="p2", chunk_id="c2", parent_chunk_id="parent-B"),
    ]
    assert parent_ids_for_results(results) == ["parent-A", "parent-B"]


def test_parent_ids_for_results_falls_back_to_chunk_id() -> None:
    """No parent_chunk_id → use chunk_id. This matches the writer-side
    contract: sub-chunks index by parent_chunk_id; raw chunks index
    by chunk_id. The helper must produce a single canonical key per
    result for the sub-chunk fetch."""
    from vo_rag.core.answer.context import parent_ids_for_results

    results = [
        _result(point_id="p1", chunk_id="c1"),
        _result(point_id="p2", chunk_id="c2"),
    ]
    assert parent_ids_for_results(results) == ["c1", "c2"]


def test_parent_ids_for_results_dedupes_in_input_order() -> None:
    """Two results pointing at the same parent should yield one entry;
    input order is preserved (this is the SQL-IN order the sub-chunk
    query sees)."""
    from vo_rag.core.answer.context import parent_ids_for_results

    results = [
        _result(point_id="p1", chunk_id="c1", parent_chunk_id="parent-A"),
        _result(point_id="p2", chunk_id="c2", parent_chunk_id="parent-A"),
        _result(point_id="p3", chunk_id="c3", parent_chunk_id="parent-B"),
    ]
    assert parent_ids_for_results(results) == ["parent-A", "parent-B"]


def test_parent_ids_for_results_falls_back_to_point_id_when_no_chunk_keys() -> None:
    """Defense-in-depth: a result with no chunk_id or parent_chunk_id
    still yields SOMETHING stable. The Qdrant point id is the last
    resort; this keeps the sub-chunk fetch from skipping the row."""
    from vo_rag.core.answer.context import parent_ids_for_results

    r = SearchResult(id="point-Z", score=0.5, collection="x", payload={})
    assert parent_ids_for_results([r]) == ["point-Z"]


# --------------------------------------------------------------------------- #
# source_ids
# --------------------------------------------------------------------------- #


def test_source_ids_returns_sorted_unique_source_ids() -> None:
    from vo_rag.core.answer.context import source_ids

    results = [
        _result(source_id="src-c"),
        _result(source_id="src-a"),
        _result(source_id="src-b"),
    ]
    assert source_ids(results) == ["src-a", "src-b", "src-c"]


def test_source_ids_deduplicates_repeats() -> None:
    from vo_rag.core.answer.context import source_ids

    results = [
        _result(source_id="src-1"),
        _result(source_id="src-1"),
        _result(source_id="src-2"),
    ]
    assert source_ids(results) == ["src-1", "src-2"]


def test_source_ids_skips_results_with_no_source_id() -> None:
    """A result without a source_id in payload contributes nothing —
    no implicit fall-back to point id. The compliance log column is
    source_id-shaped; filling it with a Qdrant point id would be
    meaningless and pollute the audit trail."""
    from vo_rag.core.answer.context import source_ids

    results = [
        _result(source_id="src-1"),
        SearchResult(id="p2", score=0.5, collection="x", payload={}),
    ]
    assert source_ids(results) == ["src-1"]


# --------------------------------------------------------------------------- #
# format_answer_context
# --------------------------------------------------------------------------- #


def test_format_answer_context_emits_s1_s2_markers() -> None:
    from vo_rag.core.answer.context import format_answer_context

    out = format_answer_context(
        [
            _result(point_id="p1", content="first chunk"),
            _result(point_id="p2", content="second chunk"),
        ],
        max_chunks=8,
    )
    assert "[S1]" in out
    assert "[S2]" in out


def test_format_answer_context_emits_point_chunk_source_id_title_page() -> None:
    from vo_rag.core.answer.context import format_answer_context

    out = format_answer_context(
        [_result(point_id="p1", chunk_id="c1", source_id="s1", title="Title", page=42)],
        max_chunks=8,
    )
    assert "point_id: p1" in out
    assert "chunk_id: c1" in out
    assert "source_id: s1" in out
    assert "title: Title" in out
    assert "page: 42" in out


def test_format_answer_context_chunk_id_takes_priority_over_parent() -> None:
    """In the rendered context block, ``chunk_id`` is the primary
    citation key — what the model writes next to ``[S1]``. The
    parent_chunk_id is the sub-chunk lookup key (handled separately
    in ``parent_ids_for_results``). Both orderings are intentional:
    citations → chunk_id, sub-chunks → parent_chunk_id."""
    from vo_rag.core.answer.context import format_answer_context

    out = format_answer_context(
        [_result(point_id="p1", chunk_id="c1", parent_chunk_id="parent-X")],
        max_chunks=8,
    )
    assert "chunk_id: c1" in out
    assert "chunk_id: parent-X" not in out


def test_format_answer_context_falls_back_to_parent_when_chunk_id_missing() -> None:
    """If a result has parent_chunk_id but no chunk_id, the rendered
    block surfaces the parent — the citation key is whatever row
    identifier we have."""
    from vo_rag.core.answer.context import format_answer_context

    out = format_answer_context(
        [_result(point_id="p1", chunk_id="", parent_chunk_id="parent-X")],
        max_chunks=8,
    )
    assert "chunk_id: parent-X" in out


def test_format_answer_context_emits_content_fallback_order() -> None:
    """The content field tries content → text → content_text → summary.
    A result with only ``text`` should still render content."""
    from vo_rag.core.answer.context import format_answer_context

    r = SearchResult(
        id="p1",
        score=0.5,
        collection="x",
        payload={"source_id": "s1", "text": "from text field"},
    )
    out = format_answer_context([r], max_chunks=8)
    assert "from text field" in out


def test_format_answer_context_respects_max_chunks() -> None:
    from vo_rag.core.answer.context import format_answer_context

    results = [_result(point_id=f"p{i}", content=f"c{i}") for i in range(10)]
    out = format_answer_context(results, max_chunks=3)
    assert "[S1]" in out
    assert "[S2]" in out
    assert "[S3]" in out
    assert "[S4]" not in out


def test_format_answer_context_empty_input_returns_empty_string() -> None:
    from vo_rag.core.answer.context import format_answer_context

    assert format_answer_context([], max_chunks=8) == ""


def test_format_answer_context_page_none_renders_empty() -> None:
    """A result with no page_number should not crash and should not
    surface a 'None' string in the rendered block."""
    from vo_rag.core.answer.context import format_answer_context

    r = _result(page=None)
    out = format_answer_context([r], max_chunks=8)
    assert "page: " in out  # the field is present but value is empty
    assert "page: None" not in out
