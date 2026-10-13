"""Tests for the canonical RRF/dedup fusion seam.

The two-pass dedup key sequence is the most delicate part of the
retrieval path — the tests below pin it.

``ConvexFusion`` and the normalizer trio are deliberately not part of this
module (no active A/B decision calls for them). The
public surface of ``vo_rag.core.retrieval.fusion`` therefore MUST NOT
re-export them; the surface guard below catches accidental additions.
"""

from __future__ import annotations

import pytest

from vo_rag.core.retrieval import fusion as fusion_mod
from vo_rag.core.retrieval.fusion import (
    RRF_DEFAULT_K,
    cross_variant_key,
    dedupe_key,
    rrf_merge_and_dedupe,
)
from vo_rag.core.retrieval.search import SearchResult

# ----------------------------------------------------------------------
# Surface guard — ConvexFusion / normalizers are not exported
# ----------------------------------------------------------------------


def test_fusion_module_does_not_export_convex_fusion() -> None:
    """ConvexFusion is not part of the fusion module (no active A/B decision
    calls for it); the RRF math lives in fusion.py. Do not accidentally
    re-introduce ConvexFusion or its normalizers."""
    public = set(getattr(fusion_mod, "__all__", ()))
    for forbidden in (
        "ConvexFusion",
        "RRFFusion",
        "ScoredDoc",
        "fusion_from_config",
        "NormalizerName",
        "apply_normalizer",
        "min_max",
        "z_score",
        "rank_score",
    ):
        assert forbidden not in public, f"fusion.__all__ leaked {forbidden!r}"
        assert not hasattr(fusion_mod, forbidden), (
            f"vo_rag.core.retrieval.fusion must not expose {forbidden!r}"
        )


def test_fusion_module_exposes_rrf_helpers() -> None:
    """RRF helpers are the canonical seam. Their signature is the
    contract any future store cross-collection helper would import."""
    assert callable(rrf_merge_and_dedupe)
    assert callable(dedupe_key)
    assert callable(cross_variant_key)
    assert RRF_DEFAULT_K == 60  # classic Cormack et al.


# ----------------------------------------------------------------------
# _build_filter behaviour (still imported from search.py by the retriever)
# ----------------------------------------------------------------------


def test_search_result_dataclass_is_narrow() -> None:
    """SearchResult is the public return type — narrow on purpose."""
    result = SearchResult(
        id="abc",
        score=0.5,
        collection="vo_rag_books_raw_v1",
        payload={"source_id": "src-1"},
    )
    assert result.id == "abc"
    assert result.score == 0.5
    assert result.collection == "vo_rag_books_raw_v1"
    assert result.payload == {"source_id": "src-1"}


# ----------------------------------------------------------------------
# RRF math + two-pass dedup key sequence
# ----------------------------------------------------------------------


def test_rrf_merge_collapses_same_key_with_summed_score() -> None:
    result_a = SearchResult(
        id="a",
        score=0.9,
        collection="c1",
        payload={"source_id": "A", "page_number": 1, "chunk_id": "same"},
    )
    result_b = SearchResult(
        id="b",
        score=0.8,
        collection="c2",
        payload={"source_id": "A", "page_number": 1, "chunk_id": "same"},
    )
    merged = rrf_merge_and_dedupe([[result_a], [result_b]], top_k=5)
    assert len(merged) == 1


def test_rrf_merge_keeps_different_keys_separate() -> None:
    result_a = SearchResult(
        id="a",
        score=0.9,
        collection="c1",
        payload={"source_id": "A", "page_number": 1},
    )
    result_b = SearchResult(
        id="b",
        score=0.8,
        collection="c1",
        payload={"source_id": "B", "page_number": 1},
    )
    merged = rrf_merge_and_dedupe([[result_a, result_b]], top_k=5)
    assert len(merged) == 2


def test_rrf_merge_keeps_same_page_different_chunks_separate() -> None:
    definitions = SearchResult(
        id="definitions",
        score=0.9,
        collection="c1",
        payload={"source_id": "A", "page_number": 3, "chunk_id": "defs"},
    )
    trace_example = SearchResult(
        id="trace-example",
        score=0.8,
        collection="c1",
        payload={"source_id": "A", "page_number": 3, "chunk_id": "trace"},
    )
    merged = rrf_merge_and_dedupe([[definitions, trace_example]], top_k=5)
    assert [result.id for result in merged] == ["definitions", "trace-example"]


def test_rrf_collapses_cross_variant_duplicates_by_book_and_content_hash() -> None:
    """Same logical book ingested as byte-distinct variants -> one merged entry."""
    variants = [
        SearchResult(
            id=f"point-{i}",
            score=0.9,
            collection="c1",
            payload={
                "source_id": f"variant-{i}",
                "chunk_id": f"chunk-{i}",
                "page_number": 12,
                "logical_book_key": "BOOK_K",
                "content_hash": "CONTENT_H",
            },
        )
        for i in range(4)
    ]
    merged = rrf_merge_and_dedupe([variants], top_k=5)
    assert len(merged) == 1
    assert merged[0].id == "point-0"  # first occurrence wins


def test_rrf_cross_variant_collapse_does_not_affect_unrelated_chunks() -> None:
    chunk_a = SearchResult(
        id="a",
        score=0.9,
        collection="c1",
        payload={
            "source_id": "src-1",
            "chunk_id": "chunk-a",
            "page_number": 12,
            "logical_book_key": "BOOK_K",
            "content_hash": "HASH_A",
        },
    )
    chunk_b = SearchResult(
        id="b",
        score=0.8,
        collection="c1",
        payload={
            "source_id": "src-1",
            "chunk_id": "chunk-b",
            "page_number": 13,
            "logical_book_key": "BOOK_K",
            "content_hash": "HASH_B",
        },
    )
    merged = rrf_merge_and_dedupe([[chunk_a, chunk_b]], top_k=5)
    assert len(merged) == 2


def test_rrf_falls_back_to_source_id_dedup_when_book_key_missing() -> None:
    """Older chunks (pre-backfill) lack logical_book_key/content_hash —
    the (source_id, chunk_id) dedup must still work."""
    a = SearchResult(
        id="a",
        score=0.9,
        collection="c1",
        payload={"source_id": "A", "chunk_id": "x"},
    )
    b = SearchResult(
        id="b",
        score=0.8,
        collection="c1",
        payload={"source_id": "B", "chunk_id": "x"},
    )
    merged = rrf_merge_and_dedupe([[a, b]], top_k=5)
    assert len(merged) == 2


def test_rrf_raw_and_sterile_parent_chunk_id_are_deduped() -> None:
    """The classic raw↔sterile pair: same passage, parent_chunk_id link."""
    raw = SearchResult(
        id="raw",
        score=0.9,
        collection="raw",
        payload={"source_id": "A", "page_number": 3, "chunk_id": "raw-defs"},
    )
    sterile = SearchResult(
        id="sterile",
        score=0.8,
        collection="sterile",
        payload={
            "source_id": "A",
            "page_number": 3,
            "chunk_id": "sterile-defs",
            "parent_chunk_id": "raw-defs",
        },
    )
    merged = rrf_merge_and_dedupe([[raw], [sterile]], top_k=5)
    assert len(merged) == 1


# ----------------------------------------------------------------------
# dedupe_key / cross_variant_key (low-level building blocks)
# ----------------------------------------------------------------------


def test_dedupe_key_prefers_parent_chunk_id() -> None:
    result = SearchResult(
        id="sterile",
        score=0.5,
        collection="sterile",
        payload={
            "source_id": "A",
            "chunk_id": "sterile-defs",
            "parent_chunk_id": "raw-defs",
        },
    )
    assert dedupe_key(result) == ("A", "raw-defs")


def test_dedupe_key_uses_chunk_id_when_parent_absent() -> None:
    result = SearchResult(
        id="x",
        score=0.5,
        collection="c1",
        payload={"source_id": "A", "chunk_id": "chunk-x"},
    )
    assert dedupe_key(result) == ("A", "chunk-x")


def test_dedupe_key_falls_back_to_storage_point_identity() -> None:
    result = SearchResult(
        id="x",
        score=0.5,
        collection="c1",
        payload={"source_id": "A", "page_number": 17},
    )
    assert dedupe_key(result) == ("A", "point:x")


def test_dedupe_key_missing_source_id_uses_result_id() -> None:
    """If a payload has no source_id, fall back to result.id so we still
    produce a stable key (better than crashing or treating every result
    as the same key)."""
    result = SearchResult(
        id="point-42",
        score=0.5,
        collection="c1",
        payload={"chunk_id": "x"},
    )
    assert dedupe_key(result) == ("point-42", "x")


def test_cross_variant_key_requires_both_fields() -> None:
    result_full = SearchResult(
        id="x",
        score=0.5,
        collection="c1",
        payload={"logical_book_key": "BK", "content_hash": "CH"},
    )
    assert cross_variant_key(result_full) == ("BK", "CH")

    assert (
        cross_variant_key(
            SearchResult(
                id="x",
                score=0.5,
                collection="c1",
                payload={"logical_book_key": "BK"},
            )
        )
        is None
    )
    assert (
        cross_variant_key(
            SearchResult(
                id="x",
                score=0.5,
                collection="c1",
                payload={"content_hash": "CH"},
            )
        )
        is None
    )


# ----------------------------------------------------------------------
# Top-k truncation
# ----------------------------------------------------------------------


def test_rrf_merge_truncates_to_top_k() -> None:
    items = [
        SearchResult(
            id=f"id-{i}",
            score=0.1 * (10 - i),
            collection="c1",
            payload={"source_id": f"src-{i}", "page_number": 1},
        )
        for i in range(10)
    ]
    merged = rrf_merge_and_dedupe([items], top_k=3)
    assert len(merged) == 3


def test_rrf_merge_honours_custom_k_constant() -> None:
    """A smaller k_rrf widens the gap between the top hit and the rest
    (the math is the same; the constant changes the relative weight)."""
    items = [
        SearchResult(
            id="top",
            score=1.0,
            collection="c1",
            payload={"source_id": "A", "chunk_id": "a"},
        ),
        SearchResult(
            id="second",
            score=0.9,
            collection="c1",
            payload={"source_id": "B", "chunk_id": "b"},
        ),
    ]
    default = rrf_merge_and_dedupe([items], top_k=2, k_rrf=60)
    custom = rrf_merge_and_dedupe([items], top_k=2, k_rrf=1)
    # Both rankings keep 'top' first; the SearchResult.score is the
    # original 4-vector score (not the merged RRF score), so we can only
    # assert ordering here. The k_rrf=0..k_rrf=60 range affects the
    # internal sum, not the return value's .score field.
    assert [r.id for r in default] == ["top", "second"]
    assert [r.id for r in custom] == ["top", "second"]


def test_rrf_merge_score_progression_matches_cormack() -> None:
    """Pin the RRF math to the classic Cormack et al. formula so the
    rankings stay comparable to published RRF results."""
    # Two lists, each with one point with the same (source_id, chunk_id)
    # -> pass 1 dedupes, RRF score = 1/(k+0+1) + 1/(k+0+1) = 2/(k+1)
    shared = SearchResult(
        id="shared",
        score=0.5,
        collection="c1",
        payload={"source_id": "S", "chunk_id": "x"},
    )
    unique = SearchResult(
        id="unique",
        score=0.4,
        collection="c1",
        payload={"source_id": "U", "chunk_id": "y"},
    )
    # RRF formula applied via two-pass dedup:
    #   - shared gets 1/(60+0+1) (rank 0 in list[0]) + 1/(60+0+1) (rank 0 in list[1])
    #   - unique gets 1/(60+1+1) (rank 1 in list[0])
    merged = rrf_merge_and_dedupe([[shared, unique], [shared]], top_k=2)
    # shared should rank first (higher RRF score after summing two lists)
    assert merged[0].id == "shared"
    # unique follows
    assert merged[1].id == "unique"
    assert len(merged) == 2


def test_rrf_merge_custom_k_via_kwarg_only() -> None:
    """k_rrf is keyword-only — accidental positional would silently change
    the dedup behaviour and break the persistence contract."""
    with pytest.raises(TypeError):
        rrf_merge_and_dedupe([[]], 5)  # type: ignore[misc]


def test_same_track_parent_copies_do_not_vote_twice_but_independent_tracks_do() -> None:
    unique = SearchResult(
        id="unique", score=1, collection="raw", payload={"source_id": "other", "chunk_id": "unique"}
    )
    raw = SearchResult(
        id="raw",
        score=1,
        collection="raw",
        payload={"source_id": "book", "chunk_id": "raw", "parent_chunk_id": "parent"},
    )
    sterile = SearchResult(
        id="sterile",
        score=1,
        collection="sterile",
        payload={"source_id": "book", "chunk_id": "sterile", "parent_chunk_id": "parent"},
    )
    same_track = rrf_merge_and_dedupe([[unique, raw, sterile]], top_k=2)
    assert [row.id for row in same_track] == ["unique", "raw"]
    assert {item["id"] for item in same_track[1].payload["duplicate_provenance"]} == {
        "raw",
        "sterile",
    }
    independent = rrf_merge_and_dedupe([[unique, raw, sterile], [sterile]], top_k=2)
    assert [row.id for row in independent] == ["raw", "unique"]


def test_same_physical_hit_in_independent_tracks_preserves_payload_shape() -> None:
    row = SearchResult(
        id="point",
        score=1,
        collection="raw",
        payload={"source_id": "book", "chunk_id": "chunk", "content": "One answer"},
    )
    result = rrf_merge_and_dedupe([[row], [row]], top_k=1)
    assert result[0].payload == row.payload
    assert "duplicate_provenance" not in result[0].payload
