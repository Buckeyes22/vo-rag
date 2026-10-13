"""Tests for the QueryMode + collection resolver leaf.

The collection-name strings ARE the persistence contract — every Qdrant
read and write hits one of them, and all of them use the ``vo_rag_*`` prefix.
"""

from __future__ import annotations

import pytest

from vo_rag.core.retrieval.query_mode import (
    MODE_AGNOSTIC_COLLECTIONS,
    RAW_COLLECTIONS,
    RAW_TO_STERILE,
    STERILE_COLLECTIONS,
    QueryMode,
    resolve_collections,
)

# --- collection-name persistence contract --------------------------- #


def test_raw_collections_use_vo_rag_prefix() -> None:
    """The RAW collections must be the ``vo_rag_*_v1`` strings."""
    assert "vo_rag_books_raw_v1" in RAW_COLLECTIONS


def test_sterile_collections_use_vo_rag_prefix() -> None:
    """The STERILE collections must be the ``vo_rag_*_v1`` strings."""
    assert "vo_rag_books_sterile_v1" in STERILE_COLLECTIONS


def test_raw_to_sterile_uses_vo_rag_prefix() -> None:
    """The sterile/raw mapping keys must be the ``vo_rag_*_v1`` strings."""
    assert RAW_TO_STERILE["vo_rag_books_raw_v1"] == "vo_rag_books_sterile_v1"


def test_mode_agnostic_collections_use_vo_rag_prefix() -> None:
    """The mode-agnostic collection list must be the ``vo_rag_*_v1`` strings."""
    for c in MODE_AGNOSTIC_COLLECTIONS:
        assert c.startswith("vo_rag_"), f"unexpected collection name: {c!r}"
        assert c.endswith("_v1"), f"unexpected collection name: {c!r}"


# --- resolve_collections behavior ----------------------------------- #


def test_internal_default_includes_raw_books() -> None:
    cols = resolve_collections(QueryMode.INTERNAL, None)
    assert "vo_rag_books_raw_v1" in cols
    assert "vo_rag_books_sterile_v1" not in cols


def test_publication_default_excludes_raw_books() -> None:
    cols = resolve_collections(QueryMode.PUBLICATION, None)
    assert "vo_rag_books_sterile_v1" in cols
    assert "vo_rag_books_raw_v1" not in cols


def test_publication_rewrites_requested_raw_to_sterile() -> None:
    cols = resolve_collections(QueryMode.PUBLICATION, ["vo_rag_books_raw_v1"])
    assert cols == ["vo_rag_books_sterile_v1"]


def test_partner_rewrites_requested_raw_to_sterile() -> None:
    cols = resolve_collections(QueryMode.PARTNER, ["vo_rag_books_raw_v1"])
    assert cols == ["vo_rag_books_sterile_v1"]


def test_internal_respects_explicit_sterile_request() -> None:
    cols = resolve_collections(QueryMode.INTERNAL, ["vo_rag_books_sterile_v1"])
    assert cols == ["vo_rag_books_sterile_v1"]


def test_non_book_collections_pass_through_unchanged() -> None:
    cols = resolve_collections(
        QueryMode.PUBLICATION,
        ["vo_rag_articles_v1", "vo_rag_repos_v1"],
    )
    assert cols == ["vo_rag_articles_v1", "vo_rag_repos_v1"]


def test_mixed_request_correctly_rewrites_and_passes_through() -> None:
    cols = resolve_collections(
        QueryMode.PUBLICATION,
        ["vo_rag_books_raw_v1", "vo_rag_articles_v1"],
    )
    assert cols == ["vo_rag_books_sterile_v1", "vo_rag_articles_v1"]


def test_order_preserved_and_deduped() -> None:
    cols = resolve_collections(
        QueryMode.PUBLICATION,
        ["vo_rag_books_raw_v1", "vo_rag_books_sterile_v1"],
    )
    assert cols == ["vo_rag_books_sterile_v1"]


def test_resolved_collections_all_use_vo_rag_prefix() -> None:
    """No code path may resolve to a collection name without the ``vo_rag_`` prefix."""
    for mode in (QueryMode.INTERNAL, QueryMode.PUBLICATION, QueryMode.PARTNER):
        cols = resolve_collections(mode, None)
        for c in cols:
            assert c.startswith("vo_rag_"), (
                f"non-vo_rag collection name leaked into resolve_collections output: {c!r}"
            )


@pytest.mark.parametrize("mode", [QueryMode.INTERNAL, QueryMode.PUBLICATION, QueryMode.PARTNER])
def test_default_resolution_includes_mode_agnostic_collections(mode: QueryMode) -> None:
    cols = resolve_collections(mode, None)
    for c in MODE_AGNOSTIC_COLLECTIONS:
        assert c in cols


# ----------------------------------------------------------------------
# VOR-1404 — the Reddit archive collection. It must be searched by a
# default query (that is the whole point of putting it in
# MODE_AGNOSTIC_COLLECTIONS) while staying out of vo_rag_books_raw_v1,
# so ~23k chunks of forum discussion never dilute the book corpus.
# ----------------------------------------------------------------------


def test_reddit_collection_is_searched_by_default() -> None:
    resolved = resolve_collections(QueryMode.INTERNAL, None)

    assert "vo_rag_reddit_v1" in resolved


def test_reddit_collection_is_searched_in_publication_mode_too() -> None:
    """It is mode-agnostic: no sterile twin, so it survives the raw->sterile rewrite."""
    resolved = resolve_collections(QueryMode.PUBLICATION, None)

    assert "vo_rag_reddit_v1" in resolved


def test_reddit_collection_is_not_a_book_collection() -> None:
    assert "vo_rag_reddit_v1" not in RAW_COLLECTIONS
    assert "vo_rag_reddit_v1" not in STERILE_COLLECTIONS
    assert "vo_rag_reddit_v1" not in RAW_TO_STERILE
