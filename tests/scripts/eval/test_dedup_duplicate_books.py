from __future__ import annotations

from typing import Any

from scripts.eval.dedup_duplicate_books import (
    SourceIdStats,
    _build_duplicate_plans,
    _choose_keep,
    _scan_filter,
)

from vo_rag.pipelines.book_rag.source_identity import CanonicalSource


def _source(
    *,
    canonical_id: str = "vo-rag-corpus-sha256:abc",
    file_hash: str = "abc",
) -> CanonicalSource:
    return CanonicalSource(
        canonical_id=canonical_id,
        title="Test Book",
        author="Test Author",
        license_class="licensed_internal",
        project_id="motorsport",
        file_hash=file_hash,
        matched_by="exact",
    )


def _filter_pairs(scroll_filter: Any) -> list[tuple[str, Any]]:
    return [(condition.key, condition.match.value) for condition in scroll_filter.must]


def test_choose_keep_prefers_canonical_source_id_over_chunk_count() -> None:
    source = _source()
    canonical = SourceIdStats(
        source_id="vo-rag-corpus-sha256:abc",
        chunk_count=10,
        source=source,
    )
    legacy = SourceIdStats(source_id="legacy-book-id", chunk_count=999, source=source)

    assert _choose_keep([legacy, canonical]) == canonical


def test_build_duplicate_plans_skips_singletons_and_marks_deletes() -> None:
    source = _source()
    canonical = SourceIdStats(
        source_id="vo-rag-corpus-sha256:abc",
        chunk_count=10,
        source=source,
    )
    legacy = SourceIdStats(source_id="legacy-book-id", chunk_count=9, source=source)
    singleton = SourceIdStats(source_id="only-book", chunk_count=5, source=_source(file_hash="def"))

    plans = _build_duplicate_plans({"abc": [legacy, canonical], "def": [singleton]})

    assert len(plans) == 1
    assert plans[0].canonical_key == "abc"
    assert plans[0].keep == canonical
    assert plans[0].delete == [legacy]


def test_scan_filter_omits_empty_filter_and_adds_source_version_conditions() -> None:
    assert _scan_filter(None, None) is None

    scroll_filter = _scan_filter("book-1", "vh-1")

    assert scroll_filter is not None
    assert _filter_pairs(scroll_filter) == [
        ("source_id", "book-1"),
        ("version_hash", "vh-1"),
    ]
