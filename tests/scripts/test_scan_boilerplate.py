"""Unit contract for the read-only boilerplate-candidate scan
(VOR-1427; the actual production scan runs separately, against the live corpus).

Only the pure aggregation/formatting functions are exercised here -- no
Qdrant connection is opened by this test module, and the script is never
pointed at anything but synthetic in-memory rows. `aggregate_candidates` is
`async` (it consumes `scan()`'s async per-book generator
directly, one row at a time, rather than a materialized list) but every
test here feeds it a plain list -- `_as_async_iter` adapts either shape.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import scan_boilerplate
from vo_rag.pipelines.book_rag import boilerplate as boilerplate_module


def test_script_imports_the_shared_normalizer() -> None:
    """The script must not carry its own copy of the normalization logic --
    it imports `boilerplate.normalize_chunk_text`, so the scan and the
    ingest-time filter can never diverge on what counts as a match."""
    assert scan_boilerplate.normalize_chunk_text is boilerplate_module.normalize_chunk_text


def test_script_imports_the_shared_qdrant_helpers() -> None:
    """The script must not reimplement facet paging
    or the per-book scroll loop over raw REST -- it imports the exact same
    async helpers `core.corpus.health` uses."""
    from vo_rag.core.corpus import health

    assert scan_boilerplate.all_source_ids is health.all_source_ids
    assert scan_boilerplate.scroll_book_payloads is health.scroll_book_payloads


def test_digest_pins_the_exact_hashing_scheme() -> None:
    """`aggregate_candidates` keys its accumulator
    on a 16-byte blake2b digest of the normalized text, not the text itself
    -- pin the exact scheme so a change to it is a visible, deliberate diff."""
    text = "this page intentionally left blank"
    assert (
        scan_boilerplate._digest(text)
        == hashlib.blake2b(  # noqa: SLF001
            text.encode("utf-8"), digest_size=16
        ).digest()
    )
    assert len(scan_boilerplate._digest(text)) == 16  # noqa: SLF001
    assert scan_boilerplate._digest(text) != scan_boilerplate._digest("a different text")  # noqa: SLF001


async def test_aggregate_groups_by_normalized_text_across_books() -> None:
    rows = [
        ("book-a", "This page intentionally left blank"),
        ("book-b", "  this PAGE intentionally left BLANK.  "),
        # same book again -- not a new distinct book
        ("book-a", "This page intentionally left blank"),
    ]

    candidates = await scan_boilerplate.aggregate_candidates(rows)
    by_text = {c.normalized_text: c for c in candidates}

    assert by_text["this page intentionally left blank"].distinct_books == 2


async def test_aggregate_drops_a_text_seen_in_only_one_book() -> None:
    """A digest's text/examples are only ever
    materialized once it has been seen in >= 2 distinct books -- this is the
    memory floor that keeps the accumulator from retaining full chunk text
    for the vast majority of a real corpus (content unique to one book)."""
    rows = [("book-c", "Some real narrative content unique to book c")]
    candidates = await scan_boilerplate.aggregate_candidates(rows)
    assert candidates == []


async def test_aggregate_accepts_an_async_iterable_too() -> None:
    """`aggregate_candidates` is fed `scan()`'s async per-book generator in
    production -- confirm it consumes an async iterable, not just a list,
    without materializing it first."""

    async def _rows() -> object:
        for row in [("book-a", "shared boilerplate"), ("book-b", "shared boilerplate")]:
            yield row

    candidates = await scan_boilerplate.aggregate_candidates(_rows())  # type: ignore[arg-type]
    assert [c.normalized_text for c in candidates] == ["shared boilerplate"]
    assert candidates[0].distinct_books == 2


async def test_aggregate_dedupes_repeats_within_the_same_book() -> None:
    """A book repeating its own boilerplate on every page must count as ONE
    book, not one per repeat -- and must not cross the 2-book floor by
    itself."""
    rows = [("book-a", "This page intentionally left blank")] * 50
    candidates = await scan_boilerplate.aggregate_candidates(rows)
    assert candidates == []  # only 1 distinct book -- below the floor


async def test_aggregate_skips_blank_normalized_text() -> None:
    """A chunk that normalizes to the empty string (pure punctuation/whitespace)
    carries no signal and must never show up as a candidate row."""
    rows = [("book-a", "   ...   "), ("book-b", "***")]
    candidates = await scan_boilerplate.aggregate_candidates(rows)
    assert candidates == []


async def test_aggregate_caps_examples_at_the_limit() -> None:
    rows = [(f"book-{i}", "Copyright 2020 all rights reserved") for i in range(5)]
    candidates = await scan_boilerplate.aggregate_candidates(rows, example_limit=3)
    assert len(candidates) == 1
    assert candidates[0].distinct_books == 5
    assert len(candidates[0].example_source_ids) == 3


async def test_aggregate_sorts_by_distinct_books_descending_then_text() -> None:
    rows = [
        ("book-a", "rare phrase"),
        ("book-e", "rare phrase"),
        ("book-b", "common phrase"),
        ("book-c", "common phrase"),
        ("book-g", "common phrase"),
        ("book-d", "another rare phrase"),
        ("book-f", "another rare phrase"),
    ]
    candidates = await scan_boilerplate.aggregate_candidates(rows)
    assert [c.normalized_text for c in candidates] == [
        "common phrase",  # 3 books -- sorts first
        "another rare phrase",  # 2 books, tied with "rare phrase" -- text breaks the tie
        "rare phrase",
    ]


async def test_filter_by_min_books() -> None:
    rows = [
        ("book-a", "seen everywhere"),
        ("book-b", "seen everywhere"),
        ("book-c", "seen everywhere"),
        ("book-d", "seen twice"),
        ("book-e", "seen twice"),
    ]
    candidates = await scan_boilerplate.aggregate_candidates(rows)
    kept = scan_boilerplate.filter_min_books(candidates, min_books=3)
    assert [c.normalized_text for c in kept] == ["seen everywhere"]


async def test_write_tsv_has_header_and_tab_separated_rows(tmp_path: Path) -> None:
    rows = [("book-a", "boilerplate line"), ("book-b", "boilerplate line")]
    candidates = await scan_boilerplate.aggregate_candidates(rows)
    out = tmp_path / "candidates.tsv"

    scan_boilerplate.write_tsv(candidates, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "normalized_text\tdistinct_books\texample_source_ids"
    assert lines[1].split("\t")[0] == "boilerplate line"
    assert lines[1].split("\t")[1] == "2"
    assert set(lines[1].split("\t")[2].split(",")) == {"book-a", "book-b"}


def test_help_documents_readonly_scan(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        scan_boilerplate.main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "Read-only" in out
    assert "--min-books" in out
    assert "--out" in out
    assert "--qdrant-url" in out


def test_main_requires_qdrant_url_and_out(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        scan_boilerplate.main([])
    assert exc_info.value.code != 0
