"""Pure unit tests for `vo_rag.core.corpus.health` (VOR-1427).

No Qdrant here -- `book_findings` and `HealthReport.ok()` are pure functions
over `BookStats` / `HealthReport` values, and `all_source_ids`'s facet
growth/truncation loop is exercised against a plain fake client (mirroring
`tests/ocr_farm/test_plan_backlog.py`'s `FakeQdrant`-based coverage of
`completed_source_books`). See `test_health_integration.py` for the
live-Qdrant `collect_stats`/`run_health` coverage.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from vo_rag.core.corpus import health
from vo_rag.core.corpus.health import (
    BookFinding,
    BookStats,
    HealthReport,
    book_findings,
    book_quality_metadata,
)


def _stats(**overrides: object) -> BookStats:
    base: dict[str, object] = {
        "source_id": "book-1",
        "source_book": "Book One",
        "chunks": 0,
        "tiny": 0,
        "top_section": None,
        "top_section_count": 0,
        "max_page": 0,
        "page_bucket_titles": 0,
        "boilerplate": 0,
    }
    base.update(overrides)
    return BookStats(**base)  # type: ignore[arg-type]


def test_degenerate_just_over_threshold() -> None:
    finding = book_findings(_stats(chunks=100, tiny=21))
    assert finding.degenerate is True


def test_degenerate_exactly_at_threshold_is_not_flagged() -> None:
    finding = book_findings(_stats(chunks=100, tiny=20))
    assert finding.degenerate is False


def test_dominant_section_over_threshold_with_enough_chunks() -> None:
    finding = book_findings(_stats(chunks=300, top_section="Bibliography", top_section_count=181))
    assert finding.dominant is not None
    title, share = finding.dominant
    assert title == "Bibliography"
    assert share == 181 / 300


def test_dominant_section_needs_more_than_200_chunks() -> None:
    finding = book_findings(_stats(chunks=200, top_section="Bibliography", top_section_count=199))
    assert finding.dominant is None


def test_dominant_section_exactly_at_60_percent_is_not_flagged() -> None:
    """The rule is `share > 0.60`, strictly -- exactly 60% (180/300) must
    not trip it."""
    finding = book_findings(_stats(chunks=300, top_section="X", top_section_count=180))
    assert finding.dominant is None


def test_zero_chunks_trips_only_no_chunks() -> None:
    """A completed book with zero chunks in the
    scanned collection is never reported clean -- it always trips
    `no_chunks`, the one finding that doesn't need a share to compute."""
    finding = book_findings(_stats(chunks=0))
    assert finding == BookFinding(False, None, False, no_chunks=True)
    assert finding.no_chunks is True


def test_small_book_with_only_page_bucket_titles_is_unpaginated() -> None:
    """A 40-chunk book stuck on page <=1 but carrying `pp. N-M` section
    titles is flagged even though it is far below the 200-chunk dominant
    threshold (defect a)."""
    finding = book_findings(_stats(chunks=40, max_page=1, page_bucket_titles=40))
    assert finding.unpaginated_page_titles is True


def test_paginated_book_with_page_bucket_titles_is_not_flagged() -> None:
    finding = book_findings(
        _stats(chunks=300, max_page=300, page_bucket_titles=40, parser_used="docling")
    )
    assert finding.unpaginated_page_titles is False


# --- empty_figures / low_text_density / image_only_suspect / too_few_chunks ---


def test_empty_figures_over_5_percent_is_flagged() -> None:
    finding = book_findings(_stats(chunks=100, figure_chunks=100, figure_chunks_no_context=6))
    assert finding.empty_figures is True


def test_empty_figures_at_exactly_5_percent_is_not_flagged() -> None:
    finding = book_findings(_stats(chunks=100, figure_chunks=100, figure_chunks_no_context=5))
    assert finding.empty_figures is False


def test_empty_figures_with_no_figure_chunks_is_not_flagged() -> None:
    finding = book_findings(_stats(chunks=100, figure_chunks=0, figure_chunks_no_context=0))
    assert finding.empty_figures is False


def test_low_text_density_flagged_under_300_chars_per_page_for_native_text_book() -> None:
    finding = book_findings(
        _stats(chunks=10, max_page=10, content_chars=2999, parser_used="docling")
    )
    assert finding.low_text_density is True


def test_low_text_density_not_flagged_at_exactly_300_chars_per_page() -> None:
    finding = book_findings(
        _stats(chunks=10, max_page=10, content_chars=3000, parser_used="docling")
    )
    assert finding.low_text_density is False


def test_low_text_density_skips_ebooks() -> None:
    """An EPUB's 'pages' are chapters, not print pages -- chars/page is
    meaningless there, so ebooks never trip this finding."""
    finding = book_findings(_stats(chunks=10, max_page=10, content_chars=1, parser_used="epub"))
    assert finding.low_text_density is False


def test_low_text_density_skips_unpaginated_books() -> None:
    finding = book_findings(_stats(chunks=10, max_page=0, content_chars=0, parser_used="docling"))
    assert finding.low_text_density is False


def test_image_only_suspect_epub_over_80_percent_figures_is_flagged() -> None:
    finding = book_findings(_stats(chunks=10, figure_chunks=9, parser_used="epub"))
    assert finding.image_only_suspect is True


def test_image_only_suspect_not_flagged_at_exactly_80_percent() -> None:
    finding = book_findings(_stats(chunks=10, figure_chunks=8, parser_used="epub"))
    assert finding.image_only_suspect is False


def test_image_only_suspect_not_flagged_for_non_epub_parser() -> None:
    finding = book_findings(_stats(chunks=10, figure_chunks=9, parser_used="docling"))
    assert finding.image_only_suspect is False


def test_too_few_chunks_flagged_over_50_pages() -> None:
    finding = book_findings(_stats(chunks=9, max_page=100))
    assert finding.too_few_chunks is True


def test_too_few_chunks_not_flagged_at_exactly_pages_over_10() -> None:
    finding = book_findings(_stats(chunks=10, max_page=100))
    assert finding.too_few_chunks is False


def test_too_few_chunks_never_flagged_at_or_under_50_pages() -> None:
    finding = book_findings(_stats(chunks=1, max_page=50))
    assert finding.too_few_chunks is False


def test_too_few_chunks_excludes_ebooks() -> None:
    """An EPUB/MOBI's `max_page` is a chapter
    count (the parser's `_split_pages`), not a print-page count, so the
    pages/10 ratio is meaningless there -- same reasoning as
    `low_text_density`'s ebook exclusion. Shape that would otherwise trip it
    (9 chunks over "100 pages")."""
    finding = book_findings(_stats(chunks=9, max_page=100, parser_used="epub"))
    assert finding.too_few_chunks is False


def test_image_only_epub_shape_is_image_only_suspect_and_empty_figures() -> None:
    """VOR-1433 regression shape: an EPUB parsed to 1 text chunk + 400
    figure_composite chunks (401 images, ~8KB of text total), none of the
    composites carrying a caption or page context."""
    finding = book_findings(
        _stats(
            chunks=401,
            text_chunks=1,
            figure_chunks=400,
            figure_chunks_no_context=400,
            parser_used="epub",
            max_page=1,
        )
    )
    assert finding.image_only_suspect is True
    assert finding.empty_figures is True
    # An ebook's synthetic "page" count must never trip the print-page-shaped
    # findings.
    assert finding.low_text_density is False
    assert finding.too_few_chunks is False


def test_book_quality_metadata_clean_book_has_no_findings() -> None:
    stats = _stats(chunks=100, max_page=100, content_chars=100 * 400, parser_used="docling")
    quality = book_quality_metadata(stats)
    assert quality["findings"] == []
    assert quality["stats"]["source_id"] == "book-1"
    assert quality["stats"]["chunks"] == 100


def test_book_quality_metadata_lists_findings_in_fixed_order() -> None:
    stats = _stats(chunks=9, tiny=21, max_page=100, content_chars=100 * 400)
    quality = book_quality_metadata(stats)
    assert quality["findings"] == ["degenerate", "too_few_chunks"]


def test_book_quality_metadata_zero_chunks_reports_no_chunks_never_clean() -> None:
    """A book that comes
    back with zero chunks -- whether never ingested here, or (the incident
    this guards against) scrolled from the wrong collection -- must never be
    reported as a clean `{"findings": []}`."""
    stats = _stats(chunks=0)
    quality = book_quality_metadata(stats)
    assert quality["findings"] == ["no_chunks"]
    assert quality["stats"]["chunks"] == 0


def test_book_quality_metadata_stats_is_the_full_book_stats_dataclass() -> None:
    stats = _stats(chunks=5, figure_chunks=2, figure_chunks_no_context=2)
    quality = book_quality_metadata(stats)
    assert quality["stats"] == {
        "source_id": stats.source_id,
        "source_book": stats.source_book,
        "chunks": stats.chunks,
        "tiny": stats.tiny,
        "top_section": stats.top_section,
        "top_section_count": stats.top_section_count,
        "max_page": stats.max_page,
        "page_bucket_titles": stats.page_bucket_titles,
        "boilerplate": stats.boilerplate,
        "tiny_boilerplate": stats.tiny_boilerplate,
        "author": stats.author,
        "parser_used": stats.parser_used,
        "ebook_source": stats.ebook_source,
        "figure_chunks": stats.figure_chunks,
        "figure_chunks_no_context": stats.figure_chunks_no_context,
        "text_chunks": stats.text_chunks,
        "content_chars": stats.content_chars,
    }


def test_health_report_ok_true_only_when_everything_is_empty() -> None:
    assert HealthReport(books=5).ok() is True
    assert HealthReport(books=5, degenerate=["book-1"]).ok() is False
    assert HealthReport(books=5, dominant_section=[("b", "t", 0.7)]).ok() is False
    assert HealthReport(books=5, test_points=["p1"]).ok() is False
    assert HealthReport(books=5, unpaginated_page_titles=["book-1"]).ok() is False
    assert HealthReport(books=5, boilerplate_points=1).ok() is False
    assert HealthReport(books=5, duplicate_groups=1).ok() is False
    assert HealthReport(books=5, empty_figures=["book-1"]).ok() is False
    assert HealthReport(books=5, low_text_density=["book-1"]).ok() is False
    assert HealthReport(books=5, image_only_suspect=["book-1"]).ok() is False
    assert HealthReport(books=5, too_few_chunks=["book-1"]).ok() is False
    assert HealthReport(books=5, no_chunks=["book-1"]).ok() is False


def test_health_report_ok_true_on_empty_corpus() -> None:
    assert HealthReport(books=0).ok() is True


class _FakeFacetClient:
    """`AsyncQdrantClient.facet` stand-in, mirroring
    `tests/ocr_farm/_farm_fakes.py`'s `FakeQdrant`."""

    def __init__(self, source_ids: list[str]) -> None:
        self.source_ids = source_ids
        self.facet_calls: list[dict[str, Any]] = []

    async def facet(
        self,
        collection_name: str,
        key: str,
        *,
        limit: int = 10,
        exact: bool = False,
        facet_filter: Filter | None = None,
    ) -> Any:
        self.facet_calls.append(
            {
                "collection_name": collection_name,
                "key": key,
                "limit": limit,
                "exact": exact,
                "facet_filter": facet_filter,
            }
        )
        values = self.source_ids[:limit]
        return SimpleNamespace(hits=[SimpleNamespace(value=v, count=1) for v in values])


async def test_all_source_ids_grows_the_facet_limit_until_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health, "_FACET_INITIAL_LIMIT", 10)
    ids = [f"book-{n}" for n in range(25)]
    client = _FakeFacetClient(ids)

    result = await health.all_source_ids(client)

    assert result == sorted(ids)
    assert [c["limit"] for c in client.facet_calls] == [10, 40]
    assert client.facet_calls[0] == {
        "collection_name": health.BOOKS_RAW_COLLECTION,
        "key": "source_id",
        "limit": 10,
        "exact": False,
        "facet_filter": None,
    }


async def test_all_source_ids_forwards_the_facet_filter_on_every_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health, "_FACET_INITIAL_LIMIT", 10)
    ids = [f"book-{n}" for n in range(25)]
    client = _FakeFacetClient(ids)
    facet_filter = Filter(
        must=[FieldCondition(key="domain_tags", match=MatchValue(value="track-and-field"))]
    )

    result = await health.all_source_ids(client, facet_filter=facet_filter)

    assert result == sorted(ids)
    assert [c["limit"] for c in client.facet_calls] == [10, 40]
    assert all(c["facet_filter"] is facet_filter for c in client.facet_calls)


async def test_all_source_ids_refuses_a_truncated_facet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health, "_FACET_INITIAL_LIMIT", 10)
    monkeypatch.setattr(health, "_FACET_MAX_LIMIT", 20)
    client = _FakeFacetClient([f"book-{n}" for n in range(30)])

    with pytest.raises(RuntimeError, match="truncated"):
        await health.all_source_ids(client)

    assert [c["limit"] for c in client.facet_calls] == [10, 20]


# --- `_collect_one`/`run_health` over a fake scroll ---


class _FakeScrollClient:
    """`AsyncQdrantClient.scroll` stand-in: one page per `source_id` filter,
    no points for the `domain_tags` (test-points) scroll."""

    def __init__(self, books: dict[str, list[dict[str, Any]]]) -> None:
        self.books = books
        self.with_payload: list[Any] = []
        self.collections: list[str] = []

    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Any,
        with_payload: Any,
        with_vectors: bool,
        limit: int,
        offset: Any,
    ) -> tuple[list[Any], Any]:
        condition = scroll_filter.must[0]
        if condition.key != "source_id":
            return [], None
        self.collections.append(collection_name)
        self.with_payload.append(with_payload)
        payloads = self.books.get(condition.match.value, [])
        requested = set(with_payload)
        points = [
            SimpleNamespace(
                id=f"{condition.match.value}-{i}",
                payload={key: value for key, value in payload.items() if key in requested},
            )
            for i, payload in enumerate(payloads)
        ]
        return points, None


#: 1 chunk per "page" in `_book`/`_paged_book` below, so each chunk's content
#: alone must clear the `low_text_density` floor (300 chars/page, VOR-1434)
#: to keep pre-existing fixtures unrelated to that finding clean.
_LONG_FILLER = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 6


def _book(
    title: str, *, author: str | None = None, n: int = 5, **extra: Any
) -> list[dict[str, Any]]:
    return [
        {
            "content": f"a normal chunk of book content number {i}, long enough. {_LONG_FILLER}",
            "source_book": title,
            "author": author,
            "page_number": i + 1,
            "parser_used": "docling",
            **extra,
        }
        for i in range(n)
    ]


async def _health(books: dict[str, list[dict[str, Any]]]) -> health.HealthReport:
    store = SimpleNamespace(_client=_FakeScrollClient(books))
    return await health.run_health(store, source_ids=list(books))  # type: ignore[arg-type]


# --- collect_stats/_collect_one must
# --- honour a caller-given collection, not hardcode BOOKS_RAW_COLLECTION ---


async def test_collect_stats_defaults_to_the_canonical_books_raw_collection() -> None:
    client = _FakeScrollClient({"x": _book("Handbook")})
    store = SimpleNamespace(_client=client)

    await health.collect_stats(store, source_ids=["x"])  # type: ignore[arg-type]

    assert client.collections == [health.BOOKS_RAW_COLLECTION]


async def test_collect_stats_scrolls_a_caller_given_collection() -> None:
    """The book_rag completion hook (VORAG_RAW_COLLECTION override, e.g.
    `ops/rc_ingest_batch.py` bulk runs) must be able to point `collect_stats`
    at the same non-default collection its own upsert used -- otherwise it
    scrolls the canonical collection, finds nothing, and reports a
    false-clean `{"findings": [], "stats": {"chunks": 0, ...}}`."""
    client = _FakeScrollClient({"x": _book("Handbook")})
    store = SimpleNamespace(_client=client)

    stats = await health.collect_stats(
        store,  # type: ignore[arg-type]
        source_ids=["x"],
        collection="vo_rag_books_raw_v2",
    )

    assert client.collections == ["vo_rag_books_raw_v2"]
    assert stats[0].chunks == 5


async def test_collect_one_scrolls_a_caller_given_collection() -> None:
    client = _FakeScrollClient({"x": _book("Handbook")})

    stats = await health._collect_one(  # noqa: SLF001
        client,  # type: ignore[arg-type]
        "x",
        None,
        collection="vo_rag_books_raw_v2",
    )

    assert client.collections == ["vo_rag_books_raw_v2"]
    assert stats.chunks == 5


async def test_keep_only_edition_and_ambiguous_year_groups_do_not_fail_health() -> None:
    report = await _health(
        {
            "a": _book("Applied Strength Methods (2013)", author="Avery"),
            "b": _book("Applied Strength Methods (2019)", author="Avery"),
            "c": _book("Applied Strength Methods", author="Avery"),
        }
    )
    assert report.duplicate_groups == 0
    assert report.ok() is True


async def test_same_title_by_different_authors_does_not_fail_health() -> None:
    report = await _health(
        {"x": _book("Handbook", author="Smith"), "y": _book("Handbook", author="Jones")}
    )
    assert report.duplicate_groups == 0
    assert report.ok() is True


async def test_real_duplicate_still_fails_health() -> None:
    report = await _health(
        {
            "x": _book("Handbook", author="Smith"),
            "y": _book("Handbook", author="smith "),
            "z": _book("Handbook", author="Jones"),
        }
    )
    assert report.duplicate_groups == 1
    assert report.ok() is False


async def test_collect_one_reads_author_and_parser_used_from_the_payload() -> None:
    client = _FakeScrollClient({"x": _book("Handbook", author="Smith", parser_used="epub")})
    stats = await health._collect_one(client, "x", None)  # type: ignore[arg-type]  # noqa: SLF001
    assert stats.author == "Smith"
    assert stats.parser_used == "epub"
    assert {"author", "parser_used"} <= set(client.with_payload[0])


async def test_tiny_boilerplate_chunk_counts_once_as_bad() -> None:
    payloads = _book("Handbook", n=4)
    payloads[0]["content"] = "Index"  # tiny AND boilerplate
    payloads[1]["content"] = "x"  # tiny only
    payloads[2]["content"] = "This page intentionally left blank, and long enough"  # boilerplate
    client = _FakeScrollClient({"x": payloads})

    def matcher(payload: dict[str, Any]) -> bool:
        return str(payload.get("content")) in {
            "Index",
            "This page intentionally left blank, and long enough",
        }

    stats = await health._collect_one(client, "x", matcher)  # type: ignore[arg-type]  # noqa: SLF001
    assert (stats.tiny, stats.boilerplate) == (2, 2)
    from vo_rag.core.corpus.duplicates import plan_duplicates

    twin = BookStats(source_id="y", source_book="Handbook", chunks=4)
    rows = {r.source_id: r for r in plan_duplicates([stats, twin], authors={}, ingested_at={})}
    assert rows["x"].bad_chunks == 3


@pytest.mark.parametrize("parser_used", ["epub", "mobi"])
def test_chapter_paginated_ebook_with_page_bucket_titles_is_flagged(parser_used: str) -> None:
    # The EPUB parser's "pages" are chapters: max_page is far above 1, yet any
    # `pp. N-M` title is still fabricated (the chunker never page-buckets them).
    finding = book_findings(
        _stats(chunks=300, max_page=40, page_bucket_titles=12, parser_used=parser_used)
    )
    assert finding.unpaginated_page_titles is True


def test_null_parser_used_book_with_page_bucket_titles_is_not_flagged() -> None:
    # Legacy GLM-era PDFs carry no parser_used, and page buckets are their
    # legitimate fallback title.
    finding = book_findings(_stats(chunks=300, max_page=300, page_bucket_titles=40))
    assert finding.unpaginated_page_titles is False


def test_null_parser_used_book_from_an_ebook_source_file_is_flagged() -> None:
    finding = book_findings(
        _stats(chunks=300, max_page=40, page_bucket_titles=12, ebook_source=True)
    )
    assert finding.unpaginated_page_titles is True


@pytest.mark.parametrize("parser_used", ["docling", "docling-ocr", "glm-ocr-0.9b"])
def test_pdf_with_page_bucket_titles_is_not_flagged(parser_used: str) -> None:
    finding = book_findings(
        _stats(chunks=300, max_page=300, page_bucket_titles=40, parser_used=parser_used)
    )
    assert finding.unpaginated_page_titles is False


# --- a null parser_used book's format comes from its source file ---

_EPUB_SHA, _PDF_SHA = "e" * 64, "f" * 64


class _FormatPool:
    """Answers the source-format lookup: garage keys per source_id."""

    def __init__(self, keys: dict[str, str]) -> None:
        self.keys = keys
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((query, args))
        source_ids = args[0]
        return [
            {"source_id": sid, "garage_key": self.keys[sid]}
            for sid in source_ids
            if sid in self.keys
        ]


def _paged_book(title: str, **extra: Any) -> list[dict[str, Any]]:
    """40 chunks on "pages" 1..40, each titled with a page bucket."""
    return [
        {
            "content": f"a normal chunk of book content number {i}, long enough. {_LONG_FILLER}",
            "source_book": title,
            "page_number": i + 1,
            "section_title": f"pp. {i + 1}-{i + 2}",
            **extra,
        }
        for i in range(40)
    ]


async def test_null_parser_used_books_are_classified_by_their_source_file() -> None:
    epub_id = f"vo-rag-corpus-sha256:{_EPUB_SHA}"
    pdf_id = f"vo-rag-corpus-sha256:{_PDF_SHA}"
    books = {epub_id: _paged_book("Ebook"), pdf_id: _paged_book("Legacy PDF", ocr_dpi=200)}
    pool = _FormatPool(
        {epub_id: f"corpus/{_EPUB_SHA}/Ebook.EPUB", pdf_id: f"corpus/{_PDF_SHA}/legacy.pdf"}
    )
    store = SimpleNamespace(_client=_FakeScrollClient(books))

    report = await health.run_health(store, source_ids=list(books), pool=pool)  # type: ignore[arg-type]

    assert report.unpaginated_page_titles == [epub_id]
    [(query, args)] = pool.calls
    assert "intake_envelopes" in query
    assert sorted(args[0]) == sorted([epub_id, pdf_id])


async def test_without_a_pool_only_explicit_ebook_parsers_count_as_non_pdf() -> None:
    books = {
        "legacy": _paged_book("Legacy PDF"),
        "ebook": _paged_book("Ebook", parser_used="epub"),
    }
    store = SimpleNamespace(_client=_FakeScrollClient(books))

    report = await health.run_health(store, source_ids=list(books))  # type: ignore[arg-type]

    assert report.unpaginated_page_titles == ["ebook"]


async def test_the_source_lookup_only_asks_about_books_it_could_change() -> None:
    parsed, plain, legacy = (f"vo-rag-corpus-sha256:{c * 64}" for c in "abc")
    books = {
        parsed: _paged_book("Parsed PDF", parser_used="docling"),
        plain: _book("No page buckets"),
        legacy: _paged_book("Legacy PDF"),
    }
    pool = _FormatPool({})
    store = SimpleNamespace(_client=_FakeScrollClient(books))

    await health.run_health(store, source_ids=list(books), pool=pool)  # type: ignore[arg-type]

    [(_, args)] = pool.calls
    assert args == ([legacy], ["c" * 64])


# --- run_health end-to-end over a fake scroll ---


def _figure(*, n: int, with_context: bool = True, **extra: Any) -> list[dict[str, Any]]:
    """`n` figure_composite chunks; `with_context=False` mimics a composite
    with neither a caption nor page context (see health.py's `empty_figures`
    docstring for why that's exactly a missing ``Caption:``/``Page context:``
    line)."""
    content = "Section: Foo\n\nCaption: a figure" if with_context else "Section: Foo"
    return [
        {
            "content": content,
            "chunk_type": "figure_composite",
            "page_number": i + 1,
            **extra,
        }
        for i in range(n)
    ]


async def test_run_health_flags_image_only_epub_shape() -> None:
    """VOR-1433 regression: an EPUB parsed to 1 text chunk + 400
    figure_composite chunks with no caption/page context."""
    text_chunk = [
        {
            "content": "the only real text in this book",
            "chunk_type": "text",
            "page_number": 1,
            "parser_used": "epub",
        }
    ]
    figures = _figure(n=400, with_context=False, parser_used="epub")
    report = await _health({"image_only": text_chunk + figures})

    assert report.image_only_suspect == ["image_only"]
    assert report.empty_figures == ["image_only"]
    assert report.ok() is False


async def test_run_health_does_not_flag_a_well_captioned_illustrated_book() -> None:
    # An EPUB (so `low_text_density`'s print-page-based ratio -- not under
    # test here -- never applies; see its own tests above).
    figures = _figure(n=10, with_context=True, parser_used="epub")
    text = _book("Illustrated Handbook", n=90, parser_used="epub")
    report = await _health({"good": text + figures})

    assert report.image_only_suspect == []
    assert report.empty_figures == []
    assert report.ok() is True
