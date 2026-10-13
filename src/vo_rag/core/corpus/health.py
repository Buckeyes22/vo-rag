"""`vo-rag corpus health` (VOR-1427).

Scans the books-raw collection, book by book, for four defects:

- **degenerate**: more than 20% of a book's chunks are near-empty
  (``len(content.strip()) < 20``) -- a bad OCR/parse pass.
- **dominant section**: one ``section_title`` swallows more than 60% of a
  book's chunks, but only once the book is over 200 chunks (below that a
  short appendix/bibliography legitimately dominates a small book) -- e.g.
  a whole book chunked as "Bibliography".
- **unpaginated page titles** (defect a): some chunks carry a ``pp. N-M``
  ``section_title`` although the book has no print pages -- every chunk
  sits on page <= 1 (``max_page``), or the book is an EPUB/MOBI (the ebook
  parser numbers its "pages" by chapter, so ``max_page`` can be far above
  1). A book is an ebook when its payload ``parser_used`` is ``epub`` or
  ``mobi``, or -- ``parser_used`` null, as on legacy GLM-era PDFs (where
  page buckets are the legitimate fallback title) and on ebook chunks
  written before the ebook path stamped it -- when its intake envelope's
  ``garage_key`` ends ``.epub``/``.mobi``/``.azw3``
  (``source_format.ebook_source_ids``, only with a Postgres ``pool``;
  without one, null counts as PDF). This mirrors the chunker's own rule
  (``TitleOnlyFallback``: never page-bucket a reflowable book) and is
  checked at *any* chunk count because the dominant-section rule (200+
  chunks) would miss a short book.
- **test points**: any point tagged ``canary``/``rr03``/``test`` in
  ``domain_tags`` -- synthetic data that leaked into a real collection.

``boilerplate``/``boilerplate_points`` are filled by the CLI: `collect_stats`
takes an optional ``boilerplate_matcher`` hook (a chunk-payload predicate,
left ``None`` here so this module stays fill-source-agnostic) and
``vo_rag.cli.corpus_ops``'s ``health`` command supplies
``vo_rag.core.corpus.boilerplate_ops.boilerplate_payload_matcher`` for it.

**Per-book findings (VOR-1434)**, computed alongside the above from
the same ``BookStats`` -- catch defects that only show up once a book's
figure/text chunk shapes are counted (the incidents motivating this: figure
composites with neither a caption nor page context, an EPUB whose image-only
detection missed and got chunked as ~all figures -- e.g. an EPUB parsed to
1 text chunk + 400 ``figure_composite`` chunks -- and books with
implausibly few chunks for their page count):

- **empty_figures**: more than 5% of a book's ``figure_composite`` chunks
  carry neither a ``Caption:`` nor a ``Page context:`` line in their
  ``content``. A composite chunk's ``content`` (``composite_emitter.figure_content``;
  before VOR-1452 it was the whole ``visual_sidecar._join_for_embedding``
  window) has ``f"Caption: {caption}"`` only when the figure has a caption
  and ``f"Page context: {excerpt}"`` only when prose was found on its page;
  a composite missing both is at best "Figure on page N (section)" plus a VLM
  description, and says essentially nothing about the figure.
- **low_text_density**: a native-text (non-EPUB/MOBI) book averages fewer
  than 300 content characters per page across all its chunks -- a parse that
  technically produced chunks but extracted almost no real text.
- **image_only_suspect**: an EPUB (``parser_used == "epub"``) where more
  than 80% of chunks are ``figure_composite`` -- the image-only-book
  detection (``vo_rag.core.parsing.parsers.epub.is_image_only``) should have
  routed this book differently but missed.
- **too_few_chunks**: a native-text (non-EPUB/MOBI) book over 50 pages
  (``max_page``) has fewer than ``max_page / 10`` chunks total -- implausibly
  sparse output for its length. Excludes ebooks for the same
  reason ``low_text_density`` does: an EPUB/MOBI's ``max_page`` is a chapter
  count (the parser's ``_split_pages``), not a print-page count, so the
  ratio is meaningless there.

These four use the same ``BookStats`` a caller already collected; no extra
Qdrant reads. ``book_quality_metadata`` packages all seven findings (these
four plus ``degenerate``/``dominant_section``/``unpaginated_page_titles``)
into the stable ``{"findings": [...], "stats": {...}}`` shape written to
``intake_envelopes.metadata->'quality'`` by the book_rag workflow's
completion step (``BookRagPipeline._record_book_quality`` in
``vo_rag.pipelines.book_rag.workflow``).

``duplicate_groups`` is the count of distinct
``vo_rag.core.corpus.duplicates.plan_duplicates`` groups over this scan's
``BookStats`` that hold at least one ``delete`` row -- i.e. work still to
do. Keep-only groups (an ``ambiguous_year`` cluster, or anything else the
keep rule leaves whole) never fail health, so a clean corpus can exit 0.
Authors come from the chunk payload's top-level ``author`` (written by
``chunk_embed.do_upsert_raw``) via ``duplicates.book_authors``, exactly as
`duplicates --report` reads them, so both group identically; only the
ingest dates (which never change grouping, only the keeper) need Postgres,
and health passes none.

`plan_duplicates` (VOR-1427) also splits each title(+author)
bucket by publication year: books with the same exact year -- or no year at
all, when at most one distinct year is present -- group together; 2+
distinct years never merge into each other, and a year-less book caught
between two different years lands in its own `ambiguous_year` group instead
of guessing which edition it belongs to. Two books with the same
normalized title, author, AND year always land in the same exact-year
subgroup, with one planned delete; an unresolved `ambiguous_year` cluster
is keep-only, so it shows in the `duplicates --report` CSV for the operator to
resolve but doesn't fail health. `duplicate_groups == 0` therefore means
no duplicate is planned for deletion.

Memory discipline: `collect_stats` scrolls each book's chunks with
``with_payload=_STATS_PAYLOAD`` and ``with_vectors=False``, 1,024 points
per page, and keeps only running counters -- never chunk content -- in
memory.

The book list itself comes from a ``source_id`` facet (unless the caller
passes explicit ids). Qdrant's facet API has no cursor, so -- mirroring
``vo_rag.ocr_farm.coordinator.completed_source_books`` -- this pages by
growing the limit until a response comes back short of it, and raises
rather than silently truncating if the facet is still full at the max
limit.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from typing import Any, NamedTuple, Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from vo_rag.core.chunking.heading_tracker import REFLOWABLE_PARSERS
from vo_rag.core.corpus.source_format import FetchPool, ebook_source_ids
from vo_rag.core.storage.qdrant import QdrantStore

BOOKS_RAW_COLLECTION = "vo_rag_books_raw_v1"

_FACET_INITIAL_LIMIT = 10_000
_FACET_MAX_LIMIT = 1_000_000
_SCROLL_PAGE = 1_024

_TINY_CONTENT_CHARS = 20
_DEGENERATE_SHARE = 0.20
_DOMINANT_MIN_CHUNKS = 200
_DOMINANT_SHARE = 0.60
_TEST_TAGS = ["canary", "rr03", "test"]
_PAGE_BUCKET_TITLE_RE = re.compile(r"^pp\. \d+-\d+$")
_FIGURE_COMPOSITE_TYPE = "figure_composite"
_TEXT_CHUNK_TYPE = "text"
_EMPTY_FIGURES_SHARE = 0.05
_LOW_TEXT_DENSITY_CHARS_PER_PAGE = 300
_IMAGE_ONLY_SUSPECT_SHARE = 0.80
_TOO_FEW_CHUNKS_MIN_PAGES = 50
_TOO_FEW_CHUNKS_PER_PAGE = 0.10
_STATS_PAYLOAD = [
    "content",
    "section_title",
    "source_book",
    "page_number",
    "author",
    "parser_used",
    "chunk_type",
]

#: A chunk payload predicate: True means "this chunk is boilerplate".
#: `boilerplate_ops` supplies the real matcher; `collect_stats` treats `None` as "don't count".
BoilerplateMatcher = Callable[[dict[str, Any]], bool]


@dataclass
class BookStats:
    """Per-book counters. Never holds chunk content -- counts only."""

    source_id: str
    source_book: str | None = None
    chunks: int = 0
    tiny: int = 0
    top_section: str | None = None
    top_section_count: int = 0
    max_page: int = 0
    page_bucket_titles: int = 0
    boilerplate: int = 0
    #: Chunks counted in BOTH ``tiny`` and ``boilerplate`` (a boilerplate
    #: text under 20 chars), so a bad chunk is counted once.
    tiny_boilerplate: int = 0
    #: The payload's top-level ``author`` (first non-empty value).
    author: str | None = None
    #: The payload's ``parser_used`` (first non-empty value); ``None`` on
    #: legacy GLM-era PDFs and on EPUB/MOBI chunks from before the ebook path
    #: stamped it.
    parser_used: str | None = None
    #: ``parser_used`` is null and the intake envelope's source file is an
    #: EPUB/MOBI (set by `collect_stats` when it has a Postgres pool).
    ebook_source: bool = False
    #: Chunks with ``chunk_type == "figure_composite"``.
    figure_chunks: int = 0
    #: Of those, chunks whose ``content`` has neither a ``Caption:`` nor a
    #: ``Page context:`` line (see module docstring, ``empty_figures``).
    figure_chunks_no_context: int = 0
    #: Chunks with ``chunk_type == "text"``.
    text_chunks: int = 0
    #: Sum of ``len(content.strip())`` across every chunk (all chunk types).
    content_chars: int = 0

    @property
    def bad_chunks(self) -> int:
        """Chunks that are tiny, boilerplate, or both -- each counted once."""
        return self.tiny + self.boilerplate - self.tiny_boilerplate


class BookFinding(NamedTuple):
    """`book_findings`'s result: which defects (if any) a book tripped."""

    degenerate: bool
    dominant: tuple[str, float] | None
    unpaginated_page_titles: bool
    #: > 5% of figure_composite chunks with no caption and no page context.
    empty_figures: bool = False
    #: A native-text book averaging < 300 content chars/page.
    low_text_density: bool = False
    #: An EPUB with > 80% figure_composite chunks.
    image_only_suspect: bool = False
    #: Fewer than pages/10 chunks for a native-text book over 50 pages
    #: (excludes ebooks, same as low_text_density).
    too_few_chunks: bool = False
    #: The book has zero chunks in this
    #: collection -- either it was never ingested here, or (the incident this
    #: guards against) the caller scrolled the wrong collection. Never
    #: "clean": a completed book with no chunks is always a defect.
    no_chunks: bool = False


@dataclass
class HealthReport:
    books: int = 0
    degenerate: list[str] = field(default_factory=list)
    dominant_section: list[tuple[str, str, float]] = field(default_factory=list)
    test_points: list[str] = field(default_factory=list)
    unpaginated_page_titles: list[str] = field(default_factory=list)
    boilerplate_points: int = 0
    duplicate_groups: int = 0
    #: Per-book findings (see health.py module docstring).
    empty_figures: list[str] = field(default_factory=list)
    low_text_density: list[str] = field(default_factory=list)
    image_only_suspect: list[str] = field(default_factory=list)
    too_few_chunks: list[str] = field(default_factory=list)
    #: Books with zero chunks in the scanned
    #: collection (see `book_findings`'s `no_chunks`).
    no_chunks: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return (
            not self.degenerate
            and not self.dominant_section
            and not self.test_points
            and not self.unpaginated_page_titles
            and self.boilerplate_points == 0
            and self.duplicate_groups == 0
            and not self.empty_figures
            and not self.low_text_density
            and not self.image_only_suspect
            and not self.too_few_chunks
            and not self.no_chunks
        )


def book_findings(stats: BookStats) -> BookFinding:
    """Which per-book defects `stats` trips, if any.

    An empty book (0 chunks) never trips any of the share-based findings --
    there's nothing to compute a share of, and `page_bucket_titles` is
    necessarily 0 too -- but it always trips `no_chunks`: a completed book
    with zero chunks in the scanned collection is never "clean", whether
    that's because it was never ingested there or (the incident this guards
    against) the caller scrolled the wrong collection.
    """
    if stats.chunks == 0:
        return BookFinding(False, None, False, no_chunks=True)

    degenerate = stats.tiny / stats.chunks > _DEGENERATE_SHARE

    dominant: tuple[str, float] | None = None
    if stats.chunks > _DOMINANT_MIN_CHUNKS and stats.top_section is not None:
        share = stats.top_section_count / stats.chunks
        if share > _DOMINANT_SHARE:
            dominant = (stats.top_section, share)

    unpaginated = stats.page_bucket_titles > 0 and (stats.max_page <= 1 or _is_ebook(stats))

    empty_figures = (
        stats.figure_chunks > 0
        and stats.figure_chunks_no_context / stats.figure_chunks > _EMPTY_FIGURES_SHARE
    )

    low_text_density = (
        not _is_ebook(stats)
        and stats.max_page > 0
        and stats.content_chars / stats.max_page < _LOW_TEXT_DENSITY_CHARS_PER_PAGE
    )

    image_only_suspect = (
        stats.parser_used == "epub"
        and stats.figure_chunks / stats.chunks > _IMAGE_ONLY_SUSPECT_SHARE
    )

    too_few_chunks = (
        not _is_ebook(stats)
        and stats.max_page > _TOO_FEW_CHUNKS_MIN_PAGES
        and stats.chunks < stats.max_page * _TOO_FEW_CHUNKS_PER_PAGE
    )

    return BookFinding(
        degenerate,
        dominant,
        unpaginated,
        empty_figures=empty_figures,
        low_text_density=low_text_density,
        image_only_suspect=image_only_suspect,
        too_few_chunks=too_few_chunks,
    )


def _is_ebook(stats: BookStats) -> bool:
    if stats.parser_used is not None:
        return stats.parser_used in REFLOWABLE_PARSERS
    return stats.ebook_source


#: `book_quality_metadata`'s finding names, in the fixed order they're
#: checked -- also the vocabulary `corpus canary-report` and the
#: alerting read from `intake_envelopes.metadata->'quality'->'findings'`.
_QUALITY_FINDING_NAMES: tuple[str, ...] = (
    "no_chunks",
    "degenerate",
    "dominant_section",
    "unpaginated_page_titles",
    "empty_figures",
    "low_text_density",
    "image_only_suspect",
    "too_few_chunks",
)


def book_quality_metadata(stats: BookStats) -> dict[str, Any]:
    """The `intake_envelopes.metadata->'quality'` shape for one book:
    ``{"findings": [...], "stats": {...}}``.

    ``findings`` is the subset of `_QUALITY_FINDING_NAMES` this book trips,
    in that fixed order -- an empty list means a clean book. ``stats`` is
    ``dataclasses.asdict(stats)``: every `BookStats` counter the findings
    above were computed from, so a consumer (`canary-report`, alerting) can
    show *why* a finding fired without a second Qdrant
    read. This shape is a stable interface: new keys
    may be added to either object, but no existing key is renamed or
    removed.
    """
    finding = book_findings(stats)
    tripped = {
        "no_chunks": finding.no_chunks,
        "degenerate": finding.degenerate,
        "dominant_section": finding.dominant is not None,
        "unpaginated_page_titles": finding.unpaginated_page_titles,
        "empty_figures": finding.empty_figures,
        "low_text_density": finding.low_text_density,
        "image_only_suspect": finding.image_only_suspect,
        "too_few_chunks": finding.too_few_chunks,
    }
    findings = [name for name in _QUALITY_FINDING_NAMES if tripped[name]]
    return {"findings": findings, "stats": asdict(stats)}


def _needs_source_format(stats: BookStats) -> bool:
    """Only a null-``parser_used`` book with page-bucket titles past page 1
    has a finding that depends on its source file's format."""
    return stats.parser_used is None and stats.page_bucket_titles > 0 and stats.max_page > 1


class _FacetClient(Protocol):
    """The slice of `AsyncQdrantClient` `all_source_ids` needs -- a
    structural protocol (mirroring `vo_rag.ocr_farm.coordinator.FacetClient`)
    so tests can exercise the growth/truncation loop against a plain fake
    instead of a real Qdrant connection."""

    async def facet(
        self,
        collection_name: str,
        key: str,
        *,
        limit: int = ...,
        exact: bool = ...,
        facet_filter: Filter | None = ...,
    ) -> Any: ...


async def all_source_ids(client: _FacetClient, *, facet_filter: Filter | None = None) -> list[str]:
    """Every distinct `source_id` in the books-raw collection.

    Mirrors `vo_rag.ocr_farm.coordinator.completed_source_books`: Qdrant's
    facet API has no cursor, so this pages by growing the limit until a
    response comes back short of it, and raises rather than silently
    truncating if it's still full at `_FACET_MAX_LIMIT`.

    Public: shared with
    `vo_rag.core.corpus.boilerplate_ops` and `scripts/scan_boilerplate.py`,
    so there is exactly one facet-paging implementation.

    `facet_filter` narrows the facet (for
    `tf_store.tf_source_ids`'s `domain_tags` match) and is re-sent on every
    attempt of the growth loop, so paging and filtering cannot drift apart.
    """
    limit = _FACET_INITIAL_LIMIT
    extra: dict[str, Any] = {} if facet_filter is None else {"facet_filter": facet_filter}
    while True:
        response = await client.facet(
            collection_name=BOOKS_RAW_COLLECTION,
            key="source_id",
            limit=limit,
            exact=False,
            **extra,
        )
        hits = list(getattr(response, "hits", None) or [])
        if len(hits) < limit:
            return sorted({str(hit.value) for hit in hits if str(hit.value or "").strip()})
        if limit >= _FACET_MAX_LIMIT:
            raise RuntimeError(
                f"{BOOKS_RAW_COLLECTION} source_id facet truncated at {limit} values"
            )
        limit = min(limit * 4, _FACET_MAX_LIMIT)


async def scroll_book_payloads(
    client: AsyncQdrantClient,
    source_id: str,
    *,
    with_payload: list[str],
    collection: str = BOOKS_RAW_COLLECTION,
) -> AsyncIterator[Any]:
    """Yield every point of one book's chunks in `collection` (the books-raw
    collection by default), paged at `_SCROLL_PAGE`, `with_vectors=False`.

    Public: the one per-book scroll loop,
    shared by `_collect_one` below, `vo_rag.core.corpus.boilerplate_ops`, and
    `scripts/scan_boilerplate.py`, so a future change to how a book's chunks
    are paged only has to happen in one place.
    """
    scroll_filter = Filter(
        must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
    )
    offset: Any = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            with_payload=with_payload,
            with_vectors=False,
            limit=_SCROLL_PAGE,
            offset=offset,
        )
        for point in points:
            yield point
        if offset is None:
            break


async def _collect_one(
    client: AsyncQdrantClient,
    source_id: str,
    boilerplate_matcher: BoilerplateMatcher | None,
    *,
    collection: str = BOOKS_RAW_COLLECTION,
) -> BookStats:
    stats = BookStats(source_id=source_id)
    section_counts: dict[str, int] = {}

    async for point in scroll_book_payloads(
        client, source_id, with_payload=_STATS_PAYLOAD, collection=collection
    ):
        payload = point.payload or {}
        stats.chunks += 1

        content = payload.get("content")
        content_text = str(content if content is not None else "")
        stats.content_chars += len(content_text.strip())
        tiny = len(content_text.strip()) < _TINY_CONTENT_CHARS
        if tiny:
            stats.tiny += 1

        chunk_type = payload.get("chunk_type")
        if chunk_type == _FIGURE_COMPOSITE_TYPE:
            stats.figure_chunks += 1
            if "Caption:" not in content_text and "Page context:" not in content_text:
                stats.figure_chunks_no_context += 1
        elif chunk_type == _TEXT_CHUNK_TYPE:
            stats.text_chunks += 1

        section_title = payload.get("section_title")
        if isinstance(section_title, str) and section_title:
            section_counts[section_title] = section_counts.get(section_title, 0) + 1
            if _PAGE_BUCKET_TITLE_RE.match(section_title):
                stats.page_bucket_titles += 1

        if stats.source_book is None:
            source_book = payload.get("source_book")
            if isinstance(source_book, str):
                stats.source_book = source_book

        if stats.author is None:
            author = payload.get("author")
            if isinstance(author, str) and author.strip():
                stats.author = author

        if stats.parser_used is None:
            parser_used = payload.get("parser_used")
            if isinstance(parser_used, str) and parser_used.strip():
                stats.parser_used = parser_used

        page_number = payload.get("page_number")
        if isinstance(page_number, int):
            stats.max_page = max(stats.max_page, page_number)

        if boilerplate_matcher is not None and boilerplate_matcher(payload):
            stats.boilerplate += 1
            if tiny:
                stats.tiny_boilerplate += 1

    if section_counts:
        top_title, top_count = max(section_counts.items(), key=lambda kv: kv[1])
        stats.top_section = top_title
        stats.top_section_count = top_count

    return stats


async def collect_stats(
    store: QdrantStore,
    *,
    source_ids: list[str] | None = None,
    boilerplate_matcher: BoilerplateMatcher | None = None,
    pool: FetchPool | None = None,
    collection: str = BOOKS_RAW_COLLECTION,
) -> list[BookStats]:
    """One `BookStats` per book: `source_ids` (in order) if given, else
    every `source_id` in the books-raw collection (via a facet).

    `boilerplate_matcher`, if given, is called with each chunk's payload
    dict; its count feeds `BookStats.boilerplate` (the CLI's `health` command
    supplies `boilerplate_ops.boilerplate_payload_matcher` -- left `None`
    here it always stays 0).

    `pool` (Postgres), if given, sets `BookStats.ebook_source` from the
    intake envelope of every book whose unpaginated finding depends on it;
    without it a null `parser_used` counts as a PDF.

    `collection` is the books-raw collection to
    scroll each book's chunks from -- defaults to `BOOKS_RAW_COLLECTION`, but
    a caller whose books live in a `VORAG_RAW_COLLECTION` override (bulk
    bench/canary runs, `ops/rc_ingest_batch.py`) must pass it, or every
    book's stats come back empty (0 chunks, no findings) from scrolling the
    wrong collection. `source_ids` resolution (the facet) always reads the
    canonical `BOOKS_RAW_COLLECTION` -- only per-book scrolling honours the
    override, matching today's only caller that needs one
    (`BookRagPipeline._record_book_quality`, which always passes explicit
    `source_ids`).
    """
    client = store._client  # noqa: SLF001
    ids = source_ids if source_ids is not None else await all_source_ids(client)
    stats = [
        await _collect_one(client, source_id, boilerplate_matcher, collection=collection)
        for source_id in ids
    ]
    if pool is not None:
        undecided = [book.source_id for book in stats if _needs_source_format(book)]
        if undecided:
            ebooks = await ebook_source_ids(pool, undecided)
            for book in stats:
                book.ebook_source = book.source_id in ebooks
    return stats


async def _test_point_ids(client: AsyncQdrantClient) -> list[str]:
    """Point ids of every chunk tagged canary/rr03/test in `domain_tags`."""
    scroll_filter = Filter(must=[FieldCondition(key="domain_tags", match=MatchAny(any=_TEST_TAGS))])
    ids: list[str] = []
    offset: Any = None
    while True:
        points, offset = await client.scroll(
            collection_name=BOOKS_RAW_COLLECTION,
            scroll_filter=scroll_filter,
            with_payload=False,
            with_vectors=False,
            limit=_SCROLL_PAGE,
            offset=offset,
        )
        ids.extend(str(point.id) for point in points)
        if offset is None:
            break
    return ids


async def run_health(
    store: QdrantStore,
    *,
    source_ids: list[str] | None = None,
    boilerplate_matcher: BoilerplateMatcher | None = None,
    pool: FetchPool | None = None,
) -> HealthReport:
    """Full corpus health report: `collect_stats` + the test-points scroll.
    `pool` is only used to tell legacy PDFs from ebooks (see `collect_stats`)."""
    # Local import (not top-level): `duplicates.py` imports `BookStats` from
    # this module, so a module-level import here would be circular.
    from vo_rag.core.corpus.duplicates import book_authors, plan_duplicates

    stats = await collect_stats(
        store, source_ids=source_ids, boilerplate_matcher=boilerplate_matcher, pool=pool
    )

    report = HealthReport(books=len(stats))
    for book in stats:
        finding = book_findings(book)
        if finding.degenerate:
            report.degenerate.append(book.source_id)
        if finding.dominant is not None:
            title, share = finding.dominant
            report.dominant_section.append((book.source_id, title, share))
        if finding.unpaginated_page_titles:
            report.unpaginated_page_titles.append(book.source_id)
        if finding.empty_figures:
            report.empty_figures.append(book.source_id)
        if finding.low_text_density:
            report.low_text_density.append(book.source_id)
        if finding.image_only_suspect:
            report.image_only_suspect.append(book.source_id)
        if finding.too_few_chunks:
            report.too_few_chunks.append(book.source_id)
        if finding.no_chunks:
            report.no_chunks.append(book.source_id)
        report.boilerplate_points += book.boilerplate

    report.test_points = await _test_point_ids(store._client)  # noqa: SLF001
    dup_rows = plan_duplicates(stats, authors=book_authors(stats), ingested_at={})
    report.duplicate_groups = len({row.group for row in dup_rows if row.action == "delete"})
    return report


__all__ = [
    "BookFinding",
    "BookStats",
    "HealthReport",
    "all_source_ids",
    "book_findings",
    "book_quality_metadata",
    "collect_stats",
    "run_health",
    "scroll_book_payloads",
]
