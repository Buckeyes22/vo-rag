"""Duplicate-title grouping, keep rule, and report rows (VOR-1427).

The operator approves a delete list before anything is actually removed;
this module only *plans* that list. ``plan_duplicates`` groups ``BookStats``
by normalized title + author + publication year and, within each
group of 2+, applies the keep rule to decide which copy survives.

Title normalization itself lives in ``vo_rag.core.corpus.titles`` (reused here
rather than reimplemented) so there is exactly
one definition of "same title" across the OCR-skip path and this report.

Year handling (VOR-1427): ``normalize_title`` folds "(YYYY)"
away, so grouping on title alone would merge different editions of the same
book -- e.g. "Applied Strength Methods (2013)" and "Applied Strength Methods
(2019)" -- into one group with one marked for deletion. ``titles.title_year`` reads that
year back out (without changing what ``normalize_title`` itself strips, so
callers like ``plan_backlog`` that already key off it are unaffected).

Within a normalized-title-plus-author bucket, ``_split_bucket`` decides how
books with an explicit year and books with none combine. Grouping is by exact year,
not by union-find, which has a transitivity bug: a year-less book unioned
with *both* an explicit-2013 book and an explicit-2019 book would silently
merge 2013 and 2019 into one group via that bridge and delete one of the
two real editions:

- 0 or 1 distinct explicit year in the bucket: unambiguous. Every book --
  explicit-year or not -- forms one group and the normal keep rule applies.
  A missing year here is not evidence of a different edition; it groups
  with the single year present (or with other missing years).
- 2+ distinct explicit years: each exact year forms its own group (never
  merged with a *different* year, and never bridged by a year-less book).
  The year-less books, instead of guessing which edition they belong to,
  become their own group with every row forced to ``action="keep"``,
  ``reason="ambiguous_year"`` -- nothing ambiguous ever reaches the delete
  list. This still surfaces in the CSV as its own group so the operator can
  resolve it manually, including a *lone* year-less book in this situation
  (never silently dropped as an ordinary singleton "group" would be).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vo_rag.core.corpus.health import BookStats
from vo_rag.core.corpus.source_format import FetchPool, source_sha256
from vo_rag.core.corpus.titles import normalize_title, title_year

#: Keep-rule sort key, in priority order (first difference wins):
#: (bad_chunks, reencoded, -chunks, ingested_at-or-"~", source_id). The
#: trailing `source_id` is a pure determinism tie-break -- without it, the
#: keeper among otherwise fully-tied books would
#: depend on `collect_stats`'s (arbitrary) input order. One reason string
#: per position, so a delete row's reason names the first key that actually
#: separated it from the keeper.
_REASONS = (
    "more degenerate/boilerplate chunks than keeper",
    "reencoded twin",
    "fewer chunks than keeper",
    "ingested later than keeper",
    "tie-break: higher source_id than keeper",
)
_KEEP_REASON = "fewest bad chunks, not reencoded, most chunks, earliest ingested"
#: Forced action/reason for a year-less book caught between 2+ distinct
#: explicit years in the same title+author bucket -- it never
#: reaches the delete list, regardless of how many other year-less books
#: share its bucket.
_AMBIGUOUS_REASON = "ambiguous_year"

_SortKey = tuple[int, bool, int, str, str]

#: Earliest completed intake envelope per book sha256 (never the legacy
#: ``sources`` table, which nothing writes any more). The envelope
#: match is ``source_format.SHA_ENVELOPE_MATCH`` (the ``garage_key`` lookup plus
#: ``raw_content.file_hash``).
ENVELOPE_INGESTED_AT_SQL = """
SELECT k.sha256, MIN(e.created_at) AS ingested_at
FROM unnest($1::text[]) AS k(sha256)
JOIN intake_envelopes e
  ON e.status = 'completed'
 AND (
        e.raw_content->>'file_hash' = k.sha256
     OR e.raw_content->>'garage_key' LIKE '%' || k.sha256 || '%'
     OR e.source_data->>'garage_key' LIKE '%' || k.sha256 || '%'
 )
GROUP BY k.sha256
"""


async def envelope_ingested_at(pool: FetchPool, source_ids: list[str]) -> dict[str, str | None]:
    """``source_id -> str(earliest completed envelope created_at)`` (``None``
    when the id carries no sha256 or no completed envelope matches)."""
    shas = {source_id: source_sha256(source_id) for source_id in source_ids}
    wanted = sorted({sha for sha in shas.values() if sha is not None})
    by_sha: dict[str, str] = {}
    if wanted:
        for row in await pool.fetch(ENVELOPE_INGESTED_AT_SQL, wanted):
            if row["ingested_at"] is not None:
                by_sha[str(row["sha256"])] = str(row["ingested_at"])
    return {
        source_id: (by_sha.get(sha) if sha is not None else None) for source_id, sha in shas.items()
    }


def book_authors(stats: list[BookStats]) -> dict[str, str | None]:
    """``source_id -> author`` from the chunk payload (``BookStats.author``):
    the one author source `corpus health` and `corpus duplicates` share."""
    return {book.source_id: book.author for book in stats}


@dataclass
class DupRow:
    """One book's row in the duplicate-title report."""

    group: str
    source_id: str
    source_book: str
    chunks: int
    bad_chunks: int
    reencoded: bool
    ingested_at: str | None
    action: Literal["keep", "delete"]
    reason: str


def _is_reencoded(source_book: str) -> bool:
    """True when the raw (un-normalized) title carries a ``.reencoded``
    marker -- the twin flag the keep rule penalizes, independent of what
    ``normalize_title`` folds away for grouping purposes."""
    return source_book.strip().lower().endswith(".reencoded")


def _bad_chunks(book: BookStats) -> int:
    return book.bad_chunks


def _sort_key(book: BookStats, ingested_at: dict[str, str | None]) -> _SortKey:
    return (
        _bad_chunks(book),
        _is_reencoded(book.source_book or ""),
        -book.chunks,
        ingested_at.get(book.source_id) or "~",
        book.source_id,
    )


def _first_diff_reason(keeper_key: _SortKey, other_key: _SortKey) -> str:
    for index, (keeper_value, other_value) in enumerate(zip(keeper_key, other_key, strict=True)):
        if keeper_value != other_value:
            return _REASONS[index]
    return _REASONS[-1]  # pragma: no cover -- source_id always differs between two books


def _split_bucket(books: list[BookStats]) -> list[tuple[list[BookStats], str, bool]]:
    """Split ``books`` (already grouped by normalized title + author) into
    ``(cluster_books, year_tag, ambiguous)`` sub-clusters.

    No transitive merging (see module docstring): a year-less
    book never bridges two *different* explicit years. Exact equality only:

    - 0 or 1 distinct explicit year present: one cluster, everyone in it,
      tagged with that year (or ``"no-year"`` if none of them has one).
    - 2+ distinct explicit years: one cluster per exact year (tagged with
      that year), plus -- if any year-less books exist -- one more cluster
      of exactly those year-less books, tagged ``"ambiguous"`` and flagged
      ``ambiguous=True`` so the caller forces every row in it to
      ``keep``/``ambiguous_year`` instead of running the normal keep rule.
    """
    years = [title_year(book.source_book or "") for book in books]
    distinct_years = {year for year in years if year is not None}

    if len(distinct_years) <= 1:
        tag = next(iter(distinct_years), "no-year")
        return [(books, tag, False)]

    by_year: dict[str, list[BookStats]] = {}
    yearless: list[BookStats] = []
    for book, year in zip(books, years, strict=True):
        if year is None:
            yearless.append(book)
        else:
            by_year.setdefault(year, []).append(book)

    clusters: list[tuple[list[BookStats], str, bool]] = [
        (year_books, year, False) for year, year_books in by_year.items()
    ]
    if yearless:
        clusters.append((yearless, "ambiguous", True))
    return clusters


def _dup_row(
    book: BookStats,
    *,
    group: str,
    action: Literal["keep", "delete"],
    reason: str,
    ingested_at: dict[str, str | None],
) -> DupRow:
    return DupRow(
        group=group,
        source_id=book.source_id,
        source_book=book.source_book or "",
        chunks=book.chunks,
        bad_chunks=_bad_chunks(book),
        reencoded=_is_reencoded(book.source_book or ""),
        ingested_at=ingested_at.get(book.source_id),
        action=action,
        reason=reason,
    )


def plan_duplicates(
    stats: list[BookStats],
    authors: dict[str, str | None],
    ingested_at: dict[str, str | None],
) -> list[DupRow]:
    """Group ``stats`` by ``(normalize_title(source_book), author)``, split
    each group by publication year (``_split_bucket``), and plan keep/delete
    for every resulting cluster.

    Books with no title (``normalize_title`` folds to ``""``) are never
    grouped -- an empty key would otherwise lump every untitled book
    together as "duplicates" of each other. An unambiguous cluster needs 2+
    books to produce any rows (same as before); an *ambiguous* year-less
    cluster (see ``_split_bucket``) always produces one ``keep``/
    ``ambiguous_year`` row per book in it, even a lone one, so it's never
    silently dropped.
    """
    groups: dict[tuple[str, str], list[BookStats]] = {}
    for book in stats:
        normalized = normalize_title(book.source_book or "")
        if not normalized:
            continue
        author_key = (authors.get(book.source_id) or "").strip().lower()
        groups.setdefault((normalized, author_key), []).append(book)

    rows: list[DupRow] = []
    for (normalized, author_key), books in groups.items():
        for cluster, year_tag, ambiguous in _split_bucket(books):
            group_label = f"{normalized}|{author_key}|{year_tag}"

            if ambiguous:
                rows.extend(
                    _dup_row(
                        book,
                        group=group_label,
                        action="keep",
                        reason=_AMBIGUOUS_REASON,
                        ingested_at=ingested_at,
                    )
                    for book in cluster
                )
                continue

            if len(cluster) < 2:
                continue

            ordered = sorted(cluster, key=lambda book: _sort_key(book, ingested_at))
            keeper = ordered[0]
            keeper_key = _sort_key(keeper, ingested_at)

            for book in ordered:
                if book is keeper:
                    rows.append(
                        _dup_row(
                            book,
                            group=group_label,
                            action="keep",
                            reason=_KEEP_REASON,
                            ingested_at=ingested_at,
                        )
                    )
                else:
                    reason = _first_diff_reason(keeper_key, _sort_key(book, ingested_at))
                    rows.append(
                        _dup_row(
                            book,
                            group=group_label,
                            action="delete",
                            reason=reason,
                            ingested_at=ingested_at,
                        )
                    )

    return rows


__all__ = [
    "ENVELOPE_INGESTED_AT_SQL",
    "DupRow",
    "book_authors",
    "envelope_ingested_at",
    "plan_duplicates",
    "source_sha256",
]
