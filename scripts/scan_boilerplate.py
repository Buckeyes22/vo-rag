#!/usr/bin/env python3
"""Read-only scan of the books-raw collection for boilerplate-chunk candidates
(VOR-1427). The scan survives the full collection -- up to ~450k chunks.

Scrolls every book in ``vo_rag_books_raw_v1`` (or the ``--source-id``
subset), one book at a time, reusing the exact same async Qdrant helpers
``vo_rag.core.corpus.health`` uses (``all_source_ids`` for the book list,
``scroll_book_payloads`` for the per-book scroll) -- there is exactly one
facet-paging implementation and one per-book scroll loop in this codebase,
not a second copy reimplemented over raw REST. Content is normalized with
the SAME function the ingest-time filter and ``corpus boilerplate``/
``corpus health`` use
(``vo_rag.pipelines.book_rag.boilerplate.normalize_chunk_text``), so a
reviewed row here is guaranteed to match what those consumers would
drop/count once added to ``BOILERPLATE_CHUNKS``.

Memory discipline: a full-collection scan must survive ~450k chunks
without holding the whole corpus in memory twice --
``scan()`` is a per-book async generator (never materializes a list of every
chunk), each book's chunks are deduped to their DISTINCT normalized texts
before ever reaching the aggregator (a running header repeated on every one
of a book's pages costs one entry, not one per page), and
``aggregate_candidates`` keys its accumulator on a 16-byte
``blake2b`` digest of the normalized text rather than the text itself,
holding the full text (and its examples) only once a digest has been seen
in 2 or more distinct books -- the vast majority of a real corpus's chunks
are unique to one book and are never that.

This script does NOT decide what is boilerplate -- it only surfaces
candidates (grouped by normalized text, counted by DISTINCT book, not raw
occurrence count) for a human to review. A
row is only added to ``BOILERPLATE_CHUNKS`` once it is confirmed to carry no
book-specific information (blank-page notices, EPUB "About This E-Book"
blocks, copyright pages, publisher marketing blocks). Reject rows such as
common equations or "Summary" -- those ARE book-specific content that happens
to repeat.

This is a maintenance tool for a real Qdrant instance -- point it
EXPLICITLY at the instance you mean to scan with ``--qdrant-url``. There is
no default: this workstation is not part of the RAG stack and must never be
pointed at production by accident.

Usage:
  uv run python scripts/scan_boilerplate.py \\
      --qdrant-url http://127.0.0.1:6400 \\
      --min-books 20 \\
      --out review-artifacts/boilerplate-candidates.tsv
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vo_rag.core.corpus.health import all_source_ids as all_source_ids
from vo_rag.core.corpus.health import scroll_book_payloads as scroll_book_payloads
from vo_rag.core.storage.client_factory import build_qdrant_client
from vo_rag.pipelines.book_rag.boilerplate import normalize_chunk_text as normalize_chunk_text

_DEFAULT_EXAMPLE_LIMIT = 3
_DIGEST_SIZE = 16
#: A digest's text/examples are only ever materialized once it has been seen
#: in at least this many distinct books -- the
#: memory floor that keeps `aggregate_candidates` from retaining full chunk
#: text for the ~450k chunks that never repeat across books. `--min-books`
#: (the CLI's user-facing knob, typically much higher, e.g. 20) is applied
#: separately by `filter_min_books`, on top of this floor.
_MIN_BOOKS_FLOOR = 2


def _digest(normalized_text: str) -> bytes:
    """16-byte blake2b digest of *normalized_text* (pins the
    exact hashing scheme `aggregate_candidates` keys its accumulator on)."""
    return hashlib.blake2b(normalized_text.encode("utf-8"), digest_size=_DIGEST_SIZE).digest()


@dataclass(frozen=True)
class CandidateRow:
    """One boilerplate candidate: a normalized text, how many DISTINCT books
    contain a chunk that normalizes to it, and up to ``example_limit`` of
    those books' source ids (for a reviewer to go look at)."""

    normalized_text: str
    distinct_books: int
    example_source_ids: tuple[str, ...]


@dataclass
class _Accumulator:
    """Per-digest running state. ``text`` stays ``None`` -- the memory this
    whole scheme exists to avoid retaining -- until ``books`` reaches
    ``_MIN_BOOKS_FLOOR``."""

    books: dict[str, None] = field(default_factory=dict)  # ordered set of book ids
    text: str | None = None


async def _as_async_iter(
    rows: Iterable[tuple[str, str]] | AsyncIterable[tuple[str, str]],
) -> AsyncIterator[tuple[str, str]]:
    """Adapt a plain (sync) iterable -- what every unit test passes -- or an
    async one -- what the live ``scan()`` generator is -- to one shape, so
    ``aggregate_candidates`` below has exactly one accumulation loop that
    NEVER buffers more than one row at a time regardless of which it's fed."""
    if isinstance(rows, AsyncIterable):
        async for row in rows:
            yield row
    else:
        for row in rows:
            yield row


async def aggregate_candidates(
    rows: Iterable[tuple[str, str]] | AsyncIterable[tuple[str, str]],
    *,
    example_limit: int = _DEFAULT_EXAMPLE_LIMIT,
) -> list[CandidateRow]:
    """*rows* is ``(source_id, content)`` pairs -- one per (deduped) chunk --
    either a plain iterable (every unit test) or ``scan()``'s async
    per-book generator: this consumes it one row
    at a time either way, never buffering the whole thing into a list first.

    Groups by a blake2b digest of ``normalize_chunk_text(content)`` (not the
    text itself, to bound memory over a ~450k-chunk scan), counting DISTINCT
    source ids per digest -- a book repeating its own boilerplate on every
    page must not inflate its candidacy past other books that only have it
    once. The normalized text (and thus the row itself) is only ever
    materialized once a digest reaches ``_MIN_BOOKS_FLOOR`` distinct books --
    singleton chunks (the bulk of a real corpus) are counted and then
    forgotten. Content that normalizes to the empty string (pure
    punctuation/whitespace) carries no signal and is never counted. Sorted
    by distinct book count (descending), then by normalized text, so the
    most cross-book-common candidates sort to the top of the reviewed TSV.
    """
    acc: dict[bytes, _Accumulator] = {}
    async for source_id, content in _as_async_iter(rows):
        normalized = normalize_chunk_text(content)
        if not normalized:
            continue
        digest = _digest(normalized)
        entry = acc.setdefault(digest, _Accumulator())
        if source_id in entry.books:
            continue  # already counted this book for this digest
        entry.books[source_id] = None
        if entry.text is None and len(entry.books) >= _MIN_BOOKS_FLOOR:
            entry.text = normalized

    candidates = [
        CandidateRow(
            normalized_text=entry.text,
            distinct_books=len(entry.books),
            example_source_ids=tuple(list(entry.books)[:example_limit]),
        )
        for entry in acc.values()
        if entry.text is not None
    ]
    candidates.sort(key=lambda c: (-c.distinct_books, c.normalized_text))
    return candidates


def filter_min_books(candidates: Sequence[CandidateRow], *, min_books: int) -> list[CandidateRow]:
    """Only rows appearing in at least ``min_books`` distinct books (on top
    of the ``_MIN_BOOKS_FLOOR`` already baked into ``aggregate_candidates``)."""
    return [c for c in candidates if c.distinct_books >= min_books]


def write_tsv(candidates: Sequence[CandidateRow], out: Path) -> None:
    """``normalized_text\\tdistinct_books\\texample_source_ids``, one header
    line plus one line per candidate (already in the caller's sort order)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["normalized_text\tdistinct_books\texample_source_ids"]
    for c in candidates:
        lines.append(f"{c.normalized_text}\t{c.distinct_books}\t{','.join(c.example_source_ids)}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Live scan (never exercised by unit tests -- those only cover the pure
# aggregate_candidates/filter_min_books/write_tsv functions above).
# ---------------------------------------------------------------------------


async def scan(*, qdrant_url: str, source_ids: list[str] | None) -> AsyncIterator[tuple[str, str]]:
    """One ``(source_id, content)`` pair per DISTINCT normalized text within
    each book (or every book in the books-raw collection when ``source_ids``
    is None). Read-only, and a per-book generator
    -- never materializes every chunk of the collection into a list.

    Dedupe happens per book (a running header on every page of a 900-page
    book costs one entry here, not 900) before anything is yielded to the
    aggregator -- `aggregate_candidates` still counts DISTINCT BOOKS
    correctly regardless, since it dedupes per-digest-per-book too, but this
    cuts the volume fed to it dramatically for the common case.
    """
    client = build_qdrant_client(url=qdrant_url, api_key=None, prefer_grpc=False)
    try:
        ids = source_ids if source_ids is not None else await all_source_ids(client)
        for source_id in ids:
            seen: set[str] = set()
            async for point in scroll_book_payloads(client, source_id, with_payload=["content"]):
                content = (point.payload or {}).get("content")
                if not content:
                    continue
                content_str = str(content)
                normalized = normalize_chunk_text(content_str)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                yield source_id, content_str
    finally:
        await client.close()


async def _run(*, qdrant_url: str, source_ids: list[str] | None, min_books: int, out: Path) -> str:
    candidates = await aggregate_candidates(scan(qdrant_url=qdrant_url, source_ids=source_ids))
    kept = filter_min_books(candidates, min_books=min_books)
    write_tsv(kept, out)
    return f"{len(kept)} candidate row(s) (of {len(candidates)} aggregated) -> {out}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--qdrant-url",
        required=True,
        help="Qdrant base URL, e.g. http://127.0.0.1:6400. Read-only: only "
        "issues facet/scroll requests, never upserts or deletes. Point this "
        "explicitly -- there is no default, on purpose.",
    )
    parser.add_argument(
        "--min-books",
        type=int,
        default=_MIN_BOOKS_FLOOR,
        help=f"only emit rows seen in at least this many distinct books "
        f"(default: {_MIN_BOOKS_FLOOR}, the same floor already baked into the scan)",
    )
    parser.add_argument(
        "--source-id",
        dest="source_ids",
        action="append",
        default=None,
        help="restrict the scan to this source id (repeatable). Default: every book.",
    )
    parser.add_argument("--out", type=Path, required=True, help="TSV output path")
    args = parser.parse_args(argv)

    summary = asyncio.run(
        _run(
            qdrant_url=args.qdrant_url,
            source_ids=args.source_ids,
            min_books=args.min_books,
            out=args.out,
        )
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
