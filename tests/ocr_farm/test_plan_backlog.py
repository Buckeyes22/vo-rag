"""``plan_backlog``: which failed scanned books the farm should OCR (VOR-1427).

Postgres and Garage are in-memory fakes; PDFs are generated with pymupdf
(a core dependency), so these run in the locked CI env.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import fitz  # type: ignore[import-untyped]
import pytest

from tests.ocr_farm._farm_fakes import BUCKET, FakeGarage, FakePool, FakeQdrant
from vo_rag.ocr_farm import coordinator
from vo_rag.ocr_farm.coordinator import (
    BOOKS_RAW_COLLECTION,
    COMPLETED_TITLES_SQL,
    FAILED_SCANNED_SQL,
    IMAGE_ONLY_EPUB_ENVELOPES,
    INGESTED_SOURCE_ID_PREFIX,
    FarmError,
    completed_source_books,
    count_pages,
    ingested_source_ids,
    plan_backlog,
)

T0 = datetime(2026, 9, 20, tzinfo=UTC)


def _pdf(pages: int, tag: str) -> bytes:
    doc = fitz.open()
    for n in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"{tag} page {n}")
    data: bytes = doc.tobytes()
    doc.close()
    return data


def _row(env_id: str, title: str | None, key: str | None, minutes: int) -> dict[str, Any]:
    return {
        "id": env_id,
        "title": title,
        "garage_key": key,
        "updated_at": T0 + timedelta(minutes=minutes),
    }


def test_failed_sql_shape() -> None:
    sql = " ".join(FAILED_SCANNED_SQL.split())
    assert "FROM intake_envelopes" in sql
    assert "status = 'failed'" in sql
    assert "error::text LIKE '%GLM-OCR forward path%'" in sql
    assert "error::text LIKE '%covered content on only%'" in sql
    assert "error::text LIKE '%OCR requires a CUDA GPU%'" in sql
    assert "id = ANY($1::text[])" in sql
    assert "metadata->>'title'" in sql
    assert "raw_content->>'garage_key'" in sql
    assert "ORDER BY updated_at DESC" in sql
    assert IMAGE_ONLY_EPUB_ENVELOPES == (
        "01M3AZDRK89N2AEX26QQJCNWK2",
        "01M3AYTR6J5R48HAKG6QH0Y5Y9",
        "01M3AZAWMR0YD10D39CG3QWBE8",
    )


def test_completed_sql_shape() -> None:
    """Completed titles come from intake_envelopes (+ Qdrant), never ``sources``."""
    sql = " ".join(COMPLETED_TITLES_SQL.split())
    assert "FROM intake_envelopes WHERE status = 'completed'" in sql
    assert "metadata->>'title'" in sql
    assert "sources" not in sql


async def test_completed_source_books_facets_the_raw_books_collection() -> None:
    qdrant = FakeQdrant(["Book A", "Book B", ""])

    values = await completed_source_books(qdrant)

    assert values == {"Book A", "Book B"}
    assert qdrant.facet_calls == [
        {
            "collection_name": "vo_rag_books_raw_v1",
            "key": "source_book",
            "limit": 10_000,
            "exact": False,
        }
    ]
    assert BOOKS_RAW_COLLECTION == "vo_rag_books_raw_v1"


async def test_completed_source_books_grows_the_facet_limit_until_complete() -> None:
    books = [f"book {n}" for n in range(25_000)]
    qdrant = FakeQdrant(books)

    values = await completed_source_books(qdrant)

    assert len(values) == 25_000
    assert [c["limit"] for c in qdrant.facet_calls] == [10_000, 40_000]


async def test_completed_source_books_refuses_a_truncated_facet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coordinator, "FACET_MAX_LIMIT", 20_000)
    qdrant = FakeQdrant([f"book {n}" for n in range(30_000)])

    with pytest.raises(FarmError, match="truncated"):
        await completed_source_books(qdrant)


async def test_ingested_source_ids_facets_the_raw_books_collection() -> None:
    sid_a, sid_b = f"vo-rag-corpus-sha256:{'a' * 64}", f"vo-rag-corpus-sha256:{'b' * 64}"
    qdrant = FakeQdrant(source_ids=[sid_a, sid_b, ""])

    values = await ingested_source_ids(qdrant)

    assert values == {sid_a, sid_b}
    assert qdrant.facet_calls == [
        {
            "collection_name": "vo_rag_books_raw_v1",
            "key": "source_id",
            "limit": 10_000,
            "exact": False,
        }
    ]
    assert INGESTED_SOURCE_ID_PREFIX == "vo-rag-corpus-sha256:"


async def test_ingested_source_ids_grows_the_facet_limit_until_complete() -> None:
    ids = [f"vo-rag-corpus-sha256:{n:064d}" for n in range(25_000)]
    qdrant = FakeQdrant(source_ids=ids)

    values = await ingested_source_ids(qdrant)

    assert len(values) == 25_000
    assert [c["limit"] for c in qdrant.facet_calls] == [10_000, 40_000]


async def test_ingested_source_ids_refuses_a_truncated_facet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coordinator, "FACET_MAX_LIMIT", 20_000)
    qdrant = FakeQdrant(source_ids=[f"vo-rag-corpus-sha256:{n:064d}" for n in range(30_000)])

    with pytest.raises(FarmError, match="truncated"):
        await ingested_source_ids(qdrant)


async def test_plan_backlog_flags_already_ingested_bytes_even_when_untitled() -> None:
    """An untitled failed envelope whose bytes
    are already ingested under a different envelope must be excluded, not
    re-OCR'd -- title matching alone can't catch it (there's no title)."""
    rows = [
        _row("UNTITLED", "", "u.pdf", 5),
        _row("OTHER", "Other Book", "o.pdf", 4),
    ]
    data_u = _pdf(3, "u")
    data_o = _pdf(2, "o")
    sha_u = hashlib.sha256(data_u).hexdigest()
    garage = FakeGarage({"u.pdf": data_u, "o.pdf": data_o})
    qdrant = FakeQdrant(source_ids=[f"{INGESTED_SOURCE_ID_PREFIX}{sha_u}"])
    pool = FakePool(failed_rows=rows)

    items = await plan_backlog(pool, garage, qdrant, bucket=BUCKET)

    by_id = {i.envelope_id: i for i in items}
    assert by_id["UNTITLED"].problem == "already_ingested"
    assert by_id["UNTITLED"].duplicate_of == f"{INGESTED_SOURCE_ID_PREFIX}{sha_u}"
    # A book whose bytes are NOT already ingested is still planned normally.
    assert by_id["OTHER"].problem is None
    assert by_id["OTHER"].sha256 == hashlib.sha256(data_o).hexdigest()


async def test_plan_backlog_filters_dedupes_counts_and_flags_cache() -> None:
    books = {
        "a.pdf": _pdf(3, "a"),
        "a-older.pdf": _pdf(3, "a-older"),
        "b.pdf": _pdf(5, "b"),
        "c.pdf": _pdf(2, "c"),
        "d.pdf": _pdf(4, "d"),
        "e.pdf": _pdf(1, "e"),
        "f.pdf": _pdf(6, "f"),
    }
    books["g-same-bytes-as-a.pdf"] = books["a.pdf"]
    rows = [
        # Newest first, as the SQL orders them.
        _row("A_NEW", "Strengthcraft [12345678]", "a.pdf", 50),
        _row("B", "Practical_Engines_Applied_Dynam_", "b.pdf", 40),
        _row("A_OLD", "Strengthcraft", "a-older.pdf", 30),  # same book, older failure
        _row("C", "Practical Rowing Skills_", "c.pdf", 20),  # truncated, replaced
        _row("D", "", "d.pdf", 10),  # untitled: cannot match, kept
        _row("E", None, "e.pdf", 5),  # untitled: kept
        _row("F", "Cached Book", "f.pdf", 1),
        _row("G", "Another Upload", "g-same-bytes-as-a.pdf", 0),  # same bytes as A_NEW
    ]
    # From intake_envelopes (completed) ...
    completed = ["Practical Engine"]  # planning false match: must NOT drop B
    # ... and from the books collection's source_book values.
    qdrant = FakeQdrant(["Practical Rowing Skills, Second Edition"])
    f_sha = hashlib.sha256(books["f.pdf"]).hexdigest()
    objects = dict(books)
    objects[f"ocr/{f_sha}/docling.json"] = b"{}"
    garage = FakeGarage(objects)
    pool = FakePool(failed_rows=rows, completed_titles=completed)

    items = await plan_backlog(pool, garage, qdrant, bucket=BUCKET)

    by_id = {i.envelope_id: i for i in items}
    assert set(by_id) == {"A_NEW", "B", "D", "E", "F", "G"}
    # Same sha256 as A_NEW: reported, not OCR'd again, tied to the kept envelope.
    assert by_id["G"].problem == "duplicate_sha256"
    assert by_id["G"].duplicate_of == "A_NEW"
    assert by_id["G"].sha256 == by_id["A_NEW"].sha256
    items = [i for i in items if i.envelope_id != "G"]
    assert by_id["A_NEW"].pages == 3
    assert by_id["A_NEW"].sha256 == hashlib.sha256(books["a.pdf"]).hexdigest()
    assert by_id["A_NEW"].garage_key == "a.pdf"
    assert by_id["A_NEW"].fmt == "pdf"
    assert by_id["B"].pages == 5
    assert by_id["D"].title == ""
    assert by_id["E"].title == ""
    assert by_id["F"].cached is True
    assert all(not by_id[k].cached for k in ("A_NEW", "B", "D", "E"))
    assert all(i.problem is None for i in items)
    # The fixed EPUB ids are passed to the failed-envelope query.
    failed_call = next(c for c in pool.fetch_calls if c[0] == FAILED_SCANNED_SQL)
    assert failed_call[1] == (list(IMAGE_ONLY_EPUB_ENVELOPES),)
    # A dropped book's bytes are never downloaded.
    assert (BUCKET, "c.pdf") not in garage.gets
    assert (BUCKET, "a-older.pdf") not in garage.gets


async def test_plan_backlog_marks_missing_and_unsupported_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row("NOKEY", "No Key", None, 5),
        _row("GONE", "Gone", "gone.pdf", 4),
        _row("MOBI", "Weird", "weird.bin", 3),
        _row("CORRUPT", "Corrupt", "corrupt.pdf", 2),
        _row("EMPTY", "Empty Epub", "empty.epub", 1),
    ]
    garage = FakeGarage(
        {
            "weird.bin": b"BOOKMOBI-not-a-pdf",
            "corrupt.pdf": b"%PDF-1.4 garbage",
            "empty.epub": b"PK\x03\x04 no spine images",
        }
    )
    monkeypatch.setattr(coordinator, "count_epub_spine_images", lambda data: 0)
    items = await plan_backlog(FakePool(failed_rows=rows), garage, FakeQdrant(), bucket=BUCKET)

    problems = {i.envelope_id: i.problem for i in items}
    assert problems == {
        "NOKEY": "missing_garage_key",
        "GONE": "missing_source_object",
        "MOBI": "unsupported_format",
        "CORRUPT": "unreadable_source",
        "EMPTY": "no_pages",
    }


async def test_plan_backlog_counts_epub_spine_images(monkeypatch: pytest.MonkeyPatch) -> None:
    epub = b"PK\x03\x04 fake epub"
    seen: list[bytes] = []

    def fake_count(data: bytes) -> int:
        seen.append(data)
        return 17

    monkeypatch.setattr(coordinator, "count_epub_spine_images", fake_count)
    rows = [_row("01M3AZDRK89N2AEX26QQJCNWK2", "Scanned Epub", "scan.epub", 1)]
    items = await plan_backlog(
        FakePool(failed_rows=rows), FakeGarage({"scan.epub": epub}), FakeQdrant(), bucket=BUCKET
    )

    assert items[0].fmt == "epub"
    assert items[0].pages == 17
    assert seen == [epub]


def test_count_pages_pdf() -> None:
    assert count_pages(_pdf(7, "x")) == ("pdf", 7)


def test_count_pages_unknown() -> None:
    assert count_pages(b"hello") == ("unknown", 0)


FIXTURE_EPUB = (
    Path(__file__).resolve().parents[1] / "fixtures" / "epub" / "image_only_25_pages.epub"
)


def test_count_epub_spine_images_with_real_ebooklib() -> None:
    pytest.importorskip("ebooklib", reason="ebooklib not installed (runtime extra)")
    data = FIXTURE_EPUB.read_bytes()

    assert coordinator.count_epub_spine_images(data) == 25
    assert count_pages(data, "scan.epub") == ("epub", 25)
    # Content-sniffed too: the EPUB mimetype entry is enough without a name.
    assert count_pages(data) == ("epub", 25)


def test_epub_to_pdf_bytes_with_real_ebooklib() -> None:
    pytest.importorskip("ebooklib", reason="ebooklib not installed (runtime extra)")
    pdf = coordinator.epub_to_pdf_bytes(FIXTURE_EPUB.read_bytes())

    assert count_pages(pdf) == ("pdf", 25)
