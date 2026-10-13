"""RunPod farm staging: <=400-page unit PDFs, EPUB -> PDF, LPT shards (VOR-1434)."""

from __future__ import annotations

import json
from pathlib import Path

import fitz  # type: ignore[import-untyped]
import pytest

from tests.ocr_farm._runpod_fakes import FakeCache, pdf_bytes, pdf_item
from vo_rag.ocr_farm.runpod import stage as stage_mod
from vo_rag.ocr_farm.runpod.stage import load_backlog, shard_units, stage_books

BUCKET = "vorag-artifacts"


def _pages(path: Path) -> list[str]:
    with fitz.open(str(path)) as doc:
        return [page.get_text().strip() for page in doc]


async def test_splits_books_into_parts_of_at_most_400_pages(tmp_path: Path) -> None:
    big, big_data = pdf_item(901, "big")
    small, small_data = pdf_item(12, "small")
    cache = FakeCache({big.garage_key: big_data, small.garage_key: small_data})

    result = await stage_books([big, small], cache, bucket=BUCKET, units_dir=tmp_path, pods=2)

    assert result.bad == []
    ranges = [(u["first"], u["last"]) for u in result.units if u["sha256"] == big.sha256]
    assert ranges == [(1, 400), (401, 800), (801, 901)]
    unit = next(u for u in result.units if u["first"] == 401)
    assert unit["unit"] == f"{big.sha256}__p001"
    texts = _pages(tmp_path / f"{unit['unit']}.pdf")
    assert len(texts) == 400 and texts[0] == "page 401" and texts[-1] == "page 800"
    assert [b["parts"] for b in result.books] == [3, 1]


async def test_epub_is_rendered_through_epub_to_pdf_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, data = pdf_item(3, "ep", fmt="epub")
    seen: list[bytes] = []

    def fake_epub_to_pdf(raw: bytes) -> bytes:
        seen.append(raw)
        return pdf_bytes(3)

    monkeypatch.setattr(stage_mod, "epub_to_pdf_bytes", fake_epub_to_pdf)
    result = await stage_books(
        [item], FakeCache({item.garage_key: data}), bucket=BUCKET, units_dir=tmp_path, pods=1
    )

    assert seen == [data]
    assert [u["pages"] for u in result.units] == [3]


async def test_sha_and_page_count_mismatches_are_reported_not_staged(tmp_path: Path) -> None:
    good, good_data = pdf_item(5, "good")
    wrong_sha, _ = pdf_item(5, "sha")
    wrong_pages, wp_data = pdf_item(5, "pages")
    wrong_pages = wrong_pages.__class__(**{**wrong_pages.to_dict(), "pages": 6})
    cache = FakeCache(
        {
            good.garage_key: good_data,
            wrong_sha.garage_key: b"other",
            wrong_pages.garage_key: wp_data,
        }
    )

    result = await stage_books(
        [good, wrong_sha, wrong_pages], cache, bucket=BUCKET, units_dir=tmp_path, pods=1
    )

    assert [b["sha256"] for b in result.books] == [good.sha256]
    assert result.bad == [
        {"envelope_id": "ENVsha", "reason": "sha256_mismatch"},
        {"envelope_id": "ENVpages", "reason": "page_count 5 != planned 6"},
    ]


def test_shards_balance_pages_longest_first() -> None:
    units = [{"unit": f"u{p}", "pages": p} for p in (400, 400, 300, 100, 100)]

    loads = shard_units(units, 2)

    assert loads == [600, 700] or loads == [700, 600]
    assert {u["shard"] for u in units} == {0, 1}


def test_load_backlog_keeps_only_books_that_need_ocr(tmp_path: Path) -> None:
    item, _ = pdf_item(4, "keep")
    rows = [
        item.to_dict(),
        {**item.to_dict(), "envelope_id": "C", "cached": True},
        {**item.to_dict(), "envelope_id": "P", "problem": "no_pages"},
        {"books": 3, "to_ocr": 1},
    ]
    path = tmp_path / "backlog.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assert load_backlog(path) == [item]
