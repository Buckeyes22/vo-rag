"""RunPod farm merge + upload + verify (VOR-1434).

Upload order is the Garage cache contract: ``ocr/<sha>/docling.json`` first,
then ``meta.json`` (watchers flip envelopes when meta.json appears), and an
existing cache is only replaced with ``overwrite=True`` (the 2026-09-26 re-dos).
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.ocr_farm._runpod_fakes import FakeCache
from vo_rag.ocr_farm.coordinator import ocr_key, ocr_meta_key
from vo_rag.ocr_farm.runpod.merge_upload import merge_and_upload, unit_state
from vo_rag.ocr_farm.runpod.verify import verify_books

BUCKET = "vorag-artifacts"
SHA = "a" * 64
OK_NOTE = "ocr_worker: ok status=success pages=400 batches=4 gpu_peak_bytes=1"


def _docling(pages: range) -> dict[str, object]:
    size = {"width": 100.0, "height": 100.0}
    return {
        "schema_name": "DoclingDocument",
        "version": "1.7.0",
        "name": "part",
        "pages": {str(p): {"size": size, "page_no": p} for p in pages},
    }


def _book(pages: int = 650, parts: int = 2) -> dict[str, object]:
    return {"envelope_id": "ENV1", "sha256": SHA, "title": "T", "fmt": "pdf", "pages": pages,
            "parts": parts}  # fmt: skip


def _units() -> list[dict[str, object]]:
    return [
        {"unit": f"{SHA}__p000", "sha256": SHA, "index": 0, "first": 1, "last": 400, "pages": 400},
        {"unit": f"{SHA}__p001", "sha256": SHA, "index": 1, "first": 401, "last": 650,
         "pages": 250},
    ]  # fmt: skip


def _write_unit(out: Path, unit: str, pages: int, note: str = OK_NOTE) -> None:
    out.mkdir(exist_ok=True)
    (out / f"{unit}.json").write_text(json.dumps(_docling(range(1, pages + 1))))
    (out / f"{unit}.note").write_text(note + "\n")


async def test_merges_parts_and_uploads_docling_then_meta(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_unit(out, f"{SHA}__p000", 400)
    _write_unit(out, f"{SHA}__p001", 250)
    cache = FakeCache()

    [outcome] = await merge_and_upload(
        [_book()], _units(), out, tmp_path / "merged", cache, bucket=BUCKET, job_id="job1"
    )

    assert outcome["status"] == "uploaded"
    assert cache.uploads == [ocr_key(SHA), ocr_meta_key(SHA)]
    merged = json.loads(cache.objects[(BUCKET, ocr_key(SHA))])
    assert sorted(int(p) for p in merged["pages"]) == list(range(1, 651))
    meta = json.loads(cache.objects[(BUCKET, ocr_meta_key(SHA))])
    assert meta["pipeline"] == "ocr" and meta["host"] == "runpod" and meta["pages"] == 650
    assert meta["parts"] == 2 and meta["worker"] == [OK_NOTE, OK_NOTE]


async def test_existing_cache_is_kept_unless_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_unit(out, f"{SHA}__p000", 400)
    _write_unit(out, f"{SHA}__p001", 250)
    old = {ocr_key(SHA): b"{}", ocr_meta_key(SHA): b'{"pipeline": "ocr", "pages": 650}'}
    cache = FakeCache(old)

    [kept] = await merge_and_upload(
        [_book()], _units(), out, tmp_path / "m", cache, bucket=BUCKET, job_id="j"
    )
    assert kept["status"] == "cached" and cache.uploads == []

    [redone] = await merge_and_upload(
        [_book()], _units(), out, tmp_path / "m", cache, bucket=BUCKET, job_id="j", overwrite=True
    )
    assert redone["status"] == "uploaded"
    assert cache.uploads == [ocr_key(SHA), ocr_meta_key(SHA)]
    assert json.loads(cache.objects[(BUCKET, ocr_meta_key(SHA))])["host"] == "runpod"


async def test_failed_and_pending_units_block_the_book_and_are_named(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_unit(out, f"{SHA}__p000", 400, note="Killed")
    cache = FakeCache()

    [outcome] = await merge_and_upload(
        [_book()], _units(), out, tmp_path / "m", cache, bucket=BUCKET, job_id="j"
    )

    assert outcome["status"] == "incomplete"
    assert outcome["units_failed"] == {f"{SHA}__p000": "Killed"}
    assert outcome["units_pending"] == [f"{SHA}__p001"]
    assert cache.uploads == []


async def test_merged_page_gap_fails_the_book(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _write_unit(out, f"{SHA}__p000", 400)
    _write_unit(out, f"{SHA}__p001", 249)  # one page short
    cache = FakeCache()

    [outcome] = await merge_and_upload(
        [_book()], _units(), out, tmp_path / "m", cache, bucket=BUCKET, job_id="j"
    )

    assert outcome["status"] == "failed(merged_pages 649 != 650)"
    assert cache.uploads == []


def test_unit_state_needs_json_and_an_ok_note(tmp_path: Path) -> None:
    assert unit_state(tmp_path, "u") == ("pending", "")
    (tmp_path / "u.note").write_text("ocr_worker: CUDA unavailable\n")
    assert unit_state(tmp_path, "u") == ("failed", "ocr_worker: CUDA unavailable")
    (tmp_path / "u.note").write_text(OK_NOTE)
    assert unit_state(tmp_path, "u") == ("failed", "ok note but no json")
    (tmp_path / "u.json").write_text("{}")
    assert unit_state(tmp_path, "u") == ("ok", OK_NOTE)


async def test_verify_requires_both_objects_ocr_pipeline_and_page_count() -> None:
    books = [
        {"envelope_id": "E1", "sha256": "1" * 64, "pages": 3},
        {"envelope_id": "E2", "sha256": "2" * 64, "pages": 3},
        {"envelope_id": "E3", "sha256": "3" * 64, "pages": 3},
        {"envelope_id": "E4", "sha256": "4" * 64, "pages": 3},
    ]
    ocr = json.dumps({"pipeline": "ocr", "pages": 3}).encode()
    cache = FakeCache(
        {
            ocr_key("1" * 64): b"{}",
            ocr_meta_key("1" * 64): ocr,
            ocr_meta_key("2" * 64): ocr,
            ocr_key("3" * 64): b"{}",
            ocr_meta_key("3" * 64): json.dumps({"pipeline": "native", "pages": 3}).encode(),
            ocr_key("4" * 64): b"{}",
            ocr_meta_key("4" * 64): json.dumps({"pipeline": "ocr", "pages": 2}).encode(),
        }
    )

    results = await verify_books(books, cache, bucket=BUCKET)

    assert [(r["envelope_id"], r["ok"], r["reason"]) for r in results] == [
        ("E1", True, ""),
        ("E2", False, "missing docling.json"),
        ("E3", False, "meta pipeline 'native' != 'ocr'"),
        ("E4", False, "meta pages 2 != 3"),
    ]
