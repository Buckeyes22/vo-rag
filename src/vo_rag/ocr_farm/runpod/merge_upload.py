"""Merge a book's finished units and upload its OCR cache (VOR-1434).

A book is uploaded only when every one of its units is ``ok`` (a non-empty
JSON plus a worker note reading ``ocr_worker: ok``) and the merged document
has exactly pages ``1..pages``. Upload order is the cache contract:
``ocr/<sha256>/docling.json`` first, then ``meta.json`` -- watchers flip an
envelope when its meta.json appears, so meta.json must never precede the
document. A book whose meta.json already exists is left alone (``cached``)
unless ``overwrite`` is set, so a re-do must pass ``overwrite`` explicitly.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from vo_rag.ocr_farm.coordinator import merge_docling_parts, ocr_key, ocr_meta_key

OK_MARKER = "ocr_worker: ok"
UnitState = Literal["ok", "failed", "pending"]
Merger = Callable[[Sequence[Path], Path, str], list[int]]


class CacheStore(Protocol):
    async def upload_file(self, bucket: str, key: str, path: Path) -> None: ...

    async def object_metadata(self, bucket: str, key: str) -> Any: ...

    async def get_bytes(self, bucket: str, key: str) -> bytes: ...


def unit_state(out_dir: Path, unit: str) -> tuple[UnitState, str]:
    """``(state, note)`` of one unit's pulled result in ``out_dir``."""
    note_path = out_dir / f"{unit}.note"
    if not note_path.exists():
        return "pending", ""
    note = note_path.read_text(encoding="utf-8", errors="replace").strip()
    if OK_MARKER not in note:
        return "failed", note or "empty note (worker killed?)"
    doc = out_dir / f"{unit}.json"
    if not doc.exists() or doc.stat().st_size == 0:
        return "failed", "ok note but no json"
    return "ok", note


async def upload_cache(
    garage: CacheStore, bucket: str, sha256: str, docling: Path, meta: Path
) -> None:
    """docling.json first, then meta.json (the object watchers act on)."""
    await garage.upload_file(bucket, ocr_key(sha256), docling)
    await garage.upload_file(bucket, ocr_meta_key(sha256), meta)


async def merge_and_upload(
    books: Sequence[dict[str, Any]],
    units: Sequence[dict[str, Any]],
    out_dir: Path,
    merged_dir: Path,
    garage: CacheStore,
    *,
    bucket: str,
    job_id: str,
    overwrite: bool = False,
    merge: Merger = merge_docling_parts,
) -> list[dict[str, Any]]:
    """One outcome per book: ``uploaded`` | ``cached`` | ``incomplete`` | ``failed(...)``.

    ``incomplete`` names each failed unit (with its worker note) and each
    unit with no result yet.
    """
    by_book: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        by_book.setdefault(str(unit["sha256"]), []).append(unit)
    outcomes: list[dict[str, Any]] = []
    for book in books:
        sha = str(book["sha256"])
        parts = sorted(by_book.get(sha, []), key=lambda u: int(u["index"]))
        outcome: dict[str, Any] = {
            "envelope_id": book["envelope_id"],
            "sha256": sha,
            "title": book.get("title"),
            "units_failed": {},
            "units_pending": [],
        }
        notes: list[str] = []
        for part in parts:
            state, note = unit_state(out_dir, str(part["unit"]))
            if state == "failed":
                outcome["units_failed"][part["unit"]] = note
            elif state == "pending":
                outcome["units_pending"].append(part["unit"])
            notes.append(note)
        if not parts or outcome["units_failed"] or outcome["units_pending"]:
            outcome["status"] = "incomplete" if parts else "failed(no_units)"
            outcomes.append(outcome)
            continue
        outcome["status"] = await _store(
            book, parts, notes, out_dir, merged_dir / sha, garage,
            bucket=bucket, job_id=job_id, overwrite=overwrite, merge=merge,
        )  # fmt: skip
        outcomes.append(outcome)
    return outcomes


async def _store(
    book: dict[str, Any],
    parts: list[dict[str, Any]],
    notes: list[str],
    out_dir: Path,
    dest: Path,
    garage: CacheStore,
    *,
    bucket: str,
    job_id: str,
    overwrite: bool,
    merge: Merger,
) -> str:
    sha, pages = str(book["sha256"]), int(book["pages"])
    if not overwrite and await garage.object_metadata(bucket, ocr_meta_key(sha)) is not None:
        return "cached"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        docling = dest / "docling.json"
        paths = [out_dir / f"{part['unit']}.json" for part in parts]
        merged = await asyncio.to_thread(merge, paths, docling, sha)
        if merged != list(range(1, pages + 1)):
            return f"failed(merged_pages {len(merged)} != {pages})"
        meta = {
            "pipeline": "ocr",
            "envelope_id": book["envelope_id"],
            "title": book.get("title"),
            "host": "runpod",
            "job": job_id,
            "pages": pages,
            "fmt": book.get("fmt", "pdf"),
            "parts": len(parts),
            "seconds": None,
            "worker": notes,
        }
        (dest / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        await upload_cache(garage, bucket, sha, docling, dest / "meta.json")
        return "uploaded"
    except Exception as exc:  # noqa: BLE001 - one book's failure must not stop the rest
        return f"failed(error_{type(exc).__name__})"
    finally:
        shutil.rmtree(dest, ignore_errors=True)
