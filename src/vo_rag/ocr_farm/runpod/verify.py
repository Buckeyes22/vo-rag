"""Check every book's OCR cache is in Garage (VOR-1434).

A book verifies when ``ocr/<sha256>/docling.json`` exists and ``meta.json``
says ``pipeline: ocr`` with the planned page count.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from vo_rag.ocr_farm.coordinator import ocr_key, ocr_meta_key
from vo_rag.ocr_farm.runpod.merge_upload import CacheStore


async def verify_book(book: dict[str, Any], garage: CacheStore, *, bucket: str) -> str:
    """``""`` when the cache is good, else why not."""
    sha = str(book["sha256"])
    if await garage.object_metadata(bucket, ocr_key(sha)) is None:
        return "missing docling.json"
    if await garage.object_metadata(bucket, ocr_meta_key(sha)) is None:
        return "missing meta.json"
    try:
        meta = json.loads(await garage.get_bytes(bucket, ocr_meta_key(sha)))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "meta.json is not JSON"
    if meta.get("pipeline") != "ocr":
        return f"meta pipeline {meta.get('pipeline')!r} != 'ocr'"
    if meta.get("pages") != book["pages"]:
        return f"meta pages {meta.get('pages')} != {book['pages']}"
    return ""


async def verify_books(
    books: Sequence[dict[str, Any]], garage: CacheStore, *, bucket: str
) -> list[dict[str, Any]]:
    results = []
    for book in books:
        reason = await verify_book(book, garage, bucket=bucket)
        results.append(
            {"envelope_id": book["envelope_id"], "sha256": book["sha256"], "ok": not reason,
             "reason": reason}
        )  # fmt: skip
    return results
