"""Stage the OCR backlog as <=400-page unit PDFs sharded across pods (VOR-1434).

Each book is read from Garage, its sha256 checked, an image-only EPUB rendered
to the page-image PDF the parser would OCR (``epub_to_pdf_bytes``), its page
count checked against the plan, and it is split with ``part_ranges`` into
``<sha256>__pNNN.pdf`` units. Units go to pods longest-processing-time first,
so every pod gets about the same number of pages.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

from vo_rag.ocr_farm.coordinator import SPLIT_PAGES, FarmItem, epub_to_pdf_bytes, part_ranges


class SourceStore(Protocol):
    async def get_bytes(self, bucket: str, key: str) -> bytes: ...


@dataclass
class StageResult:
    books: list[dict[str, Any]] = field(default_factory=list)
    units: list[dict[str, Any]] = field(default_factory=list)
    bad: list[dict[str, str]] = field(default_factory=list)
    shard_pages: list[int] = field(default_factory=list)


def load_backlog(path: Path) -> list[FarmItem]:
    """The books to OCR from ``vo-rag corpus ocr-farm plan`` JSONL output.

    Summary lines (no ``envelope_id``), cached books and books with a
    ``problem`` are dropped.
    """
    names = {f.name for f in fields(FarmItem)}
    items: list[FarmItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "envelope_id" not in row or row.get("cached") or row.get("problem"):
            continue
        items.append(FarmItem(**{k: v for k, v in row.items() if k in names}))
    return items


def unit_name(sha256: str, index: int) -> str:
    return f"{sha256}__p{index:03d}"


def _split(pdf: bytes, planned: int, sha256: str, units_dir: Path) -> list[dict[str, Any]] | str:
    import fitz  # type: ignore[import-untyped, unused-ignore]

    units: list[dict[str, Any]] = []
    with fitz.open(stream=pdf, filetype="pdf") as source:
        if int(source.page_count) != planned:
            return f"page_count {source.page_count} != planned {planned}"
        for index, (first, last) in enumerate(part_ranges(planned, SPLIT_PAGES)):
            name = unit_name(sha256, index)
            with fitz.open() as part:
                part.insert_pdf(source, from_page=first - 1, to_page=last - 1)
                part.save(str(units_dir / f"{name}.pdf"))
            units.append(
                {"unit": name, "sha256": sha256, "index": index, "first": first, "last": last,
                 "pages": last - first + 1}
            )  # fmt: skip
    return units


async def stage_books(
    items: Sequence[FarmItem],
    garage: SourceStore,
    *,
    bucket: str,
    units_dir: Path,
    pods: int,
) -> StageResult:
    """Write every book's unit PDFs into ``units_dir`` and assign shards."""
    units_dir.mkdir(parents=True, exist_ok=True)
    result = StageResult()
    for item in items:
        try:
            data = await garage.get_bytes(bucket, item.garage_key)
        except Exception as exc:  # noqa: BLE001 - one unreadable book must not stop staging
            result.bad.append({"envelope_id": item.envelope_id, "reason": f"get: {exc!s}"[:200]})
            continue
        if hashlib.sha256(data).hexdigest() != item.sha256:
            result.bad.append({"envelope_id": item.envelope_id, "reason": "sha256_mismatch"})
            continue
        try:
            pdf = data if item.fmt == "pdf" else await asyncio.to_thread(epub_to_pdf_bytes, data)
            units = await asyncio.to_thread(_split, pdf, item.pages, item.sha256, units_dir)
        except Exception as exc:  # noqa: BLE001
            reason = f"render: {type(exc).__name__}: {exc!s}"[:200]
            result.bad.append({"envelope_id": item.envelope_id, "reason": reason})
            continue
        if isinstance(units, str):
            result.bad.append({"envelope_id": item.envelope_id, "reason": units})
            continue
        result.units.extend(units)
        result.books.append({**item.to_dict(), "parts": len(units)})
    result.shard_pages = shard_units(result.units, pods)
    return result


def shard_units(units: list[dict[str, Any]], pods: int) -> list[int]:
    """Set each unit's ``shard`` (longest first onto the lightest pod); returns pages per pod."""
    if pods < 1:
        raise ValueError(f"pods must be >= 1, got {pods}")
    load = [0] * pods
    for unit in sorted(units, key=lambda u: -int(u["pages"])):
        k = load.index(min(load))
        unit["shard"] = k
        load[k] += int(unit["pages"])
    return load
