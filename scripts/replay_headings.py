#!/usr/bin/env python3
"""Dump a book's heading-candidate stream as a replay fixture (VOR-1427).

Read-only: fetches ``parsed/{ENVELOPE_ID}/parsed.json`` from the Garage
artifacts bucket (one GET), loads it as ``ParsedDocument`` and writes a JSON
fixture. It never writes to Garage or Postgres.

**Full stream** (``--full``): the element *order* of the whole document, no
body text:

* every element: ``{type, page, level}`` (``level`` is ``section_depth``);
* heading-like elements additionally carry ``text`` (normalized exactly as
  ``HierarchicalChunker`` normalizes it), ``reject`` (the
  ``is_degenerate_heading`` verdict), ``route`` (the ``ElementRoute`` that
  ``hierarchical.route_element`` -- the chunker's own routing predicate --
  assigns), ``update_title`` (whether that route calls
  ``HeadingTracker.update_title``) and ``label`` (the docling label, when the
  parser recorded one).

Heading-like means: a heading-ish ``ElementType`` (``TITLE``, ``HEADER``), a
``NARRATIVE_TEXT`` the chunker treats as a standalone heading, a short
single-line text matching chapter numbering (``^\\d+(\\.\\d+)*\\s+\\S``), or a
bare section number ("35") that a parser split off its heading.

**Compact stream** (the default): keeps a full heading entry only for

* ``title``-typed elements, elements routed to ``update_title`` and
  heading-typed elements with a non-null ``reject``;
* ``narrative_text`` candidates of at least 4 characters matching the chapter
  numbering pattern, unless the chunker routes them ``body`` (footnotes and
  sentence fragments keep only type and page);
* heading-like candidates routed ``skip_repeated`` (the running heads that make
  the chunker skip real headings) and ``narrative_text`` candidates that are
  markdown links (EPUB TOC entries and heading back-links);

and collapses every other element into run-length entries
``{type, page_start, page_end, count}`` without text, preserving order. On
disk each entry is a row (``["H", *heading_fields]`` or
``["R", *run_fields]``), one per line; ``expand_compact_entries`` turns rows
back into dicts.

Usage (on a host with the production settings loaded):
  python scripts/replay_headings.py ENVELOPE_ID OUT.json [--full]
Compact an existing full stream, or re-compact a compact one under the
current rules (local file, no Garage access):
  python scripts/replay_headings.py --compact-existing STREAM.json OUT.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from vo_rag.core.chunking.heading_filter import is_degenerate_heading
from vo_rag.core.chunking.hierarchical import ElementRoute, RoutingContext, route_element
from vo_rag.core.chunking.section_packer import SectionPacker
from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement
from vo_rag.core.parsing.text_normalization import normalize_extracted_text

#: ``ElementType`` members that denote a heading (contract.py has no
#: SECTION_HEADER; docling section headers arrive as TITLE).
HEADING_ELEMENT_TYPES: frozenset[str] = frozenset(
    {ElementType.TITLE.value, ElementType.HEADER.value}
)

COMPACT_FORMAT = "heading-stream-compact/1"
HEADING_FIELDS: tuple[str, ...] = (
    "type",
    "page",
    "level",
    "text",
    "reject",
    "route",
    "update_title",
    "label",
)
RUN_FIELDS: tuple[str, ...] = ("type", "page_start", "page_end", "count")

_CHAPTER_NUMBERING_RE = re.compile(r"^\d+(\.\d+)*\s+\S")
_BARE_SECTION_NUMBER_RE = re.compile(r"^\d+(\.\d+)*\.?$")
_MARKDOWN_LINK_RE = re.compile(r"\]\(")
#: Numbered texts longer than this (or multi-line) are body text, not headings,
#: and are kept out of the fixture.
_MAX_NUMBERED_HEADING_CHARS = 120
_MIN_COMPACT_TEXT_CHARS = 4

FetchFn = Callable[[str], Awaitable[ParsedDocument]]


def parsed_key(envelope_id: str) -> str:
    """Garage key the parse step writes (``pipelines/book_rag/steps/parse.py``)."""
    return f"parsed/{envelope_id}/parsed.json"


def _is_heading_like(element: ParsedElement, element_text: str) -> bool:
    if element.element_type.value in HEADING_ELEMENT_TYPES:
        return True
    if element.element_type is ElementType.NARRATIVE_TEXT and SectionPacker.is_standalone_heading(
        element_text
    ):
        return True
    stripped = element_text.strip()
    if not stripped or "\n" in stripped:
        return False
    if _BARE_SECTION_NUMBER_RE.match(stripped):
        return True
    return bool(
        _CHAPTER_NUMBERING_RE.match(stripped) and len(stripped) <= _MAX_NUMBERED_HEADING_CHARS
    )


def build_heading_stream(doc: ParsedDocument, *, envelope_id: str) -> dict[str, Any]:
    """Build the full heading-stream fixture for ``doc`` (pure; no I/O)."""
    routing = RoutingContext.for_document(doc)
    rows: list[dict[str, Any]] = []
    for element in doc.elements:
        row: dict[str, Any] = {
            "type": element.element_type.value,
            "page": element.page_number,
            "level": element.section_depth,
        }
        element_text = normalize_extracted_text(element.text)
        if _is_heading_like(element, element_text):
            route = route_element(element, element_text, routing)
            row["text"] = element_text.strip()
            row["reject"] = is_degenerate_heading(element_text)
            row["route"] = route.value
            row["update_title"] = route is ElementRoute.UPDATE_TITLE
            label = element.metadata.get("docling_label")
            if label:
                row["label"] = str(label)
        rows.append(row)
    return {
        "envelope_id": envelope_id,
        "title": doc.title,
        "parser_used": doc.parser_used,
        "total_pages": doc.total_pages,
        "quarantined_pages": sorted(routing.quarantined),
        "elements": rows,
    }


def _keep_in_compact(row: dict[str, Any]) -> bool:
    if "text" not in row:
        return False
    if row["type"] == ElementType.TITLE.value or row.get("update_title"):
        return True
    if row["type"] in HEADING_ELEMENT_TYPES and row.get("reject") is not None:
        return True
    if row.get("route") == ElementRoute.BODY.value:
        # Numbered footnotes / sentence fragments the chunker reads as body
        # text: the replay needs only their type and page.
        return False
    text = str(row["text"])
    if len(text) < _MIN_COMPACT_TEXT_CHARS:
        return False
    if row.get("route") == ElementRoute.SKIP_REPEATED.value:
        return True
    if row["type"] != ElementType.NARRATIVE_TEXT.value:
        return False
    return bool(_CHAPTER_NUMBERING_RE.match(text) or _MARKDOWN_LINK_RE.search(text))


def _rows_from_compact(compact: dict[str, Any]) -> list[dict[str, Any]]:
    """Full-stream rows equivalent to a compact stream, for re-compaction.

    A run of ``count`` elements expands to one row on ``page_start`` and the rest
    on ``page_end``, which compacts back to the same run.
    """
    rows: list[dict[str, Any]] = []
    for entry in expand_compact_entries(compact):
        if "count" not in entry:
            rows.append(dict(entry))
            continue
        rows.append({"type": entry["type"], "page": entry["page_start"]})
        rows.extend(
            {"type": entry["type"], "page": entry["page_end"]} for _ in range(entry["count"] - 1)
        )
    return rows


def compact_heading_stream(stream: dict[str, Any]) -> dict[str, Any]:
    """Compact a full stream (``build_heading_stream`` output), or re-compact a
    compact one under the current rules; pure and deterministic."""
    if stream.get("format") == COMPACT_FORMAT:
        header = {
            key: value
            for key, value in stream.items()
            if key not in {"format", "heading_fields", "run_fields", "entries"}
        }
        stream = {**header, "elements": _rows_from_compact(stream)}
    entries: list[list[Any]] = []
    for row in stream["elements"]:
        if _keep_in_compact(row):
            entries.append(["H", *(row.get(field) for field in HEADING_FIELDS)])
            continue
        last = entries[-1] if entries else None
        if last is not None and last[0] == "R" and last[1] == row["type"]:
            last[3] = row["page"]
            last[4] += 1
        else:
            entries.append(["R", row["type"], row["page"], row["page"], 1])
    header = {key: value for key, value in stream.items() if key != "elements"}
    return {
        "format": COMPACT_FORMAT,
        **header,
        "heading_fields": list(HEADING_FIELDS),
        "run_fields": list(RUN_FIELDS),
        "entries": entries,
    }


def expand_compact_entries(compact: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of a compact stream as dicts: heading entries and run entries."""
    if compact.get("format") != COMPACT_FORMAT:
        raise ValueError(f"not a {COMPACT_FORMAT} stream: {compact.get('format')!r}")
    heading_fields = list(compact["heading_fields"])
    run_fields = list(compact["run_fields"])
    out: list[dict[str, Any]] = []
    for row in compact["entries"]:
        kind, values = row[0], row[1:]
        fields = heading_fields if kind == "H" else run_fields
        entry = dict(zip(fields, values, strict=True))
        if kind == "H" and entry.get("label") is None:
            entry.pop("label", None)
        out.append(entry)
    return out


def dumps_compact(compact: dict[str, Any]) -> str:
    """Serialize a compact stream with one entry per line (small, diffable)."""
    header = {key: value for key, value in compact.items() if key != "entries"}
    head = json.dumps(header, ensure_ascii=False, separators=(",", ":"))
    rows = ",\n".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in compact["entries"]
    )
    return f'{head[:-1]},"entries":[\n{rows}\n]}}\n'


async def fetch_parsed_document(envelope_id: str) -> ParsedDocument:
    """GET ``parsed/{envelope_id}/parsed.json`` from the Garage artifacts bucket."""
    from vo_rag.core.config import get_settings
    from vo_rag.core.storage.garage import GarageClient

    settings = get_settings()
    garage = GarageClient(
        endpoint_url=settings.garage.endpoint,
        access_key=settings.garage.access_key,
        secret_key=settings.garage.secret_key,
        region=settings.garage.region,
    )
    raw = await garage.get_bytes(settings.garage.buckets.artifacts, parsed_key(envelope_id))
    return ParsedDocument.model_validate_json(raw)


def _write(stream: dict[str, Any], out_path: Path, *, full: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if full:
        text = json.dumps(stream, indent=1, ensure_ascii=False) + "\n"
    else:
        text = dumps_compact(compact_heading_stream(stream))
    out_path.write_text(text, encoding="utf-8")


def _summary(stream: dict[str, Any], out_path: Path) -> str:
    heading_rows = [row for row in stream["elements"] if "text" in row]
    return (
        f"{stream['envelope_id']}: {len(stream['elements'])} elements, "
        f"{len(heading_rows)} heading-like, "
        f"{sum(row['update_title'] for row in heading_rows)} routed to update_title "
        f"-> {out_path}"
    )


async def run(
    envelope_id: str,
    out_path: Path,
    *,
    full: bool = False,
    fetch: FetchFn = fetch_parsed_document,
) -> None:
    doc = await fetch(envelope_id)
    stream = build_heading_stream(doc, envelope_id=envelope_id)
    _write(stream, out_path, full=full)
    print(_summary(stream, out_path))


def compact_existing(source_path: Path, out_path: Path) -> None:
    """Compact a previously written full or compact stream (local file; no Garage)."""
    stream = json.loads(source_path.read_text(encoding="utf-8"))
    _write(stream, out_path, full=False)
    if stream.get("format") == COMPACT_FORMAT:
        print(f"{stream['envelope_id']}: re-compacted -> {out_path}")
    else:
        print(_summary(stream, out_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="ENVELOPE_ID, or FULL.json with --compact-existing")
    parser.add_argument("out", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="write the full stream")
    mode.add_argument(
        "--compact-existing",
        action="store_true",
        help="compact a local full-stream file instead of fetching from Garage",
    )
    args = parser.parse_args(argv)
    if args.compact_existing:
        compact_existing(Path(args.source), args.out)
    else:
        asyncio.run(run(args.source, args.out, full=args.full))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
