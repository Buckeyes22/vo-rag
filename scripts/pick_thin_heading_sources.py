"""Rank sources with thin heading paths from an offline chunk JSONL export.

Expected input is a local JSONL file produced by an operator-controlled export
step. This script does not query Qdrant or any live service. Each non-empty line
must be a JSON object containing at least:

    {"source_id": "vo-rag-corpus-sha256:...", "section_path": "Book > Chapter"}

For each source, the script computes:
- mean_depth: mean count of ">" separators in section_path
- shallow_pct: fraction of chunks whose depth is <= 2

Rows are printed as TSV, ranked heading-thinnest first.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceHeadingStats:
    source_id: str
    chunk_count: int
    mean_depth: float
    shallow_count: int
    shallow_pct: float


@dataclass
class _HeadingAccumulator:
    chunk_count: int = 0
    depth_sum: int = 0
    shallow_count: int = 0

    def add(self, depth: int) -> None:
        self.chunk_count += 1
        self.depth_sum += depth
        if depth <= 2:
            self.shallow_count += 1

    def stats(self, source_id: str) -> SourceHeadingStats:
        return SourceHeadingStats(
            source_id=source_id,
            chunk_count=self.chunk_count,
            mean_depth=self.depth_sum / self.chunk_count,
            shallow_count=self.shallow_count,
            shallow_pct=self.shallow_count / self.chunk_count,
        )


def section_depth(section_path: str) -> int:
    return section_path.count(">")


def rank_sources(export_path: Path) -> list[SourceHeadingStats]:
    grouped: dict[str, _HeadingAccumulator] = {}
    for source_id, depth in _iter_depths(export_path):
        grouped.setdefault(source_id, _HeadingAccumulator()).add(depth)

    rows = [acc.stats(source_id) for source_id, acc in grouped.items()]
    return sorted(rows, key=lambda row: (row.mean_depth, -row.shallow_pct, row.source_id))


def _iter_depths(export_path: Path) -> Iterator[tuple[str, int]]:
    with export_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record: Any = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: expected JSON object")

            source_id = record.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"line {line_number}: source_id must be a non-empty string")

            section_path = record.get("section_path", "")
            if section_path is None:
                section_path = ""
            if not isinstance(section_path, str):
                raise ValueError(f"line {line_number}: section_path must be a string or null")

            yield source_id, section_depth(section_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_path", type=Path, help="Offline chunk JSONL export path.")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum source rows to print. Use 0 for all rows.",
    )
    return parser


def _print_rows(rows: Sequence[SourceHeadingStats], *, limit: int) -> None:
    selected = rows if limit <= 0 else rows[:limit]
    print("source_id\tchunk_count\tmean_depth\tshallow_pct\tshallow_count")
    for row in selected:
        print(
            f"{row.source_id}\t{row.chunk_count}\t{row.mean_depth:.3f}\t"
            f"{row.shallow_pct * 100:.1f}\t{row.shallow_count}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _print_rows(rank_sources(args.export_path), limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
