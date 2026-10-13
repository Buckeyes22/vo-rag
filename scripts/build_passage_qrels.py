"""Build passage-level book_rag qrels from chunk qrels and a gold map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vo_rag.core.eval.passages import passage_key
from vo_rag.core.eval.qrels import load_qrels_tsv

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BOOK_RAG_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "eval" / "book_rag"


def build_passage_qrels(
    qrels_path: Path,
    gold_map_path: Path,
    output_path: Path,
) -> None:
    qrels = load_qrels_tsv(qrels_path, pipeline="book_rag", collection="eval")
    by_chunk_id = _load_passage_map(gold_map_path)

    lines: list[str] = []
    for judgment in qrels.judgments:
        try:
            passage_doc_id = by_chunk_id[judgment.doc_id]
        except KeyError as exc:
            msg = f"gold map missing chunk_id {judgment.doc_id!r}"
            raise ValueError(msg) from exc
        lines.append(f"{judgment.query_id}\t0\t{passage_doc_id}\t{judgment.grade}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_passage_map(gold_map_path: Path) -> dict[str, str]:
    records = json.loads(gold_map_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        msg = "gold map must be a JSON list"
        raise ValueError(msg)

    out: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            msg = "gold map records must be JSON objects"
            raise ValueError(msg)
        chunk_id = _required_str(record, "chunk_id")
        source_id = _required_str(record, "source_id")
        page_number = _required_int(record, "page_number")
        out[chunk_id] = passage_key(source_id, page_number)
    return out


def _required_str(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if value is None:
        msg = f"gold map record missing {key!r}"
        raise ValueError(msg)
    text = str(value).strip()
    if not text:
        msg = f"gold map record has empty {key!r}"
        raise ValueError(msg)
    return text


def _required_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool):
        msg = f"gold map record has invalid {key!r}: {value!r}"
        raise ValueError(msg)
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        msg = f"gold map record has invalid {key!r}: {value!r}"
        raise ValueError(msg) from exc
    if parsed < 1:
        msg = f"gold map record has invalid {key!r}: {value!r}"
        raise ValueError(msg)
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qrels",
        type=Path,
        default=_BOOK_RAG_FIXTURE_DIR / "qrels.tsv",
        help="Chunk-level qrels TSV to project.",
    )
    parser.add_argument(
        "--gold-map",
        type=Path,
        default=Path("/tmp/gold_passage_map.json"),
        help="JSON list mapping gold chunk_id to source_id and page_number.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_BOOK_RAG_FIXTURE_DIR / "qrels_passage.tsv",
        help="Output passage-level qrels TSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    # operator step: freeze the live control-arm passage baseline on localhost
    # after the target collection is populated; this local generator only binds qrels.
    build_passage_qrels(args.qrels, args.gold_map, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
