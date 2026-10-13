#!/usr/bin/env python3
"""Rebuild the deterministic book_rag eval corpus snapshot from a REAL export.

Anti-fabrication contract (load-bearing — do not relax):

The snapshot's chunk TEXT must come from the real corpus (a live Qdrant export
or a recorded ingest run), supplied via ``--source-export``. It is FORBIDDEN to
synthesize snapshot content from ``qrels.tsv``/``queries.jsonl`` — a corpus
built from the answer key makes the retrieval gate a tautology that can never
fail (this was attempted on 2026-06-10 and rejected in review). This script
enforces the contract mechanically:

1. ``--source-export`` is required; there is no fixture-derived fallback.
2. Any snapshot chunk whose text contains a query's text verbatim is rejected.
3. ``run_freeze_book_rag_baseline`` refuses to freeze a baseline whose ranking
   metrics are all perfect — uniform 1.0 is a fabrication smell, not a result.

The export format is JSONL with at least ``chunk_id``, ``source_id``, ``text``
per line (the shape produced by a Qdrant scroll over the books collection).
Provenance (export path + export sha256 + snapshot sha256) is printed and
should be recorded in the commit that updates the snapshot.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from vo_rag.core.eval.harness import run_harness
from vo_rag.core.eval.qrels import load_qrels_tsv
from vo_rag.integrations.llamaindex.eval_dataset import (
    assert_book_rag_eval_snapshot_resolves_qrels,
    freeze_baseline,
)
from vo_rag.pipelines.book_rag.eval import (
    BOOK_RAG_EVAL_CORPUS_SNAPSHOT,
    book_rag_variant,
    build_book_rag_snapshot_retriever,
    build_live_search_fn,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BOOK_RAG_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "eval" / "book_rag"
_DEFAULT_OUTPUT = _BOOK_RAG_FIXTURE_DIR / "corpus_snapshot.jsonl"

#: Ranking metrics that, if ALL exactly 1.0, indicate a fabricated corpus.
_PERFECTION_SMELL_METRICS = ("mrr", "ndcg@10", "recall@10")


class FabricatedSnapshotError(RuntimeError):
    """Raised when the snapshot fails the anti-fabrication contract."""


def _load_queries_by_id(path: Path) -> dict[str, dict[str, Any]]:
    queries: dict[str, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        queries[str(record["query_id"])] = record
    return queries


def _query_texts(fixture_dir: Path) -> list[str]:
    texts: list[str] = []
    for record in _load_queries_by_id(fixture_dir / "queries.jsonl").values():
        text = str(record.get("query_text") or record.get("text") or "").strip()
        if text:
            texts.append(text)
    return texts


def _load_export_records(source_export: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line_no, raw in enumerate(source_export.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        chunk_id = str(record.get("chunk_id") or "").strip()
        source_id = str(record.get("source_id") or "").strip()
        text = str(record.get("text") or "").strip()
        if not (chunk_id and source_id and text):
            raise ValueError(
                f"{source_export}:{line_no}: export record needs non-empty "
                "chunk_id, source_id, and text"
            )
        records.append({"chunk_id": chunk_id, "source_id": source_id, "text": text})
    if not records:
        raise ValueError(f"{source_export}: export contains no records")
    return sorted(records, key=lambda r: r["chunk_id"])


def _assert_not_fabricated(records: list[dict[str, str]], *, fixture_dir: Path) -> None:
    """Reject a snapshot whose chunks embed the eval queries verbatim."""
    offenders: list[str] = []
    for query_text in _query_texts(fixture_dir):
        for record in records:
            if query_text in record["text"]:
                offenders.append(record["chunk_id"])
    if offenders:
        raise FabricatedSnapshotError(
            "snapshot chunks contain eval query text verbatim — this is the "
            "answer-key-as-corpus tautology; rebuild from a REAL corpus "
            f"export. offending chunk_ids: {sorted(set(offenders))[:5]}"
        )


def _snapshot_bytes(records: list[dict[str, str]]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def run_rebuild_book_rag_eval_corpus(
    *,
    source_export: Path | str,
    output: Path | str = _DEFAULT_OUTPUT,
    fixture_dir: Path | str = _BOOK_RAG_FIXTURE_DIR,
) -> dict[str, Any]:
    """Rebuild the snapshot from a real corpus export; return provenance."""
    source_path = Path(source_export)
    output_path = Path(output)
    fixture_path = Path(fixture_dir)

    records = _load_export_records(source_path)
    _assert_not_fabricated(records, fixture_dir=fixture_path)
    data = _snapshot_bytes(records)
    snapshot_hash = hashlib.sha256(data).hexdigest()

    _write_atomic(output_path, data)
    assert_book_rag_eval_snapshot_resolves_qrels(fixture_path / "qrels.tsv", output_path)

    return {
        "snapshot_path": str(output_path),
        "n_chunks": len(records),
        "snapshot_hash": snapshot_hash,
        "source_export": str(source_path),
        "source_export_hash": _sha256_file(source_path),
    }


def run_freeze_book_rag_baseline(
    *,
    fixture_dir: Path | str = _BOOK_RAG_FIXTURE_DIR,
    baseline_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run retrieval against the existing snapshot and freeze ``baseline.json``.

    Requires the snapshot to already exist (built by
    ``run_rebuild_book_rag_eval_corpus`` from a real export) — there is no
    auto-rebuild fallback, by design.
    """
    fixture_path = Path(fixture_dir)
    snapshot_path = fixture_path / BOOK_RAG_EVAL_CORPUS_SNAPSHOT
    if not snapshot_path.is_file():
        raise FileNotFoundError(
            f"{snapshot_path} does not exist — rebuild it from a real corpus "
            "export first (--source-export)"
        )

    assert_book_rag_eval_snapshot_resolves_qrels(fixture_path / "qrels.tsv", snapshot_path)
    snapshot_hash = _sha256_file(snapshot_path)

    queries_by_id = _load_queries_by_id(fixture_path / "queries.jsonl")
    qrels = load_qrels_tsv(fixture_path / "qrels.tsv", pipeline="book_rag", collection="eval")
    retriever = build_book_rag_snapshot_retriever(snapshot_path)
    report = asyncio.run(
        run_harness(
            "book_rag",
            qrels,
            book_rag_variant(),
            search_fn=build_live_search_fn(retriever),
            query_text_fn=lambda query_id: str(
                queries_by_id.get(query_id, {}).get("query_text")
                or queries_by_id.get(query_id, {}).get("text")
                or query_id
            ),
        )
    )

    aggregate = report.aggregate
    if all(aggregate.get(metric) == 1.0 for metric in _PERFECTION_SMELL_METRICS):
        raise FabricatedSnapshotError(
            "refusing to freeze a baseline with uniformly perfect ranking "
            f"metrics ({', '.join(_PERFECTION_SMELL_METRICS)} all 1.0) — "
            "perfect scores over a real corpus are a fabrication smell; "
            "investigate the snapshot's provenance before freezing"
        )

    out_path = Path(baseline_path) if baseline_path is not None else fixture_path / "baseline.json"
    freeze_baseline(report, path=out_path, corpus_snapshot=f"sha256:{snapshot_hash}")
    return {
        "baseline_path": str(out_path),
        "snapshot_path": str(snapshot_path),
        "snapshot_hash": snapshot_hash,
        "n_queries": len(report.per_query),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-export",
        type=Path,
        required=True,
        help="JSONL export of REAL corpus chunks (chunk_id, source_id, text), "
        "e.g. a Qdrant scroll over the books collection",
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--freeze-baseline",
        action="store_true",
        help="also regenerate the book_rag baseline from the rebuilt snapshot",
    )
    args = parser.parse_args()

    result = run_rebuild_book_rag_eval_corpus(source_export=args.source_export, output=args.output)
    if args.freeze_baseline:
        result.update(run_freeze_book_rag_baseline(fixture_dir=args.output.parent))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
