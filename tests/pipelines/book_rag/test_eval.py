"""book_rag eval mode selection."""

from __future__ import annotations

import json
from pathlib import Path

from vo_rag.core.eval.gate import BaselineV1
from vo_rag.pipelines.book_rag import eval as book_eval

_ROOT = Path(__file__).resolve().parents[3]
_BOOK_DIR = _ROOT / "tests" / "fixtures" / "eval" / "book_rag"


def test_book_rag_evaluator_defaults_to_replay_smoke() -> None:
    entry = book_eval.book_rag_evaluator(_BOOK_DIR)

    assert entry.mode == "replay_smoke"
    assert entry.variant.name == "book_rag"

    first = json.loads((_BOOK_DIR / "queries.jsonl").read_text(encoding="utf-8").splitlines()[0])
    result = entry.search_fn(first["query_text"], 10)

    assert isinstance(result, list)
    assert result


def test_book_rag_baseline_is_schema_valid() -> None:
    baseline = json.loads((_BOOK_DIR / "baseline.json").read_text(encoding="utf-8"))

    parsed = BaselineV1.model_validate(baseline)
    assert parsed.corpus_snapshot  # provenance binding always present


def test_checked_in_snapshot_is_not_fabricated_from_queries() -> None:
    """Anti-tautology tripwire (2026-06-10 incident).

    A corpus snapshot synthesized from the eval queries makes the retrieval
    gate unfailable (chunk text == query text -> trivial perfect retrieval).
    If a snapshot is ever checked in, its chunks must not embed query text
    verbatim and the baseline must carry real sha256 provenance.
    """
    snapshot_path = _BOOK_DIR / "corpus_snapshot.jsonl"
    if not snapshot_path.is_file():
        return  # gate is blocked; nothing checked in to tripwire

    query_texts = [
        str(json.loads(line)["query_text"])
        for line in (_BOOK_DIR / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunk_texts = [
        str(json.loads(line)["text"])
        for line in snapshot_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for query_text in query_texts:
        assert not any(query_text in chunk for chunk in chunk_texts), (
            "snapshot chunk embeds eval query text verbatim — the "
            "answer-key-as-corpus tautology; rebuild from a real export"
        )

    baseline = json.loads((_BOOK_DIR / "baseline.json").read_text(encoding="utf-8"))
    assert str(baseline["corpus_snapshot"]).startswith("sha256:")


def test_retrieval_gate_blocks_without_corpus_snapshot_and_never_uses_replay(
    monkeypatch,
    tmp_path: Path,
) -> None:
    called = False

    def forbidden_replay(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("retrieval_gate must not use replay search")

    monkeypatch.setattr(book_eval, "build_replay_search_fn", forbidden_replay)

    book_dir = tmp_path / "book_rag"
    book_dir.mkdir()
    for name in ("queries.jsonl", "qrels.tsv", "baseline.json"):
        (book_dir / name).write_text((_BOOK_DIR / name).read_text(encoding="utf-8"))

    entry = book_eval.book_rag_evaluator(book_dir, mode="retrieval_gate")

    assert entry.mode == "blocked_missing_corpus_snapshot"
    assert entry.search_fn("any query", 10) == []
    assert called is False


def test_retrieval_gate_blocks_missing_retriever_when_snapshot_exists(tmp_path: Path) -> None:
    book_dir = tmp_path / "book_rag"
    book_dir.mkdir()
    for name in ("queries.jsonl", "qrels.tsv", "baseline.json"):
        (book_dir / name).write_text((_BOOK_DIR / name).read_text(encoding="utf-8"))
    (book_dir / "corpus_snapshot.jsonl").write_text("{}\n", encoding="utf-8")

    entry = book_eval.book_rag_evaluator(book_dir, mode="retrieval_gate")

    assert entry.mode == "blocked_missing_retriever"
