"""TREC-format qrels model + TSV round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from vo_rag.core.eval.qrels import (
    Judgment,
    Qrels,
    load_qrels_tsv,
    qrels_hash,
    save_qrels_tsv,
)


def _sample() -> Qrels:
    return Qrels(
        pipeline="book_rag",
        collection="vo_rag_books_sterile_v1",
        judgments=[
            Judgment(query_id="q1", doc_id="d1", grade=2),
            Judgment(query_id="q1", doc_id="d2", grade=1),
            Judgment(query_id="q2", doc_id="d3", grade=3),
        ],
    )


def test_by_query_groups_docs() -> None:
    q = _sample()
    grouped = q.by_query()
    assert grouped["q1"] == {"d1": 2, "d2": 1}
    assert grouped["q2"] == {"d3": 3}


def test_positive_set_respects_min_grade() -> None:
    q = _sample()
    highs = q.positive_set_by_query(min_grade=2)
    assert highs == {"q1": {"d1"}, "q2": {"d3"}}


def test_tsv_round_trip(tmp_path: Path) -> None:
    q = _sample()
    path = tmp_path / "qrels.tsv"
    save_qrels_tsv(q, path)
    reloaded = load_qrels_tsv(path, pipeline="book_rag", collection="vo_rag_books_sterile_v1")
    assert reloaded.judgments == q.judgments


def test_hash_stable_across_reorderings() -> None:
    q1 = _sample()
    q2 = Qrels(
        pipeline="book_rag",
        collection="vo_rag_books_sterile_v1",
        judgments=list(reversed(q1.judgments)),
    )
    assert qrels_hash(q1) == qrels_hash(q2)


def test_invalid_grade_rejected() -> None:
    with pytest.raises(ValueError):
        Qrels(
            pipeline="x",
            collection="y",
            judgments=[Judgment(query_id="q1", doc_id="d1", grade=4)],
        )


def test_loader_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "qrels.tsv"
    path.write_text(
        "# head\nq1\t0\td1\t2\n\n# mid\nq1\t0\td2\t1\n",
        encoding="utf-8",
    )
    q = load_qrels_tsv(path, pipeline="x", collection="y")
    assert len(q.judgments) == 2


def test_loader_accepts_space_delimited_fallback(tmp_path: Path) -> None:
    path = tmp_path / "qrels.tsv"
    path.write_text("q1 0 d1 2\n", encoding="utf-8")
    q = load_qrels_tsv(path, pipeline="x", collection="y")
    assert q.judgments[0].query_id == "q1"


def test_all_query_ids_preserves_insertion_order() -> None:
    q = _sample()
    assert q.all_query_ids() == ["q1", "q2"]
