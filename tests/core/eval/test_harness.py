"""Retrieval eval harness."""

from __future__ import annotations

import pytest

from vo_rag.core.eval.harness import (
    RetrievalRegression,
    RetrievalVariant,
    assert_no_regression,
    run_harness,
)
from vo_rag.core.eval.qrels import Judgment, Qrels


def _make_qrels() -> Qrels:
    return Qrels(
        pipeline="book_rag",
        collection="vo_rag_books_sterile_v1",
        judgments=[
            Judgment(query_id="q1", doc_id="d1", grade=3),
            Judgment(query_id="q1", doc_id="d2", grade=1),
            Judgment(query_id="q2", doc_id="d3", grade=2),
            Judgment(query_id="q2", doc_id="d4", grade=1),
        ],
    )


@pytest.mark.asyncio
async def test_harness_produces_per_query_and_aggregate() -> None:
    qrels = _make_qrels()
    variant = RetrievalVariant(name="identity", top_k=5)

    async def _search(query: str, top_k: int) -> list[str]:
        if query == "q1":
            return ["d1", "d2", "x", "y"]
        return ["d3", "d4", "z"]

    report = await run_harness("book_rag", qrels, variant, search_fn=_search)

    assert len(report.per_query) == 2
    assert report.aggregate["ndcg@10"] == pytest.approx(1.0)
    assert report.aggregate["mrr"] == pytest.approx(1.0)
    assert "latency_p50_ms" in report.aggregate
    assert report.qrels_hash


@pytest.mark.asyncio
async def test_harness_reverse_ranking_lowers_ndcg() -> None:
    qrels = _make_qrels()
    variant = RetrievalVariant(name="reversed", top_k=5)

    async def _search(query: str, top_k: int) -> list[str]:
        if query == "q1":
            return ["d2", "d1", "x"]
        return ["d4", "d3"]

    report = await run_harness("book_rag", qrels, variant, search_fn=_search)
    assert report.aggregate["ndcg@10"] < 1.0


@pytest.mark.asyncio
async def test_assert_no_regression_passes_at_baseline() -> None:
    qrels = _make_qrels()
    variant = RetrievalVariant(name="identity")

    async def _search(query: str, top_k: int) -> list[str]:
        if query == "q1":
            return ["d1", "d2"]
        return ["d3", "d4"]

    report = await run_harness("book_rag", qrels, variant, search_fn=_search)
    baseline = {
        "ndcg@10": report.aggregate["ndcg@10"],
        "mrr": report.aggregate["mrr"],
    }
    assert_no_regression(report, baseline)


@pytest.mark.asyncio
async def test_assert_no_regression_catches_nontrivial_drop() -> None:
    qrels = _make_qrels()
    variant = RetrievalVariant(name="degraded")

    async def _search(query: str, top_k: int) -> list[str]:
        return ["z1", "lin", "z3"]

    report = await run_harness("book_rag", qrels, variant, search_fn=_search)
    baseline = {"ndcg@10": 1.0, "recall@10": 1.0, "mrr": 1.0}
    with pytest.raises(RetrievalRegression):
        assert_no_regression(report, baseline)
