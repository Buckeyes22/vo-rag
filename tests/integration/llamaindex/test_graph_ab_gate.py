"""Graph quality A/B exit gate tests for the LlamaIndex-wrapped read path.

Validates that the exit gate harness
(:mod:`vo_rag.integrations.llamaindex.graph_ab`):

1. Computes per-query recall@10 / nDCG@10 from paired retrieval results and
   shapes them as :class:`GraphPerQueryScore` records.
2. Feeds those records through the existing :func:`evaluate_graph` verdict.
3. Enforces the non-negotiable rule: ``disable_graph_path`` is a hard FAIL.

These tests use synthetic fixtures — they do not require a live graph store.
"""

from __future__ import annotations

import pytest

from vo_rag.core.eval.qrels import Judgment, Qrels
from vo_rag.core.graph.quality_ab import (
    GraphMechanicalResult,
    GraphPerQueryScore,
)
from vo_rag.integrations.llamaindex.graph_ab import (
    build_graph_ab_scores,
    run_graph_ab_exit_gate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _passing_mechanical(
    *,
    n_books_processed: int = 10,
    n_books_extraction_failed: int = 0,
    n_total_nodes: int = 200,
    n_total_edges: int = 400,
) -> GraphMechanicalResult:
    return GraphMechanicalResult(
        n_books_processed=n_books_processed,
        n_books_extraction_failed=n_books_extraction_failed,
        n_total_nodes=n_total_nodes,
        n_total_edges=n_total_edges,
    )


@pytest.fixture
def passing_mechanical() -> GraphMechanicalResult:
    return _passing_mechanical()


@pytest.fixture
def synthetic_qrels() -> Qrels:
    """Minimal qrels with 3 queries, 2 relevant docs each."""
    judgments = [
        Judgment(query_id="q0", doc_id="d0", grade=1),
        Judgment(query_id="q0", doc_id="d1", grade=1),
        Judgment(query_id="q1", doc_id="d0", grade=1),
        Judgment(query_id="q1", doc_id="d1", grade=1),
        Judgment(query_id="q2", doc_id="d0", grade=1),
        Judgment(query_id="q2", doc_id="d1", grade=1),
    ]
    return Qrels(pipeline="test", collection="test", judgments=judgments)


# ---------------------------------------------------------------------------
# build_graph_ab_scores — metric computation from raw retrieved IDs
# ---------------------------------------------------------------------------


class TestBuildGraphAbScores:
    def test_perfect_retrieval_yields_perfect_recall_and_ndcg(
        self,
        synthetic_qrels: Qrels,
    ) -> None:
        queries = ["q0", "q1", "q2"]
        native_retrieved = {
            "q0": ["d0", "d1"],
            "q1": ["d0", "d1"],
            "q2": ["d0", "d1"],
        }
        wrapped_retrieved = {
            "q0": ["d0", "d1"],
            "q1": ["d0", "d1"],
            "q2": ["d0", "d1"],
        }
        scores = build_graph_ab_scores(
            queries=queries,
            qrels=synthetic_qrels,
            native_retrieved=native_retrieved,
            wrapped_retrieved=wrapped_retrieved,
        )
        assert len(scores) == 3
        for s in scores:
            assert s.planner_with_graph_recall_at_10 == pytest.approx(1.0)
            assert s.planner_without_graph_recall_at_10 == pytest.approx(1.0)
            assert s.planner_with_graph_ndcg_at_10 == pytest.approx(1.0)
            assert s.planner_without_graph_ndcg_at_10 == pytest.approx(1.0)

    def test_empty_wrapped_yields_zero_recall(
        self,
        synthetic_qrels: Qrels,
    ) -> None:
        queries = ["q0"]
        native_retrieved = {"q0": ["d0", "d1"]}
        wrapped_retrieved: dict[str, list[str]] = {"q0": []}
        scores = build_graph_ab_scores(
            queries=queries,
            qrels=synthetic_qrels,
            native_retrieved=native_retrieved,
            wrapped_retrieved=wrapped_retrieved,
        )
        assert len(scores) == 1
        s = scores[0]
        assert s.planner_with_graph_recall_at_10 == pytest.approx(0.0)
        assert s.planner_without_graph_recall_at_10 == pytest.approx(1.0)

    def test_query_uses_graph_path_override(self) -> None:
        qrels = Qrels(
            pipeline="test",
            collection="test",
            judgments=[Judgment(query_id="q0", doc_id="d0", grade=1)],
        )
        scores = build_graph_ab_scores(
            queries=["q0"],
            qrels=qrels,
            native_retrieved={"q0": ["d0"]},
            wrapped_retrieved={"q0": ["d0"]},
            query_uses_graph_path={"q0": False},
        )
        assert scores[0].query_uses_graph_path is False

    def test_defaults_all_queries_to_graph_path(self) -> None:
        qrels = Qrels(
            pipeline="test",
            collection="test",
            judgments=[Judgment(query_id="q0", doc_id="d0", grade=1)],
        )
        scores = build_graph_ab_scores(
            queries=["q0"],
            qrels=qrels,
            native_retrieved={"q0": ["d0"]},
            wrapped_retrieved={"q0": ["d0"]},
        )
        assert scores[0].query_uses_graph_path is True


# ---------------------------------------------------------------------------
# run_graph_ab_exit_gate — pass / fail verdicts
# ---------------------------------------------------------------------------


def _make_per_query_scores(
    n: int,
    *,
    with_recall: float = 0.80,
    without_recall: float = 0.75,
    with_ndcg: float = 0.82,
    without_ndcg: float = 0.72,
    uses_graph: bool = True,
) -> list[GraphPerQueryScore]:
    return [
        GraphPerQueryScore(
            query_id=f"q{i}",
            planner_with_graph_recall_at_10=with_recall,
            planner_without_graph_recall_at_10=without_recall,
            planner_with_graph_ndcg_at_10=with_ndcg,
            planner_without_graph_ndcg_at_10=without_ndcg,
            query_uses_graph_path=uses_graph,
        )
        for i in range(n)
    ]


class TestGraphAbGatePass:
    def test_keep_verdict_when_non_inferior_recall_and_superior_ndcg(
        self,
        passing_mechanical: GraphMechanicalResult,
    ) -> None:
        scores = _make_per_query_scores(
            15,
            with_recall=0.90,
            without_recall=0.70,
            with_ndcg=0.90,
            without_ndcg=0.70,
        )
        verdict = run_graph_ab_exit_gate(
            mechanical=passing_mechanical,
            per_query_scores=scores,
        )
        assert verdict.action == "keep_graph"
        assert verdict.keep_graph_path is True
        assert not verdict.failure_reasons

    def test_keep_verdict_with_superior_judge_when_ndcg_ties(
        self,
        passing_mechanical: GraphMechanicalResult,
    ) -> None:
        scores = [
            GraphPerQueryScore(
                query_id=f"q{i}",
                planner_with_graph_recall_at_10=0.80,
                planner_without_graph_recall_at_10=0.75,
                planner_with_graph_ndcg_at_10=0.80,
                planner_without_graph_ndcg_at_10=0.80,
                planner_with_graph_judge_score=0.95,
                planner_without_graph_judge_score=0.70,
                query_uses_graph_path=True,
            )
            for i in range(15)
        ]
        verdict = run_graph_ab_exit_gate(
            mechanical=passing_mechanical,
            per_query_scores=scores,
        )
        assert verdict.action == "keep_graph"
        assert verdict.keep_graph_path is True


class TestGraphAbGateFail:
    def test_disable_graph_path_when_recall_inferior(
        self,
        passing_mechanical: GraphMechanicalResult,
    ) -> None:
        scores = _make_per_query_scores(
            15,
            with_recall=0.40,
            without_recall=0.80,
            with_ndcg=0.90,
            without_ndcg=0.70,
        )
        verdict = run_graph_ab_exit_gate(
            mechanical=passing_mechanical,
            per_query_scores=scores,
        )
        assert verdict.action == "disable_graph_path"
        assert verdict.keep_graph_path is False
        assert any("Recall@10 is inferior" in r for r in verdict.failure_reasons)

    def test_disable_graph_path_when_no_superiority(
        self,
        passing_mechanical: GraphMechanicalResult,
    ) -> None:
        scores = _make_per_query_scores(
            15,
            with_recall=0.80,
            without_recall=0.80,
            with_ndcg=0.80,
            without_ndcg=0.80,
        )
        verdict = run_graph_ab_exit_gate(
            mechanical=passing_mechanical,
            per_query_scores=scores,
        )
        assert verdict.action == "disable_graph_path"
        assert verdict.keep_graph_path is False
        assert any("No metric is superior" in r for r in verdict.failure_reasons)

    def test_disable_graph_path_when_recall_non_inferior_but_both_metrics_tie(
        self,
        passing_mechanical: GraphMechanicalResult,
    ) -> None:
        scores = [
            GraphPerQueryScore(
                query_id=f"q{i}",
                planner_with_graph_recall_at_10=0.80,
                planner_without_graph_recall_at_10=0.80,
                planner_with_graph_ndcg_at_10=0.80,
                planner_without_graph_ndcg_at_10=0.80,
                planner_with_graph_judge_score=0.80,
                planner_without_graph_judge_score=0.80,
                query_uses_graph_path=True,
            )
            for i in range(15)
        ]
        verdict = run_graph_ab_exit_gate(
            mechanical=passing_mechanical,
            per_query_scores=scores,
        )
        assert verdict.action == "disable_graph_path"
        assert verdict.keep_graph_path is False


class TestGraphAbGateMargins:
    def test_uses_migration_default_margins(self) -> None:
        mechanical = _passing_mechanical()
        scores = _make_per_query_scores(15)
        verdict = run_graph_ab_exit_gate(
            mechanical=mechanical,
            per_query_scores=scores,
        )
        rr = verdict.recall_result
        assert rr["non_inferiority_margin_pp"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_symbols_in_all(self) -> None:
        from vo_rag.integrations.llamaindex import graph_ab as gab

        assert "build_graph_ab_scores" in gab.__all__
        assert "run_graph_ab_exit_gate" in gab.__all__
