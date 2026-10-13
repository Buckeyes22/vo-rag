from __future__ import annotations

import pytest

import vo_rag.core.graph.quality_ab as quality_ab
from vo_rag.core.eval import non_inferiority
from vo_rag.core.graph.quality_ab import (
    GraphMechanicalResult,
    GraphPerQueryScore,
    GraphPhase6bVerdict,
    emit_graph_report,
    evaluate_graph,
)

# ---------------------------------------------------------------------------
# Helpers
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


def _failing_mechanical(
    *,
    n_total_nodes: int = 10,
    n_total_edges: int = 10,
    n_books_extraction_failed: int = 5,
    n_books_processed: int = 10,
) -> GraphMechanicalResult:
    return GraphMechanicalResult(
        n_books_processed=n_books_processed,
        n_books_extraction_failed=n_books_extraction_failed,
        n_total_nodes=n_total_nodes,
        n_total_edges=n_total_edges,
    )


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


def _make_winning_verdict() -> GraphPhase6bVerdict:
    mechanical = _passing_mechanical()
    scores = _make_per_query_scores(15)
    return evaluate_graph(
        mechanical=mechanical,
        per_query_scores=scores,
        non_inferiority_margin_pp=3.0,
        superiority_margin_pp=1.0,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# GraphMechanicalResult pass_gate logic
# ---------------------------------------------------------------------------


def test_graph_verdict_uses_the_shared_non_inferiority_math() -> None:
    assert (
        quality_ab.evaluate_graph.__globals__["compute_non_inferiority"]
        is non_inferiority.compute_non_inferiority
    )


class TestMechanicalPassGate:
    def test_mechanical_pass_gate_requires_nodes_edges_and_failure_rate(self) -> None:
        # Sufficient nodes + edges + no failures → pass
        mech = _passing_mechanical()
        assert mech.pass_gate is True

    def test_mechanical_fails_when_node_count_too_low(self) -> None:
        mech = GraphMechanicalResult(
            n_books_processed=10,
            n_books_extraction_failed=0,
            n_total_nodes=50,
            n_total_edges=400,
        )
        assert mech.pass_gate is False

    def test_mechanical_fails_when_edge_count_too_low(self) -> None:
        mech = GraphMechanicalResult(
            n_books_processed=10,
            n_books_extraction_failed=0,
            n_total_nodes=200,
            n_total_edges=100,
        )
        assert mech.pass_gate is False

    def test_mechanical_fails_when_failure_rate_exceeds_threshold(self) -> None:
        mech = GraphMechanicalResult(
            n_books_processed=10,
            n_books_extraction_failed=2,
            n_total_nodes=200,
            n_total_edges=400,
        )
        # 20 % > 10 % default max
        assert mech.pass_gate is False

    def test_mechanical_failure_rate_property(self) -> None:
        mech = _passing_mechanical(n_books_processed=10, n_books_extraction_failed=1)
        assert mech.failure_rate == pytest.approx(0.1)

    def test_mechanical_failure_rate_zero_books_does_not_divide_by_zero(self) -> None:
        mech = GraphMechanicalResult(
            n_books_processed=0,
            n_books_extraction_failed=0,
            n_total_nodes=200,
            n_total_edges=400,
        )
        assert mech.failure_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate_graph — gate outcomes
# ---------------------------------------------------------------------------


class TestEvaluateGraph:
    def test_evaluate_graph_mech_fail_yields_rerun_after_fixes(self) -> None:
        mech = _failing_mechanical()
        scores = _make_per_query_scores(15)
        verdict = evaluate_graph(
            mechanical=mech,
            per_query_scores=scores,
        )
        assert verdict.action == "rerun_after_fixes"
        assert verdict.keep_graph_path is False
        assert verdict.failure_reasons

    def test_evaluate_graph_insufficient_signal_yields_rerun_after_fixes(self) -> None:
        mech = _passing_mechanical()
        # Only 5 graph-using queries, need 10
        scores = _make_per_query_scores(5, uses_graph=True)
        verdict = evaluate_graph(
            mechanical=mech,
            per_query_scores=scores,
        )
        assert verdict.action == "rerun_after_fixes"
        assert verdict.n_queries_using_graph == 5

    def test_evaluate_graph_wins_yields_keep_graph(self) -> None:
        mech = _passing_mechanical()
        # Treatment (with_graph) clearly better on recall and ndcg
        scores = _make_per_query_scores(
            15,
            with_recall=0.90,
            without_recall=0.70,
            with_ndcg=0.90,
            without_ndcg=0.70,
        )
        verdict = evaluate_graph(
            mechanical=mech,
            per_query_scores=scores,
            non_inferiority_margin_pp=3.0,
            superiority_margin_pp=1.0,
        )
        assert verdict.keep_graph_path is True
        assert verdict.action == "keep_graph"

    def test_evaluate_graph_loses_recall_yields_disable_graph_path(self) -> None:
        mech = _passing_mechanical()
        # Treatment has much lower recall (inferior)
        scores = _make_per_query_scores(
            15,
            with_recall=0.40,
            without_recall=0.80,
            with_ndcg=0.82,
            without_ndcg=0.72,
        )
        verdict = evaluate_graph(
            mechanical=mech,
            per_query_scores=scores,
            non_inferiority_margin_pp=3.0,
            superiority_margin_pp=1.0,
        )
        assert verdict.action == "disable_graph_path"
        assert verdict.keep_graph_path is False

    def test_evaluate_graph_ties_yields_disable_graph_path(self) -> None:
        """Non-inferior recall but neither metric superior → disable."""
        mech = _passing_mechanical()
        # Identical scores: recall is non-inferior but ndcg has zero delta → not superior
        scores = _make_per_query_scores(
            15,
            with_recall=0.80,
            without_recall=0.80,
            with_ndcg=0.80,
            without_ndcg=0.80,
        )
        verdict = evaluate_graph(
            mechanical=mech,
            per_query_scores=scores,
            non_inferiority_margin_pp=3.0,
            superiority_margin_pp=1.0,
        )
        assert verdict.action == "disable_graph_path"
        assert verdict.keep_graph_path is False

    def test_evaluate_graph_filters_to_graph_using_subset(self) -> None:
        """Only queries with query_uses_graph_path=True enter the comparison."""
        mech = _passing_mechanical()
        graph_scores = _make_per_query_scores(
            10,
            with_recall=0.90,
            without_recall=0.70,
            with_ndcg=0.90,
            without_ndcg=0.70,
            uses_graph=True,
        )
        non_graph_scores = _make_per_query_scores(
            20,
            with_recall=0.30,
            without_recall=0.90,
            with_ndcg=0.30,
            without_ndcg=0.90,
            uses_graph=False,
        )
        verdict = evaluate_graph(
            mechanical=mech,
            per_query_scores=graph_scores + non_graph_scores,
        )
        # Only the 10 graph-using queries should be counted
        assert verdict.n_queries_using_graph == 10
        # The winning scores on the graph-using subset should produce keep_graph
        assert verdict.keep_graph_path is True


# ---------------------------------------------------------------------------
# emit_graph_report
# ---------------------------------------------------------------------------


class TestEmitGraphReport:
    def test_emit_graph_report_returns_non_empty_multiline(self) -> None:
        verdict = _make_winning_verdict()
        report = emit_graph_report(verdict)
        assert isinstance(report, str)
        lines = report.splitlines()
        assert len(lines) > 5
        assert any("GRAPH LAYER QUALITY A/B EVALUATION REPORT" in line for line in lines)

    def test_emit_graph_report_contains_verdict_label(self) -> None:
        verdict = _make_winning_verdict()
        report = emit_graph_report(verdict)
        assert "KEEP" in report or "REJECT" in report

    def test_emit_graph_report_on_failing_verdict_shows_reject(self) -> None:
        mech = _failing_mechanical()
        scores = _make_per_query_scores(15)
        verdict = evaluate_graph(mechanical=mech, per_query_scores=scores)
        report = emit_graph_report(verdict)
        assert "REJECT" in report
