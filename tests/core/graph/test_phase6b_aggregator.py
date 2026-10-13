from __future__ import annotations

from vo_rag.core.graph.phase6b_aggregator import (
    GraphOverheadResult,
    GraphSmokeResult,
    Phase6bFinalVerdict,
    decide_phase6b,
    emit_phase6b_report,
)
from vo_rag.core.graph.quality_ab import (
    GraphMechanicalResult,
    GraphPerQueryScore,
    GraphPhase6bVerdict,
    evaluate_graph,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passing_smoke(
    *,
    n_nodes: int = 150,
    n_edges: int = 300,
    min_one_metric_improved: bool = True,
) -> GraphSmokeResult:
    return GraphSmokeResult(
        passed=True,
        n_nodes_built=n_nodes,
        n_edges_built=n_edges,
        min_one_metric_improved=min_one_metric_improved,
        improvement_breakdown={"recall_at_10": 0.05, "ndcg_at_10": 0.03},
    )


def _failing_smoke() -> GraphSmokeResult:
    return GraphSmokeResult(
        passed=False,
        n_nodes_built=5,
        n_edges_built=3,
        min_one_metric_improved=False,
        improvement_breakdown={},
    )


def _passing_overhead(
    *,
    avg_planner_latency_ms: float = 120.0,
    baseline_planner_latency_ms: float = 100.0,
    latency_overhead_pp: float = 10.0,
) -> GraphOverheadResult:
    return GraphOverheadResult(
        avg_planner_latency_ms=avg_planner_latency_ms,
        baseline_planner_latency_ms=baseline_planner_latency_ms,
        latency_overhead_pp=latency_overhead_pp,
        index_size_bytes=1024 * 1024 * 50,
    )


def _failing_overhead() -> GraphOverheadResult:
    return GraphOverheadResult(
        avg_planner_latency_ms=250.0,
        baseline_planner_latency_ms=100.0,
        latency_overhead_pp=30.0,
        index_size_bytes=1024 * 1024 * 200,
    )


def _make_per_query_scores(
    n: int,
    *,
    with_recall: float = 0.90,
    without_recall: float = 0.70,
    with_ndcg: float = 0.90,
    without_ndcg: float = 0.70,
) -> list[GraphPerQueryScore]:
    return [
        GraphPerQueryScore(
            query_id=f"q{i}",
            planner_with_graph_recall_at_10=with_recall,
            planner_without_graph_recall_at_10=without_recall,
            planner_with_graph_ndcg_at_10=with_ndcg,
            planner_without_graph_ndcg_at_10=without_ndcg,
            query_uses_graph_path=True,
        )
        for i in range(n)
    ]


def _passing_quality_ab() -> GraphPhase6bVerdict:
    mech = GraphMechanicalResult(
        n_books_processed=10,
        n_books_extraction_failed=0,
        n_total_nodes=200,
        n_total_edges=400,
    )
    scores = _make_per_query_scores(15)
    return evaluate_graph(mechanical=mech, per_query_scores=scores)


def _failing_quality_ab() -> GraphPhase6bVerdict:
    mech = GraphMechanicalResult(
        n_books_processed=10,
        n_books_extraction_failed=0,
        n_total_nodes=200,
        n_total_edges=400,
    )
    scores = _make_per_query_scores(
        15,
        with_recall=0.40,
        without_recall=0.80,
        with_ndcg=0.40,
        without_ndcg=0.80,
    )
    return evaluate_graph(mechanical=mech, per_query_scores=scores)


# ---------------------------------------------------------------------------
# GraphOverheadResult pass_gate
# ---------------------------------------------------------------------------


class TestOverheadResult:
    def test_overhead_result_computed_pass_gate(self) -> None:
        overhead = _passing_overhead(latency_overhead_pp=10.0)
        assert overhead.pass_gate is True

    def test_overhead_result_fails_when_overhead_exceeds_max(self) -> None:
        overhead = _failing_overhead()
        assert overhead.pass_gate is False

    def test_overhead_result_exactly_at_boundary_passes(self) -> None:
        # 20.0 == max_acceptable_overhead_pp (default 20.0) → should pass (<=)
        overhead = GraphOverheadResult(
            avg_planner_latency_ms=120.0,
            baseline_planner_latency_ms=100.0,
            latency_overhead_pp=20.0,
            index_size_bytes=0,
        )
        assert overhead.pass_gate is True


# ---------------------------------------------------------------------------
# decide_phase6b — gate routing
# ---------------------------------------------------------------------------


class TestDecidePhase6b:
    def test_decide_phase6b_all_pass_enables_graph_in_planner(self) -> None:
        verdict = decide_phase6b(
            smoke=_passing_smoke(),
            quality_ab=_passing_quality_ab(),
            overhead=_passing_overhead(),
        )
        assert verdict.action == "enable_graph_in_planner"
        assert verdict.keep_graph_in_default_planner is True
        assert verdict.failure_reasons == []

    def test_decide_phase6b_smoke_fail_disables_graph_path(self) -> None:
        verdict = decide_phase6b(
            smoke=_failing_smoke(),
            quality_ab=_passing_quality_ab(),
            overhead=_passing_overhead(),
        )
        assert verdict.action == "disable_graph_path"
        assert verdict.keep_graph_in_default_planner is False
        assert verdict.failure_reasons

    def test_decide_phase6b_quality_fail_disables_graph_path(self) -> None:
        verdict = decide_phase6b(
            smoke=_passing_smoke(),
            quality_ab=_failing_quality_ab(),
            overhead=_passing_overhead(),
        )
        assert verdict.action == "disable_graph_path"
        assert verdict.keep_graph_in_default_planner is False

    def test_decide_phase6b_overhead_fail_yields_rerun_after_fixes(self) -> None:
        verdict = decide_phase6b(
            smoke=_passing_smoke(),
            quality_ab=_passing_quality_ab(),
            overhead=_failing_overhead(),
        )
        assert verdict.action == "rerun_after_fixes"
        assert verdict.keep_graph_in_default_planner is False
        assert verdict.failure_reasons

    def test_decide_phase6b_returns_phase6b_final_verdict(self) -> None:
        verdict = decide_phase6b(
            smoke=_passing_smoke(),
            quality_ab=_passing_quality_ab(),
            overhead=_passing_overhead(),
        )
        assert isinstance(verdict, Phase6bFinalVerdict)
        assert isinstance(verdict.smoke, GraphSmokeResult)
        assert isinstance(verdict.overhead, GraphOverheadResult)
        assert verdict.summary


# ---------------------------------------------------------------------------
# emit_phase6b_report
# ---------------------------------------------------------------------------


class TestEmitPhase6bReport:
    def test_emit_phase6b_report_returns_non_empty_multiline(self) -> None:
        verdict = decide_phase6b(
            smoke=_passing_smoke(),
            quality_ab=_passing_quality_ab(),
            overhead=_passing_overhead(),
        )
        report = emit_phase6b_report(verdict)
        assert isinstance(report, str)
        lines = report.splitlines()
        assert len(lines) > 5
        assert any("GRAPH LAYER STOP/GO DECISION REPORT" in line for line in lines)

    def test_emit_phase6b_report_contains_action(self) -> None:
        verdict = decide_phase6b(
            smoke=_failing_smoke(),
            quality_ab=_passing_quality_ab(),
            overhead=_passing_overhead(),
        )
        report = emit_phase6b_report(verdict)
        assert (
            "disable_graph_path" in report
            or "rerun_after_fixes" in report
            or "enable_graph_in_planner" in report
        )
