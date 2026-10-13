from __future__ import annotations

from unittest.mock import patch

from vo_rag.core.raptor.evaluation import (
    RaptorMechanicalResult,
    RaptorPerQueryScore,
    emit_raptor_report,
    evaluate_raptor,
)


def _make_mechanical(
    pass_gate: bool = True,
    n_clusters_per_level: dict[str, int] | None = None,
    n_clusters_silhouette_below_threshold: int = 0,
    n_clusters_total: int = 7,
    bic_converged: bool = True,
    cluster_reproducibility_passed: bool = True,
) -> RaptorMechanicalResult:
    if n_clusters_per_level is None:
        n_clusters_per_level = {"level_0": 5, "level_1": 2}
    return RaptorMechanicalResult(
        n_clusters_per_level=n_clusters_per_level,
        n_clusters_silhouette_below_threshold=n_clusters_silhouette_below_threshold,
        n_clusters_total=n_clusters_total,
        bic_converged=bic_converged,
        cluster_reproducibility_passed=cluster_reproducibility_passed,
    )


def _make_per_query_scores(
    n: int = 20,
    raptor_recall: float = 0.70,
    baseline_recall: float = 0.65,
    raptor_ndcg: float = 0.75,
    baseline_ndcg: float = 0.65,
    raptor_judge: float = 0.0,
    baseline_judge: float = 0.0,
) -> list[RaptorPerQueryScore]:
    return [
        RaptorPerQueryScore(
            query_id=f"q-{i}",
            raptor_recall_at_10=raptor_recall,
            baseline_recall_at_10=baseline_recall,
            raptor_ndcg_at_10=raptor_ndcg,
            baseline_ndcg_at_10=baseline_ndcg,
            raptor_judge_score=raptor_judge,
            baseline_judge_score=baseline_judge,
        )
        for i in range(n)
    ]


class TestRaptorMechanicalResultPassGate:
    def test_mechanical_pass_gate_requires_silhouette_and_bic_and_reproducibility(self) -> None:
        mech = RaptorMechanicalResult(
            n_clusters_per_level={"level_0": 5, "level_1": 5},
            n_clusters_silhouette_below_threshold=5,
            n_clusters_total=10,
            bic_converged=True,
            cluster_reproducibility_passed=True,
        )
        assert mech.pass_gate is False


class TestEvaluateRaptor:
    def test_evaluate_raptor_mech_fail_yields_rerun_after_fixes(self) -> None:
        failing_mech = _make_mechanical(
            pass_gate=False,
            bic_converged=False,
            cluster_reproducibility_passed=True,
        )
        scores = _make_per_query_scores(n=20)
        verdict = evaluate_raptor(mechanical=failing_mech, per_query_scores=scores)
        assert verdict.action == "rerun_after_fixes"
        assert verdict.raptor_accepted is False

    def test_evaluate_raptor_wins_recall_non_inferior_ndcg_superior(self) -> None:
        mech = _make_mechanical(pass_gate=True)
        scores = _make_per_query_scores(
            n=30,
            raptor_recall=0.70,
            baseline_recall=0.65,
            raptor_ndcg=0.80,
            baseline_ndcg=0.60,
        )
        verdict = evaluate_raptor(mechanical=mech, per_query_scores=scores)
        assert verdict.raptor_accepted is True
        assert verdict.action == "accept_raptor"

    def test_evaluate_raptor_loses_recall_inferior_yields_reject(self) -> None:
        mech = _make_mechanical(pass_gate=True)
        scores = _make_per_query_scores(
            n=30,
            raptor_recall=0.40,
            baseline_recall=0.70,
            raptor_ndcg=0.80,
            baseline_ndcg=0.60,
        )
        verdict = evaluate_raptor(mechanical=mech, per_query_scores=scores)
        assert verdict.raptor_accepted is False
        assert verdict.action == "reject_raptor"
        assert any("Recall" in r or "recall" in r for r in verdict.failure_reasons)

    def test_evaluate_raptor_ties_yields_reject_no_superior_metric(self) -> None:
        mech = _make_mechanical(pass_gate=True)
        scores = _make_per_query_scores(
            n=30,
            raptor_recall=0.65,
            baseline_recall=0.65,
            raptor_ndcg=0.65,
            baseline_ndcg=0.65,
        )
        verdict = evaluate_raptor(mechanical=mech, per_query_scores=scores)
        assert verdict.raptor_accepted is False
        assert verdict.action == "reject_raptor"

    def test_evaluate_raptor_with_judge_scores_superior_judge_accepts(self) -> None:
        mech = _make_mechanical(pass_gate=True)
        scores = _make_per_query_scores(
            n=30,
            raptor_recall=0.68,
            baseline_recall=0.65,
            raptor_ndcg=0.65,
            baseline_ndcg=0.64,
            raptor_judge=0.85,
            baseline_judge=0.60,
        )
        verdict = evaluate_raptor(mechanical=mech, per_query_scores=scores)
        assert verdict.raptor_accepted is True
        assert verdict.judge_result is not None

    def test_evaluate_raptor_uses_vo_rag_non_inferiority(self) -> None:
        """RaptorEvalVerdict imports non-inferiority math from vo_rag.core.eval.non_inferiority."""
        mech = _make_mechanical(pass_gate=True)
        scores = _make_per_query_scores(n=10)
        with patch("vo_rag.core.raptor.evaluation.compute_non_inferiority") as mock_fn:
            mock_fn.return_value.decision = "non_inferior"
            mock_fn.return_value.delta_mean = 0.05
            mock_fn.return_value.delta_ci_low = 0.01
            mock_fn.return_value.model_dump.return_value = {
                "decision": "non_inferior",
                "delta_mean": 0.05,
                "delta_ci_low": 0.01,
            }
            evaluate_raptor(mechanical=mech, per_query_scores=scores)
            assert mock_fn.called


class TestEmitRaptorReport:
    def test_emit_raptor_report_returns_multi_line_non_empty_string(self) -> None:
        mech = _make_mechanical(pass_gate=True)
        scores = _make_per_query_scores(
            n=30,
            raptor_recall=0.70,
            baseline_recall=0.65,
            raptor_ndcg=0.80,
            baseline_ndcg=0.60,
        )
        verdict = evaluate_raptor(mechanical=mech, per_query_scores=scores)
        report = emit_raptor_report(verdict)
        assert isinstance(report, str)
        assert len(report.splitlines()) > 5
        assert "RAPTOR LAYER EVALUATION REPORT" in report
        assert "Verdict" in report
