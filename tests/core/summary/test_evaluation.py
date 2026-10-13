"""Tests for vo_rag.core.summary.evaluation.

The STOP/GO gate imports ``compute_non_inferiority`` from
``vo_rag.core.eval.non_inferiority``.
"""

from __future__ import annotations

import pytest

from vo_rag.core.summary.evaluation import (
    SummaryMechanicalResult,
    SummaryPerQueryScore,
    emit_summary_eval_report,
    evaluate_summary_layer,
)


def _make_mechanical(
    n_books_processed: int = 100,
    n_books_extraction_failed: int = 0,
    n_evidence_chunk_id_violations: int = 0,
    prompt_hash_invariance_passed: bool = True,
    max_acceptable_failure_rate: float = 0.05,
) -> SummaryMechanicalResult:
    total = n_books_processed + n_books_extraction_failed
    failure_rate = n_books_extraction_failed / total if total > 0 else 0.0
    mech = SummaryMechanicalResult(
        n_books_processed=n_books_processed,
        n_books_extraction_failed=n_books_extraction_failed,
        failure_rate=failure_rate,
        max_acceptable_failure_rate=max_acceptable_failure_rate,
        n_evidence_chunk_id_violations=n_evidence_chunk_id_violations,
        prompt_hash_invariance_passed=prompt_hash_invariance_passed,
    )
    return mech


def _make_per_query_scores(
    n: int = 20,
    with_summary_recall: float = 0.70,
    without_summary_recall: float = 0.65,
    with_summary_ndcg: float = 0.75,
    without_summary_ndcg: float = 0.65,
    with_summary_judge: float = 0.0,
    without_summary_judge: float = 0.0,
) -> list[SummaryPerQueryScore]:
    return [
        SummaryPerQueryScore(
            query_id=f"q-{i}",
            with_summary_recall_at_10=with_summary_recall,
            without_summary_recall_at_10=without_summary_recall,
            with_summary_ndcg_at_10=with_summary_ndcg,
            without_summary_ndcg_at_10=without_summary_ndcg,
            with_summary_judge_score=with_summary_judge,
            without_summary_judge_score=without_summary_judge,
        )
        for i in range(n)
    ]


class TestSummaryMechanicalResultPassGate:
    def test_summary_mechanical_pass_gate_all_ok(
        self,
    ) -> None:
        mech = _make_mechanical(
            n_books_processed=100,
            n_books_extraction_failed=2,
            n_evidence_chunk_id_violations=0,
            prompt_hash_invariance_passed=True,
        )
        assert mech.pass_gate is True

    def test_summary_mechanical_pass_gate_fails_when_evidence_chunk_id_violations_nonzero(
        self,
    ) -> None:
        mech = _make_mechanical(
            n_evidence_chunk_id_violations=3,
            prompt_hash_invariance_passed=True,
        )
        assert mech.pass_gate is False

    def test_summary_mechanical_pass_gate_fails_when_prompt_hash_invariance_false(self) -> None:
        mech = _make_mechanical(
            prompt_hash_invariance_passed=False,
            n_evidence_chunk_id_violations=0,
        )
        assert mech.pass_gate is False

    def test_summary_mechanical_pass_gate_fails_when_failure_rate_above_threshold(self) -> None:
        mech = _make_mechanical(
            n_books_processed=90,
            n_books_extraction_failed=15,
            n_evidence_chunk_id_violations=0,
            prompt_hash_invariance_passed=True,
        )
        assert mech.failure_rate == pytest.approx(15 / 105)
        assert mech.pass_gate is False


class TestEvaluateSummaryLayer:
    def test_evaluate_summary_layer_mech_fail_yields_rerun_after_fixes(self) -> None:
        failing_mech = _make_mechanical(
            n_books_processed=100,
            n_books_extraction_failed=10,
            n_evidence_chunk_id_violations=0,
            prompt_hash_invariance_passed=True,
        )
        scores = _make_per_query_scores(n=20)
        verdict = evaluate_summary_layer(mechanical=failing_mech, per_query_scores=scores)
        assert verdict.action == "rerun_after_fixes"
        assert verdict.summary_layer_accepted is False
        assert any("failure rate" in r.lower() for r in verdict.failure_reasons)

    def test_evaluate_summary_layer_wins_yields_accept_summary_layer(self) -> None:
        mech = _make_mechanical()
        scores = _make_per_query_scores(
            n=30,
            with_summary_recall=0.70,
            without_summary_recall=0.65,
            with_summary_ndcg=0.80,
            without_summary_ndcg=0.60,
        )
        verdict = evaluate_summary_layer(mechanical=mech, per_query_scores=scores)
        assert verdict.summary_layer_accepted is True
        assert verdict.action == "accept_summary_layer"

    def test_evaluate_summary_layer_loses_recall_yields_reject_summary_layer(self) -> None:
        mech = _make_mechanical()
        scores = _make_per_query_scores(
            n=30,
            with_summary_recall=0.40,
            without_summary_recall=0.70,
            with_summary_ndcg=0.80,
            without_summary_ndcg=0.60,
        )
        verdict = evaluate_summary_layer(mechanical=mech, per_query_scores=scores)
        assert verdict.summary_layer_accepted is False
        assert verdict.action == "reject_summary_layer"
        assert any("recall" in r.lower() for r in verdict.failure_reasons)

    def test_evaluate_summary_layer_ties_yields_reject_no_superior_metric(self) -> None:
        mech = _make_mechanical()
        scores = _make_per_query_scores(
            n=30,
            with_summary_recall=0.65,
            without_summary_recall=0.65,
            with_summary_ndcg=0.65,
            without_summary_ndcg=0.65,
        )
        verdict = evaluate_summary_layer(mechanical=mech, per_query_scores=scores)
        assert verdict.summary_layer_accepted is False
        assert verdict.action == "reject_summary_layer"
        assert any(
            "superior" in r.lower() or "metric" in r.lower() for r in verdict.failure_reasons
        )

    def test_evaluate_summary_layer_judge_superior_accepts_when_recall_non_inferior(self) -> None:
        mech = _make_mechanical()
        scores = _make_per_query_scores(
            n=30,
            with_summary_recall=0.68,
            without_summary_recall=0.65,
            with_summary_ndcg=0.65,
            without_summary_ndcg=0.64,
            with_summary_judge=0.85,
            without_summary_judge=0.60,
        )
        verdict = evaluate_summary_layer(mechanical=mech, per_query_scores=scores)
        assert verdict.summary_layer_accepted is True
        assert verdict.judge_result is not None


class TestEmitSummaryEvalReport:
    def test_emit_summary_eval_report_returns_non_empty_multiline_string(self) -> None:
        mech = _make_mechanical()
        scores = _make_per_query_scores(
            n=30,
            with_summary_recall=0.70,
            without_summary_recall=0.65,
            with_summary_ndcg=0.80,
            without_summary_ndcg=0.60,
        )
        verdict = evaluate_summary_layer(mechanical=mech, per_query_scores=scores)
        report = emit_summary_eval_report(verdict)
        assert isinstance(report, str)
        assert len(report.splitlines()) > 5
        assert "LAYER 1A SUMMARY EVALUATION REPORT" in report
        assert "Verdict" in report
