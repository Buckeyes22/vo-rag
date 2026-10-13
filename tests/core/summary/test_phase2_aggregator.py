"""Tests for vo_rag.core.summary.phase2_aggregator.

The summary-layer STOP/GO aggregator combines the mechanical + A/B
verdict with the per-summary coverage + faithfulness corpus quality.
"""

from __future__ import annotations

import pytest

from vo_rag.core.summary.coverage import (
    CoverageJudgment,
    CoverageResult,
    FaithfulnessClaim,
    FaithfulnessResult,
    SummaryQualityScore,
)
from vo_rag.core.summary.evaluation import (
    SummaryEvalVerdict,
    SummaryMechanicalResult,
    SummaryPerQueryScore,
    evaluate_summary_layer,
)
from vo_rag.core.summary.phase2_aggregator import (
    aggregate_corpus_quality,
    decide_phase2,
    emit_phase2_report,
)


def _coverage_result(
    summary_node_id: str,
    n_total: int,
    n_covered: int,
) -> CoverageResult:
    judgments = [
        CoverageJudgment(
            key_point_id=f"{summary_node_id}-kp{i}",
            covered=i < n_covered,
            confidence=0.9,
            evidence_in_summary="evidence text" if i < n_covered else None,
        )
        for i in range(n_total)
    ]
    return CoverageResult(
        summary_node_id=summary_node_id,
        n_key_points_total=n_total,
        n_key_points_covered=n_covered,
        coverage_rate=n_covered / n_total if n_total > 0 else 0.0,
        per_key_point_judgments=judgments,
        judge_prompt_hash="cov_hash",
    )


def _faithfulness_result(
    summary_node_id: str,
    n_extracted: int,
    n_grounded: int,
) -> FaithfulnessResult:
    judgments = [
        FaithfulnessClaim(
            claim_id=f"{summary_node_id}-cl{i}",
            claim_text=f"claim {i}",
            grounded=i < n_grounded,
            confidence=0.9,
            supporting_chunk_ids=[f"chunk-{i}"] if i < n_grounded else [],
        )
        for i in range(n_extracted)
    ]
    return FaithfulnessResult(
        summary_node_id=summary_node_id,
        n_claims_extracted=n_extracted,
        n_claims_grounded=n_grounded,
        faithfulness_rate=n_grounded / n_extracted if n_extracted > 0 else 0.0,
        per_claim_judgments=judgments,
        judge_prompt_hash="faith_hash",
    )


def _quality_score(
    summary_node_id: str,
    coverage_rate: float,
    faithfulness_rate: float,
) -> SummaryQualityScore:
    n_total = 10
    n_covered = int(round(coverage_rate * n_total))
    n_extracted = 10
    n_grounded = int(round(faithfulness_rate * n_extracted))
    coverage = _coverage_result(summary_node_id, n_total, n_covered)
    faithfulness = _faithfulness_result(summary_node_id, n_extracted, n_grounded)
    composite = 0.5 * coverage_rate + 0.5 * faithfulness_rate
    return SummaryQualityScore(
        summary_node_id=summary_node_id,
        coverage=coverage,
        faithfulness=faithfulness,
        composite_score=composite,
        composite_ci_low=composite - 0.05,
    )


def _passing_mechanical() -> SummaryMechanicalResult:
    return SummaryMechanicalResult(
        n_books_processed=10,
        n_books_extraction_failed=0,
        failure_rate=0.0,
        n_evidence_chunk_id_violations=0,
        prompt_hash_invariance_passed=True,
    )


def _passing_eval_verdict() -> SummaryEvalVerdict:
    mech = _passing_mechanical()
    per_query = [
        SummaryPerQueryScore(
            query_id=f"q{i}",
            with_summary_recall_at_10=0.85,
            without_summary_recall_at_10=0.80,
            with_summary_ndcg_at_10=0.85,
            without_summary_ndcg_at_10=0.78,
        )
        for i in range(10)
    ]
    return evaluate_summary_layer(mechanical=mech, per_query_scores=per_query)


def _failing_quality_eval_verdict() -> SummaryEvalVerdict:
    mech = _passing_mechanical()
    per_query = [
        SummaryPerQueryScore(
            query_id=f"q{i}",
            with_summary_recall_at_10=0.83,
            without_summary_recall_at_10=0.82,
            with_summary_ndcg_at_10=0.83,
            without_summary_ndcg_at_10=0.82,
        )
        for i in range(10)
    ]
    return evaluate_summary_layer(mechanical=mech, per_query_scores=per_query)


class TestAggregateCorpusQuality:
    def test_aggregate_corpus_quality_means_correct_for_uniform_scores(self) -> None:
        scores = [
            _quality_score(f"node-{i}", coverage_rate=0.80, faithfulness_rate=0.80)
            for i in range(5)
        ]
        result = aggregate_corpus_quality(scores)
        assert result.n_summaries_scored == 5
        assert result.mean_coverage_rate == 0.80
        assert result.mean_faithfulness_rate == 0.80
        assert result.mean_composite_score == 0.80

    def test_aggregate_corpus_quality_means_correct_for_mixed_scores(self) -> None:
        scores = [
            _quality_score("node-1", coverage_rate=0.60, faithfulness_rate=0.80),
            _quality_score("node-2", coverage_rate=0.80, faithfulness_rate=0.60),
            _quality_score("node-3", coverage_rate=0.70, faithfulness_rate=0.70),
        ]
        result = aggregate_corpus_quality(scores)
        assert result.n_summaries_scored == 3
        assert result.mean_coverage_rate == pytest.approx(0.7)
        assert result.mean_faithfulness_rate == pytest.approx(0.7)
        assert result.mean_composite_score == pytest.approx(0.7)

    def test_aggregate_corpus_quality_empty_list_yields_zero_means_and_fail_gate(
        self,
    ) -> None:
        result = aggregate_corpus_quality([])
        assert result.n_summaries_scored == 0
        assert result.mean_coverage_rate == 0.0
        assert result.mean_faithfulness_rate == 0.0
        assert result.mean_composite_score == 0.0
        assert result.pass_gate is False

    def test_corpus_quality_pass_gate_when_above_min_required(self) -> None:
        scores = [
            _quality_score(f"node-{i}", coverage_rate=0.90, faithfulness_rate=0.90)
            for i in range(3)
        ]
        result = aggregate_corpus_quality(scores, min_required_composite=0.70)
        assert result.pass_gate is True
        assert result.mean_composite_score >= 0.70

    def test_corpus_quality_fail_gate_when_below_min_required(self) -> None:
        scores = [
            _quality_score(f"node-{i}", coverage_rate=0.50, faithfulness_rate=0.50)
            for i in range(3)
        ]
        result = aggregate_corpus_quality(scores, min_required_composite=0.70)
        assert result.pass_gate is False
        assert result.mean_composite_score < 0.70


class TestDecidePhase2:
    def test_decide_phase2_all_pass_yields_enable_summary_layer(self) -> None:
        eval_verdict = _passing_eval_verdict()
        quality = aggregate_corpus_quality(
            [
                _quality_score(f"node-{i}", coverage_rate=0.85, faithfulness_rate=0.85)
                for i in range(5)
            ],
            min_required_composite=0.70,
        )
        verdict = decide_phase2(evaluation_verdict=eval_verdict, corpus_quality=quality)
        assert verdict.action == "enable_summary_layer"
        assert verdict.keep_summary_layer is True
        assert verdict.failure_reasons == []

    def test_decide_phase2_quality_fail_yields_disable_summary_layer(self) -> None:
        eval_verdict = _failing_quality_eval_verdict()
        quality = aggregate_corpus_quality(
            [
                _quality_score(f"node-{i}", coverage_rate=0.40, faithfulness_rate=0.40)
                for i in range(5)
            ],
            min_required_composite=0.70,
        )
        verdict = decide_phase2(evaluation_verdict=eval_verdict, corpus_quality=quality)
        assert verdict.action == "disable_summary_layer"
        assert verdict.keep_summary_layer is False
        assert len(verdict.failure_reasons) > 0

    def test_decide_phase2_eval_mech_fail_yields_rerun_after_fixes(self) -> None:
        mech_failing = SummaryMechanicalResult(
            n_books_processed=10,
            n_books_extraction_failed=2,
            failure_rate=0.1667,
            max_acceptable_failure_rate=0.05,
            n_evidence_chunk_id_violations=0,
            prompt_hash_invariance_passed=True,
        )
        per_query = [
            SummaryPerQueryScore(
                query_id=f"q{i}",
                with_summary_recall_at_10=0.85,
                without_summary_recall_at_10=0.80,
                with_summary_ndcg_at_10=0.85,
                without_summary_ndcg_at_10=0.78,
            )
            for i in range(10)
        ]
        eval_verdict = evaluate_summary_layer(mechanical=mech_failing, per_query_scores=per_query)
        quality = aggregate_corpus_quality(
            [
                _quality_score(f"node-{i}", coverage_rate=0.85, faithfulness_rate=0.85)
                for i in range(5)
            ],
            min_required_composite=0.70,
        )
        verdict = decide_phase2(evaluation_verdict=eval_verdict, corpus_quality=quality)
        assert verdict.action == "rerun_after_fixes"
        assert verdict.keep_summary_layer is False

    def test_decide_phase2_eval_ab_fail_yields_disable_summary_layer(self) -> None:
        eval_verdict = _failing_quality_eval_verdict()
        quality = aggregate_corpus_quality(
            [
                _quality_score(f"node-{i}", coverage_rate=0.85, faithfulness_rate=0.85)
                for i in range(5)
            ],
            min_required_composite=0.70,
        )
        verdict = decide_phase2(evaluation_verdict=eval_verdict, corpus_quality=quality)
        assert verdict.action == "disable_summary_layer"
        assert verdict.keep_summary_layer is False

    def test_emit_phase2_report_returns_non_empty_multiline_string(self) -> None:
        eval_verdict = _passing_eval_verdict()
        quality = aggregate_corpus_quality(
            [
                _quality_score(f"node-{i}", coverage_rate=0.85, faithfulness_rate=0.85)
                for i in range(5)
            ],
            min_required_composite=0.70,
        )
        verdict = decide_phase2(evaluation_verdict=eval_verdict, corpus_quality=quality)
        report = emit_phase2_report(verdict)
        assert isinstance(report, str)
        lines = report.splitlines()
        assert len(lines) > 5
        assert any("SUMMARY LAYER STOP/GO DECISION REPORT" in line for line in lines)
