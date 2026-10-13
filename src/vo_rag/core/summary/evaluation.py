"""STOP/GO evaluation — Layer 1A structural summary tree gate.

Decides whether the
structural summary layer (Layer 1A) earns its place in the RAG stack by
passing:

  1. A mechanical validation gate (failure-rate, evidence chunk-id provenance,
     prompt-hash invariance).
  2. A statistical A/B gate: recall non-inferiority + nDCG-or-judge superiority.

The default ``pass_gate: bool = False`` is not computed by the model itself
(Pydantic v2 forbids assigning to fields in ``model_post_init``); the
decision logic in ``evaluate_summary_layer`` does the actual gate computation.
"""

from __future__ import annotations

from typing import Any, Literal

import structlog

from vo_rag.core.eval.non_inferiority import compute_non_inferiority
from vo_rag.core.models.common import VoBaseModel

log = structlog.get_logger(__name__)

__all__ = [
    "SummaryMechanicalResult",
    "SummaryPerQueryScore",
    "SummaryEvalVerdict",
    "evaluate_summary_layer",
    "emit_summary_eval_report",
]


class SummaryMechanicalResult(VoBaseModel):
    """Mechanical STOP/GO gate for the Layer 1A structural summary tree.

    Attributes:
        n_books_processed: Number of books for which summarisation completed
            (at any level) without a hard failure.
        n_books_extraction_failed: Number of books that raised an unrecoverable
            extraction error (parser crash, embedding failure, empty output, etc.).
        failure_rate: Fraction of all attempted books that failed extraction.
            Computed from n_books_extraction_failed / max(1, total_books).
        max_acceptable_failure_rate: Maximum tolerable failure rate (default 0.05
            = 5 %).
        n_evidence_chunk_id_violations: Count of evidence_chunk_ids on any
            SummaryNode that reference chunk IDs NOT present in the node's own
            source_chunk_ids set (provenance integrity).
        prompt_hash_invariance_passed: True when re-running summarisation with
            identical inputs produces the same prompt_hash on every node checked
            (invariant 3).
        pass_gate: True when all three mechanical sub-gates pass:
            failure_rate <= max_acceptable_failure_rate,
            n_evidence_chunk_id_violations == 0, AND
            prompt_hash_invariance_passed.
    """

    n_books_processed: int
    n_books_extraction_failed: int
    failure_rate: float
    max_acceptable_failure_rate: float = 0.05
    n_evidence_chunk_id_violations: int
    prompt_hash_invariance_passed: bool
    pass_gate: bool = False

    def model_post_init(self, *args: object, **kwargs: object) -> None:
        super().model_post_init(*args, **kwargs)
        self.pass_gate = (
            self.failure_rate <= self.max_acceptable_failure_rate
            and self.n_evidence_chunk_id_violations == 0
            and self.prompt_hash_invariance_passed
        )


class SummaryPerQueryScore(VoBaseModel):
    """Pre-computed per-query retrieval metrics for the summary A/B arms.

    Live retrieval is NOT performed here; callers supply these records after
    running the evaluation harness against the with-summary and without-summary
    retrieval pipelines.

    Attributes:
        query_id: Unique identifier for the query.
        with_summary_recall_at_10: Recall@10 from the with-summary arm.
        without_summary_recall_at_10: Recall@10 from the baseline (without-summary) arm.
        with_summary_ndcg_at_10: nDCG@10 from the with-summary arm.
        without_summary_ndcg_at_10: nDCG@10 from the baseline arm.
        with_summary_judge_score: Judge-rated answer quality for with-summary arm
            (0-1 scale, 0.0 if not yet computed).
        without_summary_judge_score: Judge-rated answer quality for baseline arm
            (0-1 scale, 0.0 if not yet computed).
    """

    query_id: str
    with_summary_recall_at_10: float
    without_summary_recall_at_10: float
    with_summary_ndcg_at_10: float
    without_summary_ndcg_at_10: float
    with_summary_judge_score: float = 0.0
    without_summary_judge_score: float = 0.0


class SummaryEvalVerdict(VoBaseModel):
    """Aggregate STOP/GO verdict for the Layer 1A structural summary tree.

    Attributes:
        mechanical: Mechanical gate result (failure-rate, provenance, invariance).
        recall_result: NonInferiorityResult for Recall@10 serialised as dict.
        ndcg_result: NonInferiorityResult for nDCG@10 serialised as dict.
        judge_result: NonInferiorityResult for judge score serialised as dict,
            or None when no judge scores were provided.
        summary_layer_accepted: True when the summary layer passes the STOP/GO gate.
        action: One of accept_summary_layer / reject_summary_layer / rerun_after_fixes.
        failure_reasons: Human-readable reasons for rejection or re-run request.
        summary: One-line human-readable summary of the verdict.
    """

    mechanical: SummaryMechanicalResult
    recall_result: dict[str, Any]
    ndcg_result: dict[str, Any]
    judge_result: dict[str, Any] | None
    summary_layer_accepted: bool
    action: Literal["accept_summary_layer", "reject_summary_layer", "rerun_after_fixes"]
    failure_reasons: list[str]
    summary: str


def evaluate_summary_layer(
    *,
    mechanical: SummaryMechanicalResult,
    per_query_scores: list[SummaryPerQueryScore],
    non_inferiority_margin_pp: float = 3.0,
    superiority_margin_pp: float = 2.0,
    confidence: float = 0.95,
) -> SummaryEvalVerdict:
    """Run the STOP/GO evaluation for the Layer 1A structural summary tree.

    Reuses ``compute_non_inferiority`` from
    ``vo_rag.core.eval.non_inferiority`` for all
    statistical comparisons, treating the with-summary pipeline as the
    treatment arm and the without-summary (baseline chunks-only) pipeline
    as the control arm.

    Acceptance criteria:
      1. All mechanical sub-gates must pass.
      2. Recall@10 must be NON-INFERIOR (with_summary >= baseline - margin).
      3. At least one of (nDCG@10, judge score) must be STRICTLY SUPERIOR
         (delta_ci_low > +superiority_margin_pp / 100).

    Args:
        mechanical: Pre-computed mechanical gate result.
        per_query_scores: Pre-computed per-query metric records.
        non_inferiority_margin_pp: Pre-registered non-inferiority margin in
            percentage points (applied to Recall@10 and as the base margin for
            the nDCG/judge non-inferiority tests fed into superiority evaluation).
        superiority_margin_pp: Required superiority margin in percentage points
            (delta_ci_low must exceed this for nDCG or judge to count as superior).
        confidence: One-sided confidence level for the CI.

    Returns:
        SummaryEvalVerdict with STOP/GO decision and supporting detail.
    """
    failure_reasons: list[str] = []
    action: Literal["accept_summary_layer", "reject_summary_layer", "rerun_after_fixes"]

    # Mechanical sub-gate is computed in ``SummaryMechanicalResult.model_post_init``
    # and persists in the ``pass_gate`` field, so we read it directly here.
    mechanical_pass = mechanical.pass_gate

    if not mechanical_pass:
        if mechanical.failure_rate > mechanical.max_acceptable_failure_rate:
            total_attempted = mechanical.n_books_processed + mechanical.n_books_extraction_failed
            failure_reasons.append(
                f"Extraction failure rate {mechanical.failure_rate:.3f} exceeds "
                f"maximum {mechanical.max_acceptable_failure_rate:.3f} "
                f"({mechanical.n_books_extraction_failed}/{total_attempted} books failed)"
            )
        if mechanical.n_evidence_chunk_id_violations > 0:
            failure_reasons.append(
                f"Evidence chunk-id provenance violation: "
                f"{mechanical.n_evidence_chunk_id_violations} evidence_chunk_ids "
                f"reference chunks absent from the node's source_chunk_ids set"
            )
        if not mechanical.prompt_hash_invariance_passed:
            failure_reasons.append(
                "Prompt-hash invariance check failed — re-running summarisation "
                "with identical inputs produced a different prompt_hash"
            )

        log.warning(
            "summary_mechanical_gate_failed",
            pass_gate=mechanical_pass,
            failure_reasons=failure_reasons,
        )

        return SummaryEvalVerdict(
            mechanical=mechanical,
            recall_result={},
            ndcg_result={},
            judge_result=None,
            summary_layer_accepted=False,
            action="rerun_after_fixes",
            failure_reasons=failure_reasons,
            summary="REJECT (mechanical gate failed) | " + " | ".join(failure_reasons),
        )

    with_recalls = [s.with_summary_recall_at_10 for s in per_query_scores]
    without_recalls = [s.without_summary_recall_at_10 for s in per_query_scores]
    with_ndcgs = [s.with_summary_ndcg_at_10 for s in per_query_scores]
    without_ndcgs = [s.without_summary_ndcg_at_10 for s in per_query_scores]

    recall_result = compute_non_inferiority(
        control_scores=without_recalls,
        treatment_scores=with_recalls,
        metric="recall_at_10",
        margin_pp=non_inferiority_margin_pp,
        confidence=confidence,
    )

    ndcg_result = compute_non_inferiority(
        control_scores=without_ndcgs,
        treatment_scores=with_ndcgs,
        metric="ndcg_at_10",
        margin_pp=non_inferiority_margin_pp,
        confidence=confidence,
    )

    has_judge_scores = any(
        s.with_summary_judge_score != 0.0 or s.without_summary_judge_score != 0.0
        for s in per_query_scores
    )
    judge_result_dict: dict[str, Any] | None = None
    judge_result_obj = None
    if has_judge_scores:
        with_judges = [s.with_summary_judge_score for s in per_query_scores]
        without_judges = [s.without_summary_judge_score for s in per_query_scores]
        judge_result_obj = compute_non_inferiority(
            control_scores=without_judges,
            treatment_scores=with_judges,
            metric="judge_score",
            margin_pp=non_inferiority_margin_pp,
            confidence=confidence,
        )
        judge_result_dict = judge_result_obj.model_dump()

    superior_margin_frac = superiority_margin_pp / 100.0
    ndcg_superior = ndcg_result.delta_ci_low > superior_margin_frac
    judge_superior = (
        judge_result_obj.delta_ci_low > superior_margin_frac if judge_result_obj else False
    )

    recall_non_inferior = recall_result.decision == "non_inferior"
    summary_layer_accepted = recall_non_inferior and (ndcg_superior or judge_superior)

    if not recall_non_inferior:
        failure_reasons.append(
            f"Recall@10 is not non-inferior: delta_ci_low={recall_result.delta_ci_low:+.4f} "
            f"< -{non_inferiority_margin_pp / 100:.4f} (margin)"
        )

    if not ndcg_superior and not judge_superior:
        judge_ci_str = f"{judge_result_obj.delta_ci_low:+.4f}" if judge_result_obj else "N/A"
        failure_reasons.append(
            f"No metric is superior: nDCG delta_ci_low={ndcg_result.delta_ci_low:+.4f}, "
            f"judge delta_ci_low={judge_ci_str} "
            f"(need > +{superior_margin_frac:.4f})"
        )

    action = "accept_summary_layer" if summary_layer_accepted else "reject_summary_layer"

    summary_parts: list[str] = []
    summary_parts.append(
        f"Recall@10: delta={recall_result.delta_mean:+.4f} "
        f"CI_low={recall_result.delta_ci_low:+.4f} -> {recall_result.decision}"
    )
    summary_parts.append(
        f"nDCG@10: delta={ndcg_result.delta_mean:+.4f} "
        f"CI_low={ndcg_result.delta_ci_low:+.4f} -> {ndcg_result.decision}"
    )
    if judge_result_obj is not None:
        summary_parts.append(
            f"Judge: delta={judge_result_obj.delta_mean:+.4f} "
            f"CI_low={judge_result_obj.delta_ci_low:+.4f} -> {judge_result_obj.decision}"
        )
    verdict_label = "ACCEPT" if summary_layer_accepted else f"REJECT ({action})"
    summary = verdict_label + " | " + " | ".join(summary_parts)

    log.info(
        "summary_eval_verdict",
        summary_layer_accepted=summary_layer_accepted,
        action=action,
        n_queries=len(per_query_scores),
        failure_reasons=failure_reasons,
    )

    return SummaryEvalVerdict(
        mechanical=mechanical,
        recall_result=recall_result.model_dump(),
        ndcg_result=ndcg_result.model_dump(),
        judge_result=judge_result_dict,
        summary_layer_accepted=summary_layer_accepted,
        action=action,
        failure_reasons=failure_reasons,
        summary=summary,
    )


def emit_summary_eval_report(verdict: SummaryEvalVerdict) -> str:
    """Emit a multi-line plain-text evaluation report for the Layer 1A gate.

    Args:
        verdict: The SummaryEvalVerdict produced by evaluate_summary_layer.

    Returns:
        Multi-line human-readable plain-text report.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("LAYER 1A SUMMARY EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    status = "ACCEPT" if verdict.summary_layer_accepted else f"REJECT ({verdict.action})"
    lines.append(f"Verdict:  {status}")
    lines.append(f"Summary:  {verdict.summary}")
    lines.append("")

    lines.append("--- Mechanical Gate ---")
    mech = verdict.mechanical
    total_books = mech.n_books_processed + mech.n_books_extraction_failed
    lines.append(f"  pass_gate:                           {mech.pass_gate}")
    lines.append(f"  n_books_processed:                   {mech.n_books_processed}")
    lines.append(f"  n_books_extraction_failed:           {mech.n_books_extraction_failed}")
    lines.append(f"  total_books_attempted:               {total_books}")
    lines.append(f"  failure_rate:                        {mech.failure_rate:.4f}")
    lines.append(f"  max_acceptable_failure_rate:         {mech.max_acceptable_failure_rate:.4f}")
    lines.append(f"  n_evidence_chunk_id_violations:      {mech.n_evidence_chunk_id_violations}")
    lines.append(f"  prompt_hash_invariance_passed:       {mech.prompt_hash_invariance_passed}")
    lines.append("")

    lines.append("--- Recall@10 ---")
    if verdict.recall_result:
        rr = verdict.recall_result
        lines.append(f"  Decision:                  {rr['decision']}")
        lines.append(f"  N:                         {rr['n']}")
        lines.append(f"  With-summary mean:         {rr['treatment_mean']:.4f}")
        lines.append(f"  Without-summary mean:      {rr['control_mean']:.4f}")
        lines.append(f"  Delta mean:                {rr['delta_mean']:+.4f}")
        lines.append(f"  Delta CI low:              {rr['delta_ci_low']:+.4f}")
        lines.append(f"  Non-inferiority margin:    -{rr['non_inferiority_margin_pp']}pp")
    else:
        lines.append("  (Not computed - mechanical gate failed)")
    lines.append("")

    lines.append("--- nDCG@10 ---")
    if verdict.ndcg_result:
        nr = verdict.ndcg_result
        lines.append(f"  Decision:                  {nr['decision']}")
        lines.append(f"  N:                         {nr['n']}")
        lines.append(f"  With-summary mean:         {nr['treatment_mean']:.4f}")
        lines.append(f"  Without-summary mean:      {nr['control_mean']:.4f}")
        lines.append(f"  Delta mean:                {nr['delta_mean']:+.4f}")
        lines.append(f"  Delta CI low:              {nr['delta_ci_low']:+.4f}")
        lines.append(f"  Non-inferiority margin:    -{nr['non_inferiority_margin_pp']}pp")
    else:
        lines.append("  (Not computed - mechanical gate failed)")
    lines.append("")

    if verdict.judge_result is not None:
        jr = verdict.judge_result
        lines.append("--- Judge Score ---")
        lines.append(f"  Decision:                  {jr['decision']}")
        lines.append(f"  N:                         {jr['n']}")
        lines.append(f"  With-summary mean:         {jr['treatment_mean']:.4f}")
        lines.append(f"  Without-summary mean:      {jr['control_mean']:.4f}")
        lines.append(f"  Delta mean:                {jr['delta_mean']:+.4f}")
        lines.append(f"  Delta CI low:              {jr['delta_ci_low']:+.4f}")
        lines.append(f"  Non-inferiority margin:    -{jr['non_inferiority_margin_pp']}pp")
        lines.append("")
    else:
        lines.append("--- Judge Score ---")
        lines.append("  (Not computed)")
        lines.append("")

    if verdict.failure_reasons:
        lines.append("--- Failure Reasons ---")
        for reason in verdict.failure_reasons:
            lines.append(f"  * {reason}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
