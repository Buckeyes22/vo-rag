"""STOP/GO evaluation — semantic RAPTOR vs naive single-path baseline.

Decides whether the semantic RAPTOR layer (Layer 1B) is BETTER than naive
single-path retrieval (Layer 0 chunks only).

Metrics: Recall@10, nDCG@10, judge-rated answer quality.
Threshold: RAPTOR must be NON-INFERIOR on Recall + SUPERIOR on nDCG OR judge.
"""

from __future__ import annotations

from typing import Any, Literal

import structlog

from vo_rag.core.eval.non_inferiority import compute_non_inferiority
from vo_rag.core.models.common import VoBaseModel

log = structlog.get_logger(__name__)

__all__ = [
    "RaptorMechanicalResult",
    "RaptorPerQueryScore",
    "RaptorEvalVerdict",
    "evaluate_raptor",
    "emit_raptor_report",
]


class RaptorMechanicalResult(VoBaseModel):
    """Mechanical STOP/GO gate for the RAPTOR clustering layer.

    Attributes:
        n_clusters_per_level: Number of clusters discovered at each tree level.
        n_clusters_silhouette_below_threshold: Clusters whose silhouette score
            fell below the silhouette_threshold.
        n_clusters_total: Total number of clusters across all levels.
        silhouette_threshold: Minimum acceptable silhouette score (default 0.20).
        max_below_threshold_frac: Maximum acceptable fraction of below-threshold
            clusters (default 0.05 = 5%).
        bic_converged: True when GMM BIC selection converged reliably.
        cluster_reproducibility_passed: True when frozen-seed reproducibility
            check confirmed stable cluster membership.
        pass_gate: True when all mechanical sub-gates pass.
    """

    n_clusters_per_level: dict[str, int]
    n_clusters_silhouette_below_threshold: int
    n_clusters_total: int
    silhouette_threshold: float = 0.20
    max_below_threshold_frac: float = 0.05
    bic_converged: bool
    cluster_reproducibility_passed: bool
    pass_gate: bool = False

    @property
    def silhouette_frac(self) -> float:
        if self.n_clusters_total == 0:
            return 0.0
        return self.n_clusters_silhouette_below_threshold / self.n_clusters_total

    def model_post_init(self, *_args: Any, **_kwargs: Any) -> None:
        super().model_post_init(*_args, **_kwargs)
        silhouette_frac = self.silhouette_frac
        self.pass_gate = (
            silhouette_frac <= self.max_below_threshold_frac
            and self.bic_converged
            and self.cluster_reproducibility_passed
        )


class RaptorPerQueryScore(VoBaseModel):
    """Pre-computed per-query retrieval metrics for RAPTOR vs baseline arms.

    Attributes:
        query_id: Unique identifier for the query.
        raptor_recall_at_10: Recall@10 from the RAPTOR arm.
        baseline_recall_at_10: Recall@10 from the naive baseline arm.
        raptor_ndcg_at_10: nDCG@10 from the RAPTOR arm.
        baseline_ndcg_at_10: nDCG@10 from the naive baseline arm.
        raptor_judge_score: Judge-rated answer quality for RAPTOR (0-1 scale,
            0 if not yet computed).
        baseline_judge_score: Judge-rated answer quality for baseline (0-1 scale,
            0 if not yet computed).
    """

    query_id: str
    raptor_recall_at_10: float
    baseline_recall_at_10: float
    raptor_ndcg_at_10: float
    baseline_ndcg_at_10: float
    raptor_judge_score: float = 0.0
    baseline_judge_score: float = 0.0


class RaptorEvalVerdict(VoBaseModel):
    """Aggregate STOP/GO verdict for the semantic RAPTOR layer.

    Attributes:
        mechanical: Mechanical gate result (clustering quality, BIC, reproducibility).
        recall_result: NonInferiorityResult for Recall@10 as dict.
        ndcg_result: NonInferiorityResult for nDCG@10 as dict.
        judge_result: NonInferiorityResult for judge score as dict, or None if
            no judge scores were provided.
        raptor_accepted: True when RAPTOR passes the STOP/GO gate.
        action: One of accept_raptor / reject_raptor / rerun_after_fixes.
        failure_reasons: List of human-readable reasons for rejection or re-run.
        summary: One-line human-readable summary.
    """

    mechanical: RaptorMechanicalResult
    recall_result: dict[str, Any]
    ndcg_result: dict[str, Any]
    judge_result: dict[str, Any] | None
    raptor_accepted: bool
    action: Literal["accept_raptor", "reject_raptor", "rerun_after_fixes"]
    failure_reasons: list[str]
    summary: str


def evaluate_raptor(
    *,
    mechanical: RaptorMechanicalResult,
    per_query_scores: list[RaptorPerQueryScore],
    config_margin_pp: float = 3.0,
    config_superiority_pp: float = 2.0,
    confidence: float = 0.95,
) -> RaptorEvalVerdict:
    """Run the STOP/GO evaluation for the semantic RAPTOR layer.

    Reuses ``compute_non_inferiority`` from ``vo_rag.core.eval.non_inferiority``
    for the recall non-inferiority test. Computes non-inferiority + superiority
    for nDCG and judge independently.

    Acceptance criteria:
      1. All mechanical sub-gates must pass (silhouette fraction, BIC, reproducibility)
      2. Recall must be NON-INFERIOR (RAPTOR ≥ baseline - margin)
      3. At least one of (nDCG, judge) must be SUPERIOR
         (delta_ci_low > +config_superiority_pp / 100)

    Args:
        mechanical: Pre-computed mechanical gate result.
        per_query_scores: Pre-computed per-query metric records.
        config_margin_pp: Non-inferiority margin in percentage points.
        config_superiority_pp: Superiority margin in percentage points.
        confidence: One-sided confidence level for the CI.

    Returns:
        RaptorEvalVerdict with STOP/GO decision and supporting details.
    """
    failure_reasons: list[str] = []
    action: Literal["accept_raptor", "reject_raptor", "rerun_after_fixes"]

    if not mechanical.pass_gate:
        silhouette_frac = mechanical.silhouette_frac
        if silhouette_frac > mechanical.max_below_threshold_frac:
            failure_reasons.append(
                f"Silhouette fraction {silhouette_frac:.3f} exceeds threshold "
                f"{mechanical.max_below_threshold_frac:.3f}"
            )
        if not mechanical.bic_converged:
            failure_reasons.append("BIC did not converge — GMM component selection is unreliable")
        if not mechanical.cluster_reproducibility_passed:
            failure_reasons.append(
                "Cluster reproducibility check failed — "
                "frozen seeds did not reproduce cluster membership"
            )

        log.warning(
            "raptor_mechanical_gate_failed",
            pass_gate=mechanical.pass_gate,
            failure_reasons=failure_reasons,
        )

        return RaptorEvalVerdict(
            mechanical=mechanical,
            recall_result={},
            ndcg_result={},
            judge_result=None,
            raptor_accepted=False,
            action="rerun_after_fixes",
            failure_reasons=failure_reasons,
            summary="REJECT (mechanical gate failed) | " + " | ".join(failure_reasons),
        )

    raptor_recalls = [s.raptor_recall_at_10 for s in per_query_scores]
    baseline_recalls = [s.baseline_recall_at_10 for s in per_query_scores]
    raptor_ndcgs = [s.raptor_ndcg_at_10 for s in per_query_scores]
    baseline_ndcgs = [s.baseline_ndcg_at_10 for s in per_query_scores]

    recall_result = compute_non_inferiority(
        control_scores=baseline_recalls,
        treatment_scores=raptor_recalls,
        metric="recall_at_10",
        margin_pp=config_margin_pp,
        confidence=confidence,
    )

    ndcg_result = compute_non_inferiority(
        control_scores=baseline_ndcgs,
        treatment_scores=raptor_ndcgs,
        metric="ndcg_at_10",
        margin_pp=config_margin_pp,
        confidence=confidence,
    )

    has_judge_scores = any(
        s.raptor_judge_score != 0.0 or s.baseline_judge_score != 0.0 for s in per_query_scores
    )
    judge_result_dict: dict[str, Any] | None = None
    judge_result_obj = None
    if has_judge_scores:
        raptor_judges = [s.raptor_judge_score for s in per_query_scores]
        baseline_judges = [s.baseline_judge_score for s in per_query_scores]
        judge_result_obj = compute_non_inferiority(
            control_scores=baseline_judges,
            treatment_scores=raptor_judges,
            metric="judge_score",
            margin_pp=config_margin_pp,
            confidence=confidence,
        )
        judge_result_dict = judge_result_obj.model_dump()

    superior_margin_frac = config_superiority_pp / 100.0
    ndcg_superior = ndcg_result.delta_ci_low > superior_margin_frac
    judge_superior = (
        judge_result_obj.delta_ci_low > superior_margin_frac if judge_result_obj else False
    )

    recall_non_inferior = recall_result.decision == "non_inferior"
    raptor_accepted = recall_non_inferior and (ndcg_superior or judge_superior)

    if not recall_non_inferior:
        failure_reasons.append(
            f"Recall@10 is inferior: delta_ci_low={recall_result.delta_ci_low:+.4f} "
            f"< -{config_margin_pp / 100:.4f} (margin)"
        )

    if not ndcg_superior and not judge_superior:
        judge_ci_str = f"{judge_result_obj.delta_ci_low:+.4f}" if judge_result_obj else "N/A"
        failure_reasons.append(
            f"No metric is superior: nDCG delta_ci_low={ndcg_result.delta_ci_low:+.4f}, "
            f"judge delta_ci_low={judge_ci_str} "
            f"(need > +{superior_margin_frac:.4f})"
        )

    action = "accept_raptor" if raptor_accepted else "reject_raptor"

    summary_parts: list[str] = []
    summary_parts.append(
        f"Recall@10: delta={recall_result.delta_mean:+.4f} "
        f"CI_low={recall_result.delta_ci_low:+.4f} → {recall_result.decision}"
    )
    summary_parts.append(
        f"nDCG@10: delta={ndcg_result.delta_mean:+.4f} "
        f"CI_low={ndcg_result.delta_ci_low:+.4f} → {ndcg_result.decision}"
    )
    if judge_result_obj is not None:
        summary_parts.append(
            f"Judge: delta={judge_result_obj.delta_mean:+.4f} "
            f"CI_low={judge_result_obj.delta_ci_low:+.4f} → {judge_result_obj.decision}"
        )
    verdict_label = "ACCEPT" if raptor_accepted else f"REJECT ({action})"
    summary = verdict_label + " | " + " | ".join(summary_parts)

    log.info(
        "raptor_eval_verdict",
        raptor_accepted=raptor_accepted,
        action=action,
        n_queries=len(per_query_scores),
        failure_reasons=failure_reasons,
    )

    return RaptorEvalVerdict(
        mechanical=mechanical,
        recall_result=recall_result.model_dump(),
        ndcg_result=ndcg_result.model_dump(),
        judge_result=judge_result_dict,
        raptor_accepted=raptor_accepted,
        action=action,
        failure_reasons=failure_reasons,
        summary=summary,
    )


def emit_raptor_report(verdict: RaptorEvalVerdict) -> str:
    """Emit a multi-line plain-text evaluation report.

    Args:
        verdict: The RaptorEvalVerdict produced by evaluate_raptor.

    Returns:
        Multi-line human-readable plain-text report.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("RAPTOR LAYER EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    status = "ACCEPT" if verdict.raptor_accepted else f"REJECT ({verdict.action})"
    lines.append(f"Verdict:      {status}")
    lines.append(f"Summary:       {verdict.summary}")
    lines.append("")

    lines.append("--- Mechanical Gate ---")
    mech = verdict.mechanical
    silhouette_frac = mech.silhouette_frac
    lines.append(f"  pass_gate:                         {mech.pass_gate}")
    lines.append(f"  bic_converged:                     {mech.bic_converged}")
    lines.append(f"  cluster_reproducibility_passed:    {mech.cluster_reproducibility_passed}")
    lines.append(f"  silhouette_fraction:               {silhouette_frac:.4f}")
    lines.append(f"  silhouette_threshold:             {mech.silhouette_threshold}")
    lines.append(f"  max_below_threshold_frac:          {mech.max_below_threshold_frac:.4f}")
    lines.append(f"  n_clusters_total:                 {mech.n_clusters_total}")
    lines.append(
        f"  n_clusters_silhouette_below_threshold: {mech.n_clusters_silhouette_below_threshold}"
    )
    lines.append(f"  n_clusters_per_level:             {mech.n_clusters_per_level}")
    lines.append("")

    lines.append("--- Recall@10 ---")
    if verdict.recall_result:
        rr = verdict.recall_result
        lines.append(f"  Decision:    {rr['decision']}")
        lines.append(f"  N:           {rr['n']}")
        lines.append(f"  RAPTOR mean:    {rr['treatment_mean']:.4f}")
        lines.append(f"  Baseline mean:  {rr['control_mean']:.4f}")
        lines.append(f"  Delta mean:  {rr['delta_mean']:+.4f}")
        lines.append(f"  Delta CI low: {rr['delta_ci_low']:+.4f}")
        lines.append(f"  Non-inferiority margin: -{rr['non_inferiority_margin_pp']}pp")
    else:
        lines.append("  (Not computed — mechanical gate failed)")
    lines.append("")

    lines.append("--- nDCG@10 ---")
    if verdict.ndcg_result:
        nr = verdict.ndcg_result
        lines.append(f"  Decision:    {nr['decision']}")
        lines.append(f"  N:           {nr['n']}")
        lines.append(f"  RAPTOR mean:    {nr['treatment_mean']:.4f}")
        lines.append(f"  Baseline mean:  {nr['control_mean']:.4f}")
        lines.append(f"  Delta mean:  {nr['delta_mean']:+.4f}")
        lines.append(f"  Delta CI low: {nr['delta_ci_low']:+.4f}")
        lines.append(f"  Non-inferiority margin: -{nr['non_inferiority_margin_pp']}pp")
    else:
        lines.append("  (Not computed — mechanical gate failed)")
    lines.append("")

    if verdict.judge_result is not None:
        jr = verdict.judge_result
        lines.append("--- Judge Score ---")
        lines.append(f"  Decision:    {jr['decision']}")
        lines.append(f"  N:           {jr['n']}")
        lines.append(f"  RAPTOR mean:    {jr['treatment_mean']:.4f}")
        lines.append(f"  Baseline mean:  {jr['control_mean']:.4f}")
        lines.append(f"  Delta mean:  {jr['delta_mean']:+.4f}")
        lines.append(f"  Delta CI low: {jr['delta_ci_low']:+.4f}")
        lines.append(f"  Non-inferiority margin: -{jr['non_inferiority_margin_pp']}pp")
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
