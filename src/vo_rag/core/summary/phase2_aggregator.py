"""STOP/GO decision aggregator for the structural summary layer.

Combines the mechanical + A/B verdict from ``summary/evaluation.py`` and the
per-summary coverage + faithfulness corpus quality from ``summary/coverage.py``
into a single STOP/GO decision.

Thresholds: coverage >= 0.75,
distortion <= 0.10, unsupported <= 0.05. The corpus-quality gate
in this module enforces a related threshold via the
``min_required_composite`` parameter (default 0.70) — the per-summary
coverage / faithfulness rates are aggregated into a composite score and
compared against the threshold; downstream callers that need the strict
coverage >= 0.75 / distortion <= 0.10 / unsupported <= 0.05 constraints
should pass a ``min_required_composite`` that reflects the desired
coverage floor.
"""

from __future__ import annotations

from typing import Literal

import structlog

from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.summary.coverage import SummaryQualityScore
from vo_rag.core.summary.evaluation import SummaryEvalVerdict

__all__ = [
    "CorpusQualityResult",
    "Phase2FinalVerdict",
    "aggregate_corpus_quality",
    "decide_phase2",
    "emit_phase2_report",
]

logger = structlog.get_logger(__name__)


class CorpusQualityResult(VoBaseModel):
    """Aggregated corpus-level quality result across all summaries.

    Attributes:
        n_summaries_scored: Number of summaries that were scored.
        mean_coverage_rate: Mean coverage rate across all scored summaries.
        mean_faithfulness_rate: Mean faithfulness rate across all scored summaries.
        mean_composite_score: Mean composite score (0.5*coverage + 0.5*faithfulness)
            across all scored summaries.
        min_required_composite: Minimum acceptable mean composite score (default 0.70).
        pass_gate: True when mean_composite_score >= min_required_composite.
    """

    n_summaries_scored: int
    mean_coverage_rate: float
    mean_faithfulness_rate: float
    mean_composite_score: float
    min_required_composite: float = 0.70
    pass_gate: bool = False

    def model_post_init(self, *args: object, **kwargs: object) -> None:
        super().model_post_init(*args, **kwargs)
        self.pass_gate = self.mean_composite_score >= self.min_required_composite


class Phase2FinalVerdict(VoBaseModel):
    """Combined STOP/GO verdict for the structural summary layer.

    Attributes:
        evaluation_verdict: The mechanical + A/B verdict from summary/evaluation.py.
        corpus_quality: Aggregated corpus quality result from coverage.py scoring.
        keep_summary_layer: True when the summary layer should be kept in the RAG stack.
        action: One of enable_summary_layer / disable_summary_layer / rerun_after_fixes.
        failure_reasons: Human-readable reasons for rejection or re-run request.
        summary: One-line human-readable summary of the combined verdict.
    """

    evaluation_verdict: SummaryEvalVerdict
    corpus_quality: CorpusQualityResult
    keep_summary_layer: bool
    action: Literal["enable_summary_layer", "disable_summary_layer", "rerun_after_fixes"]
    failure_reasons: list[str]
    summary: str


def aggregate_corpus_quality(
    quality_scores: list[SummaryQualityScore],
    *,
    min_required_composite: float = 0.70,
) -> CorpusQualityResult:
    """Aggregate per-summary quality scores into a corpus-level quality result.

    Args:
        quality_scores: List of per-summary quality scores from coverage.py scoring.
        min_required_composite: Minimum acceptable mean composite score (default 0.70).

    Returns:
        CorpusQualityResult with mean coverage, faithfulness, and composite scores.
    """
    n = len(quality_scores)

    if n == 0:
        return CorpusQualityResult(
            n_summaries_scored=0,
            mean_coverage_rate=0.0,
            mean_faithfulness_rate=0.0,
            mean_composite_score=0.0,
            min_required_composite=min_required_composite,
            pass_gate=False,
        )

    mean_coverage_rate = sum(s.coverage.coverage_rate for s in quality_scores) / n
    mean_faithfulness_rate = sum(s.faithfulness.faithfulness_rate for s in quality_scores) / n
    mean_composite_score = sum(s.composite_score for s in quality_scores) / n

    return CorpusQualityResult(
        n_summaries_scored=n,
        mean_coverage_rate=mean_coverage_rate,
        mean_faithfulness_rate=mean_faithfulness_rate,
        mean_composite_score=mean_composite_score,
        min_required_composite=min_required_composite,
        pass_gate=mean_composite_score >= min_required_composite,
    )


def decide_phase2(
    *,
    evaluation_verdict: SummaryEvalVerdict,
    corpus_quality: CorpusQualityResult,
) -> Phase2FinalVerdict:
    """Determine the summary-layer STOP/GO decision.

    Both gates must pass for ``enable_summary_layer``:
      1. evaluation_verdict.action == "accept_summary_layer"
      2. corpus_quality.pass_gate

    Decision table:
      - evaluation=rerun_after_fixes  -> rerun_after_fixes (regardless of quality)
      - evaluation=reject_summary_layer -> disable_summary_layer (regardless of quality)
      - evaluation=accept BUT quality=fail -> disable_summary_layer

    Args:
        evaluation_verdict: The mechanical + A/B verdict from evaluate_summary_layer.
        corpus_quality: Aggregated corpus quality result.

    Returns:
        Phase2FinalVerdict with combined gate outcomes and recommended action.
    """
    failure_reasons: list[str] = []

    evaluation_accepts = evaluation_verdict.action == "accept_summary_layer"
    quality_passes = corpus_quality.pass_gate

    if evaluation_verdict.action == "rerun_after_fixes":
        keep_summary_layer = False
        action: Literal["enable_summary_layer", "disable_summary_layer", "rerun_after_fixes"] = (
            "rerun_after_fixes"
        )
        failure_reasons = list(evaluation_verdict.failure_reasons)
        summary = (
            f"Summary layer decision: rerun_after_fixes | "
            f"evaluation=rerun_after_fixes ({evaluation_verdict.summary}) | "
            f"quality_mean_composite={corpus_quality.mean_composite_score:.4f} | "
            f"reasons={failure_reasons}"
        )
    elif evaluation_verdict.action == "reject_summary_layer":
        keep_summary_layer = False
        action = "disable_summary_layer"
        failure_reasons = list(evaluation_verdict.failure_reasons)
        summary = (
            f"Summary layer decision: disable_summary_layer | "
            f"evaluation=reject_summary_layer ({evaluation_verdict.summary}) | "
            f"quality_mean_composite={corpus_quality.mean_composite_score:.4f} | "
            f"reasons={failure_reasons}"
        )
    elif evaluation_accepts and quality_passes:
        keep_summary_layer = True
        action = "enable_summary_layer"
        failure_reasons = []
        summary = (
            f"Summary layer decision: enable_summary_layer | "
            f"evaluation=accept_summary_layer | "
            f"quality_mean_composite={corpus_quality.mean_composite_score:.4f} "
            f"(>= {corpus_quality.min_required_composite:.2f} threshold) | "
            f"n_summaries={corpus_quality.n_summaries_scored} | "
            f"reasons=[]"
        )
    else:
        keep_summary_layer = False
        action = "disable_summary_layer"
        if not quality_passes:
            failure_reasons.append(
                f"Corpus quality gate failed: mean_composite_score="
                f"{corpus_quality.mean_composite_score:.4f} < "
                f"min_required_composite={corpus_quality.min_required_composite:.4f} "
                f"(n_summaries_scored={corpus_quality.n_summaries_scored})"
            )
        summary = (
            f"Summary layer decision: disable_summary_layer | "
            f"evaluation=accept_summary_layer | "
            f"quality=fail (mean_composite={corpus_quality.mean_composite_score:.4f}) | "
            f"reasons={failure_reasons}"
        )

    logger.info(
        "phase2_decision",
        action=action,
        keep_summary_layer=keep_summary_layer,
        evaluation_accepts=evaluation_accepts,
        quality_passes=quality_passes,
        evaluation_action=evaluation_verdict.action,
        mean_composite_score=corpus_quality.mean_composite_score,
        n_summaries=corpus_quality.n_summaries_scored,
        failure_reasons=failure_reasons,
    )

    return Phase2FinalVerdict(
        evaluation_verdict=evaluation_verdict,
        corpus_quality=corpus_quality,
        keep_summary_layer=keep_summary_layer,
        action=action,
        failure_reasons=failure_reasons,
        summary=summary,
    )


def emit_phase2_report(verdict: Phase2FinalVerdict) -> str:
    """Emit a human-readable summary-layer decision report.

    Args:
        verdict: The combined verdict.

    Returns:
        Plain-text multi-line report.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("SUMMARY LAYER STOP/GO DECISION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Action            : {verdict.action}")
    lines.append(f"Keep summary layer: {verdict.keep_summary_layer}")
    lines.append("")

    eval_v = verdict.evaluation_verdict
    lines.append("--- Evaluation Verdict (Mechanical + A/B) ---")
    lines.append(f"  action                : {eval_v.action}")
    lines.append(f"  summary_layer_accepted: {eval_v.summary_layer_accepted}")
    lines.append(f"  summary                : {eval_v.summary}")
    if eval_v.failure_reasons:
        for reason in eval_v.failure_reasons:
            lines.append(f"  failure_reason        : {reason}")
    else:
        lines.append("  failure_reasons       : []")
    lines.append("")

    cq = verdict.corpus_quality
    lines.append("--- Corpus Quality Gate ---")
    lines.append(f"  n_summaries_scored    : {cq.n_summaries_scored}")
    lines.append(f"  mean_coverage_rate    : {cq.mean_coverage_rate:.4f}")
    lines.append(f"  mean_faithfulness_rate: {cq.mean_faithfulness_rate:.4f}")
    lines.append(f"  mean_composite_score  : {cq.mean_composite_score:.4f}")
    lines.append(f"  min_required_composite: {cq.min_required_composite:.4f}")
    lines.append(f"  pass_gate             : {cq.pass_gate}")
    lines.append("")

    if verdict.failure_reasons:
        lines.append("--- Combined Failure Reasons ---")
        for reason in verdict.failure_reasons:
            lines.append(f"  * {reason}")
        lines.append("")
    else:
        lines.append("Failure reasons: []")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
