"""Shared retrieval metric math and non-inferiority tests."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from vo_rag.core.eval.judge_calibration import _normal_quantile
from vo_rag.core.models.common import VoBaseModel

__all__ = [
    "NonInferiorityResult",
    "compute_ndcg_at_k",
    "compute_non_inferiority",
    "compute_recall_at_k",
]


class NonInferiorityResult(VoBaseModel):
    """Result of a single-metric non-inferiority test.

    Attributes:
        metric: Metric name (e.g. "recall_at_10").
        n: Number of paired observations.
        control_mean: Mean of control scores.
        treatment_mean: Mean of treatment scores.
        delta_mean: treatment_mean - control_mean.
        delta_ci_low: Lower bound of the one-sided (1 - alpha) CI on delta.
        non_inferiority_margin_pp: Pre-registered margin in percentage points.
        decision: Non-inferiority verdict.
    """

    metric: str
    n: int = Field(ge=1)
    control_mean: float
    treatment_mean: float
    delta_mean: float
    delta_ci_low: float
    non_inferiority_margin_pp: float
    decision: Literal["non_inferior", "inferior", "inconclusive"]


def compute_recall_at_k(
    *,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Recall@k.

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids: Set of ground-truth relevant IDs.
        k: Cutoff depth.

    Returns:
        Fraction of relevant IDs present in the top-k retrieved IDs.
        Returns 0.0 when relevant_ids is empty.
    """
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / max(1, len(relevant_ids))


def compute_ndcg_at_k(
    *,
    retrieved_ids: list[str],
    relevance_judgments: dict[str, float],
    k: int = 10,
) -> float:
    """Compute nDCG@k using standard gain/discount formulation.

    Gain function: ``2**rel - 1`` (binary: 1.0 for rel=1, 0.0 otherwise).
    Discount:      ``1 / log2(rank + 1)`` where rank is 1-indexed.
    Ideal DCG:     computed from the same relevance_judgments sorted descending.

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevance_judgments: Mapping from ID to relevance score (typically 0/1).
        k: Cutoff depth.

    Returns:
        nDCG@k in [0.0, 1.0]. Returns 0.0 when IDCG is 0.
    """

    def _dcg(ids: list[str]) -> float:
        score = 0.0
        for rank, doc_id in enumerate(ids[:k], start=1):
            rel = relevance_judgments.get(doc_id, 0.0)
            gain = 2.0**rel - 1.0
            discount = math.log2(rank + 1)
            score += gain / discount
        return score

    dcg = _dcg(retrieved_ids)

    ideal_ids = sorted(
        relevance_judgments,
        key=lambda doc_id: relevance_judgments[doc_id],
        reverse=True,
    )
    idcg = _dcg(ideal_ids)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_non_inferiority(
    *,
    control_scores: list[float],
    treatment_scores: list[float],
    metric: str,
    margin_pp: float,
    confidence: float,
) -> NonInferiorityResult:
    """Paired non-inferiority test using a normal approximation.

    Deltas are computed as ``treatment - control`` per query. The one-sided
    CI lower bound is computed at level ``confidence`` using the inverse normal
    CDF from ``vo_rag.core.eval.judge_calibration``.

    Decision rule:
        non_inferior  - delta_ci_low > -(margin_pp / 100)
        inferior      - delta_ci_low < -(margin_pp / 100) AND delta_mean < 0
        inconclusive  - everything else

    Args:
        control_scores: Per-query scores for the control arm.
        treatment_scores: Per-query scores for the treatment arm.
        metric: Metric name used for the result record.
        margin_pp: Pre-registered non-inferiority margin in percentage points.
        confidence: Confidence level for the one-sided CI.

    Returns:
        NonInferiorityResult with the decision and supporting statistics.

    Raises:
        ValueError: If the two score lists have different lengths or are empty.
    """
    if len(control_scores) != len(treatment_scores):
        msg = (
            f"control_scores and treatment_scores must have the same length, "
            f"got {len(control_scores)} vs {len(treatment_scores)}"
        )
        raise ValueError(msg)
    if not control_scores:
        msg = "score lists must be non-empty"
        raise ValueError(msg)

    n = len(control_scores)
    deltas = [
        treatment - control
        for treatment, control in zip(treatment_scores, control_scores, strict=True)
    ]

    delta_mean = sum(deltas) / n
    control_mean = sum(control_scores) / n
    treatment_mean = sum(treatment_scores) / n

    var_delta = sum((delta - delta_mean) ** 2 for delta in deltas) / (n - 1) if n > 1 else 0.0

    se = math.sqrt(var_delta / n)

    z = _normal_quantile(confidence)
    delta_ci_low = delta_mean - z * se

    margin_frac = margin_pp / 100.0

    if delta_ci_low > -margin_frac:
        decision: Literal["non_inferior", "inferior", "inconclusive"] = "non_inferior"
    elif delta_ci_low < -margin_frac and delta_mean < 0:
        decision = "inferior"
    else:
        decision = "inconclusive"

    return NonInferiorityResult(
        metric=metric,
        n=n,
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        delta_mean=delta_mean,
        delta_ci_low=delta_ci_low,
        non_inferiority_margin_pp=margin_pp,
        decision=decision,
    )
