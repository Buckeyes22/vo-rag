"""ARES-style judge calibration via prediction-powered inference.

Pure math - no live LLM calls. Operates on already-computed JudgeScore records.

Refs:
  - Saad-Falcon et al. 2024 - ARES (arXiv:2311.09476)
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from pydantic import Field

from vo_rag.core.models.common import VoBaseModel

__all__ = [
    "CalibrationResult",
    "JudgeScore",
    "_normal_quantile",
    "calibration_acceptance",
    "compute_agreement_rate",
    "compute_ppi_calibration",
]


def _horner(coeffs: list[float], x: float) -> float:
    """Evaluate polynomial via Horner's method: sum(c[i]*x^(n-i-1) for i in range(n))."""
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result


# fmt: off
def _normal_quantile(p: float) -> float:
    """Standard normal quantile function (inverse CDF) via Acklam rational approx.

    Accurate to ~1.5e-7 absolute error across (0.0001, 0.9999).
    Based on Peter J. Acklam's algorithm, used by scipy.special.erfinv.
    """
    if not (0.0 < p < 1.0):
        msg = f"p must be in (0, 1), got {p}"
        raise ValueError(msg)

    a_coeffs = [
        -3.969683028665376e01, 2.209460984245205e02,
        -2.759285104469687e02, 1.383577518672690e02,
        -3.066479806614716e01, 2.506628277459239e00,
    ]
    b_coeffs = [
        -5.447609879822406e01, 1.615858368580409e02,
        -1.556989798598866e02, 6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c_coeffs = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e00, -2.549732539343734e00,
        4.374664141464968e00,  2.938163982698783e00,
    ]
    d_coeffs = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e00,  3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = _horner(c_coeffs, q)
        den = _horner(d_coeffs + [1.0], q)
        return num / den
    elif p < p_high:
        q = p - 0.5
        r = q * q
        poly_num = _horner(a_coeffs, r)
        poly_den = _horner(b_coeffs + [1.0], r)
        return q * poly_num / poly_den
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = _horner(c_coeffs, q)
        den = _horner(d_coeffs + [1.0], q)
        return -(num / den)
# fmt: on


class JudgeScore(VoBaseModel):
    """One query's judge score, optionally with a human label.

    Attributes:
        query_id: Unique identifier for the query.
        judge_label: LLM judge score (e.g. RAGAS faithfulness, 0-1).
        human_label: Hand-labeled score, or None if this query is not in the
            calibration subset.
    """

    query_id: str
    judge_label: float = Field(ge=0.0, le=1.0)
    human_label: float | None = Field(default=None, ge=0.0, le=1.0)


class CalibrationResult(VoBaseModel):
    """Result of ARES-style PPI calibration over a set of judged queries."""

    n_total: int = Field(ge=0)
    n_calibration: int = Field(ge=0)
    judge_mean: float
    ppi_estimate: float
    ppi_ci_lower: float
    ppi_ci_upper: float
    ci_width: float
    agreement_rate: float = Field(ge=0.0, le=1.0)
    agreement_tolerance: float


def compute_agreement_rate(scores: list[JudgeScore], tolerance: float = 0.1) -> float:
    """Fraction of calibration queries where judge and human agree within tolerance.

    Args:
        scores: List of JudgeScore records.
        tolerance: Maximum allowed absolute difference between judge_label and
            human_label for a match.

    Returns:
        Fraction of human-labeled scores within tolerance of judge scores.

    Raises:
        ValueError: If there is no calibration data (no records with human_label).
    """
    calibrated: list[JudgeScore] = [s for s in scores if s.human_label is not None]
    if not calibrated:
        raise ValueError("No calibration data: no records have human_label set")
    n_cal = len(calibrated)
    matches = sum(
        1 for s in calibrated if abs(s.judge_label - cast(float, s.human_label)) <= tolerance
    )
    return matches / n_cal


def compute_ppi_calibration(
    scores: list[JudgeScore], confidence: float = 0.95
) -> CalibrationResult:
    """Compute ARES-style PPI bias-corrected estimate and confidence interval.

    PPI rectifier (bias correction):
        delta = mean(Y_cal) - mean(f_cal)
        theta_hat = mean(f_all) + delta

    Variance (ARES Section 3.1, combined over full set and calibration subset):
        var = var(f_all) / N  +  var(Y_cal - f_cal) / n

    95% CI: theta_hat +/- z * sqrt(var) where z = norm.ppf(1 - (1-confidence)/2)

    When n == 0 (no calibration subset), PPI degrades to naive judge_mean
    with a wide fallback CI of width 2.0.

    Args:
        scores: All JudgeScore records (calibration + uncalibrated).
        confidence: Confidence level for CI (default 0.95 -> 95% CI).

    Returns:
        CalibrationResult with bias-corrected estimate and CI bounds.
    """
    n_total = len(scores)
    judge_labels = np.array([s.judge_label for s in scores])

    calibrated: list[JudgeScore] = [s for s in scores if s.human_label is not None]
    n_calibration = len(calibrated)

    judge_mean = float(np.mean(judge_labels))

    if n_calibration == 0:
        return CalibrationResult(
            n_total=n_total,
            n_calibration=0,
            judge_mean=judge_mean,
            ppi_estimate=judge_mean,
            ppi_ci_lower=judge_mean - 1.0,
            ppi_ci_upper=judge_mean + 1.0,
            ci_width=2.0,
            agreement_rate=0.0,
            agreement_tolerance=0.1,
        )

    f_cal = np.array([s.judge_label for s in calibrated])
    Y_cal = np.array([cast(float, s.human_label) for s in calibrated])

    delta = float(np.mean(Y_cal) - np.mean(f_cal))
    ppi_estimate = judge_mean + delta

    var_f = float(np.var(judge_labels, ddof=0))
    var_delta = float(np.var(Y_cal - f_cal, ddof=0))
    var_theta_hat = var_f / n_total + var_delta / n_calibration
    se = var_theta_hat**0.5

    z = _normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    ci_lower = ppi_estimate - z * se
    ci_upper = ppi_estimate + z * se
    ci_width = ci_upper - ci_lower

    try:
        agreement_rate = compute_agreement_rate(scores, tolerance=0.1)
    except ValueError:
        agreement_rate = 0.0

    return CalibrationResult(
        n_total=n_total,
        n_calibration=n_calibration,
        judge_mean=judge_mean,
        ppi_estimate=ppi_estimate,
        ppi_ci_lower=ci_lower,
        ppi_ci_upper=ci_upper,
        ci_width=ci_width,
        agreement_rate=agreement_rate,
        agreement_tolerance=0.1,
    )


def calibration_acceptance(
    result: CalibrationResult,
    *,
    min_agreement: float = 0.85,
    max_ci_width: float = 0.10,
) -> tuple[bool, list[str]]:
    """Check whether calibration result passes acceptance gates.

    Args:
        result: CalibrationResult from compute_ppi_calibration.
        min_agreement: Minimum required agreement_rate.
        max_ci_width: Maximum allowed ci_width.

    Returns:
        (passed, reasons): passed is True when both gates pass.
            reasons lists which gates failed when passed is False.
    """
    reasons: list[str] = []
    if result.agreement_rate < min_agreement:
        reasons.append(
            f"agreement_rate={result.agreement_rate:.3f} < min_agreement={min_agreement}"
        )
    if result.ci_width > max_ci_width:
        reasons.append(f"ci_width={result.ci_width:.3f} > max_ci_width={max_ci_width}")
    return (len(reasons) == 0, reasons)
