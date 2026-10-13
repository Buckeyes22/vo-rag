"""Summary key-point coverage and faithfulness scoring.

LLM-judge-based scoring with injectable judge callable:
  - coverage_rate   = fraction of human-labeled key points present in summary
  - faithfulness_rate = fraction of summary claims grounded in source chunks

Calibrated lower-bound CIs use the PPI inverse-normal helper from
``vo_rag.core.eval.judge_calibration`` (Beasley-Springer/Moro / Acklam rational approx).

Thresholds: coverage >= 0.75, distortion <= 0.10, unsupported <= 0.05. The
function inputs are open; the corresponding corpus-quality gate in
``phase2_aggregator`` enforces them via ``min_required_composite``.

Do NOT call live LLMs here — ``llm_call`` is the injection point.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.eval.judge_calibration import _normal_quantile
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.common import VoBaseModel

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Frozen prompt constants
# ---------------------------------------------------------------------------

COVERAGE_JUDGE_PROMPT_VERSION = "v1.0"

COVERAGE_JUDGE_SYSTEM_PROMPT = (
    "You are a rigorous evaluation judge. You will be shown a summary text and a single"
    " key point. Your task is to decide whether the summary meaningfully covers that key"
    " point.\n\n"
    "Respond with a JSON object with exactly these fields:\n"
    '  "covered": true or false\n'
    '  "confidence": a float in [0, 1] reflecting how certain you are\n'
    '  "evidence_in_summary": if covered is true, a short verbatim substring from the'
    " summary that demonstrates coverage; otherwise null\n\n"
    "Do not add any fields or prose outside the JSON object."
)

FAITHFULNESS_JUDGE_SYSTEM_PROMPT = (
    "You are a rigorous evaluation judge. You will be shown a claim extracted from a"
    " summary and a list of source text chunks. Your task is to decide whether the claim"
    " is directly supported by the source chunks.\n\n"
    "Respond with a JSON object with exactly these fields:\n"
    '  "grounded": true or false\n'
    '  "confidence": a float in [0, 1] reflecting how certain you are\n'
    '  "supporting_chunk_ids": a JSON array of chunk_id strings from the provided chunks'
    " that support the claim; empty array if grounded is false\n\n"
    "Do not add any fields or prose outside the JSON object."
)

FAITHFULNESS_JUDGE_PROMPT_VERSION = "v1.0"

COVERAGE_PROMPT_HASH: str = hashlib.sha256(COVERAGE_JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()

FAITHFULNESS_PROMPT_HASH: str = hashlib.sha256(
    FAITHFULNESS_JUDGE_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class CoverageJudgeConfig(VoBaseModel):
    judge_model: str = "qwen3.6-27b-fp8"
    temperature: float = 0.1
    max_tokens: int = 1500
    prompt_hash: str = COVERAGE_PROMPT_HASH


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


class KeyPoint(VoBaseModel):
    key_point_id: str
    text: str  # human-labeled or LLM-extracted gold key point
    source_chunk_ids: list[str]  # provenance for the gold


class CoverageJudgment(VoBaseModel):
    key_point_id: str
    covered: bool  # judge says summary mentions this key point
    confidence: float  # judge confidence 0..1
    evidence_in_summary: str | None  # excerpted substring from summary if covered


class FaithfulnessClaim(VoBaseModel):
    claim_id: str
    claim_text: str  # extracted from the summary
    grounded: bool  # judge says claim is supported by source chunks
    confidence: float
    supporting_chunk_ids: list[str]  # which source chunks support this claim


class CoverageResult(VoBaseModel):
    summary_node_id: str
    n_key_points_total: int
    n_key_points_covered: int
    coverage_rate: float  # n_covered / n_total
    per_key_point_judgments: list[CoverageJudgment]
    judge_prompt_hash: str


class FaithfulnessResult(VoBaseModel):
    summary_node_id: str
    n_claims_extracted: int
    n_claims_grounded: int
    faithfulness_rate: float  # n_grounded / n_claims
    per_claim_judgments: list[FaithfulnessClaim]
    judge_prompt_hash: str


class SummaryQualityScore(VoBaseModel):
    summary_node_id: str
    coverage: CoverageResult
    faithfulness: FaithfulnessResult
    composite_score: float  # 0.5*coverage + 0.5*faithfulness (simple blend)
    composite_ci_low: float  # lower-bound CI via PPI normal-approximation


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


def extract_claims_from_summary(summary_text: str) -> list[str]:
    """Split summary into individual claim strings on '. ' boundaries.

    v1 is intentionally naive — a fine-tuned claim-extractor replaces this
    in a future version. Each non-empty sentence fragment is returned as one
    claim string.

    Args:
        summary_text: Full summary text to decompose.

    Returns:
        List of claim strings (one per sentence-like segment).
    """
    raw_parts = summary_text.split(". ")
    claims: list[str] = []
    for part in raw_parts:
        stripped = part.strip()
        if stripped:
            claims.append(stripped)
    return claims


# ---------------------------------------------------------------------------
# Internal JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_coverage_response(raw: str) -> dict[str, Any]:
    """Extract JSON object from raw LLM text."""
    import json

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {"covered": False, "confidence": 0.0, "evidence_in_summary": None}
    try:
        parsed = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {"covered": False, "confidence": 0.0, "evidence_in_summary": None}
    if not isinstance(parsed, dict):
        return {"covered": False, "confidence": 0.0, "evidence_in_summary": None}
    return parsed


def _parse_faithfulness_response(raw: str) -> dict[str, Any]:
    """Extract JSON object from raw LLM text."""
    import json

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {"grounded": False, "confidence": 0.0, "supporting_chunk_ids": []}
    try:
        parsed = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {"grounded": False, "confidence": 0.0, "supporting_chunk_ids": []}
    if not isinstance(parsed, dict):
        return {"grounded": False, "confidence": 0.0, "supporting_chunk_ids": []}
    return parsed


# ---------------------------------------------------------------------------
# Core async judge functions
# ---------------------------------------------------------------------------


async def judge_coverage(
    *,
    key_points: list[KeyPoint],
    summary_text: str,
    summary_node_id: str,
    config: CoverageJudgeConfig,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
) -> CoverageResult:
    """Score how many of the given key points appear in the summary.

    For each key point a separate LLM judge call is made. The judge is asked
    to respond YES/NO (covered/not) plus a confidence score and an excerpted
    evidence substring when covered.

    Args:
        key_points: Human-labeled or LLM-extracted gold key points.
        summary_text: The generated summary to evaluate.
        summary_node_id: Stable ID of the SummaryNode being evaluated.
        config: Judge model/sampling configuration.
        llm_call: Injectable callable — signature (system, user, max_tokens, temperature)
            returning LLMResponse.

    Returns:
        CoverageResult with per-key-point judgments and aggregate rate.
    """
    judgments: list[CoverageJudgment] = []

    for kp in key_points:
        user_message = f"KEY POINT:\n{kp.text}\n\nSUMMARY:\n{summary_text}"
        log.debug(
            "judge_coverage.call",
            summary_node_id=summary_node_id,
            key_point_id=kp.key_point_id,
        )
        response = await llm_call(
            COVERAGE_JUDGE_SYSTEM_PROMPT,
            user_message,
            config.max_tokens,
            config.temperature,
        )
        parsed = _parse_coverage_response(response.text)
        covered = bool(parsed.get("covered", False))
        confidence = float(parsed.get("confidence", 0.0))
        evidence = parsed.get("evidence_in_summary")
        if not isinstance(evidence, str):
            evidence = None

        judgments.append(
            CoverageJudgment(
                key_point_id=kp.key_point_id,
                covered=covered,
                confidence=max(0.0, min(1.0, confidence)),
                evidence_in_summary=evidence,
            )
        )

    n_total = len(key_points)
    n_covered = sum(1 for j in judgments if j.covered)
    coverage_rate = n_covered / n_total if n_total > 0 else 0.0

    log.info(
        "judge_coverage.done",
        summary_node_id=summary_node_id,
        n_total=n_total,
        n_covered=n_covered,
        coverage_rate=coverage_rate,
    )

    return CoverageResult(
        summary_node_id=summary_node_id,
        n_key_points_total=n_total,
        n_key_points_covered=n_covered,
        coverage_rate=coverage_rate,
        per_key_point_judgments=judgments,
        judge_prompt_hash=COVERAGE_PROMPT_HASH,
    )


async def judge_faithfulness(
    *,
    summary_text: str,
    summary_node_id: str,
    source_chunks: list[RetrievalChunk],
    config: CoverageJudgeConfig,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
) -> FaithfulnessResult:
    """Score how many of the summary's claims are grounded in source chunks.

    Claims are extracted via extract_claims_from_summary (naive sentence split).
    Each claim is judged independently against all provided source chunks.

    Args:
        summary_text: The generated summary to evaluate.
        summary_node_id: Stable ID of the SummaryNode being evaluated.
        source_chunks: Raw retrieval chunks that were the evidence basis for the summary.
        config: Judge model/sampling configuration.
        llm_call: Injectable callable — signature (system, user, max_tokens, temperature)
            returning LLMResponse.

    Returns:
        FaithfulnessResult with per-claim judgments and aggregate rate.
    """
    claims = extract_claims_from_summary(summary_text)
    chunk_context = "\n\n".join(f"[chunk_id={c.chunk_id}]\n{c.content}" for c in source_chunks)
    claim_judgments: list[FaithfulnessClaim] = []

    for idx, claim_text in enumerate(claims):
        claim_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{summary_node_id}|claim|{idx}|{claim_text}")
        )
        user_message = f"CLAIM:\n{claim_text}\n\nSOURCE CHUNKS:\n{chunk_context}"
        log.debug(
            "judge_faithfulness.call",
            summary_node_id=summary_node_id,
            claim_id=claim_id,
            claim_index=idx,
        )
        response = await llm_call(
            FAITHFULNESS_JUDGE_SYSTEM_PROMPT,
            user_message,
            config.max_tokens,
            config.temperature,
        )
        parsed = _parse_faithfulness_response(response.text)
        grounded = bool(parsed.get("grounded", False))
        confidence = float(parsed.get("confidence", 0.0))
        supporting = parsed.get("supporting_chunk_ids", [])
        if not isinstance(supporting, list):
            supporting = []
        supporting_ids: list[str] = [str(s) for s in supporting]

        claim_judgments.append(
            FaithfulnessClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                grounded=grounded,
                confidence=max(0.0, min(1.0, confidence)),
                supporting_chunk_ids=supporting_ids,
            )
        )

    n_claims = len(claims)
    n_grounded = sum(1 for j in claim_judgments if j.grounded)
    faithfulness_rate = n_grounded / n_claims if n_claims > 0 else 0.0

    log.info(
        "judge_faithfulness.done",
        summary_node_id=summary_node_id,
        n_claims=n_claims,
        n_grounded=n_grounded,
        faithfulness_rate=faithfulness_rate,
    )

    return FaithfulnessResult(
        summary_node_id=summary_node_id,
        n_claims_extracted=n_claims,
        n_claims_grounded=n_grounded,
        faithfulness_rate=faithfulness_rate,
        per_claim_judgments=claim_judgments,
        judge_prompt_hash=FAITHFULNESS_PROMPT_HASH,
    )


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def compute_summary_quality_score(
    *,
    coverage: CoverageResult,
    faithfulness: FaithfulnessResult,
    ppi_confidence: float = 0.95,
) -> SummaryQualityScore:
    """Blend coverage + faithfulness into a composite score with a PPI-style CI lower bound.

    composite = 0.5 * coverage_rate + 0.5 * faithfulness_rate

    The lower-bound CI uses a normal-approximation over the per-judgment
    confidence scores (treating them as a calibration signal, analogous to
    ARES PPI). The standard error is estimated from the variance of the
    individual confidences; the CI lower bound is then:

        composite_ci_low = composite - z * se

    where z = _normal_quantile(1 - (1 - ppi_confidence) / 2).

    This mirrors the pattern in judge_calibration.compute_ppi_calibration but
    operates on the composite point estimate rather than a separate
    bias-corrected PPI estimate (no paired human labels available here).

    Args:
        coverage: CoverageResult from judge_coverage.
        faithfulness: FaithfulnessResult from judge_faithfulness.
        ppi_confidence: Desired confidence level for the CI (default 0.95).

    Returns:
        SummaryQualityScore with composite_score and composite_ci_low.
    """
    composite = 0.5 * coverage.coverage_rate + 0.5 * faithfulness.faithfulness_rate

    # Gather all per-judgment confidences as the calibration signal.
    all_confidences: list[float] = [j.confidence for j in coverage.per_key_point_judgments] + [
        j.confidence for j in faithfulness.per_claim_judgments
    ]

    n = len(all_confidences)
    if n > 1:
        mean_conf = sum(all_confidences) / n
        variance = sum((c - mean_conf) ** 2 for c in all_confidences) / n
        se = (variance / n) ** 0.5
        z = _normal_quantile(1.0 - (1.0 - ppi_confidence) / 2.0)
        ci_low = composite - z * se
    else:
        # Degenerate case: no calibration signal, use a wide fallback.
        ci_low = composite - 1.0

    return SummaryQualityScore(
        summary_node_id=coverage.summary_node_id,
        coverage=coverage,
        faithfulness=faithfulness,
        composite_score=composite,
        composite_ci_low=ci_low,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "COVERAGE_JUDGE_PROMPT_VERSION",
    "COVERAGE_JUDGE_SYSTEM_PROMPT",
    "COVERAGE_PROMPT_HASH",
    "FAITHFULNESS_JUDGE_PROMPT_VERSION",
    "FAITHFULNESS_JUDGE_SYSTEM_PROMPT",
    "FAITHFULNESS_PROMPT_HASH",
    "CoverageJudgeConfig",
    "KeyPoint",
    "CoverageJudgment",
    "FaithfulnessClaim",
    "CoverageResult",
    "FaithfulnessResult",
    "SummaryQualityScore",
    "extract_claims_from_summary",
    "judge_coverage",
    "judge_faithfulness",
    "compute_summary_quality_score",
]
