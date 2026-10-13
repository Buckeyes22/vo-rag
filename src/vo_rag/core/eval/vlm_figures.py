"""Grounded, reproducible evaluation for the resident figure specialist."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vo_rag.pipelines.book_rag.visual_sidecar import VLMDescription, VLMServiceLike


class FigureEvalCase(BaseModel):
    """One human-annotated figure and its minimal grounding expectations."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    image: str = Field(min_length=1)
    prompt: str = Field(default="Describe this figure for technical retrieval.", min_length=1)
    expected_page_kinds: list[
        Literal[
            "text",
            "diagram",
            "chart",
            "equation_heavy",
            "table",
            "photo",
            "mixed",
            "unclear",
        ]
    ] = Field(min_length=1)
    required_terms: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list
    )
    forbidden_terms: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list
    )


@dataclass(frozen=True)
class FigureEvalThresholds:
    """Acceptance thresholds for a staged resident-model rollout."""

    minimum_samples: int = 60
    minimum_success_rate: float = 0.95
    minimum_page_kind_accuracy: float = 0.80
    minimum_required_term_recall: float = 0.85
    maximum_forbidden_term_violations: int = 0
    maximum_needs_burst_rate: float = 0.25
    maximum_p95_latency_ms: int = 180_000

    def __post_init__(self) -> None:
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be >= 1")
        for name in (
            "minimum_success_rate",
            "minimum_page_kind_accuracy",
            "minimum_required_term_recall",
            "maximum_needs_burst_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.maximum_forbidden_term_violations < 0:
            raise ValueError("maximum_forbidden_term_violations must be >= 0")
        if self.maximum_p95_latency_ms < 1:
            raise ValueError("maximum_p95_latency_ms must be >= 1")


def load_figure_eval_cases(manifest: Path) -> list[FigureEvalCase]:
    """Load strict JSONL cases and resolve images relative to the manifest."""
    cases: list[FigureEvalCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        case = FigureEvalCase.model_validate(payload)
        if case.id in seen:
            raise ValueError(f"duplicate figure eval id at line {line_number}: {case.id}")
        seen.add(case.id)
        image = Path(case.image)
        if not image.is_absolute():
            image = (manifest.parent / image).resolve()
        if not image.is_file():
            raise ValueError(f"figure eval image is missing at line {line_number}: {image}")
        cases.append(case.model_copy(update={"image": str(image)}))
    if not cases:
        raise ValueError("figure evaluation manifest has no cases")
    return cases


def _contains_term(haystack: str, term: str) -> bool:
    return term.casefold().strip() in haystack.casefold()


def score_figure_result(case: FigureEvalCase, result: VLMDescription) -> dict[str, Any]:
    """Score model evidence only against explicit human annotations."""
    evidence = "\n".join(
        (
            result.description,
            *result.visible_text,
            *result.tables,
            *result.equations,
            *result.diagrams,
        )
    )
    required_hits = [term for term in case.required_terms if _contains_term(evidence, term)]
    forbidden_hits = [term for term in case.forbidden_terms if _contains_term(evidence, term)]
    required_recall = len(required_hits) / len(case.required_terms) if case.required_terms else 1.0
    return {
        "id": case.id,
        "ok": True,
        "page_kind": result.page_kind,
        "page_kind_correct": result.page_kind in case.expected_page_kinds,
        "required_term_recall": required_recall,
        "required_term_hits": required_hits,
        "forbidden_term_hits": forbidden_hits,
        "confidence": result.confidence,
        "needs_burst": result.needs_burst,
        "needs_burst_reasons": list(result.needs_burst_reasons),
        "attempt_count": result.attempt_count,
        "duration_ms": result.duration_ms,
        "model_used": result.model_used,
        "description": result.description,
    }


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize_figure_eval(
    results: list[dict[str, Any]],
    thresholds: FigureEvalThresholds,
) -> dict[str, Any]:
    """Aggregate case results and evaluate the rollout gate."""
    total = len(results)
    successful = [result for result in results if result.get("ok") is True]
    success_count = len(successful)
    success_rate = success_count / total if total else 0.0
    page_kind_accuracy = (
        sum(bool(result["page_kind_correct"]) for result in successful) / success_count
        if success_count
        else 0.0
    )
    required_term_recall = (
        sum(float(result["required_term_recall"]) for result in successful) / success_count
        if success_count
        else 0.0
    )
    forbidden_violations = sum(len(result["forbidden_term_hits"]) for result in successful)
    needs_burst_rate = (
        sum(bool(result["needs_burst"]) for result in successful) / success_count
        if success_count
        else 1.0
    )
    p95_latency_ms = _percentile(
        [int(result["duration_ms"]) for result in successful],
        0.95,
    )
    checks = {
        "minimum_samples": total >= thresholds.minimum_samples,
        "success_rate": success_rate >= thresholds.minimum_success_rate,
        "page_kind_accuracy": page_kind_accuracy >= thresholds.minimum_page_kind_accuracy,
        "required_term_recall": (required_term_recall >= thresholds.minimum_required_term_recall),
        "forbidden_term_violations": (
            forbidden_violations <= thresholds.maximum_forbidden_term_violations
        ),
        "needs_burst_rate": needs_burst_rate <= thresholds.maximum_needs_burst_rate,
        "p95_latency_ms": p95_latency_ms <= thresholds.maximum_p95_latency_ms,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": asdict(thresholds),
        "sample_count": total,
        "success_count": success_count,
        "success_rate": success_rate,
        "page_kind_accuracy": page_kind_accuracy,
        "required_term_recall": required_term_recall,
        "forbidden_term_violations": forbidden_violations,
        "needs_burst_rate": needs_burst_rate,
        "p95_latency_ms": p95_latency_ms,
    }


async def run_figure_eval(
    cases: list[FigureEvalCase],
    service: VLMServiceLike,
    *,
    max_tokens: int,
    maximum_image_bytes: int = 32 * 1024 * 1024,
) -> list[dict[str, Any]]:
    """Run cases sequentially so the evaluation matches the one-slot service."""
    results: list[dict[str, Any]] = []
    for case in cases:
        path = Path(case.image)
        if path.stat().st_size > maximum_image_bytes:
            results.append({"id": case.id, "ok": False, "error": "image_too_large"})
            continue
        try:
            description = await service.describe_image(
                await asyncio.to_thread(path.read_bytes),
                case.prompt,
                case.id,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - evaluation records per-case failures
            results.append({"id": case.id, "ok": False, "error": type(exc).__name__})
            continue
        results.append(score_figure_result(case, description))
    return results
