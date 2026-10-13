"""STOP/GO decision aggregator for the entity-relationship graph layer.

Combines smoke-test, quality-A/B, and overhead gates into a single unified verdict.
"""

from __future__ import annotations

from typing import Literal

import structlog

from vo_rag.core.graph.quality_ab import GraphPhase6bVerdict
from vo_rag.core.models.common import VoBaseModel

__all__ = [
    "GraphSmokeResult",
    "GraphOverheadResult",
    "Phase6bFinalVerdict",
    "decide_phase6b",
    "emit_phase6b_report",
]

logger = structlog.get_logger(__name__)


class GraphSmokeResult(VoBaseModel):
    """Result of the smoke-test gate for the graph layer."""

    passed: bool
    n_nodes_built: int
    n_edges_built: int
    min_one_metric_improved: bool
    improvement_breakdown: dict[str, float]


class GraphOverheadResult(VoBaseModel):
    """Result of the overhead gate for the graph layer."""

    avg_planner_latency_ms: float
    baseline_planner_latency_ms: float
    latency_overhead_pp: float
    max_acceptable_overhead_pp: float = 20.0
    index_size_bytes: int
    pass_gate: bool = False

    @property
    def _computed_pass_gate(self) -> bool:
        """True when latency overhead is within acceptable bounds."""
        return self.latency_overhead_pp <= self.max_acceptable_overhead_pp

    def model_post_init(self, *args: object, **kwargs: object) -> None:
        super().model_post_init(*args, **kwargs)
        self.pass_gate = self._computed_pass_gate


class Phase6bFinalVerdict(VoBaseModel):
    """Combined STOP/GO verdict."""

    smoke: GraphSmokeResult
    quality_ab: GraphPhase6bVerdict
    overhead: GraphOverheadResult
    keep_graph_in_default_planner: bool
    action: Literal["enable_graph_in_planner", "disable_graph_path", "rerun_after_fixes"]
    failure_reasons: list[str]
    summary: str


def decide_phase6b(
    *,
    smoke: GraphSmokeResult,
    quality_ab: GraphPhase6bVerdict,
    overhead: GraphOverheadResult,
) -> Phase6bFinalVerdict:
    """Determine the graph-layer STOP/GO decision.

    Args:
        smoke: Smoke-test gate result.
        quality_ab: Quality A/B evaluation result.
        overhead: Overhead gate result.

    Returns:
        Phase6bFinalVerdict with combined gate outcomes and recommended action.
    """
    failure_reasons: list[str] = []

    smoke_gate = smoke.passed and smoke.min_one_metric_improved
    quality_gate = quality_ab.keep_graph_path
    overhead_gate = overhead.pass_gate

    if not smoke_gate:
        if not smoke.passed:
            failure_reasons.append(f"smoke test failed: passed={smoke.passed}")
        if not smoke.min_one_metric_improved:
            failure_reasons.append(
                f"no metric improved in smoke test: "
                f"improvement_breakdown={smoke.improvement_breakdown}"
            )

    if not quality_gate:
        failure_reasons.append(
            f"quality A/B gate rejected graph path: keep_graph_path={quality_ab.keep_graph_path}"
        )

    if not overhead_gate:
        failure_reasons.append(
            f"overhead gate failed: latency_overhead_pp={overhead.latency_overhead_pp:.2f} "
            f"exceeds max_acceptable_overhead_pp={overhead.max_acceptable_overhead_pp:.2f}"
        )

    all_gates_pass = smoke_gate and quality_gate and overhead_gate

    if all_gates_pass:
        keep_graph_in_default_planner = True
        action: Literal["enable_graph_in_planner", "disable_graph_path", "rerun_after_fixes"] = (
            "enable_graph_in_planner"
        )
        failure_reasons = []
        summary = (
            "Graph decision: enable_graph_in_planner | "
            "smoke=pass | quality_ab=pass | overhead=pass | reasons=[]"
        )
    elif not smoke_gate:
        keep_graph_in_default_planner = False
        action = "disable_graph_path"
        summary = (
            f"Graph decision: disable_graph_path | "
            f"smoke=fail | quality_ab={'pass' if quality_gate else 'fail'} | "
            f"overhead={'pass' if overhead_gate else 'fail'} | "
            f"reasons={failure_reasons}"
        )
    elif not quality_gate:
        keep_graph_in_default_planner = False
        action = "disable_graph_path"
        summary = (
            f"Graph decision: disable_graph_path | "
            f"smoke=pass | quality_ab=fail | "
            f"overhead={'pass' if overhead_gate else 'fail'} | "
            f"reasons={failure_reasons}"
        )
    else:
        keep_graph_in_default_planner = False
        action = "rerun_after_fixes"
        summary = (
            f"Graph decision: rerun_after_fixes | "
            f"smoke=pass | quality_ab=pass | overhead=fail | "
            f"reasons={failure_reasons}"
        )

    logger.info(
        "phase6b_decision",
        action=action,
        keep_graph_in_default_planner=keep_graph_in_default_planner,
        smoke_pass=smoke_gate,
        quality_ab_pass=quality_gate,
        overhead_pass=overhead_gate,
        failure_reasons=failure_reasons,
    )

    return Phase6bFinalVerdict(
        smoke=smoke,
        quality_ab=quality_ab,
        overhead=overhead,
        keep_graph_in_default_planner=keep_graph_in_default_planner,
        action=action,
        failure_reasons=failure_reasons,
        summary=summary,
    )


def emit_phase6b_report(verdict: Phase6bFinalVerdict) -> str:
    """Emit a human-readable graph-layer decision report.

    Args:
        verdict: The combined verdict.

    Returns:
        Plain-text multi-line report.
    """
    lines: list[str] = []
    lines.append(f"{'=' * 60}")
    lines.append("GRAPH LAYER STOP/GO DECISION REPORT")
    lines.append(f"{'=' * 60}")
    lines.append("")
    lines.append(f"Action       : {verdict.action}")
    lines.append(f"Keep graph   : {verdict.keep_graph_in_default_planner}")
    lines.append("")

    smoke = verdict.smoke
    lines.append("--- Smoke Gate ---")
    lines.append(f"  passed                  : {smoke.passed}")
    lines.append(f"  n_nodes_built           : {smoke.n_nodes_built}")
    lines.append(f"  n_edges_built           : {smoke.n_edges_built}")
    lines.append(f"  min_one_metric_improved : {smoke.min_one_metric_improved}")
    lines.append(f"  improvement_breakdown    : {smoke.improvement_breakdown}")
    lines.append("")

    quality_ab = verdict.quality_ab
    lines.append("--- Quality A/B Gate ---")
    lines.append(f"  keep_graph_path : {quality_ab.keep_graph_path}")
    lines.append(f"  action           : {quality_ab.action}")
    lines.append(f"  n_queries_graph : {quality_ab.n_queries_using_graph}")
    lines.append(f"  summary          : {quality_ab.summary}")
    lines.append("")

    overhead = verdict.overhead
    lines.append("--- Overhead Gate ---")
    lines.append(f"  pass_gate                  : {overhead.pass_gate}")
    lines.append(f"  avg_planner_latency_ms    : {overhead.avg_planner_latency_ms:.2f}")
    lines.append(f"  baseline_planner_latency  : {overhead.baseline_planner_latency_ms:.2f}")
    lines.append(f"  latency_overhead_pp       : {overhead.latency_overhead_pp:.2f}")
    lines.append(f"  max_acceptable_overhead_pp: {overhead.max_acceptable_overhead_pp:.2f}")
    lines.append(f"  index_size_bytes          : {overhead.index_size_bytes}")
    lines.append("")

    reasons = verdict.failure_reasons if verdict.failure_reasons else ["All gates passed."]
    lines.append(f"Failure reasons: {reasons}")
    lines.append(f"{'=' * 60}")

    return "\n".join(lines)
