"""Pre-RAPTOR-build cost/token-budget estimator (VOR-4).

Walks the cluster corpus projection, estimates token + wall-time + dollar costs
for clustering + LLM-summary across the full RAPTOR tree, and emits an explicit
GO/STOP verdict based on configured budget caps.

No live LLM calls are made — projection only.
"""

from __future__ import annotations

import math
from typing import Literal

from vo_rag.core.models.common import VoBaseModel


class RaptorBudgetConfig(VoBaseModel):
    """Operator-configurable budget caps and cost/throughput parameters."""

    max_total_input_tokens: int = 50_000_000
    max_total_output_tokens: int = 5_000_000
    max_wall_time_hours: float = 24.0
    max_cost_usd: float = 200.0

    summary_model: str = "qwen3.6-27b-fp8"

    input_cost_per_million_tokens_usd: float = 0.40
    output_cost_per_million_tokens_usd: float = 1.20

    avg_tokens_per_chunk: int = 384
    avg_summary_input_tokens: int = 6000
    avg_summary_output_tokens: int = 600
    avg_summary_wall_time_s: float = 8.0
    concurrent_llm_calls: int = 16


class RaptorBudgetProjection(VoBaseModel):
    """Per-level and aggregate token/cost/time projection for a RAPTOR build."""

    n_total_chunks: int
    n_clusters_per_level: dict[str, int]
    total_input_tokens: int
    total_output_tokens: int
    projected_wall_time_hours: float
    projected_cost_usd: float
    breakdown_per_level: dict[str, dict[str, float]]


class RaptorPreflightVerdict(VoBaseModel):
    """GO/STOP verdict with violation details and a human-readable summary."""

    projection: RaptorBudgetProjection
    budget: RaptorBudgetConfig
    proceed: bool
    action: Literal["proceed", "abort_over_budget", "reduce_scope"]
    violations: list[str]
    summary: str


def project_n_clusters(
    *,
    n_total_chunks: int,
    level_0_target_size: int = 80,
    level_1_target_size: int = 800,
    level_2_target_size: int = 6000,
) -> dict[str, int]:
    """Return projected cluster counts for each RAPTOR level.

    LEVEL_0: ceil(n_total_chunks / level_0_target_size)
    LEVEL_1: ceil(n_total_chunks / level_1_target_size)
    LEVEL_2: ceil(n_total_chunks / level_2_target_size)
    """
    return {
        "level_0": math.ceil(n_total_chunks / level_0_target_size),
        "level_1": math.ceil(n_total_chunks / level_1_target_size),
        "level_2": math.ceil(n_total_chunks / level_2_target_size),
    }


def project_budget(
    *,
    n_total_chunks: int,
    config: RaptorBudgetConfig,
) -> RaptorBudgetProjection:
    """Project token, cost, and wall-time budgets across all RAPTOR levels.

    For each level:
      - n_summaries  = n_clusters at that level
      - input_tokens = n_summaries * avg_summary_input_tokens
      - output_tokens = n_summaries * avg_summary_output_tokens
      - cost_usd = (input * input_cost / 1e6) + (output * output_cost / 1e6)
      - wall_time_hours = (n_summaries * avg_summary_wall_time_s)
                          / concurrent_llm_calls / 3600
    """
    n_clusters = project_n_clusters(n_total_chunks=n_total_chunks)

    breakdown: dict[str, dict[str, float]] = {}
    total_input = 0
    total_output = 0
    total_wall_s = 0.0
    total_cost = 0.0

    for level_key, n_clusters_at_level in n_clusters.items():
        input_tok = n_clusters_at_level * config.avg_summary_input_tokens
        output_tok = n_clusters_at_level * config.avg_summary_output_tokens
        cost = (
            input_tok * config.input_cost_per_million_tokens_usd / 1_000_000
            + output_tok * config.output_cost_per_million_tokens_usd / 1_000_000
        )
        wall_time_h = (
            (n_clusters_at_level * config.avg_summary_wall_time_s)
            / config.concurrent_llm_calls
            / 3600
        )

        breakdown[level_key] = {
            "input_tokens": float(input_tok),
            "output_tokens": float(output_tok),
            "wall_time_hours": wall_time_h,
            "cost_usd": cost,
        }

        total_input += input_tok
        total_output += output_tok
        total_wall_s += n_clusters_at_level * config.avg_summary_wall_time_s
        total_cost += cost

    total_wall_h = total_wall_s / config.concurrent_llm_calls / 3600

    return RaptorBudgetProjection(
        n_total_chunks=n_total_chunks,
        n_clusters_per_level=n_clusters,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        projected_wall_time_hours=total_wall_h,
        projected_cost_usd=total_cost,
        breakdown_per_level=breakdown,
    )


def decide_preflight(
    *,
    projection: RaptorBudgetProjection,
    budget: RaptorBudgetConfig,
) -> RaptorPreflightVerdict:
    """Evaluate a projection against budget caps and return a GO/STOP verdict.

    - No violations        -> action="proceed", proceed=True
    - Single-axis breach   -> action="reduce_scope"
    - Multi-axis breach    -> action="abort_over_budget"
    """
    violations: list[str] = []

    if projection.total_input_tokens > budget.max_total_input_tokens:
        violations.append(
            f"total_input_tokens {projection.total_input_tokens:,} "
            f"exceeds cap {budget.max_total_input_tokens:,}"
        )
    if projection.total_output_tokens > budget.max_total_output_tokens:
        violations.append(
            f"total_output_tokens {projection.total_output_tokens:,} "
            f"exceeds cap {budget.max_total_output_tokens:,}"
        )
    if projection.projected_wall_time_hours > budget.max_wall_time_hours:
        violations.append(
            f"projected_wall_time_hours {projection.projected_wall_time_hours:.2f} "
            f"exceeds cap {budget.max_wall_time_hours:.2f}"
        )
    if projection.projected_cost_usd > budget.max_cost_usd:
        violations.append(
            f"projected_cost_usd {projection.projected_cost_usd:.4f} "
            f"exceeds cap {budget.max_cost_usd:.2f}"
        )

    n_violations = len(violations)
    if n_violations == 0:
        action: Literal["proceed", "abort_over_budget", "reduce_scope"] = "proceed"
    elif n_violations == 1:
        action = "reduce_scope"
    else:
        action = "abort_over_budget"

    proceed = n_violations == 0

    if proceed:
        summary = (
            f"All budget caps satisfied for {projection.n_total_chunks:,} chunks "
            f"({sum(projection.n_clusters_per_level.values()):,} total summaries). "
            f"Estimated cost: ${projection.projected_cost_usd:.4f}, "
            f"wall time: {projection.projected_wall_time_hours:.2f}h. "
            f"RAPTOR build may proceed."
        )
    else:
        cap_list = "; ".join(violations)
        summary = (
            f"Budget caps exceeded for {projection.n_total_chunks:,} chunks: {cap_list}. "
            f"Action: {action}."
        )

    return RaptorPreflightVerdict(
        projection=projection,
        budget=budget,
        proceed=proceed,
        action=action,
        violations=violations,
        summary=summary,
    )


def emit_preflight_report(verdict: RaptorPreflightVerdict) -> str:
    """Render a multi-line plain-text preflight report with per-level breakdown."""
    p = verdict.projection
    b = verdict.budget

    lines: list[str] = [
        "=" * 72,
        "RAPTOR PREFLIGHT REPORT",
        "=" * 72,
        f"Corpus size      : {p.n_total_chunks:,} chunks",
        f"Summary model    : {b.summary_model}",
        f"Concurrency      : {b.concurrent_llm_calls} parallel LLM calls",
        "",
        "Per-level projection:",
        f"  {'Level':<12} {'Clusters':>10} {'Input tok':>12} {'Output tok':>12} "
        f"{'Wall (h)':>10} {'Cost ($)':>10}",
        "  " + "-" * 68,
    ]

    for level_key, n in p.n_clusters_per_level.items():
        bd = p.breakdown_per_level[level_key]
        lines.append(
            f"  {level_key:<12} {n:>10,} {int(bd['input_tokens']):>12,} "
            f"{int(bd['output_tokens']):>12,} {bd['wall_time_hours']:>10.2f} "
            f"{bd['cost_usd']:>10.4f}"
        )

    lines += [
        "  " + "-" * 68,
        f"  {'TOTAL':<12} {sum(p.n_clusters_per_level.values()):>10,} "
        f"{p.total_input_tokens:>12,} {p.total_output_tokens:>12,} "
        f"{p.projected_wall_time_hours:>10.2f} {p.projected_cost_usd:>10.4f}",
        "",
        "Budget caps:",
        f"  max_total_input_tokens   : {b.max_total_input_tokens:,}",
        f"  max_total_output_tokens  : {b.max_total_output_tokens:,}",
        f"  max_wall_time_hours      : {b.max_wall_time_hours:.2f}",
        f"  max_cost_usd             : ${b.max_cost_usd:.2f}",
        "",
        f"Verdict          : {verdict.action.upper()}",
        f"Proceed          : {verdict.proceed}",
    ]

    if verdict.violations:
        lines.append("Violations:")
        for v in verdict.violations:
            lines.append(f"  - {v}")
    else:
        lines.append("Violations       : none")

    lines += [
        "",
        verdict.summary,
        "=" * 72,
    ]

    return "\n".join(lines)


__all__ = [
    "RaptorBudgetConfig",
    "RaptorBudgetProjection",
    "RaptorPreflightVerdict",
    "project_n_clusters",
    "project_budget",
    "decide_preflight",
    "emit_preflight_report",
]
