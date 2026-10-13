"""STOP/GO gate — planner-with-graph vs planner-without-graph A/B test.

Decides whether the GRAPH path stays enabled or is silently disabled at the
planner level.
"""

from __future__ import annotations

from typing import Any, Literal

import structlog

from vo_rag.core.eval.non_inferiority import compute_non_inferiority
from vo_rag.core.models.common import VoBaseModel

log = structlog.get_logger(__name__)

__all__ = [
    "GraphMechanicalResult",
    "GraphPerQueryScore",
    "GraphPhase6bVerdict",
    "evaluate_graph",
    "emit_graph_report",
]

# Minimum number of queries that must use the graph path before the A/B
# comparison is meaningful.
_MIN_GRAPH_QUERIES = 10


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class GraphMechanicalResult(VoBaseModel):
    """Mechanical STOP/GO gate for the entity-relationship graph layer.

    Attributes:
        n_books_processed: Number of books passed to graph extraction.
        n_books_extraction_failed: Number of books where extraction failed.
        n_total_nodes: Total entity nodes across all extracted books.
        n_total_edges: Total relation edges across all extracted books.
        min_node_count: Minimum acceptable node count (default 100).
        min_edge_count: Minimum acceptable edge count (default 200).
        max_failure_rate: Maximum acceptable book-level extraction failure rate
            (default 0.10 = 10 %).
        pass_gate: Computed — True when node count, edge count and failure rate
            all satisfy their thresholds.
    """

    n_books_processed: int
    n_books_extraction_failed: int
    n_total_nodes: int
    n_total_edges: int
    min_node_count: int = 100
    min_edge_count: int = 200
    max_failure_rate: float = 0.10
    pass_gate: bool = False

    @property
    def failure_rate(self) -> float:
        """Book-level extraction failure rate."""
        return self.n_books_extraction_failed / max(1, self.n_books_processed)

    def model_post_init(self, *args: object, **kwargs: object) -> None:
        super().model_post_init(*args, **kwargs)
        self.pass_gate = (
            self.n_total_nodes >= self.min_node_count
            and self.n_total_edges >= self.min_edge_count
            and self.failure_rate <= self.max_failure_rate
        )


class GraphPerQueryScore(VoBaseModel):
    """Pre-computed per-query retrieval metrics for the graph A/B comparison.

    Live retrieval is NOT performed here; callers supply these records after
    running the harness against both planner arms.

    Attributes:
        query_id: Unique identifier for the query.
        planner_with_graph_recall_at_10: Recall@10 from the graph-enabled arm.
        planner_without_graph_recall_at_10: Recall@10 from the baseline arm.
        planner_with_graph_ndcg_at_10: nDCG@10 from the graph-enabled arm.
        planner_without_graph_ndcg_at_10: nDCG@10 from the baseline arm.
        planner_with_graph_judge_score: Judge-rated answer quality, graph arm
            (0–1 scale, 0.0 when not yet computed).
        planner_without_graph_judge_score: Judge-rated answer quality, baseline
            arm (0–1 scale, 0.0 when not yet computed).
        query_uses_graph_path: True when the planner's plan included the GRAPH
            retrieval path for this specific query.  Only queries where this is
            True enter the superiority computation — mixed-population dilution
            is explicitly avoided.
    """

    query_id: str
    planner_with_graph_recall_at_10: float
    planner_without_graph_recall_at_10: float
    planner_with_graph_ndcg_at_10: float
    planner_without_graph_ndcg_at_10: float
    planner_with_graph_judge_score: float = 0.0
    planner_without_graph_judge_score: float = 0.0
    query_uses_graph_path: bool


class GraphPhase6bVerdict(VoBaseModel):
    """Aggregate STOP/GO verdict for the entity-relationship graph layer.

    Attributes:
        mechanical: Mechanical gate result (node/edge counts, failure rate).
        recall_result: NonInferiorityResult for Recall@10 as dict (graph-using
            query subset).
        ndcg_result: NonInferiorityResult for nDCG@10 as dict (graph-using
            query subset).
        judge_result: NonInferiorityResult for judge score as dict, or None if
            no judge scores were provided.
        n_queries_using_graph: Number of queries where query_uses_graph_path
            is True — the evaluation subset.
        keep_graph_path: True when the graph path passes the STOP/GO gate.
        action: One of keep_graph / disable_graph_path / rerun_after_fixes.
        failure_reasons: Human-readable list of reasons for rejection or re-run.
        summary: One-line human-readable summary.
    """

    mechanical: GraphMechanicalResult
    recall_result: dict[str, Any]
    ndcg_result: dict[str, Any]
    judge_result: dict[str, Any] | None
    n_queries_using_graph: int
    keep_graph_path: bool
    action: Literal["keep_graph", "disable_graph_path", "rerun_after_fixes"]
    failure_reasons: list[str]
    summary: str


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------


def evaluate_graph(
    *,
    mechanical: GraphMechanicalResult,
    per_query_scores: list[GraphPerQueryScore],
    non_inferiority_margin_pp: float = 3.0,
    superiority_margin_pp: float = 1.0,
    confidence: float = 0.95,
) -> GraphPhase6bVerdict:
    """Run the STOP/GO evaluation for the entity-relationship graph.

    Acceptance criteria:
      1. Mechanical gate must pass (node/edge counts, failure rate).
      2. At least ``_MIN_GRAPH_QUERIES`` queries must use the graph path —
         otherwise there is insufficient signal for the comparison.
      3. Recall@10 must be NON-INFERIOR on the graph-using query subset
         (treatment = with_graph, control = without_graph, margin =
         ``non_inferiority_margin_pp``).
      4. At least one of (nDCG@10, judge score) must be SUPERIOR on the
         graph-using query subset (delta_ci_low > +superiority_margin_pp/100).

    If all four conditions hold → ``keep_graph`` (graph path stays enabled).
    If the mechanical gate fails → ``rerun_after_fixes``.
    If the signal gate fails (too few graph-using queries) → ``rerun_after_fixes``.
    If recall is inferior → ``disable_graph_path``.
    If neither metric is superior (but recall is non-inferior) → ``disable_graph_path``.

    Args:
        mechanical: Pre-computed mechanical gate result.
        per_query_scores: Pre-computed per-query metric records (all queries).
        non_inferiority_margin_pp: Non-inferiority margin for recall in
            percentage points (default 3.0).
        superiority_margin_pp: Superiority margin for nDCG/judge in
            percentage points (default 1.0).
        confidence: One-sided confidence level for the CI (default 0.95).

    Returns:
        GraphPhase6bVerdict with the STOP/GO decision and supporting details.
    """
    failure_reasons: list[str] = []
    action: Literal["keep_graph", "disable_graph_path", "rerun_after_fixes"]

    # ------------------------------------------------------------------
    # Gate 1: mechanical
    # ------------------------------------------------------------------
    if not mechanical.pass_gate:
        if mechanical.n_total_nodes < mechanical.min_node_count:
            failure_reasons.append(
                f"Node count {mechanical.n_total_nodes} < minimum {mechanical.min_node_count}"
            )
        if mechanical.n_total_edges < mechanical.min_edge_count:
            failure_reasons.append(
                f"Edge count {mechanical.n_total_edges} < minimum {mechanical.min_edge_count}"
            )
        if mechanical.failure_rate > mechanical.max_failure_rate:
            failure_reasons.append(
                f"Extraction failure rate {mechanical.failure_rate:.3f} "
                f"exceeds maximum {mechanical.max_failure_rate:.3f}"
            )

        log.warning(
            "graph_mechanical_gate_failed",
            pass_gate=mechanical.pass_gate,
            failure_reasons=failure_reasons,
        )

        return GraphPhase6bVerdict(
            mechanical=mechanical,
            recall_result={},
            ndcg_result={},
            judge_result=None,
            n_queries_using_graph=0,
            keep_graph_path=False,
            action="rerun_after_fixes",
            failure_reasons=failure_reasons,
            summary="RERUN (mechanical gate failed) | " + " | ".join(failure_reasons),
        )

    # ------------------------------------------------------------------
    # Gate 2: sufficient signal — only graph-using queries count
    # ------------------------------------------------------------------
    graph_queries = [s for s in per_query_scores if s.query_uses_graph_path]
    n_graph = len(graph_queries)

    if n_graph < _MIN_GRAPH_QUERIES:
        reason = (
            f"Insufficient graph-path signal: only {n_graph} queries used the GRAPH "
            f"path; minimum required is {_MIN_GRAPH_QUERIES}"
        )
        failure_reasons.append(reason)

        log.warning(
            "graph_insufficient_signal",
            n_graph=n_graph,
            min_required=_MIN_GRAPH_QUERIES,
        )

        return GraphPhase6bVerdict(
            mechanical=mechanical,
            recall_result={},
            ndcg_result={},
            judge_result=None,
            n_queries_using_graph=n_graph,
            keep_graph_path=False,
            action="rerun_after_fixes",
            failure_reasons=failure_reasons,
            summary=(
                f"RERUN (insufficient signal: {n_graph} < {_MIN_GRAPH_QUERIES} graph-using queries)"
            ),
        )

    # ------------------------------------------------------------------
    # Extract per-query score arrays (graph-using subset only)
    # ------------------------------------------------------------------
    with_graph_recalls = [s.planner_with_graph_recall_at_10 for s in graph_queries]
    without_graph_recalls = [s.planner_without_graph_recall_at_10 for s in graph_queries]

    with_graph_ndcgs = [s.planner_with_graph_ndcg_at_10 for s in graph_queries]
    without_graph_ndcgs = [s.planner_without_graph_ndcg_at_10 for s in graph_queries]

    # ------------------------------------------------------------------
    # Gate 3: non-inferiority on Recall@10
    # (treatment = with_graph, control = without_graph)
    # ------------------------------------------------------------------
    recall_result = compute_non_inferiority(
        control_scores=without_graph_recalls,
        treatment_scores=with_graph_recalls,
        metric="recall_at_10",
        margin_pp=non_inferiority_margin_pp,
        confidence=confidence,
    )

    # ------------------------------------------------------------------
    # Gate 4: superiority on nDCG@10
    # ------------------------------------------------------------------
    ndcg_result = compute_non_inferiority(
        control_scores=without_graph_ndcgs,
        treatment_scores=with_graph_ndcgs,
        metric="ndcg_at_10",
        margin_pp=non_inferiority_margin_pp,
        confidence=confidence,
    )

    # ------------------------------------------------------------------
    # Gate 4 (optional): superiority on judge score
    # ------------------------------------------------------------------
    has_judge_scores = any(
        s.planner_with_graph_judge_score != 0.0 or s.planner_without_graph_judge_score != 0.0
        for s in graph_queries
    )
    judge_result_obj = None
    judge_result_dict: dict[str, Any] | None = None
    if has_judge_scores:
        with_graph_judges = [s.planner_with_graph_judge_score for s in graph_queries]
        without_graph_judges = [s.planner_without_graph_judge_score for s in graph_queries]
        judge_result_obj = compute_non_inferiority(
            control_scores=without_graph_judges,
            treatment_scores=with_graph_judges,
            metric="judge_score",
            margin_pp=non_inferiority_margin_pp,
            confidence=confidence,
        )
        judge_result_dict = judge_result_obj.model_dump()

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------
    superiority_frac = superiority_margin_pp / 100.0
    recall_non_inferior = recall_result.decision == "non_inferior"
    ndcg_superior = ndcg_result.delta_ci_low > superiority_frac
    judge_superior = judge_result_obj.delta_ci_low > superiority_frac if judge_result_obj else False

    keep_graph_path = recall_non_inferior and (ndcg_superior or judge_superior)

    if not recall_non_inferior:
        failure_reasons.append(
            f"Recall@10 is inferior: delta_ci_low={recall_result.delta_ci_low:+.4f} "
            f"< -{non_inferiority_margin_pp / 100:.4f} (margin) "
            f"on {n_graph} graph-using queries"
        )

    if not ndcg_superior and not judge_superior:
        judge_ci_str = f"{judge_result_obj.delta_ci_low:+.4f}" if judge_result_obj else "N/A"
        failure_reasons.append(
            f"No metric is superior: nDCG delta_ci_low={ndcg_result.delta_ci_low:+.4f}, "
            f"judge delta_ci_low={judge_ci_str} "
            f"(need > +{superiority_frac:.4f})"
        )

    action = "keep_graph" if keep_graph_path else "disable_graph_path"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    verdict_label = "KEEP" if keep_graph_path else f"DISABLE ({action})"
    summary_parts: list[str] = [
        f"n_graph_queries={n_graph}",
        f"Recall@10: delta={recall_result.delta_mean:+.4f} "
        f"CI_low={recall_result.delta_ci_low:+.4f} → {recall_result.decision}",
        f"nDCG@10: delta={ndcg_result.delta_mean:+.4f} "
        f"CI_low={ndcg_result.delta_ci_low:+.4f} → "
        f"{'superior' if ndcg_superior else 'not_superior'}",
    ]
    if judge_result_obj is not None:
        summary_parts.append(
            f"Judge: delta={judge_result_obj.delta_mean:+.4f} "
            f"CI_low={judge_result_obj.delta_ci_low:+.4f} → "
            f"{'superior' if judge_superior else 'not_superior'}"
        )
    summary = verdict_label + " | " + " | ".join(summary_parts)

    log.info(
        "graph_phase6b_verdict",
        keep_graph_path=keep_graph_path,
        action=action,
        n_queries_total=len(per_query_scores),
        n_queries_using_graph=n_graph,
        failure_reasons=failure_reasons,
    )

    return GraphPhase6bVerdict(
        mechanical=mechanical,
        recall_result=recall_result.model_dump(),
        ndcg_result=ndcg_result.model_dump(),
        judge_result=judge_result_dict,
        n_queries_using_graph=n_graph,
        keep_graph_path=keep_graph_path,
        action=action,
        failure_reasons=failure_reasons,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Report emitter
# ---------------------------------------------------------------------------


def emit_graph_report(verdict: GraphPhase6bVerdict) -> str:
    """Emit a multi-line plain-text graph evaluation report.

    Args:
        verdict: The GraphPhase6bVerdict produced by evaluate_graph.

    Returns:
        Multi-line human-readable plain-text report string.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("GRAPH LAYER QUALITY A/B EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    status = "KEEP graph path" if verdict.keep_graph_path else f"REJECT ({verdict.action})"
    lines.append(f"Verdict:  {status}")
    lines.append(f"Summary:  {verdict.summary}")
    lines.append("")

    lines.append("--- Mechanical Gate ---")
    mech = verdict.mechanical
    lines.append(f"  pass_gate:              {mech.pass_gate}")
    lines.append(f"  n_books_processed:      {mech.n_books_processed}")
    lines.append(f"  n_books_failed:         {mech.n_books_extraction_failed}")
    lines.append(
        f"  failure_rate:           {mech.failure_rate:.4f}  (max {mech.max_failure_rate:.4f})"
    )
    lines.append(f"  n_total_nodes:          {mech.n_total_nodes}  (min {mech.min_node_count})")
    lines.append(f"  n_total_edges:          {mech.n_total_edges}  (min {mech.min_edge_count})")
    lines.append("")

    lines.append(
        f"--- Query Population (graph-using subset: {verdict.n_queries_using_graph} queries) ---"
    )
    lines.append("")

    lines.append("--- Recall@10 (non-inferiority) ---")
    if verdict.recall_result:
        rr = verdict.recall_result
        lines.append(f"  Decision:          {rr['decision']}")
        lines.append(f"  N:                 {rr['n']}")
        lines.append(f"  With-graph mean:   {rr['treatment_mean']:.4f}")
        lines.append(f"  Without-graph mean:{rr['control_mean']:.4f}")
        lines.append(f"  Delta mean:        {rr['delta_mean']:+.4f}")
        lines.append(f"  Delta CI low:      {rr['delta_ci_low']:+.4f}")
        lines.append(f"  Non-inf margin:    -{rr['non_inferiority_margin_pp']}pp")
    else:
        lines.append("  (Not computed — earlier gate failed)")
    lines.append("")

    lines.append("--- nDCG@10 (superiority) ---")
    if verdict.ndcg_result:
        nr = verdict.ndcg_result
        lines.append(f"  N:                 {nr['n']}")
        lines.append(f"  With-graph mean:   {nr['treatment_mean']:.4f}")
        lines.append(f"  Without-graph mean:{nr['control_mean']:.4f}")
        lines.append(f"  Delta mean:        {nr['delta_mean']:+.4f}")
        lines.append(f"  Delta CI low:      {nr['delta_ci_low']:+.4f}")
    else:
        lines.append("  (Not computed — earlier gate failed)")
    lines.append("")

    lines.append("--- Judge Score (superiority, optional) ---")
    if verdict.judge_result is not None:
        jr = verdict.judge_result
        lines.append(f"  N:                 {jr['n']}")
        lines.append(f"  With-graph mean:   {jr['treatment_mean']:.4f}")
        lines.append(f"  Without-graph mean:{jr['control_mean']:.4f}")
        lines.append(f"  Delta mean:        {jr['delta_mean']:+.4f}")
        lines.append(f"  Delta CI low:      {jr['delta_ci_low']:+.4f}")
    else:
        lines.append("  (Not computed)")
    lines.append("")

    if verdict.failure_reasons:
        lines.append("--- Failure Reasons ---")
        for reason in verdict.failure_reasons:
            lines.append(f"  * {reason}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
