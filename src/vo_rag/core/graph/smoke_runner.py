"""End-to-end smoke runner — STOP/GO gate for the entity graph layer.

Orchestrates the graph smoke test:
  1. Accepts pre-built graph state (node/edge counts) and two retriever callables.
  2. Runs pilot queries concurrently against both arms (with-graph vs without-graph).
  3. Computes Recall@10 and nDCG@10 per query using gold relevance labels.
  4. Aggregates per-metric mean improvements (with − without in pp).
  5. Returns a SmokeRunResult that is mapped to GraphSmokeResult for phase6b_aggregator.

No real Qdrant, LLM, or network calls are made here — retrievers are callable
injection points supplied by the caller.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import structlog
from pydantic import Field

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.eval.non_inferiority import (
    compute_ndcg_at_k,
    compute_recall_at_k,
)
from vo_rag.core.graph.phase6b_aggregator import GraphSmokeResult
from vo_rag.core.models.common import VoBaseModel

log = structlog.get_logger(__name__)

__all__ = [
    "SmokeRunConfig",
    "PilotQueryWithEvidence",
    "PerQuerySmokeScore",
    "SmokeRunResult",
    "compute_per_query_score",
    "aggregate_smoke_metrics",
    "run_phase6a_smoke",
    "to_phase6b_smoke_result",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SmokeRunConfig(VoBaseModel):
    """Configuration for the graph smoke run."""

    min_node_count_pilot: int = 100
    min_edge_count_pilot: int = 200
    judge_min_improvement_pp: float = 1.0
    metrics_to_test: list[str] = Field(default_factory=lambda: ["recall_at_10", "ndcg_at_10"])
    pilot_query_count: int = 20


# ---------------------------------------------------------------------------
# Per-query input and output models
# ---------------------------------------------------------------------------


class PilotQueryWithEvidence(VoBaseModel):
    """A pilot query with its gold relevance annotations."""

    query_id: str
    query_text: str
    gold_chunk_ids: set[str]
    gold_relevance_scores: dict[str, float]


class PerQuerySmokeScore(VoBaseModel):
    """Retrieval metric scores for one pilot query under both arms."""

    query_id: str
    with_graph_recall_at_10: float
    without_graph_recall_at_10: float
    with_graph_ndcg_at_10: float
    without_graph_ndcg_at_10: float


# ---------------------------------------------------------------------------
# Aggregate result
# ---------------------------------------------------------------------------


class SmokeRunResult(VoBaseModel):
    """Aggregate result of a graph smoke run."""

    n_nodes_built: int
    n_edges_built: int
    per_query_scores: list[PerQuerySmokeScore]
    mean_improvements: dict[str, float]
    min_one_metric_improved: bool
    smoke_passed: bool


# ---------------------------------------------------------------------------
# Per-query scoring
# ---------------------------------------------------------------------------


def compute_per_query_score(
    *,
    query: PilotQueryWithEvidence,
    with_graph_chunks: list[RetrievalChunk],
    without_graph_chunks: list[RetrievalChunk],
) -> PerQuerySmokeScore:
    """Compute Recall@10 and nDCG@10 for both arms on a single pilot query.

    Args:
        query: Pilot query with gold chunk IDs and graded relevance scores.
        with_graph_chunks: Retrieved chunks from the graph-augmented arm.
        without_graph_chunks: Retrieved chunks from the baseline arm.

    Returns:
        PerQuerySmokeScore with both arms' metric values.
    """
    with_ids = [c.chunk_id for c in with_graph_chunks]
    without_ids = [c.chunk_id for c in without_graph_chunks]

    with_recall = compute_recall_at_k(
        retrieved_ids=with_ids,
        relevant_ids=query.gold_chunk_ids,
        k=10,
    )
    without_recall = compute_recall_at_k(
        retrieved_ids=without_ids,
        relevant_ids=query.gold_chunk_ids,
        k=10,
    )

    with_ndcg = compute_ndcg_at_k(
        retrieved_ids=with_ids,
        relevance_judgments=query.gold_relevance_scores,
        k=10,
    )
    without_ndcg = compute_ndcg_at_k(
        retrieved_ids=without_ids,
        relevance_judgments=query.gold_relevance_scores,
        k=10,
    )

    log.debug(
        "smoke_runner.per_query_score",
        query_id=query.query_id,
        with_recall=with_recall,
        without_recall=without_recall,
        with_ndcg=with_ndcg,
        without_ndcg=without_ndcg,
    )

    return PerQuerySmokeScore(
        query_id=query.query_id,
        with_graph_recall_at_10=with_recall,
        without_graph_recall_at_10=without_recall,
        with_graph_ndcg_at_10=with_ndcg,
        without_graph_ndcg_at_10=without_ndcg,
    )


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

# Maps metric name to (with_field, without_field) on PerQuerySmokeScore.
_METRIC_FIELD_MAP: dict[str, tuple[str, str]] = {
    "recall_at_10": ("with_graph_recall_at_10", "without_graph_recall_at_10"),
    "ndcg_at_10": ("with_graph_ndcg_at_10", "without_graph_ndcg_at_10"),
}


def aggregate_smoke_metrics(
    *,
    per_query_scores: list[PerQuerySmokeScore],
    config: SmokeRunConfig,
) -> tuple[dict[str, float], bool]:
    """Aggregate per-query scores into mean improvements.

    For each metric in config.metrics_to_test, computes:
        mean(with_score − without_score) across all queries.

    min_one_metric_improved is True when at least one metric's mean improvement
    is >= config.judge_min_improvement_pp (in percentage points, i.e. the raw
    float difference is compared after multiplying by 100).

    Args:
        per_query_scores: Per-query score records.
        config: Smoke run configuration.

    Returns:
        Tuple of (mean_improvements_dict, min_one_metric_improved).
    """
    if not per_query_scores:
        return dict.fromkeys(config.metrics_to_test, 0.0), False

    n = len(per_query_scores)
    mean_improvements: dict[str, float] = {}

    for metric in config.metrics_to_test:
        field_pair = _METRIC_FIELD_MAP.get(metric)
        if field_pair is None:
            log.warning(
                "smoke_runner.aggregate.unknown_metric",
                metric=metric,
                supported=list(_METRIC_FIELD_MAP),
            )
            mean_improvements[metric] = 0.0
            continue

        with_field, without_field = field_pair
        total_improvement = sum(
            getattr(score, with_field) - getattr(score, without_field) for score in per_query_scores
        )
        mean_improvements[metric] = (total_improvement / n) * 100.0

    min_one_metric_improved = any(
        improvement >= config.judge_min_improvement_pp for improvement in mean_improvements.values()
    )

    log.info(
        "smoke_runner.aggregate.done",
        n_queries=n,
        mean_improvements=mean_improvements,
        min_one_metric_improved=min_one_metric_improved,
    )

    return mean_improvements, min_one_metric_improved


# ---------------------------------------------------------------------------
# Main smoke runner
# ---------------------------------------------------------------------------


async def run_phase6a_smoke(
    *,
    pilot_queries: list[PilotQueryWithEvidence],
    n_nodes_built: int,
    n_edges_built: int,
    with_graph_retriever: Callable[[str], Awaitable[list[RetrievalChunk]]],
    without_graph_retriever: Callable[[str], Awaitable[list[RetrievalChunk]]],
    config: SmokeRunConfig,
) -> SmokeRunResult:
    """Execute the graph smoke run against all pilot queries.

    For each pilot query, both retrievers are called concurrently. Results
    are scored per-query and aggregated into a SmokeRunResult.

    smoke_passed is True iff:
      - n_nodes_built >= config.min_node_count_pilot
      - n_edges_built >= config.min_edge_count_pilot
      - min_one_metric_improved is True

    Args:
        pilot_queries: Pilot queries with gold relevance annotations.
        n_nodes_built: Node count from the pre-built graph.
        n_edges_built: Edge count from the pre-built graph.
        with_graph_retriever: Async callable: query_text → chunks (graph arm).
        without_graph_retriever: Async callable: query_text → chunks (baseline).
        config: Smoke run configuration.

    Returns:
        SmokeRunResult with per-query scores and aggregate gate outcome.
    """
    log.info(
        "smoke_runner.run.start",
        n_queries=len(pilot_queries),
        n_nodes_built=n_nodes_built,
        n_edges_built=n_edges_built,
    )

    per_query_scores: list[PerQuerySmokeScore] = []

    async def _score_query(query: PilotQueryWithEvidence) -> PerQuerySmokeScore:
        with_chunks, without_chunks = await asyncio.gather(
            with_graph_retriever(query.query_text),
            without_graph_retriever(query.query_text),
        )
        return compute_per_query_score(
            query=query,
            with_graph_chunks=with_chunks,
            without_graph_chunks=without_chunks,
        )

    per_query_scores = list(await asyncio.gather(*(_score_query(q) for q in pilot_queries)))

    mean_improvements, min_one_metric_improved = aggregate_smoke_metrics(
        per_query_scores=per_query_scores,
        config=config,
    )

    nodes_ok = n_nodes_built >= config.min_node_count_pilot
    edges_ok = n_edges_built >= config.min_edge_count_pilot
    smoke_passed = nodes_ok and edges_ok and min_one_metric_improved

    log.info(
        "smoke_runner.run.done",
        nodes_ok=nodes_ok,
        edges_ok=edges_ok,
        min_one_metric_improved=min_one_metric_improved,
        smoke_passed=smoke_passed,
    )

    return SmokeRunResult(
        n_nodes_built=n_nodes_built,
        n_edges_built=n_edges_built,
        per_query_scores=per_query_scores,
        mean_improvements=mean_improvements,
        min_one_metric_improved=min_one_metric_improved,
        smoke_passed=smoke_passed,
    )


# ---------------------------------------------------------------------------
# Mapping to phase6b_aggregator shape
# ---------------------------------------------------------------------------


def to_phase6b_smoke_result(smoke: SmokeRunResult) -> GraphSmokeResult:
    """Map SmokeRunResult to the GraphSmokeResult consumed by phase6b_aggregator.

    Fields forwarded:
      - passed              ← smoke.smoke_passed
      - n_nodes_built       ← smoke.n_nodes_built
      - n_edges_built       ← smoke.n_edges_built
      - min_one_metric_improved ← smoke.min_one_metric_improved
      - improvement_breakdown ← smoke.mean_improvements

    Args:
        smoke: The SmokeRunResult produced by run_phase6a_smoke.

    Returns:
        GraphSmokeResult ready for decide_phase6b.
    """
    return GraphSmokeResult(
        passed=smoke.smoke_passed,
        n_nodes_built=smoke.n_nodes_built,
        n_edges_built=smoke.n_edges_built,
        min_one_metric_improved=smoke.min_one_metric_improved,
        improvement_breakdown=smoke.mean_improvements,
    )
