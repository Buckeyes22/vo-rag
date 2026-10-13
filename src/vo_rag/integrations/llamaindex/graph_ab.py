"""Graph quality A/B exit gate for the LlamaIndex-wrapped read path.

Wires the existing vo-rag graph A/B evaluation (:func:`evaluate_graph`) over
the LlamaIndex-wrapped graph read path
(:class:`VoRagPropertyGraphStore` + :class:`VoRagPGRetriever` +
:class:`PropertyGraphIndex` from VOR-211-217).

The harness computes per-query recall@10 and nDCG@10 from paired retrieval
results, shapes them as :class:`GraphPerQueryScore` records, and feeds them
through the existing :func:`evaluate_graph` verdict.  A
``disable_graph_path`` verdict is a hard FAIL of the exit gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vo_rag.core.eval.metrics import ndcg_at_k, recall_at_k
from vo_rag.core.eval.qrels import Qrels
from vo_rag.core.graph.quality_ab import (
    GraphMechanicalResult,
    GraphPerQueryScore,
    GraphPhase6bVerdict,
    evaluate_graph,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "build_graph_ab_scores",
    "run_graph_ab_exit_gate",
]


# ---------------------------------------------------------------------------
# Score builder
# ---------------------------------------------------------------------------


def build_graph_ab_scores(
    *,
    queries: list[str],
    qrels: Qrels,
    native_retrieved: dict[str, list[str]],
    wrapped_retrieved: dict[str, list[str]],
    query_uses_graph_path: dict[str, bool] | None = None,
) -> list[GraphPerQueryScore]:
    """Build :class:`GraphPerQueryScore` records from paired retrieval results.

    Computes recall@10 and nDCG@10 for each query using vo-rag's existing
    metric implementations.  The ``wrapped_retrieved`` dict must contain
    chunk IDs keyed by query ID (deterministic content-addressed IDs so
    that recall/nDCG are comparable across arms).

    Args:
        queries: Ordered list of query IDs.
        qrels: Ground-truth qrels for metric computation.
        native_retrieved: ``{query_id: [chunk_id, ...]}`` from the native
            (control) arm, best-first.
        wrapped_retrieved: ``{query_id: [chunk_id, ...]}`` from the wrapped
            (treatment / LlamaIndex) arm, best-first.
        query_uses_graph_path: Optional override for which queries used the
            graph path.  Defaults to ``True`` for all queries.

    Returns:
        List of :class:`GraphPerQueryScore` records, one per query.
    """
    positive_by_query = qrels.positive_set_by_query(min_grade=1)
    judgments_by_query = qrels.by_query()

    scores: list[GraphPerQueryScore] = []
    for qid in queries:
        native_ids = native_retrieved.get(qid, [])
        wrapped_ids = wrapped_retrieved.get(qid, [])
        relevant = positive_by_query.get(qid, set())
        judgments = judgments_by_query.get(qid, {})

        scores.append(
            GraphPerQueryScore(
                query_id=qid,
                planner_with_graph_recall_at_10=recall_at_k(wrapped_ids, relevant, k=10),
                planner_without_graph_recall_at_10=recall_at_k(native_ids, relevant, k=10),
                planner_with_graph_ndcg_at_10=ndcg_at_k(wrapped_ids, judgments, k=10),
                planner_without_graph_ndcg_at_10=ndcg_at_k(native_ids, judgments, k=10),
                query_uses_graph_path=(
                    query_uses_graph_path.get(qid, True)
                    if query_uses_graph_path is not None
                    else True
                ),
            )
        )
    return scores


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------


def run_graph_ab_exit_gate(
    *,
    mechanical: GraphMechanicalResult,
    per_query_scores: list[GraphPerQueryScore],
    non_inferiority_margin_pp: float = 2.0,
    superiority_margin_pp: float = 1.0,
    confidence: float = 0.95,
) -> GraphPhase6bVerdict:
    """Run the graph quality A/B exit gate.

    Wraps the existing :func:`evaluate_graph` with fixed
    defaults (``margin_pp=2.0``, ``confidence=0.95``).

    A ``disable_graph_path`` verdict is a hard FAIL of the exit gate.
    Callers should assert ``verdict.keep_graph_path is True`` or
    ``verdict.action == "keep_graph"``.

    Args:
        mechanical: Pre-computed mechanical gate result.
        per_query_scores: Pre-computed per-query metric records.
        non_inferiority_margin_pp: Non-inferiority margin for recall in
            percentage points (default 2.0).
        superiority_margin_pp: Superiority margin for nDCG / judge in
            percentage points (default 1.0).
        confidence: One-sided confidence level for the CI (default 0.95).

    Returns:
        :class:`GraphPhase6bVerdict` with the STOP/GO decision.
    """
    return evaluate_graph(
        mechanical=mechanical,
        per_query_scores=per_query_scores,
        non_inferiority_margin_pp=non_inferiority_margin_pp,
        superiority_margin_pp=superiority_margin_pp,
        confidence=confidence,
    )
