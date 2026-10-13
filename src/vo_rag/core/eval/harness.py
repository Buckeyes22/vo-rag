"""Retrieval evaluation harness.

Runs a ranked-retrieval callable against a `Qrels` set and reports
`HarnessReport` with per-query + aggregate metrics. The harness is deliberately
agnostic about HOW retrieval is executed - callers pass a coroutine factory
`search_fn(query_text, top_k)` so the same harness drives unit-test mocks, the
real live retriever, and ablation variants.

Sync search callables are also supported: the harness inspects the returned
value and awaits it only if it's awaitable. This keeps the gate's
synchronous fake-search adapters usable from the async harness without
forcing every test fixture to be async.
"""

from __future__ import annotations

import inspect
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import Field

from vo_rag.core.eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from vo_rag.core.eval.passages import (
    GoldPassageProjectionFn,
    RelevanceMode,
    RetrievedItem,
    project_qrels_to_passage_keys,
    project_retrieved_to_doc_ids,
    project_retrieved_to_passage_keys,
)
from vo_rag.core.eval.qrels import Qrels, qrels_hash
from vo_rag.core.models.common import VoBaseModel


class RetrievalVariant(VoBaseModel):
    name: str
    top_k: int = 10
    prefetch_k: int = 100
    fusion: dict[str, Any] = Field(default_factory=dict)
    reranker_enabled: bool = True
    query_transform: str = "identity"


class PerQueryResult(VoBaseModel):
    query_id: str
    retrieved: list[str]
    latency_ms: float
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    mrr: float
    precision_at_k: dict[int, float]


class HarnessReport(VoBaseModel):
    pipeline: str
    variant: RetrievalVariant
    qrels_hash: str
    git_sha: str | None = None
    per_query: list[PerQueryResult]
    aggregate: dict[str, float]
    wall_clock_s: float


SearchFn = Callable[
    [str, int],
    "Sequence[RetrievedItem] | Awaitable[Sequence[RetrievedItem]]",
]
QueryTextFn = Callable[[str], str]

_DEFAULT_K_VALUES: tuple[int, ...] = (1, 5, 10, 20)


class RetrievalRegression(AssertionError):
    """Raised by assert_no_regression when a baseline bar is breached."""


async def _call_search(search_fn: SearchFn, query: str, top_k: int) -> list[RetrievedItem]:
    out = search_fn(query, top_k)
    if inspect.isawaitable(out):
        result = await out
    else:
        result = out
    return list(result)


async def run_harness(
    pipeline: str,
    qrels: Qrels,
    variant: RetrievalVariant,
    *,
    search_fn: SearchFn,
    query_text_fn: QueryTextFn | None = None,
    k_values: tuple[int, ...] = _DEFAULT_K_VALUES,
    git_sha: str | None = None,
    relevance_mode: RelevanceMode = "chunk_id",
    gold_doc_projection_fn: GoldPassageProjectionFn | None = None,
) -> HarnessReport:
    """Execute the harness.

    `search_fn(query_text, top_k)` returns a list of doc_ids in rank
    order. `query_text_fn(query_id)` maps a qrels query_id to the text
    the search function expects; defaults to identity so tests can use
    query_ids directly.
    """
    text_fn = query_text_fn or (lambda qid: qid)
    eval_qrels = (
        project_qrels_to_passage_keys(qrels, gold_doc_projection_fn)
        if relevance_mode == "passage"
        else qrels
    )
    grouped = eval_qrels.by_query()
    positives = eval_qrels.positive_set_by_query()
    per_query: list[PerQueryResult] = []
    start = time.perf_counter()

    for query_id in qrels.all_query_ids():
        q_text = text_fn(query_id)
        t0 = time.perf_counter()
        raw_retrieved = await _call_search(search_fn, q_text, variant.top_k)
        retrieved = (
            project_retrieved_to_passage_keys(raw_retrieved)
            if relevance_mode == "passage"
            else project_retrieved_to_doc_ids(raw_retrieved)
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        judgments = grouped.get(query_id, {})
        relevant = positives.get(query_id, set())

        per_query.append(
            PerQueryResult(
                query_id=query_id,
                retrieved=list(retrieved),
                latency_ms=latency_ms,
                recall_at_k={k: recall_at_k(retrieved, relevant, k) for k in k_values},
                ndcg_at_k={k: ndcg_at_k(retrieved, judgments, k) for k in k_values},
                mrr=mrr(retrieved, relevant),
                precision_at_k={k: precision_at_k(retrieved, relevant, k) for k in k_values},
            )
        )

    wall_clock = time.perf_counter() - start
    aggregate = _aggregate(per_query, k_values)

    return HarnessReport(
        pipeline=pipeline,
        variant=variant,
        qrels_hash=qrels_hash(eval_qrels),
        git_sha=git_sha,
        per_query=per_query,
        aggregate=aggregate,
        wall_clock_s=wall_clock,
    )


def _aggregate(per_query: list[PerQueryResult], k_values: tuple[int, ...]) -> dict[str, float]:
    if not per_query:
        return {}
    agg: dict[str, float] = {}
    for k in k_values:
        agg[f"recall@{k}"] = statistics.mean(p.recall_at_k[k] for p in per_query)
        agg[f"ndcg@{k}"] = statistics.mean(p.ndcg_at_k[k] for p in per_query)
        agg[f"precision@{k}"] = statistics.mean(p.precision_at_k[k] for p in per_query)
    agg["mrr"] = statistics.mean(p.mrr for p in per_query)
    latencies = sorted(p.latency_ms for p in per_query)
    agg["latency_p50_ms"] = latencies[len(latencies) // 2]
    agg["latency_p95_ms"] = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    return agg


def assert_no_regression(
    report: HarnessReport,
    baseline: dict[str, float],
    *,
    max_ndcg_drop: float = 0.02,
    max_recall_drop: float = 0.03,
) -> None:
    """Raise RetrievalRegression if any gated metric is worse than baseline
    by more than the allowed drop. Silently passes if the metric isn't in
    the baseline (e.g., the baseline was taken at different k_values)."""
    current = report.aggregate
    violations: list[str] = []

    for key, base in baseline.items():
        if key.startswith("ndcg@") or key == "mrr":
            delta = base - current.get(key, 0.0)
            if delta > max_ndcg_drop:
                violations.append(
                    f"{key}: {current.get(key, 0.0):.4f} vs baseline {base:.4f} (drop {delta:.4f})"
                )
        elif key.startswith("recall@"):
            delta = base - current.get(key, 0.0)
            if delta > max_recall_drop:
                violations.append(
                    f"{key}: {current.get(key, 0.0):.4f} vs baseline {base:.4f} (drop {delta:.4f})"
                )

    if violations:
        raise RetrievalRegression("retrieval quality regressed:\n  " + "\n  ".join(violations))


__all__ = [
    "HarnessReport",
    "PerQueryResult",
    "RetrievalRegression",
    "RetrievalVariant",
    "SearchFn",
    "assert_no_regression",
    "run_harness",
]
