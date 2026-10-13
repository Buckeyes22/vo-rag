"""Pure-Python ranking metrics used by the retrieval eval harness.

Pure-numpy-free implementations of recall@k, precision@k, MRR, DCG, nDCG, and
average precision. Reference cross-check tests in tests/core/eval/test_metrics.py
pin these against hand-computed values; pytrec_eval is not required.

All functions accept a ranked `retrieved` list (doc_ids, best-first) and
either a `relevant` set (binary) or a `judgments` dict (graded 0/1/2/3).
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of `relevant` docs that appear in the top-k of retrieved.

    Returns 0.0 when there are no relevant docs (vacuous query); this is
    deliberate so aggregate averages don't blow up.
    """
    if not relevant:
        return 0.0
    if k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(top_k)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant doc in retrieved."""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved: list[str], judgments: dict[str, int], k: int) -> float:
    """Discounted cumulative gain. 0-based `i`; discount log2(i+2).

    Graded relevance: the gain is the raw judgment value (0, 1, 2, 3).
    """
    if k <= 0:
        return 0.0
    total = 0.0
    for i, doc in enumerate(retrieved[:k]):
        grade = judgments.get(doc, 0)
        if grade <= 0:
            continue
        total += grade / math.log2(i + 2)
    return total


def ndcg_at_k(retrieved: list[str], judgments: dict[str, int], k: int) -> float:
    """nDCG - DCG divided by IDCG (the DCG of the ideal ranking)."""
    if k <= 0:
        return 0.0
    ideal_order = sorted(judgments.values(), reverse=True)[:k]
    idcg = 0.0
    for i, grade in enumerate(ideal_order):
        if grade <= 0:
            continue
        idcg += grade / math.log2(i + 2)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, judgments, k) / idcg


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """AP - mean of precision@i for each rank where a relevant doc appears."""
    if not relevant:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / len(relevant) if len(relevant) else 0.0


__all__ = [
    "average_precision",
    "dcg_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
