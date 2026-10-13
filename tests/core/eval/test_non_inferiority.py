from __future__ import annotations

import pytest

from vo_rag.core.eval.non_inferiority import (
    NonInferiorityResult,
    compute_ndcg_at_k,
    compute_non_inferiority,
    compute_recall_at_k,
)


def test_compute_recall_at_k_partial_match() -> None:
    assert compute_recall_at_k(
        retrieved_ids=["a", "b", "x"],
        relevant_ids={"a", "b", "c", "d"},
        k=3,
    ) == pytest.approx(0.5)


def test_compute_ndcg_at_k_prefers_better_ordering() -> None:
    relevance = {"a": 1.0, "b": 0.5, "c": 0.25}
    best = compute_ndcg_at_k(retrieved_ids=["a", "b", "c"], relevance_judgments=relevance, k=3)
    worse = compute_ndcg_at_k(retrieved_ids=["c", "b", "a"], relevance_judgments=relevance, k=3)
    assert best > worse


def test_compute_non_inferiority_positive_delta_is_non_inferior() -> None:
    result = compute_non_inferiority(
        control_scores=[0.80],
        treatment_scores=[0.83],
        metric="recall_at_10",
        margin_pp=3.0,
        confidence=0.95,
    )
    assert isinstance(result, NonInferiorityResult)
    assert result.decision == "non_inferior"
    assert result.delta_mean == pytest.approx(0.03)
