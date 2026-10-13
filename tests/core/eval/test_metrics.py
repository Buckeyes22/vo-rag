"""Pure-Python ranking metrics."""

from __future__ import annotations

import math

import pytest

from vo_rag.core.eval.metrics import (
    average_precision,
    dcg_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_all_relevant_in_top_k() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b", "c"}
    assert recall_at_k(retrieved, relevant, 5) == 1.0


def test_recall_at_k_partial() -> None:
    retrieved = ["x", "a", "y", "b"]
    relevant = {"a", "b", "c"}
    assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)


def test_recall_at_k_empty_relevant_returns_zero() -> None:
    assert recall_at_k(["a", "b"], set(), 5) == 0.0


def test_precision_at_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, 4) == pytest.approx(0.5)
    assert precision_at_k(retrieved, relevant, 2) == pytest.approx(0.5)


def test_mrr_first_rank() -> None:
    assert mrr(["a", "b", "c"], {"a"}) == 1.0


def test_mrr_second_rank() -> None:
    assert mrr(["a", "b", "c"], {"b"}) == pytest.approx(0.5)


def test_mrr_no_hit() -> None:
    assert mrr(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_perfect_ranking_is_one() -> None:
    retrieved = ["a", "b", "c"]
    judgments = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(retrieved, judgments, 3) == pytest.approx(1.0)


def test_ndcg_reverse_ranking_less_than_one() -> None:
    retrieved = ["c", "b", "a"]
    judgments = {"a": 3, "b": 2, "c": 1}
    score = ndcg_at_k(retrieved, judgments, 3)
    assert 0 < score < 1.0


def test_ndcg_zero_when_no_positive_judgments() -> None:
    retrieved = ["a", "b", "c"]
    judgments = {"a": 0, "b": 0, "c": 0}
    assert ndcg_at_k(retrieved, judgments, 3) == 0.0


def test_dcg_matches_hand_computed() -> None:
    dcg = dcg_at_k(["a", "b", "c", "d"], {"a": 3, "b": 2, "c": 0, "d": 1}, 4)
    expected = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(5)
    assert dcg == pytest.approx(expected)


def test_average_precision_hand_computed() -> None:
    retrieved = ["a", "x", "c", "d", "y"]
    relevant = {"a", "c", "d"}
    expected = (1.0 + 2 / 3 + 3 / 4) / 3
    assert average_precision(retrieved, relevant) == pytest.approx(expected)


def test_k_zero_returns_zero() -> None:
    assert recall_at_k(["a"], {"a"}, 0) == 0.0
    assert precision_at_k(["a"], {"a"}, 0) == 0.0
    assert ndcg_at_k(["a"], {"a": 3}, 0) == 0.0
