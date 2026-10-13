"""Tests for variant selection."""

from __future__ import annotations

import pytest

from vo_rag.core.llm.variants import select_variant


def test_weight_validation_empty() -> None:
    with pytest.raises(ValueError):
        select_variant("p", {})


def test_weight_validation_negative() -> None:
    with pytest.raises(ValueError):
        select_variant("p", {"v1": 1.5, "v2": -0.5})


def test_weight_validation_sum() -> None:
    with pytest.raises(ValueError):
        select_variant("p", {"v1": 0.5, "v2": 0.3})


def test_sticky_selection_by_envelope_id_is_deterministic() -> None:
    weights = {"v1": 0.5, "v2": 0.5}
    first = select_variant("p", weights, envelope_id="env-abc")
    for _ in range(10):
        assert select_variant("p", weights, envelope_id="env-abc") == first


def test_weighted_split_chi_squared() -> None:
    """10k calls with 90/10 weights — chi² must be below 6.635 (p=0.01, df=1)."""
    import random as _random

    rng = _random.Random(1234)
    weights = {"v1": 0.9, "v2_canary": 0.1}
    counts = {"v1": 0, "v2_canary": 0}
    n = 10_000
    for _ in range(n):
        counts[select_variant("sterilizer", weights, rng=rng)] += 1

    expected_v1 = n * weights["v1"]
    expected_v2 = n * weights["v2_canary"]
    chi2 = (counts["v1"] - expected_v1) ** 2 / expected_v1 + (
        counts["v2_canary"] - expected_v2
    ) ** 2 / expected_v2

    assert chi2 < 6.635, f"chi² = {chi2}, counts = {counts}"
