from __future__ import annotations

import pytest

from vo_rag.core.eval.judge_calibration import _normal_quantile


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.5, 0.0), (0.95, 1.6448536), (0.975, 1.9599640)],
)
def test_normal_quantile_acklam_accuracy(p: float, expected: float) -> None:
    assert _normal_quantile(p) == pytest.approx(expected, abs=2e-6)


def test_normal_quantile_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        _normal_quantile(1.0)
