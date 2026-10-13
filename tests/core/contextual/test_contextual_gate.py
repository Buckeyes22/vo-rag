from __future__ import annotations

import importlib.util
from pathlib import Path

from vo_rag.core.eval.non_inferiority import compute_non_inferiority


def test_contextual_shared_math_lives_in_core_eval() -> None:
    result = compute_non_inferiority(
        control_scores=[0.8],
        treatment_scores=[0.83],
        metric="recall_at_10",
        margin_pp=3.0,
        confidence=0.95,
    )
    assert result.decision == "non_inferior"


def test_contextual_enrichment_is_not_silently_ported() -> None:
    contextual_dir = Path(__file__).resolve().parents[3] / "src" / "vo_rag" / "core" / "contextual"
    assert not (contextual_dir / "enrichment.py").exists()
    assert not (contextual_dir / "decision_gate.py").exists()
    assert importlib.util.find_spec("vo_rag.core.contextual.enrichment") is None
