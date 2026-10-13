"""Tests for retrieval_smoke gate policy."""

from __future__ import annotations

from vo_rag.pipelines.book_rag.retrieval_smoke import (
    GateDecision,
    SmokeReport,
    gate_decision,
)


def test_gate_decision_zero_score_positive_n_fail() -> None:
    report = SmokeReport(
        source_id="s1",
        n_requested=5,
        n_scored=5,
        top_k=10,
        score=0.0,
    )
    result = gate_decision(report=report)
    assert result.decision == GateDecision.FAIL


def test_gate_decision_zero_score_zero_n_quarantine() -> None:
    report = SmokeReport(
        source_id="s1",
        n_requested=5,
        n_scored=0,
        top_k=10,
        score=0.0,
    )
    result = gate_decision(report=report)
    assert result.decision == GateDecision.QUARANTINE


def test_gate_decision_above_threshold_pass() -> None:
    report = SmokeReport(
        source_id="s1",
        n_requested=5,
        n_scored=5,
        top_k=10,
        score=0.65,
    )
    result = gate_decision(report=report)
    assert result.decision == GateDecision.PASS
