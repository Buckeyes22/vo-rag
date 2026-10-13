"""Baseline freezing and non-inferiority contract tests (VOR-236)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vo_rag.core.eval.harness import (
    HarnessReport,
    PerQueryResult,
    RetrievalVariant,
    assert_no_regression,
)
from vo_rag.integrations.llamaindex.eval_dataset import (
    assert_non_inferior_to_baseline,
    freeze_baseline,
)


def _known_report() -> HarnessReport:
    aggregate = {
        "recall@10": 0.95,
        "ndcg@10": 0.9,
        "mrr": 0.875,
    }
    return HarnessReport(
        pipeline="book_rag",
        variant=RetrievalVariant(name="book_rag", top_k=10),
        qrels_hash="qrels-hash",
        per_query=[
            PerQueryResult(
                query_id="q-001",
                retrieved=["doc-a"],
                latency_ms=1.0,
                recall_at_k={10: 1.0},
                ndcg_at_k={10: 1.0},
                mrr=1.0,
                precision_at_k={10: 0.1},
            )
        ],
        aggregate=aggregate,
        wall_clock_s=0.01,
    )


class TestFreezeBaseline:
    """Contract tests for ``freeze_baseline``."""

    def test_writes_aggregate_json_and_round_trip_feeds_regression_gate(
        self,
        tmp_path: Path,
    ) -> None:
        report = _known_report()
        baseline_path = tmp_path / "baseline.json"

        baseline = freeze_baseline(report, path=baseline_path, corpus_snapshot="fixture:test")

        assert baseline["aggregate"] == report.aggregate
        loaded = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert loaded["aggregate"] == report.aggregate
        assert loaded["schema_version"] == 1
        assert loaded["qrels_hash"] == report.qrels_hash
        assert_no_regression(report, baseline["aggregate"])


class TestAssertNonInferiorToBaseline:
    """Contract tests for the governed non-inferiority wrapper."""

    def test_passes_when_treatment_matches_or_beats_control(self) -> None:
        result = assert_non_inferior_to_baseline(
            [0.9, 0.8, 0.7],
            [0.91, 0.8, 0.72],
            metric="ndcg@10",
        )

        assert result.decision == "non_inferior"
        assert result.non_inferiority_margin_pp == 2.0

    def test_raises_when_treatment_exceeds_default_margin_drop(self) -> None:
        with pytest.raises(AssertionError, match="margin_pp=2.0"):
            assert_non_inferior_to_baseline(
                [1.0, 1.0, 1.0],
                [0.97, 0.97, 0.97],
                metric="recall@10",
            )
