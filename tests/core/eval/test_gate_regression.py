"""Per-pipeline regression orchestration in core/eval/gate.py.

Contract: an active pipeline MUST have
  - a queries.jsonl + qrels.tsv + baseline.json fixture set,
  - a registered evaluator that exposes a `search_fn` + `variant`,
and the gate MUST run the harness and call assert_no_regression.

A pipeline with no entries in the evaluators map is a regression failure
(active pipeline, baseline present, no evaluator) - same exit code as a
metric regression so CI cannot silently green-light a missing component.

No active pipelines => pass.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from vo_rag.core.config.pipelines import PipelineSpec, PipelineStatus
from vo_rag.core.eval.gate import EvalGateResult, run_eval_gate
from vo_rag.core.eval.harness import RetrievalVariant
from vo_rag.core.eval.qrels import load_qrels_tsv, qrels_hash

_SEARCH_FN = Callable[[str, int], "list[str]"]


def _spec(name: str, status: PipelineStatus) -> PipelineSpec:
    return PipelineSpec(name=name, status=status, summary="s")


def _write_fixture(root: Path, *, baseline: dict[str, float]) -> Path:
    """Materialise the spine_smoke fixture set under `root/<pipeline>/`."""
    pipe_dir = root / "spine_smoke"
    pipe_dir.mkdir(parents=True, exist_ok=True)
    (pipe_dir / "queries.jsonl").write_text(
        '{"query_id": "q1", "text": "q1"}\n'
        '{"query_id": "q2", "text": "q2"}\n'
        '{"query_id": "q3", "text": "q3"}\n',
        encoding="utf-8",
    )
    (pipe_dir / "qrels.tsv").write_text(
        "q1\t0\td1a\t3\nq1\t0\td1b\t1\nq2\t0\td2a\t2\nq2\t0\td2b\t1\nq3\t0\td3a\t3\n",
        encoding="utf-8",
    )
    qrels = load_qrels_tsv(pipe_dir / "qrels.tsv", pipeline="spine_smoke", collection="eval")
    baseline_data = {
        "schema_version": 1,
        "pipeline": "spine_smoke",
        "qrels_hash": qrels_hash(qrels),
        "corpus_snapshot": "fixture:test",
        "created_at": "2026-06-09T00:00:00Z",
        "aggregate": baseline,
    }
    (pipe_dir / "baseline.json").write_text(json.dumps(baseline_data), encoding="utf-8")
    return pipe_dir


def _perfect_search(query: str, top_k: int) -> list[str]:
    return {
        "q1": ["d1a", "d1b", "junk1", "junk2"],
        "q2": ["d2a", "d2b", "junk3"],
        "q3": ["d3a", "junk4", "junk5"],
    }[query][:top_k]


def _degraded_search(query: str, top_k: int) -> list[str]:
    return ["junk"] * top_k


def _evaluator(search_fn: _SEARCH_FN) -> tuple[RetrievalVariant, _SEARCH_FN]:
    return RetrievalVariant(name="smoke", top_k=10), search_fn


def test_gate_passes_when_no_pipeline_is_active() -> None:
    registry = {"spine_smoke": _spec("spine_smoke", PipelineStatus.PLANNED)}
    result = run_eval_gate(registry, evaluators={}, fixture_root=Path("/nope"))
    assert isinstance(result, EvalGateResult)
    assert result.passed is True
    assert result.checked == []
    assert result.failed == []


def test_gate_passes_at_baseline(tmp_path: Path) -> None:
    _write_fixture(tmp_path, baseline={"ndcg@10": 1.0, "recall@10": 1.0, "mrr": 1.0})
    variant, search = _evaluator(_perfect_search)
    registry = {"spine_smoke": _spec("spine_smoke", PipelineStatus.ACTIVE)}
    evaluators = {"spine_smoke": (variant, search)}
    result = run_eval_gate(registry, evaluators=evaluators, fixture_root=tmp_path)
    assert result.passed is True, result
    assert "spine_smoke" in result.checked
    assert result.failed == []
    assert result.missing_evaluators == []
    assert result.missing_fixtures == []
    assert result.regressions == []


def test_gate_fails_when_search_degrades(tmp_path: Path) -> None:
    _write_fixture(tmp_path, baseline={"ndcg@10": 1.0, "recall@10": 1.0, "mrr": 1.0})
    variant, search = _evaluator(_degraded_search)
    registry = {"spine_smoke": _spec("spine_smoke", PipelineStatus.ACTIVE)}
    evaluators = {"spine_smoke": (variant, search)}
    result = run_eval_gate(registry, evaluators=evaluators, fixture_root=tmp_path)
    assert result.passed is False
    assert "spine_smoke" in result.failed
    assert "spine_smoke" not in result.checked
    assert result.regressions, "degraded search must trip assert_no_regression"


def test_gate_fails_when_active_pipeline_has_no_evaluator(tmp_path: Path) -> None:
    _write_fixture(tmp_path, baseline={"ndcg@10": 1.0, "recall@10": 1.0, "mrr": 1.0})
    registry = {"spine_smoke": _spec("spine_smoke", PipelineStatus.ACTIVE)}
    # Evaluators map is empty - this is the regression path.
    result = run_eval_gate(registry, evaluators={}, fixture_root=tmp_path)
    assert result.passed is False
    assert "spine_smoke" in result.missing_evaluators
    assert "spine_smoke" not in result.checked


def test_gate_fails_when_active_pipeline_has_no_fixture(tmp_path: Path) -> None:
    registry = {"spine_smoke": _spec("spine_smoke", PipelineStatus.ACTIVE)}
    variant, search = _evaluator(_perfect_search)
    evaluators = {"spine_smoke": (variant, search)}
    result = run_eval_gate(registry, evaluators=evaluators, fixture_root=tmp_path)
    assert result.passed is False
    assert "spine_smoke" in result.missing_fixtures
