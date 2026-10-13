import asyncio
import json
import warnings

from vo_rag.core.config.pipelines import PipelineSpec, PipelineStatus
from vo_rag.core.eval.gate import BaselineV1, EvalGateError, EvalGateResult, run_eval_gate
from vo_rag.core.eval.harness import RetrievalVariant
from vo_rag.core.eval.qrels import load_qrels_tsv, qrels_hash
from vo_rag.core.models.common import VoBaseModel


def _spec(name, status):
    return PipelineSpec(name=name, status=status, summary="s")


def _fake_search_fn(query, top_k):
    return ["doc1"]


def _make_fixture_dir(tmp_path, baseline_data):
    fixture_dir = tmp_path / "book_rag"
    fixture_dir.mkdir()

    queries = [
        {"query_id": "q1", "text": "test query"},
    ]
    (fixture_dir / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")

    (fixture_dir / "qrels.tsv").write_text("q1\t0\tdoc1\t1\n")
    (fixture_dir / "baseline.json").write_text(json.dumps(baseline_data, indent=2))

    return fixture_dir


def test_eval_gate_result_passed_is_derived_from_failure_fields():
    constructed = EvalGateResult(failed=["book_rag"])
    assert constructed.passed is False
    assert constructed.model_dump()["passed"] is False

    mutated = EvalGateResult()
    assert mutated.passed is True
    mutated.blocked_preconditions["book_rag"] = "blocked_missing_retriever"
    assert mutated.passed is False
    assert mutated.model_dump()["passed"] is False


def test_gate_passes_when_no_pipeline_is_active():
    reg = {"book_rag": _spec("book_rag", PipelineStatus.NEXT)}
    result = run_eval_gate(reg, baseline_exists=lambda name: False)
    assert isinstance(result, EvalGateResult)
    assert result.passed is True
    assert result.checked == []


def test_gate_fails_when_active_pipeline_lacks_baseline():
    reg = {"book_rag": _spec("book_rag", PipelineStatus.ACTIVE)}
    result = run_eval_gate(reg, baseline_exists=lambda name: False)
    assert result.passed is False
    assert "book_rag" in result.missing_baselines


def test_gate_passes_when_active_pipeline_has_baseline():
    reg = {"book_rag": _spec("book_rag", PipelineStatus.ACTIVE)}
    result = run_eval_gate(reg, baseline_exists=lambda name: True)
    assert result.passed is True
    assert result.missing_baselines == []


def test_gate_raises_evalgateerror_on_qrels_hash_mismatch(tmp_path):
    baseline_data = {
        "schema_version": 1,
        "pipeline": "book_rag",
        "qrels_hash": "sha256:wrong_hash",
        "corpus_snapshot": "fixture:test",
        "created_at": "2026-06-09T00:00:00Z",
        "aggregate": {"recall@5": 0.0, "ndcg@10": 0.0, "mrr": 0.0},
    }
    _make_fixture_dir(tmp_path, baseline_data)

    reg = {"book_rag": _spec("book_rag", PipelineStatus.ACTIVE)}
    evaluators = {"book_rag": (RetrievalVariant(name="test", top_k=10), _fake_search_fn)}

    result = run_eval_gate(reg, evaluators=evaluators, fixture_root=tmp_path)

    assert result.passed is False
    assert any("qrels_hash" in r for r in result.regressions)
    assert "book_rag" in result.failed


def test_gate_passes_on_matching_qrels_hash(tmp_path):
    fixture_dir = tmp_path / "book_rag"
    fixture_dir.mkdir()

    queries = [
        {"query_id": "q1", "text": "test query"},
    ]
    (fixture_dir / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    (fixture_dir / "qrels.tsv").write_text("q1\t0\tdoc1\t1\n")

    qrels = load_qrels_tsv(fixture_dir / "qrels.tsv", pipeline="book_rag", collection="eval")
    correct_hash = qrels_hash(qrels)

    baseline_data = {
        "schema_version": 1,
        "pipeline": "book_rag",
        "qrels_hash": correct_hash,
        "corpus_snapshot": "fixture:test",
        "created_at": "2026-06-09T00:00:00Z",
        "aggregate": {"recall@5": 0.0, "ndcg@10": 0.0, "mrr": 0.0},
    }
    (fixture_dir / "baseline.json").write_text(json.dumps(baseline_data, indent=2))

    reg = {"book_rag": _spec("book_rag", PipelineStatus.ACTIVE)}
    evaluators = {"book_rag": (RetrievalVariant(name="test", top_k=10), _fake_search_fn)}

    result = run_eval_gate(reg, evaluators=evaluators, fixture_root=tmp_path)

    assert result.passed is True
    assert "book_rag" in result.checked


def test_gate_rejects_corpus_snapshot_mismatch(tmp_path, monkeypatch):
    fixture_dir = tmp_path / "book_rag"
    fixture_dir.mkdir()

    queries = [
        {"query_id": "q1", "text": "test query"},
    ]
    (fixture_dir / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    (fixture_dir / "qrels.tsv").write_text("q1\t0\tdoc1\t1\n")

    qrels = load_qrels_tsv(fixture_dir / "qrels.tsv", pipeline="book_rag", collection="eval")
    baseline_data = {
        "schema_version": 1,
        "pipeline": "book_rag",
        "qrels_hash": qrels_hash(qrels),
        "corpus_snapshot": "fixture:old",
        "created_at": "2026-06-09T00:00:00Z",
        "aggregate": {"recall@5": 0.0, "ndcg@10": 0.0, "mrr": 0.0},
    }
    (fixture_dir / "baseline.json").write_text(json.dumps(baseline_data, indent=2))

    monkeypatch.delenv("VO_RAG_ALLOW_CORPUS_SNAPSHOT_MISMATCH", raising=False)

    reg = {"book_rag": _spec("book_rag", PipelineStatus.ACTIVE)}
    evaluators = {"book_rag": (RetrievalVariant(name="test", top_k=10), _fake_search_fn)}

    result = run_eval_gate(
        reg,
        evaluators=evaluators,
        fixture_root=tmp_path,
        corpus_snapshot="fixture:current",
    )

    assert result.passed is False
    assert any("corpus_snapshot" in r and "fixture:old" in r for r in result.regressions)
    assert "book_rag" in result.failed


def test_gate_allows_corpus_snapshot_mismatch_with_env_var(tmp_path, monkeypatch):
    fixture_dir = tmp_path / "book_rag"
    fixture_dir.mkdir()

    queries = [
        {"query_id": "q1", "text": "test query"},
    ]
    (fixture_dir / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    (fixture_dir / "qrels.tsv").write_text("q1\t0\tdoc1\t1\n")

    qrels = load_qrels_tsv(fixture_dir / "qrels.tsv", pipeline="book_rag", collection="eval")
    baseline_data = {
        "schema_version": 1,
        "pipeline": "book_rag",
        "qrels_hash": qrels_hash(qrels),
        "corpus_snapshot": "fixture:old",
        "created_at": "2026-06-09T00:00:00Z",
        "aggregate": {"recall@5": 0.0, "ndcg@10": 0.0, "mrr": 0.0},
    }
    (fixture_dir / "baseline.json").write_text(json.dumps(baseline_data, indent=2))

    monkeypatch.setenv("VO_RAG_ALLOW_CORPUS_SNAPSHOT_MISMATCH", "1")

    reg = {"book_rag": _spec("book_rag", PipelineStatus.ACTIVE)}
    evaluators = {"book_rag": (RetrievalVariant(name="test", top_k=10), _fake_search_fn)}

    result = run_eval_gate(
        reg,
        evaluators=evaluators,
        fixture_root=tmp_path,
        corpus_snapshot="fixture:current",
    )

    assert result.passed is True
    assert "book_rag" in result.checked


def test_gate_reports_malformed_schema_baseline_as_evalgateerror(tmp_path):
    baseline_data = {
        "schema_version": 1,
        "pipeline": "book_rag",
        "corpus_snapshot": "fixture:test",
        "created_at": "2026-06-09T00:00:00Z",
        "aggregate": {"recall@5": 0.0, "ndcg@10": 0.0, "mrr": 0.0},
    }
    _make_fixture_dir(tmp_path, baseline_data)

    reg = {"book_rag": _spec("book_rag", PipelineStatus.ACTIVE)}
    evaluators = {"book_rag": (RetrievalVariant(name="test", top_k=10), _fake_search_fn)}

    result = run_eval_gate(
        reg,
        evaluators=evaluators,
        fixture_root=tmp_path,
        corpus_snapshot="fixture:test",
    )

    assert result.passed is False
    assert any("baseline schema invalid" in r for r in result.regressions)
    assert not any("harness error" in r for r in result.regressions)
    assert "book_rag" in result.failed


def test_gate_rejects_legacy_baseline_without_env_var(tmp_path, monkeypatch):
    baseline_data = {
        "recall@5": 0.0,
        "ndcg@10": 0.0,
        "mrr": 0.0,
    }
    _make_fixture_dir(tmp_path, baseline_data)

    monkeypatch.delenv("VO_RAG_ALLOW_LEGACY_BASELINE", raising=False)

    reg = {"book_rag": _spec("book_rag", PipelineStatus.ACTIVE)}
    evaluators = {"book_rag": (RetrievalVariant(name="test", top_k=10), _fake_search_fn)}

    result = run_eval_gate(reg, evaluators=evaluators, fixture_root=tmp_path)

    assert result.passed is False
    assert any("legacy" in r for r in result.regressions)
    assert "book_rag" in result.failed


def test_gate_allows_legacy_baseline_with_env_var(tmp_path, monkeypatch):
    baseline_data = {
        "recall@5": 0.0,
        "ndcg@10": 0.0,
        "mrr": 0.0,
    }
    _make_fixture_dir(tmp_path, baseline_data)

    monkeypatch.setenv("VO_RAG_ALLOW_LEGACY_BASELINE", "1")

    reg = {"book_rag": _spec("book_rag", PipelineStatus.ACTIVE)}
    evaluators = {"book_rag": (RetrievalVariant(name="test", top_k=10), _fake_search_fn)}

    result = run_eval_gate(reg, evaluators=evaluators, fixture_root=tmp_path)

    assert result.passed is True
    assert "book_rag" in result.checked


def test_evalgateerror_is_exception():
    assert issubclass(EvalGateError, Exception)


def test_baseline_v1_uses_vo_base_model_config():
    assert issubclass(BaselineV1, VoBaseModel)

    baseline = BaselineV1.model_validate(
        {
            "schema_version": 1,
            "pipeline": " book_rag ",
            "qrels_hash": " sha256:test ",
            "corpus_snapshot": " fixture:test ",
            "created_at": " 2026-06-09T00:00:00Z ",
            "aggregate": {"recall@5": 0.0},
        }
    )

    assert baseline.pipeline == "book_rag"
    assert baseline.qrels_hash == "sha256:test"


def test_gate_inside_running_loop_fails_without_unawaited_warning(tmp_path):
    fixture_dir = tmp_path / "book_rag"
    fixture_dir.mkdir()

    queries = [
        {"query_id": "q1", "text": "test query"},
    ]
    (fixture_dir / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    (fixture_dir / "qrels.tsv").write_text("q1\t0\tdoc1\t1\n")

    qrels = load_qrels_tsv(fixture_dir / "qrels.tsv", pipeline="book_rag", collection="eval")
    (fixture_dir / "baseline.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "book_rag",
                "qrels_hash": qrels_hash(qrels),
                "corpus_snapshot": "fixture:test",
                "created_at": "2026-06-09T00:00:00Z",
                "aggregate": {"recall@5": 1.0, "ndcg@10": 1.0, "mrr": 1.0},
            },
            indent=2,
        )
    )

    reg = {"book_rag": _spec("book_rag", PipelineStatus.ACTIVE)}
    evaluators = {"book_rag": (RetrievalVariant(name="test", top_k=10), _fake_search_fn)}

    async def _call_gate() -> EvalGateResult:
        return run_eval_gate(reg, evaluators=evaluators, fixture_root=tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = asyncio.run(_call_gate())

    assert result.passed is False
    assert any("active event loop" in regression for regression in result.regressions)
    assert not any("was never awaited" in str(warning.message) for warning in caught)
