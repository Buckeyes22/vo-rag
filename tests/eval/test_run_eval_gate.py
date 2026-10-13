"""Script-level eval gate mode selection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import scripts.run_eval_gate as run_eval_gate_script

from vo_rag.core.config.pipelines import PipelineSpec, PipelineStatus, load_pipeline_registry
from vo_rag.core.eval import registered as registered_mod
from vo_rag.core.eval.gate import EvalGateResult
from vo_rag.core.eval.qrels import load_qrels_tsv
from vo_rag.core.eval.registered import run_registered_eval_gate
from vo_rag.pipelines.book_rag import eval as book_eval


def test_build_evaluators_defaults_to_replay_smoke(monkeypatch) -> None:
    monkeypatch.delenv("VO_RAG_EVAL_MODE", raising=False)

    evaluators = run_eval_gate_script._build_evaluators(load_pipeline_registry())

    assert evaluators["book_rag"].mode == "replay_smoke"


def test_ci_eval_gate_does_not_use_replay_search(monkeypatch) -> None:
    called = False

    def forbidden_replay(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("CI gate must not use replay search")

    monkeypatch.setenv("VO_RAG_EVAL_MODE", "retrieval_gate")
    monkeypatch.setattr(book_eval, "build_replay_search_fn", forbidden_replay)

    evaluators = run_eval_gate_script._build_evaluators(load_pipeline_registry())

    # The re-grounded snapshot is checked in and the script supplies the
    # retriever factory, so CI retrieval mode arms fully — and never replays.
    assert evaluators["book_rag"].mode == "retrieval_gate"
    assert called is False


_REPO_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eval" / "book_rag"


def _write_armed_tmp_fixtures(tmp_path: Path, monkeypatch) -> Path:
    """Re-root the script's fixture lookup at a tmp copy with a SYNTHETIC
    corpus snapshot so the armed retrieval-gate wiring can be pinned without
    any checked-in snapshot artifact.

    The snapshot text is deliberately meaningless and the tmp baseline's
    ranking metrics are zeroed: these tests pin WIRING (the armed path runs
    end-to-end through the real SnapshotBookRagRetriever), never retrieval
    quality. A checked-in snapshot fabricated from the eval queries is the
    forbidden tautology — see scripts/rebuild_book_rag_eval_corpus.py.
    """
    book_dir = tmp_path / "book_rag"
    book_dir.mkdir()
    for name in ("queries.jsonl", "qrels.tsv"):
        (book_dir / name).write_text((_REPO_FIXTURES / name).read_text(encoding="utf-8"))

    baseline = json.loads((_REPO_FIXTURES / "baseline.json").read_text(encoding="utf-8"))
    baseline["aggregate"] = {
        key: (0.0 if key.startswith(("ndcg@", "recall@", "precision@")) or key == "mrr" else value)
        for key, value in baseline["aggregate"].items()
    }
    (book_dir / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")

    qrels = load_qrels_tsv(book_dir / "qrels.tsv", pipeline="book_rag", collection="eval")
    doc_ids = sorted({judgment.doc_id for judgment in qrels.judgments})
    snapshot_lines = [
        json.dumps(
            {"chunk_id": doc_id, "source_id": "tmp-wiring-test", "text": f"synthetic chunk {i}"}
        )
        for i, doc_id in enumerate(doc_ids)
    ]
    (book_dir / book_eval.BOOK_RAG_EVAL_CORPUS_SNAPSHOT).write_text(
        "\n".join(snapshot_lines) + "\n", encoding="utf-8"
    )

    monkeypatch.setattr(run_eval_gate_script, "_FIXTURE_ROOT", tmp_path)
    monkeypatch.setattr(
        run_eval_gate_script,
        "run_registered_eval_gate",
        lambda registry, **kw: run_registered_eval_gate(registry, fixture_root=tmp_path, **kw),
    )
    return book_dir


def test_build_evaluators_can_arm_retrieval_gate_with_retriever_factory(
    monkeypatch, tmp_path: Path
) -> None:
    book_dir = tmp_path / "book_rag"
    book_dir.mkdir()
    source_book_dir = (
        Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eval" / "book_rag"
    )
    for name in ("queries.jsonl", "qrels.tsv", "baseline.json"):
        (book_dir / name).write_text((source_book_dir / name).read_text(encoding="utf-8"))
    (book_dir / book_eval.BOOK_RAG_EVAL_CORPUS_SNAPSHOT).write_text("{}\n", encoding="utf-8")

    class _FakeRetriever:
        async def hybrid_search(self, *args, **kwargs):
            return []

    monkeypatch.setenv("VO_RAG_EVAL_MODE", "retrieval_gate")

    evaluators = run_eval_gate_script.build_eval_gate_evaluators(
        load_pipeline_registry(),
        fixture_root=tmp_path,
        retriever_factory=lambda: _FakeRetriever(),
    )

    assert evaluators["book_rag"].mode == "retrieval_gate"


def test_retrieval_gate_uses_live_search_when_snapshot_exists(monkeypatch, tmp_path: Path) -> None:
    book_dir = tmp_path / "book_rag"
    book_dir.mkdir()
    source_book_dir = (
        Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eval" / "book_rag"
    )
    for name in ("queries.jsonl", "qrels.tsv", "baseline.json"):
        (book_dir / name).write_text((source_book_dir / name).read_text(encoding="utf-8"))

    qrels = load_qrels_tsv(book_dir / "qrels.tsv", pipeline="book_rag", collection="eval")
    # Synthetic snapshot resolving the qrels ids; content is irrelevant here
    # because the retriever below is a fake — this pins gate wiring only.
    snapshot_doc_ids = sorted({judgment.doc_id for judgment in qrels.judgments})
    (book_dir / book_eval.BOOK_RAG_EVAL_CORPUS_SNAPSHOT).write_text(
        "\n".join(
            json.dumps({"chunk_id": doc_id, "source_id": "tmp", "text": f"synthetic {i}"})
            for i, doc_id in enumerate(snapshot_doc_ids)
        )
        + "\n",
        encoding="utf-8",
    )
    by_query = qrels.by_query()
    query_text_to_id: dict[str, str] = {}
    for raw in (book_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        query_text_to_id[str(record["query_text"])] = str(record["query_id"])

    class _FakeRetriever:
        async def hybrid_search(self, query, *args, **kwargs):
            query_id = query_text_to_id[query]
            return [
                type("_Result", (), {"id": doc_id, "payload": {"chunk_id": doc_id}})()
                for doc_id in by_query[query_id]
            ]

    monkeypatch.setenv("VO_RAG_EVAL_MODE", "retrieval_gate")

    result = run_registered_eval_gate(
        load_pipeline_registry(),
        fixture_root=tmp_path,
        retriever_factory=lambda: _FakeRetriever(),
    )

    assert result.passed is True
    assert result.checked == ["book_rag"]
    assert result.blocked_preconditions == {}


def test_registered_eval_gate_reports_blocked_retrieval_mode(monkeypatch) -> None:
    """Snapshot checked in but NO retriever factory -> blocked_missing_retriever."""
    monkeypatch.setenv("VO_RAG_EVAL_MODE", "retrieval_gate")

    registry = load_pipeline_registry()
    result = run_registered_eval_gate(registry)

    assert result.passed is False
    assert result.blocked_preconditions == {"book_rag": "blocked_missing_retriever"}
    assert result.failed == ["book_rag"]
    assert result.skipped == [
        name for name, spec in registry.items() if spec.status.value != "active"
    ]


def test_registered_eval_gate_reports_blocked_missing_retriever(monkeypatch, tmp_path) -> None:
    """Snapshot present but no retriever wired -> the OTHER blocked label."""
    book_dir = tmp_path / "book_rag"
    book_dir.mkdir()
    for name in ("queries.jsonl", "qrels.tsv", "baseline.json"):
        (book_dir / name).write_text((_REPO_FIXTURES / name).read_text(encoding="utf-8"))
    (book_dir / book_eval.BOOK_RAG_EVAL_CORPUS_SNAPSHOT).write_text(
        '{"chunk_id": "x", "source_id": "tmp", "text": "t"}\n', encoding="utf-8"
    )
    monkeypatch.setenv("VO_RAG_EVAL_MODE", "retrieval_gate")

    result = run_registered_eval_gate(load_pipeline_registry(), fixture_root=tmp_path)

    assert result.passed is False
    assert result.blocked_preconditions == {"book_rag": "blocked_missing_retriever"}


def test_registered_eval_gate_partitions_blocked_and_checked_active_pipelines(
    monkeypatch,
) -> None:
    blocked_entry = object()
    healthy_entry = object()
    registry = {
        "book_rag": PipelineSpec(name="book_rag", status=PipelineStatus.ACTIVE, summary="book"),
        "code_rag": PipelineSpec(name="code_rag", status=PipelineStatus.ACTIVE, summary="code"),
        "idea_capture": PipelineSpec(
            name="idea_capture", status=PipelineStatus.PLANNED, summary="idea"
        ),
    }

    def _fake_build_eval_gate_evaluators(
        registry_arg, *, fixture_root, retriever_factory=None, relevance_mode=None, collection=None
    ):
        assert registry_arg is registry
        assert retriever_factory is None
        return {"book_rag": blocked_entry, "code_rag": healthy_entry}

    def _fake_blocked_eval_gate_modes(evaluators):
        assert evaluators == {"book_rag": blocked_entry, "code_rag": healthy_entry}
        return {"book_rag": "blocked_missing_retriever"}

    def _fake_run_eval_gate(registry_arg, *, evaluators, fixture_root, relevance_mode=None):
        assert set(registry_arg) == {"code_rag", "idea_capture"}
        assert evaluators == {"code_rag": healthy_entry}
        return EvalGateResult(checked=["code_rag"], skipped=["idea_capture"])

    monkeypatch.setattr(
        registered_mod, "build_eval_gate_evaluators", _fake_build_eval_gate_evaluators
    )
    monkeypatch.setattr(registered_mod, "blocked_eval_gate_modes", _fake_blocked_eval_gate_modes)
    monkeypatch.setattr(registered_mod, "run_eval_gate", _fake_run_eval_gate)

    result = registered_mod.run_registered_eval_gate(registry)

    assert result.passed is False
    assert result.checked == ["code_rag"]
    assert result.skipped == ["idea_capture"]
    assert result.failed == ["book_rag"]
    assert result.blocked_preconditions == {"book_rag": "blocked_missing_retriever"}


def test_script_eval_gate_runs_retrieval_mode_when_snapshot_exists(
    monkeypatch, capsys, tmp_path
) -> None:
    """Armed-path wiring pin via a tmp synthetic snapshot (never checked in)."""
    _write_armed_tmp_fixtures(tmp_path, monkeypatch)
    monkeypatch.setenv("VO_RAG_EVAL_MODE", "retrieval_gate")
    monkeypatch.delenv("VO_RAG_LIVE_CORPUS_SNAPSHOT", raising=False)

    exit_code = run_eval_gate_script.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "checked pipelines (matched baseline): ['book_rag']" in captured.out
    assert "eval gate PASS" in captured.out


def test_script_eval_gate_rejects_live_snapshot_mismatch(monkeypatch, capsys, tmp_path) -> None:
    _write_armed_tmp_fixtures(tmp_path, monkeypatch)
    monkeypatch.setenv("VO_RAG_EVAL_MODE", "retrieval_gate")
    monkeypatch.setenv("VO_RAG_LIVE_CORPUS_SNAPSHOT", "sha256:not-the-baseline")

    exit_code = run_eval_gate_script.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "baseline corpus_snapshot does not match live corpus identity" in captured.out
    assert "sha256:not-the-baseline" in captured.out


def test_script_eval_gate_supplies_retriever_factory_in_retrieval_mode(
    monkeypatch, tmp_path
) -> None:
    _write_armed_tmp_fixtures(tmp_path, monkeypatch)
    monkeypatch.setenv("VO_RAG_EVAL_MODE", "retrieval_gate")

    def fake_run_registered_eval_gate(registry, *, retriever_factory=None):
        assert "book_rag" in registry
        assert retriever_factory is not None
        assert hasattr(retriever_factory(), "hybrid_search")
        return EvalGateResult(checked=["book_rag"])

    monkeypatch.setattr(
        run_eval_gate_script,
        "run_registered_eval_gate",
        fake_run_registered_eval_gate,
    )

    exit_code = run_eval_gate_script.main()

    assert exit_code == 0


def test_script_eval_gate_reports_failed_pipelines(monkeypatch, capsys) -> None:
    def fake_run_registered_eval_gate(registry, **kwargs):
        assert "book_rag" in registry
        return EvalGateResult(
            checked=["book_rag"],
            failed=["book_rag"],
            regressions=["book_rag: nDCG@10 regressed"],
        )

    monkeypatch.setattr(
        run_eval_gate_script,
        "run_registered_eval_gate",
        fake_run_registered_eval_gate,
    )

    exit_code = run_eval_gate_script.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "checked pipelines (matched baseline): ['book_rag']" in captured.out
    assert "failed: retrieval regression detected:" in captured.out
    assert "book_rag: nDCG@10 regressed" in captured.out
    assert "eval gate FAIL" in captured.out
    assert "book_rag" in captured.out


def test_script_eval_gate_blocked_does_not_mask_other_failures(monkeypatch, capsys) -> None:
    """A blocked pipeline must not whitewash a genuine failure elsewhere.

    ci.sh tolerates only BLOCKED_EXIT_CODE; if a co-occurring real failure
    were folded into exit 3, CI would stay green on a regression.
    """

    def fake_run_registered_eval_gate(registry, **kwargs):
        return EvalGateResult(
            checked=["code_rag"],
            failed=["book_rag", "code_rag"],
            regressions=["code_rag: nDCG@10 regressed"],
            blocked_preconditions={"book_rag": "blocked_missing_corpus_snapshot"},
        )

    monkeypatch.setattr(
        run_eval_gate_script,
        "run_registered_eval_gate",
        fake_run_registered_eval_gate,
    )

    exit_code = run_eval_gate_script.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "eval gate BLOCKED" in captured.out
    assert "code_rag: nDCG@10 regressed" in captured.out
    assert "eval gate FAIL" in captured.out


def test_script_eval_gate_no_model_backend_warning() -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("VO_RAG_EVAL_MODE", None)

    result = subprocess.run(
        [sys.executable, "scripts/run_eval_gate.py"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "eval gate PASS" in result.stdout
    assert result.stderr == ""


def test_cli_eval_gate_no_model_backend_warning() -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("VO_RAG_EVAL_MODE", None)

    result = subprocess.run(
        [sys.executable, "-m", "vo_rag.cli.main", "eval-gate"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "eval gate PASS" in result.stdout
    assert result.stderr == ""
