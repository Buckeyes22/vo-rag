"""Gate-wiring contract tests for ``refresh_book_rag_qrels`` (VOR-235).

Verifies that a regenerated, provenance-passing ``Qrels`` is written to the
fixture directory and is consumable by ``run_eval_gate`` without modification.
Also verifies the non-destructive failure path: a provenance-failing qrels set
raises and writes no fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from scripts.rebuild_book_rag_eval_corpus import (
    FabricatedSnapshotError,
    run_freeze_book_rag_baseline,
    run_rebuild_book_rag_eval_corpus,
)

from vo_rag.core.config.pipelines import PipelineSpec, PipelineStatus
from vo_rag.core.eval.gate import BaselineV1, run_eval_gate
from vo_rag.core.eval.qrels import load_qrels_tsv, qrels_hash
from vo_rag.integrations.llamaindex.eval_dataset import (
    QrelsProvenanceError,
    refresh_book_rag_qrels,
)

if TYPE_CHECKING:
    from llama_index.core.llama_dataset.legacy.embedding import (
        EmbeddingQAFinetuneDataset,
    )


def _make_qa_dataset(
    query_ids: list[str],
    doc_ids: list[list[str]],
) -> EmbeddingQAFinetuneDataset:
    """Build a deterministic ``EmbeddingQAFinetuneDataset``."""
    from llama_index.core.llama_dataset.legacy.embedding import (
        EmbeddingQAFinetuneDataset,
    )

    queries = {qid: f"Query text for {qid}" for qid in query_ids}
    relevant_docs = dict(zip(query_ids, doc_ids, strict=True))
    corpus = {doc_id: f"Corpus for {doc_id}" for group in doc_ids for doc_id in group}
    return EmbeddingQAFinetuneDataset(
        queries=queries,
        relevant_docs=relevant_docs,
        corpus=corpus,
    )


def _make_queries_jsonl(query_ids: list[str]) -> str:
    """Build a ``queries.jsonl`` string."""
    lines: list[str] = []
    for qid in query_ids:
        record = {"query_id": qid, "text": f"Query text for {qid}"}
        lines.append(json.dumps(record))
    return "\n".join(lines) + "\n"


def _make_baseline_json(qrels_hash_value: str) -> str:
    """Build a provenance-bound baseline (F18 schema) with floors low enough for
    a perfect replay evaluator. The qrels_hash must match the refreshed qrels so
    the gate's provenance check clears."""
    return json.dumps(
        {
            "schema_version": 1,
            "pipeline": "book_rag",
            "qrels_hash": qrels_hash_value,
            "corpus_snapshot": "fixture:test",
            "created_at": "2026-06-09T00:00:00+00:00",
            "aggregate": {
                "ndcg@10": 0.5,
                "recall@10": 0.5,
                "mrr": 0.5,
            },
        }
    )


def _write_rebuild_fixtures(fixture_dir: Path) -> Path:
    """Two queries, two judged chunks; q2's chunk shares no terms with its
    query so frozen metrics are deliberately imperfect (the perfection guard
    rejects uniformly-1.0 baselines as a fabrication smell)."""
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "queries.jsonl").write_text(
        json.dumps({"query_id": "q1", "query_text": "alpha bravo signal"})
        + "\n"
        + json.dumps({"query_id": "q2", "query_text": "delta echo signal"})
        + "\n",
        encoding="utf-8",
    )
    (fixture_dir / "qrels.tsv").write_text(
        "q1\t0\tc1\t1\nq2\t0\tc2\t1\n",
        encoding="utf-8",
    )
    export = fixture_dir / "source_export.jsonl"
    export.write_text(
        json.dumps(
            {"chunk_id": "c1", "source_id": "book-a", "text": "alpha bravo corpus prose racing"}
        )
        + "\n"
        + json.dumps({"chunk_id": "c2", "source_id": "book-b", "text": "zulu yankee corpus prose"})
        + "\n",
        encoding="utf-8",
    )
    return export


def test_rebuild_book_rag_eval_corpus_writes_snapshot_from_real_export(tmp_path):
    fixture_dir = tmp_path / "book_rag"
    export = _write_rebuild_fixtures(fixture_dir)
    output = fixture_dir / "corpus_snapshot.jsonl"

    result = run_rebuild_book_rag_eval_corpus(
        source_export=export, output=output, fixture_dir=fixture_dir
    )

    assert output.is_file()
    assert result["snapshot_path"] == str(output)
    assert result["n_chunks"] == 2
    assert result["source_export"] == str(export)
    assert len(result["source_export_hash"]) == 64


def test_rebuild_rejects_snapshot_fabricated_from_queries(tmp_path):
    """The answer-key-as-corpus tautology must be mechanically impossible."""
    fixture_dir = tmp_path / "book_rag"
    export = _write_rebuild_fixtures(fixture_dir)
    export.write_text(
        json.dumps({"chunk_id": "c1", "source_id": "x", "text": "alpha bravo signal"})
        + "\n"
        + json.dumps({"chunk_id": "c2", "source_id": "x", "text": "real prose"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FabricatedSnapshotError, match="answer-key-as-corpus"):
        run_rebuild_book_rag_eval_corpus(
            source_export=export,
            output=fixture_dir / "corpus_snapshot.jsonl",
            fixture_dir=fixture_dir,
        )


def test_freeze_book_rag_baseline_binds_snapshot_hash(tmp_path):
    fixture_dir = tmp_path / "book_rag"
    export = _write_rebuild_fixtures(fixture_dir)
    run_rebuild_book_rag_eval_corpus(
        source_export=export,
        output=fixture_dir / "corpus_snapshot.jsonl",
        fixture_dir=fixture_dir,
    )
    baseline_path = fixture_dir / "baseline.json"

    result = run_freeze_book_rag_baseline(
        fixture_dir=fixture_dir,
        baseline_path=baseline_path,
    )

    parsed = BaselineV1.model_validate(json.loads(baseline_path.read_text(encoding="utf-8")))
    assert result["baseline_path"] == str(baseline_path)
    assert parsed.corpus_snapshot == f"sha256:{result['snapshot_hash']}"
    assert parsed.aggregate["mrr"] < 1.0  # imperfect by fixture design


def test_freeze_refuses_uniformly_perfect_metrics(tmp_path):
    """All-1.0 ranking metrics are a fabrication smell, not a result."""
    fixture_dir = tmp_path / "book_rag"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "queries.jsonl").write_text(
        json.dumps({"query_id": "q1", "query_text": "alpha bravo signal"}) + "\n",
        encoding="utf-8",
    )
    (fixture_dir / "qrels.tsv").write_text("q1\t0\tc1\t1\n", encoding="utf-8")
    # Full term overlap WITHOUT verbatim query embedding (different order), so
    # the rebuild-time check passes but retrieval is trivially perfect.
    export = fixture_dir / "source_export.jsonl"
    export.write_text(
        json.dumps({"chunk_id": "c1", "source_id": "x", "text": "signal bravo alpha corpus"})
        + "\n",
        encoding="utf-8",
    )
    run_rebuild_book_rag_eval_corpus(
        source_export=export,
        output=fixture_dir / "corpus_snapshot.jsonl",
        fixture_dir=fixture_dir,
    )

    with pytest.raises(FabricatedSnapshotError, match="uniformly perfect"):
        run_freeze_book_rag_baseline(fixture_dir=fixture_dir)


class TestRefreshBookRagQrels:
    """Contract tests for ``refresh_book_rag_qrels``."""

    def test_provenance_passing_writes_fixture_and_gate_passes(
        self,
        tmp_path: Path,
    ) -> None:
        """A provenance-passing qa set writes ``qrels.tsv`` and the gate clears."""
        query_ids = ["q-001", "q-002"]
        doc_ids = [["doc-a", "doc-b"], ["doc-c"]]
        qa = _make_qa_dataset(query_ids, doc_ids)

        fixture_root = tmp_path / "fixtures" / "eval"
        book_rag_dir = fixture_root / "book_rag"
        book_rag_dir.mkdir(parents=True)

        # Seed the queries companion fixture required by run_eval_gate.
        (book_rag_dir / "queries.jsonl").write_text(
            _make_queries_jsonl(query_ids),
            encoding="utf-8",
        )

        def resolve_fn(ids: set[str]) -> set[str]:
            return ids  # all resolve

        qrels_path = refresh_book_rag_qrels(
            qa,
            fixture_root=fixture_root,
            resolve_fn=resolve_fn,
        )

        assert qrels_path == (book_rag_dir / "qrels.tsv").resolve()
        assert qrels_path.is_file()

        # Bind the baseline to the refreshed qrels so the F18 provenance check
        # clears (qrels_hash must match what run_eval_gate computes from qrels.tsv).
        loaded_qrels = load_qrels_tsv(qrels_path, pipeline="book_rag", collection="eval")
        (book_rag_dir / "baseline.json").write_text(
            _make_baseline_json(qrels_hash(loaded_qrels)),
            encoding="utf-8",
        )

        # Wire the gate: ACTIVE book_rag with a replay evaluator.
        registry = {
            "book_rag": PipelineSpec(
                name="book_rag",
                status=PipelineStatus.ACTIVE,
                summary="test",
            ),
        }
        from vo_rag.pipelines.book_rag.eval import book_rag_evaluator

        evaluators = {
            "book_rag": book_rag_evaluator(book_rag_dir),
        }

        result = run_eval_gate(
            registry,
            evaluators=evaluators,
            fixture_root=fixture_root,
        )
        assert result.passed is True
        assert "book_rag" in result.checked

    def test_provenance_failing_raises_and_writes_no_fixture(
        self,
        tmp_path: Path,
    ) -> None:
        """A provenance-failing qa set raises and writes no ``qrels.tsv``."""
        query_ids = ["q-001"]
        doc_ids = [["doc-missing"]]
        qa = _make_qa_dataset(query_ids, doc_ids)

        fixture_root = tmp_path / "fixtures" / "eval"

        def resolve_fn(ids: set[str]) -> set[str]:
            return set()  # nothing resolves

        with pytest.raises(QrelsProvenanceError):
            refresh_book_rag_qrels(
                qa,
                fixture_root=fixture_root,
                resolve_fn=resolve_fn,
            )

        assert not (fixture_root / "book_rag" / "qrels.tsv").exists()
