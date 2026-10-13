"""Provenance-check contract tests for ``assert_qrels_resolve`` (VOR-234).

Verifies the exact failure mode that orphaned the old qrels set:
a corpus re-ingest changed chunk IDs, so the labelled qrels no longer
point at live chunks.  The provenance check catches this **before**
the set is trusted, by asserting every ``Judgment.doc_id`` resolves
via an injected ``resolve_fn``.
"""

from __future__ import annotations

import pytest

from vo_rag.core.chunking.ids import _chunk_id
from vo_rag.core.eval.qrels import Judgment, Qrels
from vo_rag.integrations.llamaindex.eval_dataset import (
    QrelsProvenanceError,
    QrelsProvenanceReport,
    assert_book_rag_eval_snapshot_resolves_qrels,
    assert_qrels_resolve,
)


def _make_qrels(doc_ids: list[str]) -> Qrels:
    """Build a ``Qrels`` with one judgment per *doc_id* (all query_id='q-0')."""
    judgments = [Judgment(query_id="q-0", doc_id=doc_id, grade=1) for doc_id in doc_ids]
    return Qrels(pipeline="test_pipeline", collection="test_collection", judgments=judgments)


def _make_chunk_ids(n: int) -> list[str]:
    """Return *n* distinct content-addressed chunk ids."""
    return [_chunk_id(f"src-{i}", f"body-{i}", "text") for i in range(n)]


def test_book_rag_eval_corpus_snapshot_resolves_qrels_doc_ids(tmp_path):
    qrels_path = tmp_path / "qrels.tsv"
    snapshot_path = tmp_path / "corpus_snapshot.jsonl"
    qrels_path.write_text("q1\t0\tchunk-1\t1\nq2\t0\tchunk-2\t1\n", encoding="utf-8")
    snapshot_path.write_text(
        '{"chunk_id": "chunk-1", "source_id": "book-a", "text": "one"}\n'
        '{"chunk_id": "chunk-2", "source_id": "book-b", "text": "two"}\n',
        encoding="utf-8",
    )

    assert_book_rag_eval_snapshot_resolves_qrels(qrels_path, snapshot_path)


class TestAssertQrelsResolve:
    """Contract tests for ``assert_qrels_resolve``."""

    def test_all_resolvable_returns_report(self) -> None:
        """Every doc_id resolves → report with resolved == total, no raise."""
        doc_ids = _make_chunk_ids(3)
        qrels = _make_qrels(doc_ids)

        def resolve_fn(ids: set[str]) -> set[str]:
            return ids  # full set is live

        report = assert_qrels_resolve(qrels, resolve_fn=resolve_fn)

        assert isinstance(report, QrelsProvenanceReport)
        assert report.resolved == 3
        assert report.total == 3

    def test_one_missing_raises_with_unresolved_id_named(self) -> None:
        """One doc_id not in the live set → raises, message names the missing id."""
        doc_ids = _make_chunk_ids(3)
        qrels = _make_qrels(doc_ids)
        missing_id = doc_ids[1]

        def resolve_fn(ids: set[str]) -> set[str]:
            return ids - {missing_id}

        with pytest.raises(QrelsProvenanceError) as exc_info:
            assert_qrels_resolve(qrels, resolve_fn=resolve_fn)

        exc = exc_info.value
        assert isinstance(exc, ValueError)
        assert missing_id in exc.unresolved
        assert missing_id in str(exc)
        # Must NOT silently pass or drop the row.
        assert "1 qrel doc_id(s) not found" in str(exc)

    def test_multiple_missing_raises_all_named(self) -> None:
        """Multiple missing ids → all appear in the exception."""
        doc_ids = _make_chunk_ids(5)
        qrels = _make_qrels(doc_ids)
        missing = {doc_ids[0], doc_ids[3]}

        def resolve_fn(ids: set[str]) -> set[str]:
            return ids - missing

        with pytest.raises(QrelsProvenanceError) as exc_info:
            assert_qrels_resolve(qrels, resolve_fn=resolve_fn)

        exc = exc_info.value
        assert exc.unresolved == missing
        for mid in missing:
            assert mid in str(exc)

    def test_empty_qrels_returns_zero_counts(self) -> None:
        """An empty qrels set trivially passes (no ids to resolve)."""
        qrels = Qrels(pipeline="test", collection="test", judgments=[])

        def resolve_fn(ids: set[str]) -> set[str]:
            return set()

        report = assert_qrels_resolve(qrels, resolve_fn=resolve_fn)
        assert report.resolved == 0
        assert report.total == 0

    def test_resolver_receives_exact_doc_id_set(self) -> None:
        """The injected resolver is called with the exact set of doc_ids."""
        doc_ids = _make_chunk_ids(4)
        qrels = _make_qrels(doc_ids)
        received: set[str] | None = None

        def resolve_fn(ids: set[str]) -> set[str]:
            nonlocal received
            received = ids
            return ids

        assert_qrels_resolve(qrels, resolve_fn=resolve_fn)
        assert received == set(doc_ids)

    def test_duplicate_doc_ids_deduped_before_resolve(self) -> None:
        """Duplicate doc_ids across judgments are deduplicated before resolve_fn."""
        doc_id = _chunk_id("src-0", "body-0", "text")
        # Two judgments pointing at the same doc_id.
        judgments = [
            Judgment(query_id="q-0", doc_id=doc_id, grade=1),
            Judgment(query_id="q-1", doc_id=doc_id, grade=1),
        ]
        qrels = Qrels(pipeline="test", collection="test", judgments=judgments)
        call_count = 0

        def resolve_fn(ids: set[str]) -> set[str]:
            nonlocal call_count
            call_count += 1
            assert len(ids) == 1
            return ids

        report = assert_qrels_resolve(qrels, resolve_fn=resolve_fn)
        assert call_count == 1
        assert report.total == 1
        assert report.resolved == 1
