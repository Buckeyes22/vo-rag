"""Contract tests for ``qa_dataset_to_qrels`` / ``write_qrels_tsv`` (VOR-233).

Lowers a LlamaIndex ``EmbeddingQAFinetuneDataset`` into vo-rag ``Qrels`` and
verifies:

1. **Lowering fidelity** — one ``Judgment`` per ``(query_id, node_id)`` pair,
   every ``grade == 1``, ``pipeline`` / ``collection`` set on the result.
2. **No doc-id mutation** — every ``Judgment.doc_id`` is *exactly* a node id
   from the source ``relevant_docs`` (no re-mint, no point-id substitution).
   This is the load-bearing invariant for VOR-234's provenance check.
3. **TSV round-trip** — ``save_qrels_tsv`` → ``load_qrels_tsv`` reproduces a
   ``Qrels`` with the same ``qrels_hash`` (the canonical sha256 over the
   sorted judgment triples).

Sister to ``test_eval_dataset.py`` (VOR-232) — they exercise the generator
side; this one exercises the lowering side.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vo_rag.core.chunking.ids import _chunk_id
from vo_rag.core.eval.qrels import (
    load_qrels_tsv,
    qrels_hash,
)
from vo_rag.integrations.llamaindex.eval_dataset import (
    qa_dataset_to_qrels,
    write_qrels_tsv,
)

if TYPE_CHECKING:
    from llama_index.core.llama_dataset.legacy.embedding import (
        EmbeddingQAFinetuneDataset,
    )


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_embedding_qa(
    *,
    n_queries: int = 3,
    n_docs_per_query: int = 2,
) -> tuple[EmbeddingQAFinetuneDataset, list[str], list[list[str]]]:
    """Build a deterministic ``EmbeddingQAFinetuneDataset`` for the contract tests.

    Returns the dataset plus the *expected* node-id groups in input order so
    the assertions can pin order-independent properties (every node id
    appears, every judgment has grade=1) without coupling to dict-iteration
    order, which Python preserves but shouldn't be load-bearing here.
    """
    from llama_index.core.llama_dataset.legacy.embedding import (
        EmbeddingQAFinetuneDataset,
    )

    queries: dict[str, str] = {}
    relevant_docs: dict[str, list[str]] = {}
    corpus: dict[str, str] = {}
    expected_node_ids: list[list[str]] = []

    for q_idx in range(n_queries):
        query_id = f"q-{q_idx:03d}"
        queries[query_id] = f"What does chunk group {q_idx} say?"

        node_ids: list[str] = []
        for n_idx in range(n_docs_per_query):
            node_id = _chunk_id(
                source_id=f"src-{q_idx}-{n_idx}",
                content=f"body text for query {q_idx} doc {n_idx}",
                stage="text",
            )
            node_ids.append(node_id)
            corpus[node_id] = f"body text for query {q_idx} doc {n_idx}"
        relevant_docs[query_id] = node_ids
        expected_node_ids.append(node_ids)

    qa = EmbeddingQAFinetuneDataset(queries=queries, relevant_docs=relevant_docs, corpus=corpus)
    return qa, [nid for group in expected_node_ids for nid in group], expected_node_ids


# ---------------------------------------------------------------------------
# Lowering fidelity
# ---------------------------------------------------------------------------


class TestQaDatasetToQrels:
    """Contract tests for ``qa_dataset_to_qrels``."""

    def test_yields_one_judgment_per_query_node_pair(self) -> None:
        qa, _flat, expected_groups = _make_embedding_qa(n_queries=3, n_docs_per_query=2)
        qrels = qa_dataset_to_qrels(qa, pipeline="book_rag", collection="harness_collection")

        # 3 queries × 2 nodes = 6 judgments, one per (query, node) pair.
        assert len(qrels.judgments) == 3 * 2
        for j in qrels.judgments:
            assert j.grade == 1
            assert j.query_id in qa.queries

    def test_pipeline_and_collection_propagate(self) -> None:
        qa, _, _ = _make_embedding_qa()
        qrels = qa_dataset_to_qrels(qa, pipeline="book_rag", collection="harness_collection")
        assert qrels.pipeline == "book_rag"
        assert qrels.collection == "harness_collection"

    def test_doc_id_is_exactly_a_source_node_id(self) -> None:
        """No mutation, no re-mint — the load-bearing ID-fidelity invariant.

        Every ``Judgment.doc_id`` must appear verbatim in the source
        ``relevant_docs``; this is what VOR-234's live-collection provenance
        check joins on. Re-keying would reintroduce the orphaned-qrels hole.
        """
        qa, _flat, _groups = _make_embedding_qa(n_queries=4, n_docs_per_query=3)
        qrels = qa_dataset_to_qrels(qa, pipeline="book_rag", collection="harness_collection")

        # Flatten source node ids preserving duplicates (one query can
        # share a positive node with another, or list the same node twice).
        all_source_node_ids: set[str] = set()
        for node_ids in qa.relevant_docs.values():
            all_source_node_ids.update(node_ids)

        for j in qrels.judgments:
            assert j.doc_id in all_source_node_ids, (
                f"doc_id {j.doc_id!r} was re-minted; must be a verbatim "
                f"node id from the source relevant_docs"
            )

    def test_judgment_count_matches_relevant_docs_cardinality(self) -> None:
        qa, _flat, _groups = _make_embedding_qa(n_queries=5, n_docs_per_query=1)
        qrels = qa_dataset_to_qrels(qa, pipeline="book_rag", collection="harness_collection")
        expected = sum(len(v) for v in qa.relevant_docs.values())
        assert len(qrels.judgments) == expected

    def test_by_query_helper_recovers_grade_one(self) -> None:
        qa, _flat, _groups = _make_embedding_qa()
        qrels = qa_dataset_to_qrels(qa, pipeline="book_rag", collection="harness_collection")
        by_q = qrels.by_query()
        for query_id, expected_nodes in qa.relevant_docs.items():
            assert query_id in by_q
            for node_id in expected_nodes:
                assert by_q[query_id][node_id] == 1

    def test_empty_relevant_docs_yields_empty_judgments(self) -> None:
        from llama_index.core.llama_dataset.legacy.embedding import (
            EmbeddingQAFinetuneDataset,
        )

        qa = EmbeddingQAFinetuneDataset(
            queries={"q-empty": "anything"},
            relevant_docs={},
            corpus={},
        )
        qrels = qa_dataset_to_qrels(qa, pipeline="book_rag", collection="harness_collection")
        assert qrels.judgments == []
        assert qrels.pipeline == "book_rag"
        assert qrels.collection == "harness_collection"


# ---------------------------------------------------------------------------
# TSV round-trip
# ---------------------------------------------------------------------------


class TestWriteQrelsTsvRoundTrip:
    """``save_qrels_tsv`` / ``load_qrels_tsv`` must reproduce the same ``Qrels``."""

    def test_round_trip_preserves_qrels_hash(self, tmp_path: Path) -> None:
        qa, _flat, _groups = _make_embedding_qa(n_queries=3, n_docs_per_query=2)
        qrels = qa_dataset_to_qrels(qa, pipeline="book_rag", collection="harness_collection")

        out = tmp_path / "qrels.tsv"
        write_qrels_tsv(qrels, out)

        assert out.is_file()
        # TREC format: 4 tab-separated fields, one per line.
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(qrels.judgments)
        for line, j in zip(lines, qrels.judgments, strict=True):
            assert line.split("\t") == [j.query_id, "0", j.doc_id, str(j.grade)]

        loaded = load_qrels_tsv(out, pipeline=qrels.pipeline, collection=qrels.collection)
        # Equal qrels_hash ⇒ byte-identical canonical serialisation ⇒ same
        # set of (query_id, doc_id, grade) triples (modulo ordering, which
        # qrels_hash canonicalises via sort).
        assert qrels_hash(loaded) == qrels_hash(qrels)
        assert loaded.pipeline == qrels.pipeline
        assert loaded.collection == qrels.collection

    def test_round_trip_preserves_doc_ids_byte_for_byte(self, tmp_path: Path) -> None:
        """Beyond the hash, every doc_id must survive a save→load cycle intact."""
        qa, _flat, _groups = _make_embedding_qa(n_queries=2, n_docs_per_query=2)
        qrels = qa_dataset_to_qrels(qa, pipeline="book_rag", collection="harness_collection")

        out = tmp_path / "qrels.tsv"
        write_qrels_tsv(qrels, out)
        loaded = load_qrels_tsv(out, pipeline=qrels.pipeline, collection=qrels.collection)

        assert {j.doc_id for j in loaded.judgments} == {j.doc_id for j in qrels.judgments}

    def test_does_not_mutate_source_relevant_docs(self, tmp_path: Path) -> None:
        """Lowering + writing must not mutate the source dataset's relevant_docs."""
        qa, _flat, _groups = _make_embedding_qa()
        # Snapshot via copy (relevant_docs is a dict of lists).
        before: dict[str, list[str]] = {k: list(v) for k, v in qa.relevant_docs.items()}
        qrels = qa_dataset_to_qrels(qa, pipeline="book_rag", collection="harness_collection")
        write_qrels_tsv(qrels, tmp_path / "qrels.tsv")
        assert dict(qa.relevant_docs) == before
