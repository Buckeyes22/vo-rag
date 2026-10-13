"""Contract tests for eval-set generation wrappers.

Exercises ``generate_eval_pairs`` and ``generate_rag_dataset`` with a
fake/stub LLM to prove:

1. Chunk-ID fidelity — ``EmbeddingQAFinetuneDataset.relevant_docs`` values are
   the *input* ``node.id_`` handles, never a Qdrant point id.
2. Governed-LLM enforcement — ``llm`` is a required kwarg; the wrappers raise
   before LlamaIndex can fall back to ``Settings.llm``.
3. ``generate_rag_dataset`` returns a ``LabelledRagDataset`` without touching
   the optional ``spacy`` dependency.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest

from vo_rag.core.chunking.ids import _chunk_id
from vo_rag.core.eval.harness import HarnessReport, RetrievalVariant
from vo_rag.integrations.llamaindex.eval_dataset import (
    freeze_baseline,
    generate_eval_pairs,
    generate_rag_dataset,
)

if TYPE_CHECKING:
    from llama_index.core.schema import Document, TextNode


@pytest.fixture
def mock_llm():
    from llama_index.core.llms import MockLLM

    return MockLLM()


def _make_text_node(text: str, node_id: str) -> TextNode:
    from llama_index.core.schema import TextNode

    return TextNode(text=text, id_=node_id)


def _make_document(text: str, doc_id: str) -> Document:
    from llama_index.core.schema import Document

    return Document(text=text, id_=doc_id)


class TestGenerateEvalPairs:
    """Contract tests for ``generate_eval_pairs``."""

    def test_relevant_docs_uses_live_chunk_ids(self, mock_llm) -> None:
        chunk_id_1 = _chunk_id("src-1", "The quick brown fox.", "text")
        chunk_id_2 = _chunk_id("src-2", "Python is a language.", "text")
        nodes = [
            _make_text_node("The quick brown fox.", chunk_id_1),
            _make_text_node("Python is a language.", chunk_id_2),
        ]

        qa = generate_eval_pairs(nodes, llm=mock_llm, num_questions_per_chunk=1)

        # Every relevant_docs key must resolve to a query.
        for query_id, doc_ids in qa.relevant_docs.items():
            assert query_id in qa.queries
            for doc_id in doc_ids:
                assert doc_id in {chunk_id_1, chunk_id_2}

    def test_llm_none_raises(self) -> None:
        from llama_index.core.schema import TextNode

        nodes = [TextNode(text="hello", id_="chunk-1")]
        bad_llm: Any = None
        with pytest.raises(ValueError, match="llm is required"):
            generate_eval_pairs(nodes, llm=bad_llm)

    def test_does_not_read_settings_llm(self, mock_llm) -> None:
        chunk_id_1 = _chunk_id("src-1", "The quick brown fox.", "text")
        nodes = [_make_text_node("The quick brown fox.", chunk_id_1)]

        # If LlamaIndex fell back to Settings.llm it would call resolve_llm,
        # which tries to import llama-index-llms-openai and fails in this
        # environment.  Patching resolve_llm to raise makes any fallback
        # unmistakably loud.
        with patch(
            "llama_index.core.llms.utils.resolve_llm",
            side_effect=RuntimeError("Settings.llm was read"),
        ):
            # Should succeed because we pass llm explicitly.
            generate_eval_pairs(nodes, llm=mock_llm, num_questions_per_chunk=1)


class TestGenerateRagDataset:
    """Contract tests for ``generate_rag_dataset``."""

    def test_returns_labelled_rag_dataset(self, mock_llm) -> None:
        docs = [
            _make_document("The quick brown fox.", "doc-1"),
            _make_document("Python is a language.", "doc-2"),
        ]

        ds = generate_rag_dataset(docs, llm=mock_llm, num_questions_per_chunk=1)

        from llama_index.core.llama_dataset import LabelledRagDataExample, LabelledRagDataset

        assert isinstance(ds, LabelledRagDataset)
        assert len(ds.examples) == 2
        for raw_ex in ds.examples:
            ex = cast(LabelledRagDataExample, raw_ex)
            assert ex.query
            assert ex.reference_contexts
            assert any(
                ctx in ["The quick brown fox.", "Python is a language."]
                for ctx in ex.reference_contexts
            )

    def test_llm_none_raises(self) -> None:
        docs = [_make_document("hello", "doc-1")]
        bad_llm: Any = None
        with pytest.raises(ValueError, match="llm is required"):
            generate_rag_dataset(docs, llm=bad_llm)

    def test_does_not_read_settings_llm(self, mock_llm) -> None:
        docs = [_make_document("The quick brown fox.", "doc-1")]

        # If LlamaIndex fell back to Settings.llm it would call resolve_llm,
        # which tries to import llama-index-llms-openai and fails in this
        # environment.  Patching resolve_llm to raise makes any fallback
        # unmistakably loud.
        with patch(
            "llama_index.core.llms.utils.resolve_llm",
            side_effect=RuntimeError("Settings.llm was read"),
        ):
            # Should succeed because we pass llm explicitly.
            generate_rag_dataset(docs, llm=mock_llm, num_questions_per_chunk=1)


class TestFreezeBaseline:
    """Contract tests for ``freeze_baseline``."""

    def test_freeze_baseline_writes_qrels_hash_and_aggregate(self, tmp_path) -> None:
        report = HarnessReport(
            pipeline="book_rag",
            aggregate={"recall@5": 0.8},
            qrels_hash="sha256:abc",
            variant=RetrievalVariant(name="test"),
            per_query=[],
            wall_clock_s=0.0,
        )
        baseline = freeze_baseline(
            report, path=tmp_path / "baseline.json", corpus_snapshot="fixture:test"
        )
        assert baseline["qrels_hash"] == "sha256:abc"
        assert baseline["aggregate"]["recall@5"] == 0.8
        assert baseline["schema_version"] == 1
        assert baseline["corpus_snapshot"] == "fixture:test"
        assert "created_at" in baseline

        # Verify round-trip through disk
        loaded = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
        assert loaded["qrels_hash"] == "sha256:abc"
        assert loaded["aggregate"]["recall@5"] == 0.8
