"""Passage-level eval projection tests for chunking-invariant retrieval gates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from vo_rag.core.eval.harness import RetrievalVariant, run_harness
from vo_rag.core.eval.passages import passage_key, passage_keys_from_payload
from vo_rag.core.eval.qrels import Judgment, Qrels, load_qrels_tsv

_ROOT = Path(__file__).resolve().parents[3]
_BOOK_DIR = _ROOT / "tests" / "fixtures" / "eval" / "book_rag"
_PASSAGE_KEY_RE = re.compile(r"^.+#p[1-9][0-9]*$")


def test_passage_projection_distinguishes_source_and_page() -> None:
    gold = passage_key("source-a", 7)

    assert gold in passage_keys_from_payload({"source_id": "source-a", "page_number": 7})
    assert gold not in passage_keys_from_payload({"source_id": "source-a", "page_number": 8})
    assert gold in passage_keys_from_payload({"source_id": "source-a", "page_range": "6-8"})
    assert gold not in passage_keys_from_payload({"source_id": "source-b", "page_number": 7})


@pytest.mark.asyncio
async def test_passage_mode_scores_projected_payloads_with_existing_metrics() -> None:
    qrels = Qrels(
        pipeline="book_rag",
        collection="eval",
        judgments=[
            Judgment(query_id="q1", doc_id=passage_key("source-a", 3), grade=1),
            Judgment(query_id="q2", doc_id=passage_key("source-b", 10), grade=1),
        ],
    )
    variant = RetrievalVariant(name="passage", top_k=5)

    async def _search(query: str, top_k: int) -> list[dict[str, Any]]:
        if query == "q1":
            return [
                {"id": "wrong-page", "payload": {"source_id": "source-a", "page_number": 99}},
                {"id": "span-hit", "payload": {"source_id": "source-a", "page_range": "3-5"}},
            ][:top_k]
        return [
            {"id": "wrong-source", "payload": {"source_id": "source-c", "page_number": 10}},
        ][:top_k]

    report = await run_harness(
        "book_rag",
        qrels,
        variant,
        search_fn=_search,
        k_values=(1, 2),
        relevance_mode="passage",
    )

    assert report.per_query[0].retrieved[:2] == [
        passage_key("source-a", 99),
        passage_key("source-a", 3),
    ]
    assert report.aggregate["recall@1"] == pytest.approx(0.0)
    assert report.aggregate["recall@2"] == pytest.approx(0.5)
    assert report.aggregate["mrr"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_legacy_chunk_id_mode_is_noop_for_existing_rankings() -> None:
    qrels = Qrels(
        pipeline="book_rag",
        collection="eval",
        judgments=[
            Judgment(query_id="q1", doc_id="chunk-a", grade=3),
            Judgment(query_id="q1", doc_id="chunk-b", grade=1),
        ],
    )
    variant = RetrievalVariant(name="legacy", top_k=3)

    async def _search(query: str, top_k: int) -> list[str]:
        return ["chunk-b", "chunk-a", "noise"][:top_k]

    default_report = await run_harness("book_rag", qrels, variant, search_fn=_search)
    explicit_legacy_report = await run_harness(
        "book_rag",
        qrels,
        variant,
        search_fn=_search,
        relevance_mode="chunk_id",
    )

    assert explicit_legacy_report.per_query[0].retrieved == default_report.per_query[0].retrieved
    metric_keys = {
        key
        for key in default_report.aggregate
        if key.startswith(("recall@", "ndcg@", "precision@")) or key == "mrr"
    }
    assert {key: explicit_legacy_report.aggregate[key] for key in metric_keys} == {
        key: default_report.aggregate[key] for key in metric_keys
    }


def test_generated_passage_qrels_fixture_is_well_formed() -> None:
    chunk_qrels = load_qrels_tsv(_BOOK_DIR / "qrels.tsv", pipeline="book_rag", collection="eval")
    passage_qrels = load_qrels_tsv(
        _BOOK_DIR / "qrels_passage.tsv",
        pipeline="book_rag",
        collection="eval",
    )

    assert len(passage_qrels.judgments) == len(chunk_qrels.judgments)
    assert all(_PASSAGE_KEY_RE.match(judgment.doc_id) for judgment in passage_qrels.judgments)
