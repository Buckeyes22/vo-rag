"""book_rag eval-gate wiring (VOR-7).

Proves the keystone integration contract: once book_rag is ACTIVE, the eval gate
requires a registered evaluator + fixtures or it fails — and the CI-safe replay
evaluator satisfies it offline (no Qdrant/GPU). Also locks the two latent gate
bugs the shipped fixtures exposed: the ``query_text`` field fallback and
the flat regression-floor baseline.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from vo_rag.core.config.pipelines import PipelineSpec, PipelineStatus
from vo_rag.core.eval.gate import _load_queries_jsonl, run_eval_gate
from vo_rag.core.eval.passages import is_passage_key
from vo_rag.core.eval.registered import build_eval_gate_evaluators
from vo_rag.pipelines.book_rag.eval import (
    BOOK_RAG_EVAL_COLLECTION,
    book_rag_evaluator,
    book_rag_variant,
    build_live_search_fn,
    build_replay_search_fn,
)

_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_ROOT = _ROOT / "tests" / "fixtures" / "eval"
_BOOK_DIR = _FIXTURE_ROOT / "book_rag"


def _spec(status: PipelineStatus) -> PipelineSpec:
    return PipelineSpec(name="book_rag", status=status, summary="book rag")


# ── variant ─────────────────────────────────────────────────────────────────


def test_variant_is_book_rag_config() -> None:
    v = book_rag_variant()
    assert v.name == "book_rag"
    assert v.top_k == 10
    assert v.prefetch_k == 100
    assert v.reranker_enabled is True


# ── replay search_fn (CI-safe) ──────────────────────────────────────────────


def test_replay_returns_relevant_ranked_chunk_ids() -> None:
    search_fn = build_replay_search_fn(_BOOK_DIR / "queries.jsonl", _BOOK_DIR / "qrels.tsv")
    # First fixture query: text -> its single relevant chunk_id from qrels.
    first = json.loads((_BOOK_DIR / "queries.jsonl").read_text().splitlines()[0])
    text = first["query_text"]
    result = search_fn(text, 10)
    assert isinstance(result, list)
    assert result, "replay must return the qrels-relevant chunk(s) for a known query"
    # The known relevant doc for that query id must be present in the ranking.
    qrels_first = (_BOOK_DIR / "qrels.tsv").read_text().splitlines()[0].split("\t")
    assert qrels_first[2] in result


def test_replay_unknown_query_returns_empty() -> None:
    search_fn = build_replay_search_fn(_BOOK_DIR / "queries.jsonl", _BOOK_DIR / "qrels.tsv")
    assert search_fn("a query that does not exist in the fixture", 10) == []


def test_replay_passage_mode_returns_gold_passage_keys() -> None:
    entry = book_rag_evaluator(_BOOK_DIR, relevance_mode="passage")
    first = json.loads((_BOOK_DIR / "queries.jsonl").read_text().splitlines()[0])

    result = entry.search_fn(first["query_text"], 10)

    assert isinstance(result, list)
    assert result
    assert all(isinstance(doc_id, str) and is_passage_key(doc_id) for doc_id in result)


# ── gate bug: query_text field fallback ──────────────────────────────────


def test_queries_loader_accepts_query_text_field() -> None:
    loaded = _load_queries_jsonl(_BOOK_DIR / "queries.jsonl")
    assert loaded, "loader returned no queries"
    # Every value must be non-empty — the bug resolved them all to "".
    assert all(text.strip() for text in loaded.values())


# ── synthetic demo baseline: aggregate metrics are 0.0, not a quality floor ──


def test_synthetic_demo_baseline_is_explicitly_not_a_quality_floor() -> None:
    # Real corpus scores are excluded from this public snapshot.
    baseline = json.loads((_BOOK_DIR / "baseline.json").read_text())
    assert baseline["pipeline"] == "book_rag"
    for key in ("ndcg@10", "recall@10", "mrr"):
        assert baseline["aggregate"][key] == 0.0


# ── keystone: gate behaviour when book_rag is ACTIVE ────────────────────────


def test_gate_passes_with_active_book_rag_and_replay_evaluator() -> None:
    registry = {"book_rag": _spec(PipelineStatus.ACTIVE)}
    evaluators = {"book_rag": book_rag_evaluator(_BOOK_DIR)}
    result = run_eval_gate(registry, evaluators=evaluators, fixture_root=_FIXTURE_ROOT)
    assert result.passed is True, (result.regressions, result.failed)
    assert "book_rag" in result.checked


def test_gate_fails_active_book_rag_without_evaluator() -> None:
    # Activating book_rag without an evaluator must fail the gate.
    registry = {"book_rag": _spec(PipelineStatus.ACTIVE)}
    result = run_eval_gate(registry, evaluators={}, fixture_root=_FIXTURE_ROOT)
    assert result.passed is False
    assert "book_rag" in result.missing_evaluators


def test_gate_skips_book_rag_while_next() -> None:
    registry = {"book_rag": _spec(PipelineStatus.NEXT)}
    result = run_eval_gate(registry, evaluators={}, fixture_root=_FIXTURE_ROOT)
    assert result.passed is True
    assert "book_rag" in result.skipped


# ── live search_fn shape (unit, fake retriever — no Qdrant) ─────────────────


def test_live_search_fn_extracts_chunk_ids_from_payload() -> None:
    class _FakeRetriever:
        async def hybrid_search(
            self, query: str, *, collection: str, top_k: int, prefetch_depth: int
        ) -> list[SimpleNamespace]:
            assert collection == BOOK_RAG_EVAL_COLLECTION
            return [
                SimpleNamespace(id="point-1", payload={"chunk_id": "chunk-A"}),
                SimpleNamespace(id="point-2", payload={}),  # falls back to id
            ]

    search_fn = build_live_search_fn(_FakeRetriever())
    coro = search_fn("q", 10)
    assert inspect.iscoroutine(coro)
    out = asyncio.run(coro)
    assert out == ["chunk-A", "point-2"]


def test_live_search_fn_threads_collection_override() -> None:
    seen_collections: list[str] = []

    class _FakeRetriever:
        async def hybrid_search(
            self, query: str, *, collection: str, top_k: int, prefetch_depth: int
        ) -> list[SimpleNamespace]:
            seen_collections.append(collection)
            return [SimpleNamespace(id="point-1", payload={"chunk_id": "chunk-A"})]

    search_fn = build_live_search_fn(_FakeRetriever(), collection="vo_rag_books_arm_v1")
    result = search_fn("q", 10)
    assert inspect.iscoroutine(result)
    asyncio.run(result)

    assert seen_collections == ["vo_rag_books_arm_v1"]


def test_registered_evaluator_threads_collection_override(monkeypatch) -> None:
    seen_collections: list[str] = []

    class _FakeRetriever:
        async def hybrid_search(
            self, query: str, *, collection: str, top_k: int, prefetch_depth: int
        ) -> list[SimpleNamespace]:
            seen_collections.append(collection)
            return [SimpleNamespace(id="point-1", payload={"chunk_id": "chunk-A"})]

    monkeypatch.setenv("VO_RAG_EVAL_MODE", "retrieval_gate")
    evaluators = build_eval_gate_evaluators(
        {"book_rag": _spec(PipelineStatus.ACTIVE)},
        fixture_root=_FIXTURE_ROOT,
        retriever_factory=_FakeRetriever,
        collection="vo_rag_books_arm_v1",
    )

    result = evaluators["book_rag"].search_fn("q", 10)
    assert inspect.iscoroutine(result)
    asyncio.run(result)

    assert seen_collections == ["vo_rag_books_arm_v1"]
