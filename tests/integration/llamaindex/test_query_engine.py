"""Contract tests for build_vo_rag_query_engine async reranker path.

TEST 1 (regression): VoRagReranker with an async backend raises TypeError at
construction — the sync-bridge via run_coro_sync is never reached.

TEST 2 (supported path): an async rerank wired via VoRagRetriever(rerank_fn=...)
is awaited natively under engine.aquery and succeeds.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import Filter, ScoredPoint

from vo_rag.core.answer.synthesizer import (
    AnswerDraft,
    AnswerProfile,
    AnswerSynthesis,
)
from vo_rag.core.llm.output_validator import ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import HybridRetriever, SearchResult
from vo_rag.integrations.llamaindex.query_engine import build_vo_rag_query_engine
from vo_rag.integrations.llamaindex.reranker import VoRagReranker
from vo_rag.integrations.llamaindex.retriever import VoRagRetriever

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _FakeQdrantStore:
    async def search_hybrid(
        self,
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        *,
        top_k: int = 20,
        filters: Filter | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]:
        return []


class _FakeHybridRetriever(HybridRetriever):
    def __init__(self, results: list[SearchResult] | None = None) -> None:
        super().__init__(qdrant=_FakeQdrantStore(), embedding_client=None)
        self.results = results or []

    async def cross_collection_search(
        self,
        query: str,
        collections: list[str],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        mode: QueryMode = QueryMode.INTERNAL,
        visual_mode: str | None = None,
    ) -> list[SearchResult]:
        return self.results


def _answer_synth():
    a = MagicMock()
    a.synthesize = AsyncMock(
        return_value=AnswerSynthesis(
            draft=AnswerDraft(answer="the answer", confidence=0.8, cited_chunk_ids=["c1"]),
            llm_response=LLMResponse(text="the answer", model="claude-x"),
            validation=ValidationResult(ok=True),
            should_quarantine=False,
            model="claude-x",
        )
    )
    return a


# --------------------------------------------------------------------------- #
# TEST 1 — async backend rejected at construction (no sync-bridge reached)
# --------------------------------------------------------------------------- #


def test_query_engine_uses_reranker_public_boundary_only() -> None:
    import vo_rag.integrations.llamaindex.query_engine as query_engine_module

    tree = ast.parse(inspect.getsource(query_engine_module))

    private_imports: list[str] = []
    private_attr_literals: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "vo_rag.integrations.llamaindex.reranker"
        ):
            private_imports.extend(alias.name for alias in node.names if alias.name.startswith("_"))
        if isinstance(node, ast.Constant) and node.value == "_reranker":
            private_attr_literals.append(node.value)

    assert private_imports == []
    assert private_attr_literals == []


@pytest.mark.asyncio
async def test_reranker_rejects_async_backend_at_construction():
    class _AsyncBackend:
        async def rerank(self, query, candidates, top_k=20):
            return candidates[:top_k]

    with pytest.raises(TypeError, match="cannot wrap an async reranker backend"):
        VoRagReranker(reranker=_AsyncBackend(), top_k=1)


# --------------------------------------------------------------------------- #
# TEST 2 — async rerank via VoRagRetriever(rerank_fn=...) under aquery
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_aquery_with_async_rerank_fn_succeeds():
    """Async rerank_fn is awaited natively in _aretrieve; no sync-bridge is hit."""
    rerank_calls: list[tuple[str, list[SearchResult]]] = []

    async def _async_rerank(query: str, hits: list[SearchResult]) -> list[SearchResult]:
        rerank_calls.append((query, hits))
        # Reverse order to prove the rerank actually ran.
        return list(reversed(hits))

    results = [
        SearchResult(id="p1", score=0.9, collection="books", payload={"chunk_id": "c1"}),
        SearchResult(id="p2", score=0.8, collection="books", payload={"chunk_id": "c2"}),
    ]

    retriever = _FakeHybridRetriever(results=results)
    vorag_retriever = VoRagRetriever(
        retriever,
        mode=QueryMode.PUBLICATION,
        top_k=10,
        rerank_fn=_async_rerank,
        requested_collections=["books"],
    )

    engine = build_vo_rag_query_engine(
        retriever=vorag_retriever,
        answer=_answer_synth(),
        profile=AnswerProfile.QUALITY,
    )

    response = await engine.aquery("test query")

    # Assert the async rerank_fn was actually called and awaited.
    assert len(rerank_calls) == 1
    assert rerank_calls[0][0] == "test query"
    assert len(rerank_calls[0][1]) == 2

    # Because the rerank reversed the order, the nodes arriving at synthesis
    # should be in reversed order.
    source_nodes = response.source_nodes
    assert [n.node.id_ for n in source_nodes] == ["c2", "c1"]
