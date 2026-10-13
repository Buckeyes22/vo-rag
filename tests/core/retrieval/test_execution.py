"""Tests for the framework-neutral ``core.retrieval.execution`` helper.

The retrieval flow is an importable ``execute_retrieval(query, ...)`` that:

  1. resolves collections (via ``QueryMode`` + ``resolve_collections``),
  2. calls ``HybridRetriever.cross_collection_search`` (which itself
     delegates to ``QdrantStore.search_hybrid``),
  3. optionally calls a ``rerank_fn`` callable (for example the ``GteReranker``
     — kept as an injected dependency so this module stays
     framework-neutral),
  4. truncates to ``top_k`` and returns the list of ``SearchResult``.

No FastAPI, no auth, no DB, no observability imports — the helper is
plumbed the same way in tests, in scripts, and in the API surface when
the route handler lands.
"""

from __future__ import annotations

from typing import Any

import pytest
from qdrant_client.models import ScoredPoint

from vo_rag.core.retrieval.execution import (
    QueryExecution,
    execute_retrieval,
)
from vo_rag.core.retrieval.query_mode import (
    MODE_AGNOSTIC_COLLECTIONS,
    QueryMode,
)
from vo_rag.core.retrieval.search import HybridRetriever, SearchResult

# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class FakeEmbedClient:
    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, Any]:
        return {
            "dense": [[0.1]],
            "sparse": [{"indices": [1], "values": [0.2]}],
            "colbert": [[[0.3]]],
        }


def _scored(point_id: str, score: float, payload: dict[str, Any] | None = None) -> ScoredPoint:
    return ScoredPoint(
        id=point_id,
        score=score,
        payload=payload or {},
        vector=None,
        version=0,
        shard_key=None,
    )


class FakeQdrantStore:
    """Returns scripted results for each collection, in order."""

    def __init__(self, scripted: list[list[ScoredPoint]] | None = None) -> None:
        self.scripted = list(scripted or [])
        self.calls: list[dict[str, Any]] = []

    async def search_hybrid(
        self,
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]:
        self.calls.append(
            {
                "collection": collection,
                "filters": filters,
                "top_k": top_k,
            }
        )
        if not self.scripted:
            return []
        return list(self.scripted.pop(0))


# ----------------------------------------------------------------------
# resolve_collections delegation
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_resolves_collections_for_internal_default() -> None:
    """Internal mode defaults to the raw books collection + every
    mode-agnostic collection. execute_retrieval must use the same
    resolver that the API surface will use."""
    store = FakeQdrantStore()
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]

    await execute_retrieval(
        retriever=retriever,
        query="trail braking",
        mode=QueryMode.INTERNAL,
    )

    called = {c["collection"] for c in store.calls}
    assert "vo_rag_books_raw_v1" in called
    assert set(MODE_AGNOSTIC_COLLECTIONS).issubset(called)


@pytest.mark.asyncio
async def test_execution_resolves_publication_to_sterile() -> None:
    store = FakeQdrantStore()
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]

    await execute_retrieval(
        retriever=retriever,
        query="trail braking",
        mode=QueryMode.PUBLICATION,
    )

    called = {c["collection"] for c in store.calls}
    assert "vo_rag_books_sterile_v1" in called
    # raw must NOT be searched in publication mode
    assert "vo_rag_books_raw_v1" not in called


@pytest.mark.asyncio
async def test_execution_honours_explicit_requested_collections() -> None:
    """The caller can pin the collection set; the resolver still applies
    the sterile/raw mapping for restricted modes."""
    store = FakeQdrantStore()
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]

    await execute_retrieval(
        retriever=retriever,
        query="trail braking",
        mode=QueryMode.PUBLICATION,
        requested_collections=["vo_rag_books_raw_v1"],  # must be re-mapped
    )

    called = {c["collection"] for c in store.calls}
    assert called == {"vo_rag_books_sterile_v1"}


# ----------------------------------------------------------------------
# Rerank hook
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_optional_rerank_hook_is_invoked() -> None:
    store = FakeQdrantStore(
        scripted=[
            [_scored("a", 0.9, {"source_id": "A", "chunk_id": "a"})],
            [_scored("b", 0.8, {"source_id": "B", "chunk_id": "b"})],
        ]
    )
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]

    # Trivial rerank: reverse the order the helper receives, but keep
    # the same identities so the test is observable.
    seen: list[str] = []

    def rerank_fn(query: str, hits: list[SearchResult]) -> list[SearchResult]:
        seen.append(query)
        return list(reversed(hits))

    results = await execute_retrieval(
        retriever=retriever,
        query="trail braking",
        mode=QueryMode.INTERNAL,
        rerank_fn=rerank_fn,
    )

    assert seen == ["trail braking"]
    assert [r.id for r in results] == ["b", "a"]


async def test_execution_awaits_async_rerank_hook() -> None:
    """The remote RerankerClient.rerank is async — execute_retrieval must await it
    (regression for the unawaited-coroutine bug, VOR-8)."""
    store = FakeQdrantStore(
        scripted=[
            [_scored("a", 0.9, {"source_id": "A", "chunk_id": "a"})],
            [_scored("b", 0.8, {"source_id": "B", "chunk_id": "b"})],
        ]
    )
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]

    async def async_rerank_fn(query: str, hits: list[SearchResult]) -> list[SearchResult]:
        return list(reversed(hits))

    results = await execute_retrieval(
        retriever=retriever,
        query="trail braking",
        mode=QueryMode.INTERNAL,
        rerank_fn=async_rerank_fn,
    )

    # Must be the resolved list, NOT a coroutine, in reversed order.
    assert [r.id for r in results] == ["b", "a"]


@pytest.mark.asyncio
async def test_execution_without_rerank_returns_top_k_in_rrf_order() -> None:
    store = FakeQdrantStore(
        scripted=[
            [
                _scored("a", 0.9, {"source_id": "A", "chunk_id": "a"}),
                _scored("c", 0.6, {"source_id": "C", "chunk_id": "c"}),
            ],
            [_scored("b", 0.8, {"source_id": "B", "chunk_id": "b"})],
        ]
    )
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]

    # Use baseline visual_mode so the helper goes through the simple
    # canonical RRF path (not the text/visual quota dispatcher).
    results = await execute_retrieval(
        retriever=retriever,
        query="trail braking",
        mode=QueryMode.INTERNAL,
        top_k=2,
        visual_mode="baseline",
        requested_collections=["vo_rag_books_raw_v1", "vo_rag_books_sterile_v1"],
    )

    assert len(results) == 2
    # a is rank 1 in list[0], b is rank 1 in list[1] -> a wins
    assert results[0].id == "a"


@pytest.mark.asyncio
async def test_execution_rerank_can_truncate() -> None:
    store = FakeQdrantStore(
        scripted=[
            [
                _scored("a", 0.9, {"source_id": "A", "chunk_id": "a"}),
                _scored("b", 0.8, {"source_id": "B", "chunk_id": "b"}),
                _scored("c", 0.7, {"source_id": "C", "chunk_id": "c"}),
            ],
        ]
    )
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]

    def rerank_fn(query: str, hits: list[SearchResult]) -> list[SearchResult]:
        return hits[:1]  # keep only the top

    results = await execute_retrieval(
        retriever=retriever,
        query="trail braking",
        mode=QueryMode.INTERNAL,
        top_k=5,
        rerank_fn=rerank_fn,
    )

    assert [r.id for r in results] == ["a"]


# ----------------------------------------------------------------------
# Auto-merge post-processing
# ----------------------------------------------------------------------


def _auto_merge_child_points() -> list[ScoredPoint]:
    return [
        _scored(
            "child-a",
            0.9,
            {
                "source_id": "book-1",
                "chunk_id": "child-a",
                "parent_id": "parent-1",
                "parent_child_count": 3,
            },
        ),
        _scored(
            "child-b",
            0.8,
            {
                "source_id": "book-1",
                "chunk_id": "child-b",
                "parent_id": "parent-1",
                "parent_child_count": 3,
            },
        ),
        _scored("solo", 0.7, {"source_id": "book-1", "chunk_id": "solo"}),
    ]


@pytest.mark.asyncio
async def test_execution_auto_merge_replaces_majority_child_group() -> None:
    store = FakeQdrantStore(scripted=[_auto_merge_child_points()])
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]
    fetched_parent_ids: list[str] = []

    def fetch_parent(parent_id: str) -> SearchResult | None:
        fetched_parent_ids.append(parent_id)
        return SearchResult(
            id=parent_id,
            score=0.0,
            collection="vo_rag_books_raw_v1",
            payload={"source_id": "book-1", "chunk_id": parent_id},
        )

    results = await execute_retrieval(
        retriever=retriever,
        query="trail braking",
        requested_collections=["vo_rag_books_raw_v1"],
        top_k=5,
        visual_mode="baseline",
        auto_merge=True,
        fetch_parent_fn=fetch_parent,
    )

    assert fetched_parent_ids == ["parent-1"]
    assert [result.id for result in results] == ["parent-1", "solo"]


@pytest.mark.asyncio
async def test_execution_auto_merge_default_off_returns_unmerged_hits() -> None:
    store = FakeQdrantStore(scripted=[_auto_merge_child_points()])
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]

    results = await execute_retrieval(
        retriever=retriever,
        query="trail braking",
        requested_collections=["vo_rag_books_raw_v1"],
        top_k=5,
        visual_mode="baseline",
    )

    assert [result.id for result in results] == ["child-a", "child-b", "solo"]


@pytest.mark.asyncio
async def test_execution_auto_merge_disabled_does_not_fetch_parent() -> None:
    store = FakeQdrantStore(scripted=[_auto_merge_child_points()])
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]
    fetched_parent_ids: list[str] = []

    def fetch_parent(parent_id: str) -> SearchResult | None:
        fetched_parent_ids.append(parent_id)
        return SearchResult(
            id=parent_id,
            score=0.0,
            collection="vo_rag_books_raw_v1",
            payload={"source_id": "book-1", "chunk_id": parent_id},
        )

    results = await execute_retrieval(
        retriever=retriever,
        query="trail braking",
        requested_collections=["vo_rag_books_raw_v1"],
        top_k=5,
        visual_mode="baseline",
        auto_merge=False,
        fetch_parent_fn=fetch_parent,
    )

    assert fetched_parent_ids == []
    assert [result.id for result in results] == ["child-a", "child-b", "solo"]


# ----------------------------------------------------------------------
# Visual mode propagation
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_passes_visual_mode_to_retriever() -> None:
    store = FakeQdrantStore()
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]

    # Patch cross_collection_search on this instance to capture args.
    captured: dict[str, Any] = {}

    async def _capture(
        query: str,
        collections: list[str],
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        mode: QueryMode = QueryMode.INTERNAL,
        visual_mode: str | None = None,
    ) -> list[SearchResult]:
        captured["query"] = query
        captured["collections"] = list(collections)
        captured["top_k"] = top_k
        captured["mode"] = mode
        captured["visual_mode"] = visual_mode
        return []

    retriever.cross_collection_search = _capture  # type: ignore[method-assign, assignment]

    await execute_retrieval(
        retriever=retriever,
        query="trail braking",
        mode=QueryMode.PUBLICATION,
        top_k=7,
        visual_mode="visual_only",
    )

    assert captured["query"] == "trail braking"
    assert captured["top_k"] == 7
    assert captured["mode"] is QueryMode.PUBLICATION
    assert captured["visual_mode"] == "visual_only"


# ----------------------------------------------------------------------
# QueryExecution dataclass
# ----------------------------------------------------------------------


def test_query_execution_dataclass_shape() -> None:
    """QueryExecution is the public dataclass passed to execute_retrieval.
    It holds the retriever plus an optional rerank hook."""
    store = FakeQdrantStore()
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]
    qe = QueryExecution(retriever=retriever)
    assert qe.retriever is retriever
    assert qe.rerank_fn is None

    # rerank_fn is assignable
    def _hook(q: str, hits: list[SearchResult]) -> list[SearchResult]:
        return hits

    qe.rerank_fn = _hook
    assert qe.rerank_fn is _hook


# ----------------------------------------------------------------------
# Imports stay framework-neutral
# ----------------------------------------------------------------------


def test_execution_module_does_not_import_fastapi_or_db() -> None:
    """execute_retrieval must be usable from a script, a worker, or a
    route handler. A side effect import of FastAPI / asyncpg / structlog
    would break that."""
    import vo_rag.core.retrieval.execution as execution_mod

    module_source = execution_mod.__file__
    assert module_source is not None
    with open(module_source, encoding="utf-8") as f:
        text = f.read()
    for forbidden in (
        "fastapi",
        "asyncpg",
        "vo_rag.core.db",
        "vo_rag.core.observability",
    ):
        assert forbidden not in text, f"core.retrieval.execution must not depend on {forbidden!r}"


def test_execution_module_does_not_import_reranker_directly() -> None:
    """The rerank is a CALLABLE parameter, not a hard import. This lets
    the reranker ship separately and keeps tests deterministic."""
    import ast

    import vo_rag.core.retrieval.execution as execution_mod

    module_source = execution_mod.__file__
    assert module_source is not None
    with open(module_source, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=module_source)

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported.append(f"{module}.{alias.name}")

    for forbidden in (
        "GteReranker",
        "vo_rag.core.reranking",
        "vo_rag.core.reranking.gte",
    ):
        for imp in imported:
            assert forbidden not in imp, (
                f"core.retrieval.execution must not import {forbidden!r}; got {imp!r}"
            )
