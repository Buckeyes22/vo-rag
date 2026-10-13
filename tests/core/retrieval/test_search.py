"""Tests for ``HybridRetriever`` — the public retrieve path.

The retriever does not reach past the store into
``self.qdrant._client.query_points``. It delegates to
``QdrantStore.search_hybrid`` (and per-collection
``search_hybrid`` + canonical RRF for cross-collection fan-out).

The ``test_hybrid_retriever_delegates_to_qdrant_store`` test below is
the regression guard against a parallel retrieval implementation that
bypasses the store.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    ScoredPoint,
    VectorParams,
)

from vo_rag.core.compliance.license_filter import (
    license_filter_for_mode,
    merge_with_user_filter,
)
from vo_rag.core.dsi.routing import BookRoute, RoutingDecision
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import (
    HybridRetriever,
    SearchResult,
    _build_filter,
    _with_generation_filter,
)


def _field_conds(filt: Filter) -> list[FieldCondition]:
    """Return the ``FieldCondition`` entries in ``filt.must``.

    The qdrant-client types ``Filter.must`` as a broad union; in the
    filters the retriever builds every entry is a ``FieldCondition``.
    This helper narrows the union so the tests can read ``.key`` and
    ``.match.any`` without tripping mypy's invariance check.
    """
    must = filt.must or []
    return [c for c in must if isinstance(c, FieldCondition)]


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class FakeEmbedClient:
    """Returns a deterministic 4-vector response keyed off the call args."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "texts": list(texts),
                "return_dense": return_dense,
                "return_sparse": return_sparse,
                "return_colbert": return_colbert,
            }
        )
        return {
            "dense": [[0.1, 0.2, 0.3]],
            "sparse": [{"indices": [1, 2, 3], "values": [0.4, 0.5, 0.6]}],
            "colbert": [[[0.7, 0.8]]],
        }


class FakeQdrantClient:
    """Bare Qdrant client — should NEVER be reached by HybridRetriever."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def query_points(self, **kwargs: Any) -> Any:  # pragma: no cover - guarded
        self.calls.append(kwargs)
        raise AssertionError(
            "HybridRetriever must not reach self.qdrant._client.query_points "
            "directly; use QdrantStore.search_hybrid instead."
        )


class FakeQdrantStore:
    """Records every call to the canonical seam (search_hybrid, etc.)."""

    def __init__(self) -> None:
        self._client = FakeQdrantClient()
        self.search_hybrid_calls: list[dict[str, Any]] = []
        self.next_results: list[ScoredPoint] = []

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
        self.search_hybrid_calls.append(
            {
                "collection": collection,
                "query_dense": list(query_dense),
                "query_sparse": dict(query_sparse),
                "query_bm25": dict(query_bm25),
                "query_colbert": [list(v) for v in query_colbert],
                "top_k": top_k,
                "filters": filters,
                "prefetch_depth": prefetch_depth,
            }
        )
        return list(self.next_results)


def _scored_point(
    point_id: str,
    score: float,
    payload: dict[str, Any] | None = None,
) -> ScoredPoint:
    return ScoredPoint(
        id=point_id,
        score=score,
        payload=payload or {},
        vector=None,
        version=0,
        shard_key=None,
    )


# ----------------------------------------------------------------------
# _build_filter — the small private helper for Qdrant filter construction
# ----------------------------------------------------------------------


def test_build_filter_scalar_value_uses_match_value() -> None:
    built = _build_filter({"license_class": "cleared"})
    assert isinstance(built, Filter)
    conds = _field_conds(built)
    assert conds, "filter should have at least one must clause"
    assert isinstance(conds[0].match, MatchValue)


def test_build_filter_list_value_uses_match_any() -> None:
    built = _build_filter({"license_class": ["cleared", "public_domain"]})
    assert isinstance(built, Filter)
    conds = _field_conds(built)
    assert conds, "filter should have at least one must clause"
    assert isinstance(conds[0].match, MatchAny)


def test_build_filter_none_or_empty_returns_none() -> None:
    assert _build_filter(None) is None
    assert _build_filter({}) is None


def test_build_filter_turns_a_bounds_dict_into_a_range() -> None:
    f = _build_filter({"tf_pub_year": {"gte": 1990, "lte": 2010}, "tf_event_family": ["sprint"]})
    assert f is not None
    by_key = {c.key: c for c in _field_conds(f)}
    assert by_key["tf_pub_year"].range == Range(gte=1990, lte=2010)
    family = by_key["tf_event_family"].match
    assert isinstance(family, MatchAny)
    assert family.any == ["sprint"]


def test_build_filter_rejects_an_unknown_bound() -> None:
    with pytest.raises(ValueError, match="unsupported bound"):
        _build_filter({"tf_pub_year": {"between": [1, 2]}})


def test_build_filter_rejects_an_empty_bounds_dict() -> None:
    """``{}`` carries no bound, so a Range built from it would match
    everything while looking like a real constraint (VOR-1443)."""
    with pytest.raises(ValueError, match="unsupported bound"):
        _build_filter({"tf_pub_year": {}})


@pytest.mark.parametrize(
    "bounds",
    [
        {"gte": None},
        {"gte": "1990"},
        {"gte": True},
        {"gte": float("nan")},
        {"gte": float("inf")},
        {"gte": float("-inf")},
    ],
)
def test_build_filter_rejects_a_non_numeric_bound(bounds: dict[str, Any]) -> None:
    """A ``None``/string/boolean/non-finite-float bound never reaches Qdrant
    as a usable bound — the client drops it or Qdrant treats it as unbounded,
    so the clause reads like a constraint while matching every point
    (VOR-1443)."""
    with pytest.raises(ValueError, match="unsupported bound"):
        _build_filter({"tf_pub_year": bounds})


def test_build_filter_unknown_bound_message_handles_non_str_keys() -> None:
    """Building the "unsupported bound(s)" message must not call ``sorted()``
    directly on a set that can hold both ``str`` and non-``str`` keys — Python
    cannot order ``int`` against ``str``, so a non-str key must still raise
    the intended ``ValueError``, not a ``TypeError`` (VOR-1443 r3)."""
    with pytest.raises(ValueError, match="unsupported bound"):
        _build_filter({"tf_pub_year": {1: 5, "x": 2}})


def test_build_filter_reads_a_range_from_any_mapping() -> None:
    """A read-only view (``MappingProxyType``, a wrapped request body) is a
    ``Mapping`` without being a ``dict``; treating it as a scalar would match
    on the mapping object instead of the year (VOR-1443)."""
    f = _build_filter({"tf_pub_year": MappingProxyType({"gte": 1990, "lte": 2010})})
    assert f is not None
    by_key = {c.key: c for c in _field_conds(f)}
    assert by_key["tf_pub_year"].range == Range(gte=1990, lte=2010)


def test_range_condition_survives_merge_with_user_filter() -> None:
    """A Range clause must survive the license/user merge — a dropped clause
    silently widens a ``tf_pub_year`` range into "any year" (VOR-1443)."""
    user_filter = _build_filter(
        {"tf_pub_year": {"gte": 1990, "lte": 2010}, "source_book": "race_engineering"}
    )
    assert user_filter is not None

    merged = merge_with_user_filter(license_filter_for_mode(QueryMode.PUBLICATION), user_filter)
    assert merged is not None

    by_key = {c.key: c for c in _field_conds(merged)}
    assert by_key["tf_pub_year"].range == Range(gte=1990, lte=2010)
    book = by_key["source_book"].match
    assert isinstance(book, MatchValue)
    assert book.value == "race_engineering"


# ----------------------------------------------------------------------
# HybridRetriever.hybrid_search — the delegation contract
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_search_delegates_to_qdrant_store_search_hybrid() -> None:
    """HybridRetriever delegates to
    QdrantStore.search_hybrid. It does NOT call
    ``qdrant._client.query_points`` directly."""
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    await retriever.hybrid_search(
        "trail braking",
        "vo_rag_books_raw_v1",
        top_k=7,
    )

    assert len(store.search_hybrid_calls) == 1
    call = store.search_hybrid_calls[0]
    assert call["collection"] == "vo_rag_books_raw_v1"
    assert call["top_k"] == 7
    # Vector plumbing round-trip — dense/sparse/colbert from embed,
    # bm25 produced locally from the BM25 tokenizer.
    assert call["query_dense"] == [0.1, 0.2, 0.3]
    assert call["query_sparse"] == {1: 0.4, 2: 0.5, 3: 0.6}
    assert call["query_colbert"] == [[0.7, 0.8]]
    assert call["query_bm25"]  # non-empty — query text was tokenized

    # The Qdrant client's query_points MUST NOT have been called.
    assert store._client.calls == []


@pytest.mark.asyncio
async def test_hybrid_search_requests_dense_sparse_colbert_in_one_call() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    await retriever.hybrid_search("trail braking", "vo_rag_books_raw_v1")

    assert embed.calls, "embedding client must have been called"
    embed_call = embed.calls[0]
    assert embed_call["texts"] == ["trail braking"]
    assert embed_call["return_dense"] is True
    assert embed_call["return_sparse"] is True
    assert embed_call["return_colbert"] is True


@pytest.mark.asyncio
async def test_generation_rollout_filters_default_book_retrieval_to_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VORAG_INGEST_GENERATIONS", raising=False)
    store = FakeQdrantStore()
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]

    await retriever.hybrid_search("trail braking", "vo_rag_books_raw_v1")

    query_filter = store.search_hybrid_calls[0]["filters"]
    assert isinstance(query_filter, Filter)
    clauses = query_filter.should
    if clauses is None:
        clauses = []
    elif not isinstance(clauses, list):
        clauses = [clauses]
    active = [
        condition
        for condition in clauses
        if isinstance(condition, FieldCondition) and condition.key == "ingest_state"
    ]
    assert len(active) == 1
    assert isinstance(active[0].match, MatchValue)
    assert active[0].match.value == "active"
    legacy = [condition for condition in clauses if isinstance(condition, IsEmptyCondition)]
    assert len(legacy) == 1
    assert legacy[0].is_empty.key == "ingest_state"


@pytest.mark.asyncio
async def test_generation_filter_is_always_on_for_every_searched_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeQdrantStore()
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]

    monkeypatch.delenv("VORAG_INGEST_GENERATIONS", raising=False)
    await retriever.hybrid_search("one", "vo_rag_books_raw_v1")
    await retriever.hybrid_search("two", "vo_rag_research_v1")

    book_filter = store.search_hybrid_calls[0]["filters"]
    assert isinstance(book_filter, Filter)
    assert book_filter.should is not None
    non_book_filter = store.search_hybrid_calls[1]["filters"]
    assert isinstance(non_book_filter, Filter)
    assert non_book_filter.should is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "collection",
    ["vo_rag_books_raw_v1", "vo_rag_reddit_v1", "vo_rag_rc_custom_v1"],
)
async def test_default_filter_rejects_batch_staged_points_in_local_qdrant(
    collection: str,
) -> None:
    client = AsyncQdrantClient(location=":memory:")
    try:
        await client.create_collection(
            collection,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
        states = ["active", "staging", "failed", "quarantined", "superseded"]
        points = [
            PointStruct(
                id=index,
                vector=[1.0],
                payload={"ingest_state": state, "generation_id": f"gen-{index}"},
            )
            for index, state in enumerate(states, start=1)
        ]
        points.append(PointStruct(id=99, vector=[1.0], payload={"legacy": True}))
        await client.upsert(collection, points, wait=True)

        visible, _offset = await client.scroll(
            collection,
            scroll_filter=_with_generation_filter(collection, None),
            limit=10,
        )

        assert {point.id for point in visible} == {1, 99}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_internal_validator_can_explicitly_inspect_one_staged_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VORAG_INGEST_GENERATIONS", raising=False)
    store = FakeQdrantStore()
    retriever = HybridRetriever(qdrant=store, embedding_client=FakeEmbedClient())  # type: ignore[arg-type]

    await retriever.hybrid_search(
        "validation query",
        "vo_rag_books_raw_v1",
        filters={"generation_id": "gen-1"},
        ingest_state_override="staging",
    )

    query_filter = store.search_hybrid_calls[0]["filters"]
    assert isinstance(query_filter, Filter)
    values = {
        condition.key: getattr(condition.match, "value", None)
        for condition in _field_conds(query_filter)
    }
    assert values == {"generation_id": "gen-1", "ingest_state": "staging"}


@pytest.mark.asyncio
async def test_prepared_vector_bundle_preserves_result_ranking() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]
    store.next_results = [
        _scored_point("first", 0.9, {"source_id": "one"}),
        _scored_point("second", 0.8, {"source_id": "two"}),
    ]

    implicit = await retriever.hybrid_search("trail braking", "books")
    bundle = await retriever.prepare_query("trail braking")
    prepared = await retriever.hybrid_search("trail braking", "books", vector_bundle=bundle)

    assert [(result.id, result.score) for result in prepared] == [
        (result.id, result.score) for result in implicit
    ]


@pytest.mark.asyncio
async def test_hybrid_search_bm25_uses_same_tokenizer_as_writers() -> None:
    """The BM25 id scheme is the persistence contract — query and write
    paths share ``tokens_to_bm25_sparse``. Drift breaks every Qdrant
    point. Pin the sparse shape: a non-empty {int: float} dict built
    from the same tokenizer used by every Qdrant writer."""
    from vo_rag.core.retrieval.bm25_tokenizer import (
        tokenize,
        tokens_to_bm25_sparse,
    )

    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    await retriever.hybrid_search("trail braking balance", "vo_rag_books_raw_v1")

    expected = tokens_to_bm25_sparse(tokenize("trail braking balance"))
    assert store.search_hybrid_calls[0]["query_bm25"] == expected


@pytest.mark.asyncio
async def test_hybrid_search_wraps_scored_points_as_search_results() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    store.next_results = [
        _scored_point(
            "p1",
            0.91,
            {"source_id": "src-1", "page_number": 1, "chunk_id": "c1"},
        ),
        _scored_point(
            "p2",
            0.84,
            {"source_id": "src-2", "page_number": 2, "chunk_id": "c2"},
        ),
    ]

    results = await retriever.hybrid_search("trail braking", "vo_rag_books_raw_v1")

    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].id == "p1"
    assert results[0].score == pytest.approx(0.91)
    assert results[0].collection == "vo_rag_books_raw_v1"
    assert results[0].payload["source_id"] == "src-1"


@pytest.mark.asyncio
async def test_hybrid_search_passes_user_filters_through_to_store() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    await retriever.hybrid_search(
        "trail braking",
        "vo_rag_books_raw_v1",
        mode=QueryMode.PUBLICATION,  # exercise the merge path
        filters={"source_book": "race_engineering"},
    )

    call_filters = store.search_hybrid_calls[0]["filters"]
    assert isinstance(call_filters, Filter)
    keys = {cond.key for cond in _field_conds(call_filters)}
    assert "source_book" in keys


@pytest.mark.asyncio
async def test_hybrid_search_publication_adds_license_filter_to_store() -> None:
    """License filter merge is security-sensitive — it must reach the
    store as a Qdrant Filter (not the raw user dict)."""
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    await retriever.hybrid_search(
        "trail braking",
        "vo_rag_books_sterile_v1",
        top_k=5,
        mode=QueryMode.PUBLICATION,
    )

    call_filters = store.search_hybrid_calls[0]["filters"]
    # User filter is None -> store receives the license filter directly.
    assert isinstance(call_filters, Filter)
    conds = _field_conds(call_filters)
    assert conds[0].key == "license_class"
    assert isinstance(conds[0].match, MatchAny)
    assert set(conds[0].match.any) == {"cleared", "public_domain"}


@pytest.mark.asyncio
async def test_hybrid_search_internal_merges_license_filter_with_user_filter() -> None:
    """Internal mode = no license restriction, but the user's filters
    still need to reach the store intact."""
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    await retriever.hybrid_search(
        "trail braking",
        "vo_rag_books_raw_v1",
        top_k=5,
        mode=QueryMode.INTERNAL,
        filters={"source_book": "race_engineering"},
    )

    call_filters = store.search_hybrid_calls[0]["filters"]
    assert isinstance(call_filters, Filter)
    keys = {cond.key for cond in _field_conds(call_filters)}
    assert "source_book" in keys
    assert "license_class" not in keys  # internal mode = no license filter


@pytest.mark.asyncio
async def test_hybrid_search_publication_merges_user_filter_with_license() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    await retriever.hybrid_search(
        "trail braking",
        "vo_rag_books_sterile_v1",
        top_k=5,
        mode=QueryMode.PUBLICATION,
        filters={"source_book": "race_engineering"},
    )

    call_filters = store.search_hybrid_calls[0]["filters"]
    assert isinstance(call_filters, Filter)
    keys = {cond.key for cond in _field_conds(call_filters)}
    assert "license_class" in keys
    assert "source_book" in keys


@pytest.mark.asyncio
async def test_hybrid_search_annotates_unlicensed_in_internal_mode() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    store.next_results = [
        _scored_point(
            "p1",
            0.9,
            {"source_id": "src-1", "license_class": "pending"},
        ),
    ]

    results = await retriever.hybrid_search(
        "trail braking",
        "vo_rag_books_raw_v1",
        mode=QueryMode.INTERNAL,
    )

    assert results[0].payload.get("_vo_rag_license_warning") == "source_license_class=pending"


@pytest.mark.asyncio
async def test_hybrid_search_does_not_annotate_in_publication_mode() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    store.next_results = [
        _scored_point("p1", 0.9, {"source_id": "src-1", "license_class": "cleared"}),
    ]

    results = await retriever.hybrid_search(
        "trail braking",
        "vo_rag_books_sterile_v1",
        mode=QueryMode.PUBLICATION,
    )

    assert "_vo_rag_license_warning" not in results[0].payload


@pytest.mark.asyncio
async def test_hybrid_search_propagates_prefetch_depth() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    await retriever.hybrid_search(
        "trail braking",
        "vo_rag_books_raw_v1",
        prefetch_depth=200,
    )

    assert store.search_hybrid_calls[0]["prefetch_depth"] == 200


# ----------------------------------------------------------------------
# HybridRetriever.cross_collection_search — visual-mode dispatch + RRF
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_collection_baseline_uses_canonical_rrf() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    # Two collections, one shared (source_id, chunk_id) point -> dedup
    # in the canonical retriever RRF (NOT the store's
    # (source_id, page_number) key).
    store.next_results = [
        _scored_point(
            "a-raw",
            0.9,
            {"source_id": "src-1", "page_number": 1, "chunk_id": "shared"},
        ),
    ]

    # Two different collections, same id, same chunk_id -> per-list hits
    # carry the same (source_id, chunk_id) dedup key -> one merged entry.
    async def _route_one_per_collection(
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
        if collection == "vo_rag_books_raw_v1":
            return [_scored_point("a-raw", 0.9, {"source_id": "src-1", "chunk_id": "shared"})]
        return [_scored_point("a-sterile", 0.85, {"source_id": "src-1", "chunk_id": "shared"})]

    store.search_hybrid = _route_one_per_collection  # type: ignore[method-assign]

    results = await retriever.cross_collection_search(
        query="trail braking",
        collections=["vo_rag_books_raw_v1", "vo_rag_books_sterile_v1"],
        top_k=5,
        mode=QueryMode.INTERNAL,
        visual_mode="baseline",
    )

    assert len(results) == 1
    assert results[0].id in {"a-raw", "a-sterile"}


@pytest.mark.asyncio
async def test_cross_collection_search_visual_only_restricts_to_visual_chunks() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    observed_filters: list[dict[str, Any] | None] = []

    async def _route(
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
        observed_filters.append(filters)
        return []

    store.search_hybrid = _route  # type: ignore[method-assign]

    await retriever.cross_collection_search(
        query="trail braking",
        collections=["vo_rag_books_raw_v1"],
        top_k=10,
        mode=QueryMode.INTERNAL,
        visual_mode="visual_only",
    )

    assert observed_filters, "store.search_hybrid must have been called"
    filt = observed_filters[0]
    assert isinstance(filt, Filter)
    keys = {cond.key for cond in _field_conds(filt)}
    assert "chunk_type" in keys


@pytest.mark.asyncio
async def test_cross_collection_search_top_k_one_uses_unsplit_baseline() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    seen_calls: list[dict[str, Any]] = []

    async def _route(
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
        seen_calls.append({"collection": collection, "top_k": top_k, "filters": filters})
        return [
            _scored_point(
                f"{collection}-{top_k}",
                0.5,
                {"source_id": "x", "chunk_id": f"{collection}-{top_k}"},
            )
        ]

    store.search_hybrid = _route  # type: ignore[method-assign]

    results = await retriever.cross_collection_search(
        query="trail braking",
        collections=["vo_rag_books_raw_v1"],
        top_k=1,
        mode=QueryMode.INTERNAL,
        visual_mode=None,
    )

    assert len(results) <= 1
    assert [call["top_k"] for call in seen_calls] == [1]
    assert isinstance(seen_calls[0]["filters"], Filter)


@pytest.mark.asyncio
async def test_cross_collection_search_intent_aware_splits_text_and_visual() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    seen_chunk_type_filters: list[Any] = []

    async def _route(
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
        if isinstance(filters, Filter):
            for cond in filters.must:
                if cond.key == "chunk_type":
                    seen_chunk_type_filters.append(cond.match.any)
        return [_scored_point(f"{collection}-{top_k}", 0.5, {"source_id": "x"})]

    store.search_hybrid = _route  # type: ignore[method-assign]

    await retriever.cross_collection_search(
        query="show me the diagram of slip angle",  # visual-intent
        collections=["vo_rag_books_raw_v1"],
        top_k=10,
        mode=QueryMode.INTERNAL,
        visual_mode="intent_aware",
    )

    # 70/30 visual/text split when visual intent is detected
    assert any(set(c) == {"figure_composite", "figure", "table"} for c in seen_chunk_type_filters)
    assert any(set(c) == {"text", "body"} for c in seen_chunk_type_filters)


@pytest.mark.asyncio
async def test_cross_collection_two_track_fanout_embeds_exactly_once() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    await retriever.cross_collection_search(
        query="show a diagram of trail braking",
        collections=[f"collection-{index}" for index in range(10)],
        top_k=20,
        visual_mode="intent_aware",
    )

    assert len(embed.calls) == 1
    assert len(store.search_hybrid_calls) == 20  # ten collections, two tracks
    assert {tuple(call["query_dense"]) for call in store.search_hybrid_calls} == {(0.1, 0.2, 0.3)}


@pytest.mark.asyncio
async def test_cross_collection_search_rejects_unknown_visual_mode() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="visual_mode"):
        await retriever.cross_collection_search(
            query="trail braking",
            collections=["vo_rag_books_raw_v1"],
            top_k=10,
            mode=QueryMode.INTERNAL,
            visual_mode="not-a-real-mode",
        )


@pytest.mark.asyncio
async def test_cross_collection_search_internal_mode_returns_annotated_results() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    async def _route(
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
        return [
            _scored_point(
                f"{collection}-p",
                0.5,
                {"source_id": "x", "license_class": "pending"},
            )
        ]

    store.search_hybrid = _route  # type: ignore[method-assign]

    results = await retriever.cross_collection_search(
        query="trail braking",
        collections=["vo_rag_books_raw_v1", "vo_rag_articles_v1"],
        top_k=10,
        mode=QueryMode.INTERNAL,
        visual_mode="baseline",
    )

    assert all("_vo_rag_license_warning" in (r.payload or {}) for r in results)


@pytest.mark.asyncio
async def test_hybrid_search_never_touches_qdrant_client_query_points() -> None:
    """End-to-end guard: no matter what filters / mode / visual_mode
    the user picks, the retriever MUST NOT reach past the store."""
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    await retriever.hybrid_search(
        "trail braking", "vo_rag_books_raw_v1", mode=QueryMode.PUBLICATION
    )
    await retriever.cross_collection_search(
        "trail braking",
        ["vo_rag_books_raw_v1", "vo_rag_articles_v1"],
        mode=QueryMode.INTERNAL,
        visual_mode="visual_only",
    )

    assert store._client.calls == []


# ----------------------------------------------------------------------
# HybridRetriever.dsi_routed_search — additive DSI-routed path
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dsi_routed_search_issues_per_book_calls_and_merges() -> None:
    """Each BookRoute with total_budget > 0 triggers one search_hybrid call
    sized to that route's budget, filtered by source_id = book_id."""
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    routing = RoutingDecision(
        book_routes=[
            BookRoute(
                book_id="book-a",
                confidence=0.5,
                primary_budget=5,
                fallback_budget=1,
                total_budget=6,
            ),
            BookRoute(
                book_id="book-b",
                confidence=0.3,
                primary_budget=4,
                fallback_budget=1,
                total_budget=5,
            ),
            BookRoute(
                book_id="book-c",
                confidence=0.05,
                primary_budget=0,
                fallback_budget=0,
                total_budget=0,
            ),
        ],
        top_k=2,
        total_budget_chunks=20,
        primary_share=0.95,
        fallback_share=0.05,
    )

    async def _route(
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        *,
        top_k: int = 20,
        filters: Any | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]:
        # Extract book_id from the filter
        book_id: str | None = None
        if isinstance(filters, Filter) and filters.must:
            for cond in filters.must:
                if (
                    isinstance(cond, FieldCondition)
                    and cond.key == "source_id"
                    and isinstance(cond.match, MatchValue)
                ):
                    book_id = str(cond.match.value)
        if book_id == "book-a":
            return [_scored_point(f"{book_id}-p{i}", 0.9 - i * 0.01) for i in range(top_k)]
        if book_id == "book-b":
            return [_scored_point(f"{book_id}-p{i}", 0.8 - i * 0.01) for i in range(top_k)]
        return []

    store.search_hybrid = _route  # type: ignore[method-assign]

    results = await retriever.dsi_routed_search(
        "trail braking",
        routing=routing,
        raw_collection="vo_rag_books_raw_v1",
    )

    # book-a budget = 6, book-b budget = 5, book-c budget = 0 (skipped)
    book_a_results = [r for r in results if r.id.startswith("book-a")]
    book_b_results = [r for r in results if r.id.startswith("book-b")]
    assert len(book_a_results) == 6
    assert len(book_b_results) == 5

    # Results should be SearchResult instances from the raw collection
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.collection == "vo_rag_books_raw_v1" for r in results)


@pytest.mark.asyncio
async def test_dsi_routed_search_merges_user_filter_with_book_filter() -> None:
    """The user's filters are AND-combined with the per-book source_id filter."""
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    observed_filters: list[Filter | None] = []

    async def _route(
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        *,
        top_k: int = 20,
        filters: Any | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]:
        observed_filters.append(filters)
        return []

    store.search_hybrid = _route  # type: ignore[method-assign]

    routing = RoutingDecision(
        book_routes=[
            BookRoute(
                book_id="book-a",
                confidence=0.5,
                primary_budget=3,
                fallback_budget=0,
                total_budget=3,
            ),
        ],
        top_k=1,
        total_budget_chunks=10,
        primary_share=1.0,
        fallback_share=0.0,
    )

    await retriever.dsi_routed_search(
        "trail braking",
        routing=routing,
        raw_collection="vo_rag_books_raw_v1",
        filters={"license_class": "cleared"},
    )

    assert len(observed_filters) == 1
    filt = observed_filters[0]
    assert isinstance(filt, Filter)
    keys = {cond.key for cond in _field_conds(filt)}
    assert "source_id" in keys
    assert "license_class" in keys


@pytest.mark.asyncio
async def test_dsi_routed_search_applies_license_annotation() -> None:
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    async def _route(
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        *,
        top_k: int = 20,
        filters: Any | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]:
        return [
            _scored_point(
                "p1",
                0.9,
                {"source_id": "book-a", "license_class": "pending"},
            )
        ]

    store.search_hybrid = _route  # type: ignore[method-assign]

    routing = RoutingDecision(
        book_routes=[
            BookRoute(
                book_id="book-a",
                confidence=0.5,
                primary_budget=3,
                fallback_budget=0,
                total_budget=3,
            ),
        ],
        top_k=1,
        total_budget_chunks=10,
        primary_share=1.0,
        fallback_share=0.0,
    )

    results = await retriever.dsi_routed_search(
        "trail braking",
        routing=routing,
        raw_collection="vo_rag_books_raw_v1",
        mode=QueryMode.INTERNAL,
    )

    assert len(results) == 1
    assert results[0].payload.get("_vo_rag_license_warning") == "source_license_class=pending"


@pytest.mark.asyncio
async def test_dsi_routed_search_does_not_touch_qdrant_client() -> None:
    """The additive DSI path must also stay behind the store seam."""
    store = FakeQdrantStore()
    embed = FakeEmbedClient()
    retriever = HybridRetriever(qdrant=store, embedding_client=embed)  # type: ignore[arg-type]

    routing = RoutingDecision(
        book_routes=[
            BookRoute(
                book_id="book-a",
                confidence=0.5,
                primary_budget=1,
                fallback_budget=0,
                total_budget=1,
            ),
        ],
        top_k=1,
        total_budget_chunks=10,
        primary_share=1.0,
        fallback_share=0.0,
    )

    await retriever.dsi_routed_search(
        "trail braking",
        routing=routing,
        raw_collection="vo_rag_books_raw_v1",
    )

    assert store._client.calls == []


# Sanity: SearchResult is a regular dataclass (not pydantic) — that's the
# public contract (no coercion overhead, the wrapper function is the only
# writer).
def test_search_result_is_dataclass_not_pydantic() -> None:
    import dataclasses

    assert dataclasses.is_dataclass(SearchResult)


def test_search_result_collection_required() -> None:
    with pytest.raises(TypeError):
        SearchResult(id="x", score=0.0)  # type: ignore[call-arg]
