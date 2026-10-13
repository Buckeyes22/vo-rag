"""Contract tests for VoRagRetriever adapter parity + quarantine + mode safety."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from llama_index.core.schema import QueryBundle
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, ScoredPoint

from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import HybridRetriever, SearchResult
from vo_rag.integrations.llamaindex.retriever import VoRagRetriever

if TYPE_CHECKING:
    from collections.abc import Callable


def _field_conds(filt: Filter) -> list[FieldCondition]:
    """Return the ``FieldCondition`` entries in ``filt.must``."""
    must = filt.must or []
    return [c for c in must if isinstance(c, FieldCondition)]


class _FakeQdrantStore:
    """Minimal fake satisfying ``_QdrantStoreProtocol`` for typing."""

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


class _RecordingQdrantStore:
    """Store spy for exercising the real ``HybridRetriever`` compliance path."""

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
        filters: Filter | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]:
        self.calls.append(
            {
                "collection": collection,
                "filters": filters,
                "top_k": top_k,
                "prefetch_depth": prefetch_depth,
            }
        )
        if not self.scripted:
            return []
        return list(self.scripted.pop(0))


class _FakeEmbeddingClient:
    """Minimal four-vector embedder for the real ``HybridRetriever`` seam."""

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


class _FakeHybridRetriever(HybridRetriever):
    """Fake that returns canned SearchResults for parity tests."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        super().__init__(qdrant=_FakeQdrantStore(), embedding_client=None)
        self.results = results or []
        self.last_call: dict[str, Any] | None = None

    async def cross_collection_search(
        self,
        query: str,
        collections: list[str],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        mode: QueryMode = QueryMode.INTERNAL,
        visual_mode: str | None = None,
    ) -> list[SearchResult]:
        self.last_call = {
            "query": query,
            "collections": collections,
            "top_k": top_k,
            "filters": filters,
            "mode": mode,
            "visual_mode": visual_mode,
        }
        return self.results


def _scored_point(point_id: str, score: float, payload: dict[str, Any]) -> ScoredPoint:
    return ScoredPoint(
        id=point_id,
        score=score,
        payload=payload,
        vector=None,
        version=0,
        shard_key=None,
    )


@pytest.fixture
def fake_hybrid_retriever() -> _FakeHybridRetriever:
    return _FakeHybridRetriever()


@pytest.mark.asyncio
async def test_parity_with_execute_retrieval(
    fake_hybrid_retriever: _FakeHybridRetriever,
    make_search_result: Callable[..., SearchResult],
) -> None:
    """Adapter output == native execute_retrieval output, in order, by id+score."""
    from vo_rag.core.retrieval import execution

    fake_hybrid_retriever.results = [
        make_search_result(id="p1", score=0.9, payload_chunk_id="c1"),
        make_search_result(id="p2", score=0.8, payload_chunk_id="c2"),
        make_search_result(id="p3", score=0.7, payload_chunk_id="c3"),
    ]

    native = await execution.execute_retrieval(
        retriever=fake_hybrid_retriever,
        query="brake bias",
        mode=QueryMode.PUBLICATION,
        top_k=10,
    )
    adapter = VoRagRetriever(fake_hybrid_retriever, mode=QueryMode.PUBLICATION, top_k=10)
    nodes = await adapter._aretrieve(QueryBundle(query_str="brake bias"))

    assert [n.node.id_ for n in nodes] == [r.payload["chunk_id"] for r in native]
    assert [n.node.metadata["qdrant_point_id"] for n in nodes] == [r.id for r in native]
    assert [n.score for n in nodes] == [r.score for r in native]


def test_internal_mode_is_refused_by_default(
    fake_hybrid_retriever: _FakeHybridRetriever,
) -> None:
    with pytest.raises(ValueError, match="UNREDACTED"):
        VoRagRetriever(fake_hybrid_retriever, mode=QueryMode.INTERNAL)
    # explicit opt-in is allowed
    VoRagRetriever(fake_hybrid_retriever, mode=QueryMode.INTERNAL, allow_internal=True)


@pytest.mark.asyncio
async def test_internal_mode_is_rechecked_per_call(
    fake_hybrid_retriever: _FakeHybridRetriever,
) -> None:
    adapter = VoRagRetriever(fake_hybrid_retriever, mode=QueryMode.PUBLICATION)
    adapter._mode = QueryMode.INTERNAL

    with pytest.raises(ValueError, match="UNREDACTED"):
        await adapter._aretrieve(QueryBundle(query_str="raw excerpt"))


@pytest.mark.asyncio
async def test_publication_resolves_sterile_collections(
    fake_hybrid_retriever: _FakeHybridRetriever,
) -> None:
    adapter = VoRagRetriever(
        fake_hybrid_retriever,
        requested_collections=["vo_rag_books_raw_v1"],
    )

    await adapter._aretrieve(QueryBundle(query_str="anything"))

    assert fake_hybrid_retriever.last_call is not None
    assert fake_hybrid_retriever.last_call["mode"] == QueryMode.PUBLICATION
    assert fake_hybrid_retriever.last_call["collections"] == ["vo_rag_books_sterile_v1"]
    assert "vo_rag_books_raw_v1" not in fake_hybrid_retriever.last_call["collections"]


@pytest.mark.asyncio
async def test_publication_carries_license_filter() -> None:
    store = _RecordingQdrantStore()
    retriever = HybridRetriever(qdrant=store, embedding_client=_FakeEmbeddingClient())
    adapter = VoRagRetriever(
        retriever,
        mode=QueryMode.PUBLICATION,
        requested_collections=["vo_rag_books_raw_v1"],
        filters={"source_id": "book-1"},
        visual_mode="baseline",
    )

    await adapter._aretrieve(QueryBundle(query_str="trail braking"))

    assert len(store.calls) == 1
    assert store.calls[0]["collection"] == "vo_rag_books_sterile_v1"
    recorded_filter = store.calls[0]["filters"]
    assert isinstance(recorded_filter, Filter)
    conditions_by_key = {cond.key: cond for cond in _field_conds(recorded_filter)}
    assert {"license_class", "source_id"}.issubset(conditions_by_key)
    license_match = conditions_by_key["license_class"].match
    assert isinstance(license_match, MatchAny)
    assert set(license_match.any) == {"cleared", "public_domain"}
    source_match = conditions_by_key["source_id"].match
    assert isinstance(source_match, MatchValue)
    assert source_match.value == "book-1"


@pytest.mark.asyncio
async def test_visual_mode_quotas_threaded() -> None:
    store = _RecordingQdrantStore()
    retriever = HybridRetriever(qdrant=store, embedding_client=_FakeEmbeddingClient())
    adapter = VoRagRetriever(
        retriever,
        mode=QueryMode.PUBLICATION,
        requested_collections=["vo_rag_books_raw_v1"],
        top_k=10,
        visual_mode="visual_track",
    )

    await adapter._aretrieve(QueryBundle(query_str="any query"))

    assert sorted(call["top_k"] for call in store.calls) == [10, 10]
    text_filter = store.calls[0]["filters"]
    visual_filter = store.calls[1]["filters"]
    assert isinstance(text_filter, Filter)
    assert isinstance(visual_filter, Filter)
    text_conditions = {cond.key: cond for cond in _field_conds(text_filter)}
    visual_conditions = {cond.key: cond for cond in _field_conds(visual_filter)}
    text_match = text_conditions["chunk_type"].match
    visual_match = visual_conditions["chunk_type"].match
    assert isinstance(text_match, MatchAny)
    assert isinstance(visual_match, MatchAny)
    assert set(text_match.any) == {"text", "body"}
    assert set(visual_match.any) == {"figure_composite", "figure", "table"}


@pytest.mark.asyncio
async def test_publication_mode_never_returns_raw_content() -> None:
    pending_payload = {
        "chunk_id": "chunk-pending",
        "content": "Internal-only book excerpt",
        "section_path": "Chapter 1 > Internal",
        "license_class": "pending",
    }
    store = _RecordingQdrantStore(
        scripted=[[_scored_point("point-pending", 0.91, pending_payload)]]
    )
    retriever = HybridRetriever(qdrant=store, embedding_client=_FakeEmbeddingClient())
    adapter = VoRagRetriever(
        retriever,
        mode=QueryMode.PUBLICATION,
        requested_collections=["vo_rag_books_raw_v1"],
        visual_mode="baseline",
    )

    nodes = await adapter._aretrieve(QueryBundle(query_str="anything"))

    assert len(nodes) == 1
    assert "_vo_rag_license_warning" not in nodes[0].node.metadata
    collections = [call["collection"] for call in store.calls]
    assert collections == ["vo_rag_books_sterile_v1"]
    assert "vo_rag_books_raw_v1" not in collections


@pytest.mark.asyncio
async def test_quarantine_missing_chunk_id(
    fake_hybrid_retriever: _FakeHybridRetriever,
    make_search_result: Callable[..., SearchResult],
    monkeypatch: Any,
) -> None:
    """One hit lacking payload['chunk_id'] is dropped (warned), not allowed to fail the query."""
    fake_hybrid_retriever.results = [
        make_search_result(id="p1", score=0.9, payload_chunk_id="c1"),
        make_search_result(id="p2", score=0.8, payload={"content": "bad", "section_path": "A > B"}),
        make_search_result(
            id="p3",
            score=0.7,
            payload={
                "chunk_id": "c3",
                "content": "Distinct valid third passage",
                "section_path": "A > B",
            },
        ),
    ]

    adapter = VoRagRetriever(
        fake_hybrid_retriever,
        mode=QueryMode.PUBLICATION,
        top_k=10,
        requested_collections=["harness_collection"],
    )

    warnings: list[str] = []

    def _capture_warning(msg: str, *args: Any) -> None:
        warnings.append(msg % args)

    monkeypatch.setattr("vo_rag.integrations.llamaindex.retriever.logger.warning", _capture_warning)

    nodes = await adapter._aretrieve(QueryBundle(query_str="anything"))

    assert len(nodes) == 2
    assert nodes[0].node.id_ == "c1"
    assert nodes[1].node.id_ == "c3"
    assert len(warnings) == 1
    assert "quarantined" in warnings[0]
    assert "p2" in warnings[0]


@pytest.mark.asyncio
async def test_all_quarantined_hits_are_logged_as_error(
    fake_hybrid_retriever: _FakeHybridRetriever,
    make_search_result: Callable[..., SearchResult],
    monkeypatch: Any,
) -> None:
    fake_hybrid_retriever.results = [
        make_search_result(id="p1", score=0.9, payload={"content": "bad"}),
        make_search_result(id="p2", score=0.8, payload={"content": "also bad"}),
    ]
    adapter = VoRagRetriever(
        fake_hybrid_retriever,
        mode=QueryMode.PUBLICATION,
        top_k=10,
        requested_collections=["harness_collection"],
    )
    errors: list[str] = []

    def _capture_error(msg: str, *args: Any) -> None:
        errors.append(msg % args)

    monkeypatch.setattr("vo_rag.integrations.llamaindex.retriever.logger.error", _capture_error)

    nodes = await adapter._aretrieve(QueryBundle(query_str="anything"))

    assert nodes == []
    assert len(errors) == 1
    assert "all retrieval hits quarantined" in errors[0]
    assert "2/2" in errors[0]
