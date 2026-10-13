"""Prove publication filtering at Qdrant read seams with local storage."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSelectorExclude,
    PayloadSelectorInclude,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from vo_rag.core.storage.publication_client import PublicationClient

_BOOKS = "vo_rag_books_raw_v1"
_BRAIN = "vo_rag_brain_v1"


class _PointerReader:
    def __init__(self, active: dict[str, int | None]) -> None:
        self.active = active
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, corpus: str, collection: str) -> dict[str, int | None]:
        self.calls.append((corpus, collection))
        return dict(self.active)


class _RecordingRaw:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self.client = client
        self.last_query: dict[str, Any] | None = None

    async def query_points(self, collection_name: str, **kwargs: Any) -> Any:
        self.last_query = dict(kwargs)
        return await self.client.query_points(collection_name=collection_name, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


async def _client(collections: tuple[str, ...] = (_BOOKS,)) -> AsyncQdrantClient:
    client = AsyncQdrantClient(location=":memory:")
    for collection in collections:
        await client.create_collection(
            collection,
            vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(), "bm25": SparseVectorParams()},
        )
    return client


def _point(
    point_id: int,
    source_id: str,
    revision: int | None,
    dense: list[float],
    *,
    sparse_index: int = 1,
    book_id: str = "book-1",
) -> PointStruct:
    payload: dict[str, Any] = {"source_id": source_id, "book_id": book_id, "title": "Book"}
    if revision is not None:
        payload["source_revision"] = revision
    sparse = SparseVector(indices=[sparse_index], values=[1.0])
    return PointStruct(
        id=point_id,
        vector={"dense": dense, "sparse": sparse, "bm25": sparse},
        payload=payload,
    )


async def _seed_books(client: AsyncQdrantClient) -> None:
    await client.upsert(
        _BOOKS,
        points=[
            _point(1, "book-1", 1, [1.0, 0.0], sparse_index=1),
            _point(2, "book-1", 2, [0.9, 0.1], sparse_index=1),
            _point(3, "other-legacy", None, [0.8, 0.2], sparse_index=1, book_id="other-book"),
            _point(4, "unregistered-stage", 9, [1.0, 0.0], sparse_index=1),
        ],
        wait=True,
    )


@pytest.mark.asyncio
async def test_dense_only_query_keeps_active_revision_and_legacy_other_source() -> None:
    client = await _client()
    try:
        await _seed_books(client)
        pointers = _PointerReader({"book-1": 2})
        wrapped = PublicationClient(client, pointers)

        result = await wrapped.query_points(
            _BOOKS,
            query=[1.0, 0.0],
            using="dense",
            limit=10,
        )

        assert {point.id for point in result.points} == {2, 3}
        assert pointers.calls == [("books", _BOOKS)]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_recursive_lexical_prefetch_and_final_filter_share_one_pointer_snapshot() -> None:
    client = await _client()
    try:
        await _seed_books(client)
        pointers = _PointerReader({"book-1": 2})
        raw = _RecordingRaw(client)
        wrapped = PublicationClient(raw, pointers)
        nested_sparse = Prefetch(
            query=SparseVector(indices=[1], values=[1.0]), using="sparse", limit=10
        )
        dense_stage = Prefetch(query=[1.0, 0.0], using="dense", limit=10, prefetch=[nested_sparse])
        lexical_stage = Prefetch(
            query=SparseVector(indices=[1], values=[1.0]), using="bm25", limit=10
        )

        result = await wrapped.query_points(
            _BOOKS,
            prefetch=[dense_stage, lexical_stage],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=10,
            query_filter=Filter(
                must=[FieldCondition(key="book_id", match=MatchValue(value="book-1"))]
            ),
        )

        assert {point.id for point in result.points} == {2}
        assert pointers.calls == [("books", _BOOKS)]
        assert raw.last_query is not None
        final_filter = raw.last_query["query_filter"]
        top_prefetch = raw.last_query["prefetch"]
        assert final_filter is not None and final_filter.must is not None
        assert len(top_prefetch) == 2
        for stage in top_prefetch:
            assert stage.filter is not None
        assert top_prefetch[0].prefetch is not None
        assert top_prefetch[0].prefetch[0].filter is not None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_scroll_and_count_keep_caller_scope_and_apply_book_publication() -> None:
    client = await _client()
    try:
        await _seed_books(client)
        pointers = _PointerReader({"book-1": 2})
        wrapped = PublicationClient(client, pointers)
        caller_filter = Filter(
            must=[FieldCondition(key="book_id", match=MatchValue(value="book-1"))]
        )

        rows, next_offset = await wrapped.scroll(
            _BOOKS,
            scroll_filter=caller_filter,
            with_payload=True,
            with_vectors=False,
            limit=10,
        )
        count = await wrapped.count(
            _BOOKS,
            count_filter=caller_filter,
            exact=True,
        )

        assert next_offset is None
        assert [row.id for row in rows] == [2]
        assert count.count == 1
        assert pointers.calls == [("books", _BOOKS), ("books", _BOOKS)]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_retrieve_filters_before_restoring_payload_selection() -> None:
    client = await _client()
    try:
        await _seed_books(client)
        pointers = _PointerReader({"book-1": 2})
        wrapped = PublicationClient(client, pointers)

        without_payload = await wrapped.retrieve(
            _BOOKS, ids=[1, 2, 4], with_payload=False, with_vectors=False
        )
        selected_payload = await wrapped.retrieve(
            _BOOKS, ids=[1, 2, 4], with_payload=["source_id", "source_revision"]
        )

        assert [row.id for row in without_payload] == [2]
        assert without_payload[0].payload is None
        assert [row.id for row in selected_payload] == [2]
        assert selected_payload[0].payload == {"source_id": "book-1", "source_revision": 2}
        assert pointers.calls == [("books", _BOOKS), ("books", _BOOKS)]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_retrieve_payload_selector_objects_preserve_include_and_exclude_fields() -> None:
    client = await _client()
    try:
        await _seed_books(client)
        wrapped = PublicationClient(client, _PointerReader({"book-1": 2}))

        included = await wrapped.retrieve(
            _BOOKS,
            ids=[2],
            with_payload=PayloadSelectorInclude(include=["source_id", "source_revision"]),
            with_vectors=False,
        )
        excluded = await wrapped.retrieve(
            _BOOKS,
            ids=[2],
            with_payload=PayloadSelectorExclude(exclude=["title"]),
            with_vectors=False,
        )

        assert included[0].payload == {"source_id": "book-1", "source_revision": 2}
        assert excluded[0].payload == {
            "source_id": "book-1",
            "book_id": "book-1",
            "source_revision": 2,
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_retrieve_payload_selectors_preserve_nested_field_selection() -> None:
    client = await _client()
    try:
        await client.upsert(
            _BOOKS,
            points=[
                PointStruct(
                    id=20,
                    vector={"dense": [1.0, 0.0]},
                    payload={
                        "source_id": "book-1",
                        "source_revision": 2,
                        "metadata": {"element_index": 17, "parser": "epub"},
                    },
                )
            ],
            wait=True,
        )
        wrapped = PublicationClient(client, _PointerReader({"book-1": 2}))

        included = await wrapped.retrieve(
            _BOOKS,
            ids=[20],
            with_payload=PayloadSelectorInclude(
                include=["metadata.element_index", "metadata.parser"]
            ),
            with_vectors=False,
        )
        excluded = await wrapped.retrieve(
            _BOOKS,
            ids=[20],
            with_payload=PayloadSelectorExclude(exclude=["metadata.parser"]),
            with_vectors=False,
        )

        assert included[0].payload == {"metadata": {"element_index": 17, "parser": "epub"}}
        assert excluded[0].payload == {
            "source_id": "book-1",
            "source_revision": 2,
            "metadata": {"element_index": 17},
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_brain_reads_are_passed_through_without_publication_pointer_lookup() -> None:
    client = await _client((_BRAIN,))
    try:
        await client.upsert(
            _BRAIN,
            points=[_point(7, "brain-source", 9, [1.0, 0.0])],
            wait=True,
        )
        pointers = _PointerReader({})
        wrapped = PublicationClient(client, pointers)

        result = await wrapped.query_points(_BRAIN, query=[1.0, 0.0], using="dense", limit=10)

        assert [point.id for point in result.points] == [7]
        assert pointers.calls == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_derived_dependency_advance_hides_legacy_and_stale_shared_graph() -> None:
    import hashlib

    derived = "vo_rag_books_graph_nodes_v1"
    source = 'book.with.dots-and-quote"'
    fingerprint = hashlib.sha256(source.encode()).hexdigest()
    client = await _client((derived,))
    try:
        await client.upsert(
            derived,
            points=[
                PointStruct(
                    id=10,
                    vector={"dense": [1.0, 0.0]},
                    payload={"source_id": "node-a", "book_ids": [source, "other"]},
                ),
                PointStruct(
                    id=11,
                    vector={"dense": [1.0, 0.0]},
                    payload={
                        "source_id": "node-b",
                        "book_ids": [source],
                        "dependency_revision_keys": {fingerprint: 1},
                    },
                ),
                PointStruct(
                    id=12,
                    vector={"dense": [0.9, 0.1]},
                    payload={
                        "source_id": "node-c",
                        "book_ids": [source],
                        "dependency_revision_keys": {fingerprint: 2},
                    },
                ),
                PointStruct(
                    id=13,
                    vector={"dense": [0.8, 0.2]},
                    payload={"source_id": "node-d", "book_ids": ["other"]},
                ),
            ],
        )

        async def pointers(_corpus: str, collection: str) -> dict[str, int | None]:
            return {source: 2} if collection == _BOOKS else {}

        wrapped = PublicationClient(client, pointers)
        result = await wrapped.query_points(derived, query=[1.0, 0.0], using="dense", limit=10)
        assert {point.id for point in result.points} == {12, 13}
        assert (await wrapped.count(derived, exact=True)).count == 2
        rows, _ = await wrapped.scroll(derived, limit=10)
        assert {row.id for row in rows} == {12, 13}
        rows = await wrapped.retrieve(derived, [10, 11, 12, 13], with_payload=False)
        assert {row.id for row in rows} == {12, 13}
        assert all(row.payload is None for row in rows)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_unknown_qdrant_methods_cannot_bypass_publication() -> None:
    client = await _client()
    try:
        wrapped = PublicationClient(client, _PointerReader({}))
        for method in (
            "search_matrix_offsets",
            "query_batch_points",
            "recommend_batch",
            "clear_payload",
            "delete_collection",
            "get_client",
        ):
            with pytest.raises(AttributeError, match="explicit source publication contract"):
                getattr(wrapped, method)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_logical_node_resolves_active_base_while_stale_physical_id_stays_hidden() -> None:
    from uuid import uuid4

    from vo_rag.core.storage.logical_points import retrieve_logical_points

    collection = "vo_rag_books_summary_v1"
    client = await _client((collection,))
    base, old, current = (str(uuid4()) for _ in range(3))
    pointers = _PointerReader({"book-1": 2})
    wrapped = PublicationClient(client, pointers)
    try:
        await client.upsert(
            collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector={"dense": [1.0, 0.0]},
                    payload={
                        "source_id": "book-1",
                        "source_revision": revision,
                        "publication_base_id": base,
                        "node_id": base,
                        "dependency_revision_keys": {hashlib.sha256(b"book-1").hexdigest(): 2},
                    },
                )
                for point_id, revision in ((old, 1), (current, 2))
            ],
        )
        assert await wrapped.retrieve(collection, ids=[old]) == []
        rows = await retrieve_logical_points(wrapped, collection=collection, ids=[base])
        assert [str(row.id) for row in rows] == [current]
        assert rows[0].payload["node_id"] == base
        rows = await wrapped.resolve_logical_points(collection, [base.replace("-", "")])
        assert [str(row.id) for row in rows] == [current]
        with pytest.raises(ValueError, match="duplicate UUID aliases"):
            await wrapped.resolve_logical_points(collection, [base, base.replace("-", "")])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_logical_node_refuses_two_visible_points_with_the_same_base_identity() -> None:
    from uuid import uuid4

    client = await _client()
    base = str(uuid4())
    wrapped = PublicationClient(client, _PointerReader({"book-1": 2}))
    try:
        await client.upsert(
            _BOOKS,
            points=[
                PointStruct(
                    id=str(uuid4()),
                    vector={"dense": [1.0, 0.0]},
                    payload={
                        "source_id": "book-1",
                        "source_revision": 2,
                        "publication_base_id": base,
                    },
                )
                for _ in range(2)
            ],
        )
        with pytest.raises(ValueError, match="ambiguous"):
            await wrapped.resolve_logical_points(_BOOKS, [base])
    finally:
        await client.close()
