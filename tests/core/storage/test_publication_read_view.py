"""A multi-call query keeps one view while independent queries see new revisions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from vo_rag.core.storage.publication_client import PublicationClient
from vo_rag.core.storage.publication_read_view import captured_pointers, publication_query
from vo_rag.core.storage.qdrant import QdrantStore

COLLECTION = "vo_rag_books_raw_v1"


class Pool:
    revision = 1
    fetches = 0

    async def fetch(self, _sql: str, corpora: list[str]) -> list[dict[str, Any]]:
        assert corpora == ["books", "reddit"]
        self.fetches += 1
        return [
            {
                "corpus": "books",
                "collection": COLLECTION,
                "source_id": "book",
                "active_revision": self.revision,
            }
        ]


@pytest.mark.asyncio
async def test_query_view_retains_previous_revision_and_resets_after_completion() -> None:
    pool = Pool()
    coordinator = SimpleNamespace(pool=pool, corpus="books", list_active=AsyncMock())
    store = object.__new__(QdrantStore)
    store._publication_enabled = True  # noqa: SLF001
    store._source_coordinator = AsyncMock(return_value=coordinator)  # type: ignore[method-assign]  # noqa: SLF001
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        COLLECTION, vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)}
    )
    await client.upsert(
        COLLECTION,
        points=[
            PointStruct(
                id=1,
                vector={"dense": [1.0, 0.0]},
                payload={"source_id": "book", "source_revision": 1},
            ),
            PointStruct(
                id=2,
                vector={"dense": [1.0, 0.0]},
                payload={"source_id": "book", "source_revision": 2},
            ),
        ],
    )
    wrapped = PublicationClient(client, store._publication_pointers)  # noqa: SLF001

    @publication_query
    async def query(*, retriever: Any) -> list[int]:
        first = await wrapped.query_points(COLLECTION, query=[1.0, 0.0], using="dense")
        pool.revision = 2
        second = await wrapped.query_points(COLLECTION, query=[1.0, 0.0], using="dense")
        assert captured_pointers(object(), "books", COLLECTION) is None
        return [first.points[0].id, second.points[0].id]

    try:
        assert await query(retriever=SimpleNamespace(qdrant=store)) == [1, 1]
        assert pool.fetches == 1
        coordinator.list_active.assert_not_awaited()
        assert captured_pointers(pool, "books", COLLECTION) is None
        assert await query(retriever=SimpleNamespace(qdrant=store)) == [2, 2]
        assert pool.fetches == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_exception_resets_the_read_view() -> None:
    pool = Pool()
    store = object.__new__(QdrantStore)
    store._publication_enabled = True  # noqa: SLF001
    store._source_coordinator = AsyncMock(return_value=SimpleNamespace(pool=pool))  # type: ignore[method-assign]  # noqa: SLF001

    @publication_query
    async def query(*, store: QdrantStore) -> None:
        assert captured_pointers(pool, "books", COLLECTION) == {"book": 1}
        raise RuntimeError("query fixture failure")

    with pytest.raises(RuntimeError, match="query fixture failure"):
        await query(store=store)
    assert captured_pointers(pool, "books", COLLECTION) is None


@pytest.mark.asyncio
async def test_overlapping_queries_keep_independent_pointer_snapshots() -> None:
    class ConcurrentPool:
        revision = 1
        first_captured = asyncio.Event()
        release_first = asyncio.Event()

        async def fetch(self, _sql: str, corpora: list[str]) -> list[dict[str, Any]]:
            assert corpora == ["books", "reddit"]
            snapshot = self.revision
            if snapshot == 1:
                self.first_captured.set()
                await self.release_first.wait()
            return [
                {
                    "corpus": "books",
                    "collection": COLLECTION,
                    "source_id": "book",
                    "active_revision": snapshot,
                }
            ]

    pool = ConcurrentPool()
    coordinator = SimpleNamespace(pool=pool, corpus="books")
    store = object.__new__(QdrantStore)
    store._publication_enabled = True  # noqa: SLF001
    store._source_coordinator = AsyncMock(return_value=coordinator)  # type: ignore[method-assign]  # noqa: SLF001
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        COLLECTION, vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)}
    )
    await client.upsert(
        COLLECTION,
        points=[
            PointStruct(
                id=1,
                vector={"dense": [1.0, 0.0]},
                payload={"source_id": "book", "source_revision": 1},
            ),
            PointStruct(
                id=2,
                vector={"dense": [1.0, 0.0]},
                payload={"source_id": "book", "source_revision": 2},
            ),
        ],
    )
    wrapped = PublicationClient(client, store._publication_pointers)  # noqa: SLF001

    @publication_query
    async def query(*, retriever: Any) -> int:
        result = await wrapped.query_points(COLLECTION, query=[1.0, 0.0], using="dense")
        return int(result.points[0].id)

    first: asyncio.Future[int] = asyncio.ensure_future(
        query(retriever=SimpleNamespace(qdrant=store))
    )
    try:
        await asyncio.wait_for(pool.first_captured.wait(), timeout=2)
        pool.revision = 2
        second: asyncio.Future[int] = asyncio.ensure_future(
            query(retriever=SimpleNamespace(qdrant=store))
        )
        assert await asyncio.wait_for(second, timeout=2) == 2
        pool.release_first.set()
        assert await asyncio.wait_for(first, timeout=2) == 1
    finally:
        pool.release_first.set()
        await asyncio.gather(first, return_exceptions=True)
        await client.close()
