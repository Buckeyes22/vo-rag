"""Integration tests for `vo_rag.core.corpus.boilerplate_ops`'s Qdrant
helpers (VOR-1427): `boilerplate_point_ids` and
`delete_boilerplate_points` -- the core logic behind `corpus boilerplate
--scan|--delete`. Imported from `core.corpus`, not `vo_rag.cli`, since the
CLI module is a thin wrapper over this one (mirrors `purge.py`/`health.py`).

Gated on live testinfra (``VORAG_TEST_QDRANT_URL`` -- ``make up`` starts
throwaway local containers, see the Makefile ``test-int`` target). Never
points at production -- this workstation is not part of the RAG stack.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from vo_rag.core.corpus import boilerplate_ops
from vo_rag.core.corpus.boilerplate_ops import boilerplate_point_ids, delete_boilerplate_points
from vo_rag.core.storage import VoPoint
from vo_rag.core.storage.qdrant import COLLECTION_NAMES, QdrantStore, setup_collections

pytestmark = pytest.mark.integration

_RAW = "vo_rag_books_raw_v1"
_DENSE = [0.01] * 1024
_COLBERT = [[0.01] * 1024]
_BLANK_PAGE = "This page intentionally left blank"


def _require_qdrant_url() -> str:
    url = os.environ.get("VORAG_TEST_QDRANT_URL")
    if not url:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail("VORAG_TEST_QDRANT_URL is required by the testinfra lane")
        pytest.skip("VORAG_TEST_QDRANT_URL not set -- skipping integration test")
    return url


@pytest.fixture(scope="module")
def qdrant_url() -> str:
    return _require_qdrant_url()


@pytest.fixture(scope="module", autouse=True)
async def _prepare_qdrant(qdrant_url: str) -> AsyncGenerator[None, None]:
    """Wipe and recreate vo-rag's declared collections once for this module."""
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=qdrant_url, api_key=None, prefer_grpc=False)

    async def _clean() -> None:
        existing = {item.name for item in (await client.get_collections()).collections}
        for name in COLLECTION_NAMES:
            if name in existing:
                await client.delete_collection(collection_name=name)

    try:
        await _clean()
        await setup_collections(client)
        yield
    finally:
        await _clean()
        await client.close()


@pytest.fixture
async def store(qdrant_url: str) -> AsyncGenerator[QdrantStore, None]:
    s = QdrantStore(url=qdrant_url, api_key=None, prefer_grpc=False)
    try:
        yield s
    finally:
        await s._client.close()  # noqa: SLF001


@pytest.fixture
def seeded_source_ids() -> list[str]:
    """Every test appends every `source_id` it seeds into `_RAW` here;
    `_cleanup_seeded` deletes exactly those points on teardown (mirroring
    `test_health_integration.py`'s isolation pattern -- collections are
    shared module-scoped, tests are not)."""
    return []


@pytest.fixture(autouse=True)
async def _cleanup_seeded(
    store: QdrantStore, seeded_source_ids: list[str]
) -> AsyncGenerator[None, None]:
    yield
    if not seeded_source_ids:
        return
    await store._client.delete(  # noqa: SLF001
        collection_name=_RAW,
        points_selector=Filter(
            must=[FieldCondition(key="source_id", match=MatchAny(any=seeded_source_ids))]
        ),
        wait=True,
    )


def _points(source_id: str, contents: list[str]) -> list[VoPoint]:
    return [
        VoPoint(
            source_id=f"{source_id}-{i}",
            chunk_index=i,
            stage="itest",
            dense=_DENSE,
            sparse={1: 0.5},
            bm25={2: 1.0},
            colbert=_COLBERT,
            payload={"project_id": "itest", "source_id": source_id, "content": content},
        )
        for i, content in enumerate(contents)
    ]


async def _count(store: QdrantStore, source_id: str, *, collection: str = _RAW) -> int:
    flt = Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))])
    result = await store._client.count(  # noqa: SLF001
        collection_name=collection,
        count_filter=flt,
        exact=True,
    )
    return int(result.count)


async def test_scan_finds_boilerplate_points_by_book_and_leaves_them_in_place(
    store: QdrantStore, seeded_source_ids: list[str]
) -> None:
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-bp-a-{marker}"
    source_b = f"itest-bp-b-{marker}"
    seeded_source_ids.extend([source_a, source_b])

    await store.upsert_vectors(
        _RAW,
        _points(
            source_a,
            [
                "This page intentionally left blank",
                "A perfectly ordinary paragraph about motorcycles.",
                "  this PAGE intentionally left BLANK.  ",
            ],
        ),
    )
    await store.upsert_vectors(
        _RAW,
        _points(source_b, ["This page intentionally left blank"]),
    )

    by_book = await boilerplate_point_ids(store, source_ids=[source_a, source_b])

    assert set(by_book.keys()) == {source_a, source_b}
    assert len(by_book[source_a]) == 2
    assert len(by_book[source_b]) == 1

    assert await _count(store, source_a) == 3  # scan never deletes
    assert await _count(store, source_b) == 1


async def test_scan_excludes_a_boilerplate_prefixed_real_chunk(
    store: QdrantStore, seeded_source_ids: list[str]
) -> None:
    """A chunk that merely starts with boilerplate text but
    carries real content afterward must never be reported as a match."""
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-bp-prefix-{marker}"
    seeded_source_ids.append(source_a)

    await store.upsert_vectors(
        _RAW,
        _points(
            source_a,
            [
                "This page intentionally left blank. Chapter 3 begins with load "
                "transfer under braking."
            ],
        ),
    )

    by_book = await boilerplate_point_ids(store, source_ids=[source_a])

    assert by_book == {}


async def test_delete_removes_only_the_matched_points(
    store: QdrantStore, seeded_source_ids: list[str]
) -> None:
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-bp-del-{marker}"
    seeded_source_ids.append(source_a)

    await store.upsert_vectors(
        _RAW,
        _points(
            source_a,
            [
                "This page intentionally left blank",
                "A perfectly ordinary paragraph about motorcycles.",
            ],
        ),
    )

    by_book = await boilerplate_point_ids(store, source_ids=[source_a])
    assert len(by_book[source_a]) == 1

    deleted = await delete_boilerplate_points(store, by_book[source_a])
    assert deleted == 1

    assert await _count(store, source_a) == 1  # only the real chunk survives

    remaining = await boilerplate_point_ids(store, source_ids=[source_a])
    assert remaining == {}


async def test_delete_batches_at_the_configured_batch_size(
    store: QdrantStore, monkeypatch: pytest.MonkeyPatch, seeded_source_ids: list[str]
) -> None:
    """Forces `delete_boilerplate_points`'s batching loop to run more than
    one batch (patching the batch size down rather than seeding 256+ real
    points, to keep this test fast) and confirms every id is still deleted."""
    monkeypatch.setattr(boilerplate_ops, "DELETE_BATCH", 2)

    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-bp-batch-{marker}"
    seeded_source_ids.append(source_a)

    await store.upsert_vectors(
        _RAW,
        _points(source_a, ["This page intentionally left blank"] * 5),
    )

    by_book = await boilerplate_point_ids(store, source_ids=[source_a])
    assert len(by_book[source_a]) == 5

    deleted = await delete_boilerplate_points(store, by_book[source_a])
    assert deleted == 5

    assert await _count(store, source_a) == 0


async def test_derived_collections_are_scanned_and_cleared_per_source_id(
    store: QdrantStore, seeded_source_ids: list[str]
) -> None:
    from vo_rag.core.corpus.boilerplate_ops import derived_boilerplate_point_ids

    sterile, contextualized = "vo_rag_books_sterile_v1", "vo_rag_books_contextualized_v1"
    marker = uuid.uuid4().hex[:8]
    source_a = f"itest-bp-derived-{marker}"
    seeded_source_ids.append(source_a)
    contents = [_BLANK_PAGE, "A perfectly ordinary paragraph about motorcycles."]
    await store.upsert_vectors(_RAW, _points(source_a, contents))
    await store.upsert_vectors(sterile, _points(source_a, contents))
    await store.upsert_vectors(contextualized, _points(source_a, contents))
    try:
        found = await derived_boilerplate_point_ids(store, source_ids=[source_a])
        assert {name: len(by_book[source_a]) for name, by_book in found.items() if by_book} == {
            sterile: 1,
            contextualized: 1,
        }
        for name, by_book in found.items():
            assert by_book is not None
            assert await delete_boilerplate_points(store, by_book[source_a], collection=name) == 1
        assert await derived_boilerplate_point_ids(store, source_ids=[source_a]) == {
            sterile: {},
            contextualized: {},
        }
        for name in (sterile, contextualized):
            assert await _count(store, source_a, collection=name) == 1  # real chunk survives
    finally:
        for name in (sterile, contextualized):
            await store._client.delete(  # noqa: SLF001
                collection_name=name,
                points_selector=Filter(
                    must=[FieldCondition(key="source_id", match=MatchValue(value=source_a))]
                ),
                wait=True,
            )
