"""Exercise source-scoped mutation planning and copy-on-write on local Qdrant."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointsList,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    UpdateStatus,
    VectorParams,
)

from vo_rag.core.ingest.source_publication import (
    ActivePublication,
    PublicationState,
    SourceFence,
    SourcePublicationJob,
    StalePublicationError,
)
from vo_rag.core.storage.source_mutations import (
    SourceMutationBatchError,
    SourceMutationWriter,
)


class _MemoryCoordinator:
    """Small pointer/checkpoint fake; all point data lives in real local Qdrant."""

    corpus = "books"

    def __init__(self) -> None:
        self.active: dict[tuple[str, str], ActivePublication] = {}
        self.jobs: dict[str, SourcePublicationJob] = {}

    @asynccontextmanager
    async def fence(
        self,
        source_id: str,
        *,
        job_id: str | None = None,
        dependency_sources: Sequence[str] = (),
    ) -> AsyncGenerator[SourceFence, None]:
        yield SourceFence(
            None,
            self.corpus,
            source_id,
            job_id,
            frozenset({source_id, *dependency_sources}),
        )

    async def assert_legacy_write_allowed(
        self, fence: SourceFence, source_id: str, dependency_sources: Sequence[str] = ()
    ) -> None:
        required = {source_id, *dependency_sources}
        if not required.issubset(fence.locked_sources):
            raise ValueError("legacy write fence does not cover every source dependency")
        if any(
            pointer.revision is not None
            for (_, candidate), pointer in self.active.items()
            if candidate in required
        ):
            raise ValueError("legacy write refused for an active revision")
        if any(
            job.source_id in required and job.state is not PublicationState.PRESERVED_NO_WRITE
            for job in self.jobs.values()
        ):
            raise ValueError("legacy write refused for pending publication job")

    async def begin(
        self,
        _fence: SourceFence,
        collection: str,
        source_id: str,
        job_id: str,
        source_sha256: str,
        expected_active_revision: int | None,
        recipe: Mapping[str, Any],
    ) -> SourcePublicationJob:
        existing = self.jobs.get(job_id)
        if existing is not None:
            return existing
        pointer = self.active.get((collection, source_id))
        actual = pointer.revision if pointer else None
        if actual != expected_active_revision:
            raise StalePublicationError("stale in-memory pointer")
        previous_revisions = [
            job.revision for job in self.jobs.values() if job.source_id == source_id
        ]
        job = SourcePublicationJob(
            job_id=job_id,
            corpus=self.corpus,
            collection=collection,
            source_id=source_id,
            revision=max(previous_revisions, default=0) + 1,
            expected_active_revision=expected_active_revision,
            source_sha256=source_sha256,
            state=PublicationState.PREPARING,
            checkpoint={},
            retention_disposition=None,
        )
        self.jobs[job_id] = job
        return job

    async def checkpoint(
        self, _fence: SourceFence, job_id: str, state: PublicationState, data: Mapping[str, Any]
    ) -> SourcePublicationJob:
        job = self.jobs[job_id]
        updated = replace(job, state=state, checkpoint=dict(data))
        self.jobs[job_id] = updated
        return updated

    async def publish(self, _fence: SourceFence, job_id: str) -> ActivePublication:
        job = self.jobs[job_id]
        pointer = ActivePublication(
            self.corpus,
            job.collection,
            job.source_id,
            job.revision,
            job.source_sha256,
            job.job_id,
        )
        self.active[(job.collection, job.source_id)] = pointer
        self.jobs[job_id] = replace(job, state=PublicationState.PUBLISHED)
        return pointer

    async def read_active(self, collection: str, source_id: str) -> ActivePublication | None:
        return self.active.get((collection, source_id))

    async def list_active(self, collection: str) -> dict[str, int | None]:
        return {
            source_id: pointer.revision
            for (name, source_id), pointer in self.active.items()
            if name == collection
        }


@pytest.fixture
async def qdrant() -> AsyncGenerator[tuple[AsyncQdrantClient, str], None]:
    client = AsyncQdrantClient(location=":memory:")
    collection = "mutation_" + uuid4().hex
    await client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    try:
        yield client, collection
    finally:
        await client.close()


@pytest.fixture
async def raw_books_qdrant() -> AsyncGenerator[tuple[AsyncQdrantClient, str], None]:
    """Use the canonical raw collection name to exercise dependency scope."""
    client = AsyncQdrantClient(location=":memory:")
    collection = "vo_rag_books_raw_v1"
    await client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    try:
        yield client, collection
    finally:
        await client.close()


def _points(source_id: str, count: int = 3) -> list[PointStruct]:
    return [
        PointStruct(
            id=_point_id(source_id, index),
            vector={
                "dense": [1.0, float(index + 1)],
                "sparse": SparseVector(indices=[1], values=[0.5]),
            },
            payload={"source_id": source_id, "chunk_index": index, "content": f"passage {index}"},
        )
        for index in range(count)
    ]


def _point_id(source_id: str, index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"vo-rag-test:{source_id}:{index}"))


def _register_active_revision(
    coordinator: _MemoryCoordinator, collection: str, source_id: str, revision: int
) -> None:
    """Register a published fake pointer and job matching seeded physical points."""
    job_id = f"seed-{collection}-{source_id}"
    source_sha256 = "a" * 64
    coordinator.active[(collection, source_id)] = ActivePublication(
        coordinator.corpus, collection, source_id, revision, source_sha256, job_id
    )
    coordinator.jobs[job_id] = SourcePublicationJob(
        job_id=job_id,
        corpus=coordinator.corpus,
        collection=collection,
        source_id=source_id,
        revision=revision,
        expected_active_revision=None,
        source_sha256=source_sha256,
        state=PublicationState.PUBLISHED,
        checkpoint={},
        retention_disposition="retained_old_revision",
    )


async def _seed(client: AsyncQdrantClient, collection: str, source_id: str) -> None:
    await client.upsert(collection_name=collection, points=_points(source_id), wait=True)


async def _active_rows(
    client: AsyncQdrantClient, collection: str, source_id: str, revision: int
) -> list[Any]:
    from vo_rag.core.storage.publication_filter import publication_filter

    rows, _ = await client.scroll(
        collection_name=collection,
        scroll_filter=publication_filter({source_id: revision}),
        limit=100,
        with_payload=True,
        with_vectors=True,
    )
    return [row for row in rows if (row.payload or {}).get("source_id") == source_id]


@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("append_matching_point", [True, False])
async def test_selection_drift_before_fence_preserves_current_source_without_mutation(
    qdrant: tuple[AsyncQdrantClient, str], legacy: bool, append_matching_point: bool
) -> None:
    client, collection = qdrant
    source_id = "vo-rag-corpus-sha256:" + "c" * 64
    await _seed(client, collection, source_id)

    class RacingCoordinator(_MemoryCoordinator):
        injected = False

        @asynccontextmanager
        async def fence(
            self,
            source_id: str,
            *,
            job_id: str | None = None,
            dependency_sources: Sequence[str] = (),
        ) -> AsyncGenerator[SourceFence, None]:
            if not self.injected:
                self.injected = True
                if append_matching_point:
                    replacement = _points(source_id, 4)[-1]
                else:
                    replacement = _points(source_id)[0]
                    replacement.payload = {
                        **(replacement.payload or {}),
                        "content": "new content committed before maintenance lock",
                    }
                await client.upsert(collection_name=collection, points=[replacement], wait=True)
            async with super().fence(
                source_id, job_id=job_id, dependency_sources=dependency_sources
            ) as acquired:
                yield acquired

    coordinator = RacingCoordinator()
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]
    selector: Any = [_point_id(source_id, 0)]
    if append_matching_point:
        selector = Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))])
    mutation = writer.mutate_legacy if legacy else writer.mutate
    with pytest.raises(SourceMutationBatchError) as error:
        await mutation("set_payload", collection, {"points": selector, "payload": {"review": True}})
    assert error.value.failed == {source_id: "StalePublicationError"}
    assert await coordinator.list_active(collection) == {}
    assert coordinator.jobs == {}
    rows, _ = await client.scroll(collection, limit=100, with_payload=True)
    assert len(rows) == (4 if append_matching_point else 3)
    assert all("review" not in (row.payload or {}) for row in rows)
    if not append_matching_point:
        current = next(row for row in rows if str(row.id) == _point_id(source_id, 0))
        assert (current.payload or {})["content"] == "new content committed before maintenance lock"


async def test_reddit_legacy_writer_has_no_books_dependency_requirement(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    coordinator = _MemoryCoordinator()
    coordinator.corpus = "reddit"
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]
    result = await writer.mutate_legacy("upsert", collection, {"points": _points("reddit-thread")})
    assert result.status is UpdateStatus.COMPLETED
    assert (await client.count(collection, exact=True)).count == 3
    assert coordinator.jobs == {}


async def test_payload_patch_publishes_new_revision_and_retains_old_points(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    source_id = "book-a"
    await _seed(client, collection, source_id)
    coordinator = _MemoryCoordinator()
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]

    result = await writer.mutate(
        "set_payload",
        collection,
        {"points": [_point_id(source_id, 1)], "payload": {"quality": "reviewed"}},
    )

    assert result.status is UpdateStatus.COMPLETED
    assert await coordinator.list_active(collection) == {source_id: 1}
    visible = await _active_rows(client, collection, source_id, 1)
    assert len(visible) == 3
    patched = next(row for row in visible if row.payload["chunk_index"] == 1)
    assert patched.payload["quality"] == "reviewed"
    assert patched.payload["content"] == "passage 1"
    all_rows, _ = await client.scroll(collection, limit=100, with_payload=True)
    assert len([row for row in all_rows if (row.payload or {}).get("source_id") == source_id]) == 6


@pytest.mark.parametrize("selector_kind", ["filter", "ids"])
async def test_raw_quality_mutation_is_eligible_without_self_dependency(
    raw_books_qdrant: tuple[AsyncQdrantClient, str], selector_kind: str
) -> None:
    client, collection = raw_books_qdrant
    source_id = "raw-quality-source"
    other_source = "raw-quality-unrelated"
    seeded = [
        point.model_copy(
            update={
                "vector": {
                    **cast(dict[str, Any], point.vector),
                    "dense": [1.0, 0.0],
                },
                "payload": {
                    **(point.payload or {}),
                    "source_revision": 1,
                    "publication_base_id": str(point.id),
                    "trust_level": "trusted",
                    "tf_event_family": ["relay"],
                },
            }
        )
        for point in _points(source_id)
    ]
    outside = _points(other_source, 1)[0]
    outside = outside.model_copy(
        update={"vector": {**cast(dict[str, Any], outside.vector), "dense": [1.0, 0.0]}}
    )
    await client.upsert(collection_name=collection, points=[*seeded, outside], wait=True)
    before_rows, _ = await client.scroll(
        collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
        ),
        limit=100,
        with_payload=True,
        with_vectors=True,
    )
    before_by_id = {str(row.id): row for row in before_rows}
    coordinator = _MemoryCoordinator()
    _register_active_revision(coordinator, collection, source_id, 1)
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]
    if selector_kind == "filter":
        selector: Any = Filter(
            must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
        )
    else:
        selector = [str(seeded[1].id)]

    warnings = [{"stage": "book_quality", "code": "low_text_density"}]
    result = await writer.mutate(
        "set_payload", collection, {"points": selector, "payload": {"quality_warnings": warnings}}
    )

    assert result.status is UpdateStatus.COMPLETED
    assert await coordinator.list_active(collection) == {source_id: 2}
    physical, offset = await client.scroll(
        collection, limit=100, with_payload=True, with_vectors=True
    )
    assert offset is None
    source_points = [row for row in physical if (row.payload or {}).get("source_id") == source_id]
    assert len(source_points) == 6  # every old physical point is retained
    assert sum((row.payload or {}).get("source_revision") == 1 for row in source_points) == 3
    current = [row for row in source_points if (row.payload or {}).get("source_revision") == 2]
    assert len(current) == 3
    changed = [row for row in current if (row.payload or {}).get("chunk_index") == 1]
    assert len(changed) == 1
    changed_payload = changed[0].payload or {}
    assert changed_payload["quality_warnings"] == warnings
    assert changed_payload["content"] == "passage 1"
    assert changed_payload["tf_event_family"] == ["relay"]
    assert changed[0].vector == before_by_id[str(seeded[1].id)].vector
    assert all(
        "dependency_revisions" not in (row.payload or {})
        and "dependency_revision_keys" not in (row.payload or {})
        for row in current
    )
    old_after = {
        str(row.id): row for row in source_points if (row.payload or {}).get("source_revision") == 1
    }
    assert set(old_after) == set(before_by_id)
    for point_id, row in old_after.items():
        assert row.payload == before_by_id[point_id].payload
        assert row.vector == before_by_id[point_id].vector
    outside_after = next(
        row for row in physical if (row.payload or {}).get("source_id") == other_source
    )
    assert outside_after.payload == outside.payload
    assert outside_after.vector == outside.vector


async def test_raw_id_selector_never_reads_book_dependency_pointers(
    raw_books_qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = raw_books_qdrant
    source_id = "raw-id-selector"
    points = [
        point.model_copy(
            update={
                "payload": {
                    **(point.payload or {}),
                    "source_revision": 1,
                    "publication_base_id": str(point.id),
                }
            }
        )
        for point in _points(source_id)
    ]
    await client.upsert(collection_name=collection, points=points, wait=True)

    class NoBooksDependencyCoordinator(_MemoryCoordinator):
        async def list_active(self, name: str) -> dict[str, int | None]:
            if name == "vo_rag_books_raw_v1":
                raise AssertionError("raw source selection must not query dependency pointers")
            return await super().list_active(name)

    coordinator = NoBooksDependencyCoordinator()
    _register_active_revision(coordinator, collection, source_id, 1)
    rows, failures = await SourceMutationWriter(
        client,
        coordinator,  # type: ignore[arg-type]
    )._select(collection, [str(points[0].id)])
    assert failures == {}
    assert [str(row.id) for row in rows] == [str(points[0].id)]


async def test_derived_books_id_mutation_rejects_stale_raw_dependency(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    import hashlib

    client, collection = qdrant
    raw_source = "raw-dependency-source"
    derived_source = "derived-summary-source"
    stale_point = _points(derived_source, 1)[0].model_copy(
        update={
            "vector": {"dense": [1.0, 0.0], "sparse": SparseVector(indices=[1], values=[0.5])},
            "payload": {
                "source_id": derived_source,
                "book_id": raw_source,
                "source_revision": 1,
                "dependency_revisions": {raw_source: 1},
                "dependency_revision_keys": {hashlib.sha256(raw_source.encode()).hexdigest(): 1},
                "content": "summary derived from raw revision one",
            },
        }
    )
    await client.upsert(collection_name=collection, points=[stale_point], wait=True)
    coordinator = _MemoryCoordinator()
    _register_active_revision(coordinator, collection, derived_source, 1)
    _register_active_revision(coordinator, "vo_rag_books_raw_v1", raw_source, 2)

    with pytest.raises(SourceMutationBatchError) as caught:
        await SourceMutationWriter(client, coordinator).mutate(  # type: ignore[arg-type]
            "set_payload",
            collection,
            {"points": [str(stale_point.id)], "payload": {"quality_warnings": []}},
        )

    assert caught.value.failed == {derived_source: "StalePublicationError"}
    assert await coordinator.list_active(collection) == {derived_source: 1}
    rows, _ = await client.scroll(collection, limit=100, with_payload=True, with_vectors=True)
    assert len(rows) == 1
    assert (rows[0].payload or {}) == (stale_point.payload or {})
    assert rows[0].vector == stale_point.vector


async def test_reddit_mutation_does_not_read_books_raw_pointers(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    source_id = "reddit-thread-source"
    await _seed(client, collection, source_id)

    class RedditCoordinator(_MemoryCoordinator):
        corpus = "reddit"

        async def list_active(self, name: str) -> dict[str, int | None]:
            if name == "vo_rag_books_raw_v1":
                raise AssertionError("Reddit mutations must not require Books raw pointers")
            return await super().list_active(name)

    coordinator = RedditCoordinator()
    result = await SourceMutationWriter(client, coordinator).mutate(  # type: ignore[arg-type]
        "set_payload",
        collection,
        {"points": [_point_id(source_id, 0)], "payload": {"reviewed": True}},
    )

    assert result.status is UpdateStatus.COMPLETED
    assert await coordinator.list_active(collection) == {source_id: 1}
    assert coordinator.jobs


async def test_tag_patch_requires_matching_source_and_current_content_receipt(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    from vo_rag.core.corpus.tf_schema import content_sha256

    client, collection = qdrant
    source_id = "book-tag-bound"
    await _seed(client, collection, source_id)
    coordinator = _MemoryCoordinator()
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]

    await writer.mutate(
        "set_payload",
        collection,
        {
            "points": [_point_id(source_id, 1)],
            "payload": {
                "tf_tag_source_id": source_id,
                "tf_tag_content_sha256": content_sha256("passage 1"),
                "tf_event_family": ["relay"],
            },
        },
    )
    visible = await _active_rows(client, collection, source_id, 1)
    tagged = next(row for row in visible if row.payload["chunk_index"] == 1)
    assert tagged.payload["tf_event_family"] == ["relay"]


async def test_tag_patch_rejects_wrong_content_receipt_before_pointer_switch(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    source_id = "book-tag-wrong-content"
    await _seed(client, collection, source_id)
    coordinator = _MemoryCoordinator()
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]

    with pytest.raises(SourceMutationBatchError):
        await writer.mutate(
            "set_payload",
            collection,
            {
                "points": [_point_id(source_id, 1)],
                "payload": {
                    "tf_tag_source_id": source_id,
                    "tf_tag_content_sha256": "0" * 64,
                    "tf_event_family": ["relay"],
                },
            },
        )

    assert await coordinator.list_active(collection) == {}


async def test_maintenance_cannot_promote_trust_level(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    source_id = "book-trust-immutable"
    await client.upsert(
        collection_name=collection,
        points=[
            point.model_copy(
                update={"payload": {**(point.payload or {}), "trust_level": "quarantined"}}
            )
            for point in _points(source_id)
        ],
        wait=True,
    )
    writer = SourceMutationWriter(client, _MemoryCoordinator())  # type: ignore[arg-type]

    with pytest.raises(SourceMutationBatchError):
        await writer.mutate(
            "set_payload",
            collection,
            {"points": [_point_id(source_id, 0)], "payload": {"trust_level": "trusted"}},
        )


async def test_legacy_maintenance_cannot_promote_trust_level(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    source_id = "book-legacy-trust-immutable"
    points = [
        point.model_copy(
            update={"payload": {**(point.payload or {}), "trust_level": "quarantined"}}
        )
        for point in _points(source_id)
    ]
    await client.upsert(collection_name=collection, points=points, wait=True)
    writer = SourceMutationWriter(client, _MemoryCoordinator())  # type: ignore[arg-type]

    with pytest.raises(SourceMutationBatchError):
        await writer.mutate_legacy(
            "set_payload",
            collection,
            {"points": [points[0].id], "payload": {"trust_level": "trusted"}},
        )


async def test_partial_delete_creates_revision_and_hides_deleted_tail(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    source_id = "book-delete-one"
    await _seed(client, collection, source_id)
    coordinator = _MemoryCoordinator()
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]

    await writer.mutate("delete", collection, {"points_selector": [_point_id(source_id, 1)]})

    assert await coordinator.list_active(collection) == {source_id: 1}
    visible = await _active_rows(client, collection, source_id, 1)
    assert {row.payload["chunk_index"] for row in visible} == {0, 2}
    all_rows, _ = await client.scroll(collection, limit=100, with_payload=True)
    assert len([row for row in all_rows if (row.payload or {}).get("source_id") == source_id]) == 5


async def test_full_delete_is_tombstone_and_never_physically_removes_old_points(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    source_id = "book-tombstone"
    await _seed(client, collection, source_id)
    coordinator = _MemoryCoordinator()
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]

    await writer.mutate(
        "delete",
        collection,
        {"points_selector": [_point_id(source_id, index) for index in range(3)]},
    )

    assert await coordinator.list_active(collection) == {source_id: 1}
    assert await _active_rows(client, collection, source_id, 1) == []
    all_rows, _ = await client.scroll(collection, limit=100, with_payload=True)
    assert len([row for row in all_rows if (row.payload or {}).get("source_id") == source_id]) == 3
    job = next(iter(coordinator.jobs.values()))
    verification = job.checkpoint["verification"]
    assert verification["empty_source_tombstone"] is True
    assert set(verification["tombstone_before_payloads"]) == {
        _point_id(source_id, index) for index in range(3)
    }


async def test_stale_id_selector_is_rejected_and_does_not_retarget_current_revision(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    source_id = "book-stale-selector"
    await _seed(client, collection, source_id)
    coordinator = _MemoryCoordinator()
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]
    await writer.mutate(
        "set_payload", collection, {"points": [_point_id(source_id, 0)], "payload": {"flag": True}}
    )

    with pytest.raises(SourceMutationBatchError) as caught:
        await writer.mutate(
            "set_payload",
            collection,
            {"points": [_point_id(source_id, 0)], "payload": {"flag": False}},
        )
    assert caught.value.failed == {source_id: "StalePublicationError"}
    assert await coordinator.list_active(collection) == {source_id: 1}
    current = await _active_rows(client, collection, source_id, 1)
    assert next(row for row in current if row.payload["chunk_index"] == 0).payload["flag"] is True


async def test_source_upsert_replaces_matching_logical_id_without_changing_content(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    source_id = "book-upsert"
    await _seed(client, collection, source_id)
    coordinator = _MemoryCoordinator()
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]
    replacement = _points(source_id, 1)[0].model_copy(
        update={
            "payload": {
                "source_id": source_id,
                "chunk_index": 0,
                "content": "passage 0",
                "quality": "producer-reviewed",
            }
        }
    )

    await writer.mutate("upsert", collection, {"points": [replacement]})

    visible = await _active_rows(client, collection, source_id, 1)
    assert len(visible) == 3
    replaced = next(row for row in visible if row.payload["chunk_index"] == 0)
    assert replaced.payload["content"] == "passage 0"
    assert replaced.payload["quality"] == "producer-reviewed"


async def test_legacy_compatibility_write_preserves_ids_when_source_is_unmanaged(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    source_id = "vo-rag-corpus-sha256:" + "a" * 64
    await _seed(client, collection, source_id)
    writer = SourceMutationWriter(client, _MemoryCoordinator())  # type: ignore[arg-type]

    await writer.mutate_legacy(
        "set_payload",
        collection,
        {"points": [_point_id(source_id, 1)], "payload": {"reviewed": True}},
    )

    rows, _ = await client.scroll(collection, limit=100, with_payload=True)
    assert len(rows) == 3
    patched = next(row for row in rows if str(row.id) == _point_id(source_id, 1))
    assert (patched.payload or {})["reviewed"] is True
    assert "source_revision" not in (patched.payload or {})


async def test_legacy_upsert_batches_eight_points_under_one_source_fence(
    qdrant: tuple[AsyncQdrantClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, collection = qdrant
    source_id = "vo-rag-corpus-sha256:" + "c" * 64
    coordinator = _MemoryCoordinator()
    fence_count = 0
    original_fence = coordinator.fence

    @asynccontextmanager
    async def counted_fence(*args: Any, **kwargs: Any) -> AsyncGenerator[SourceFence, None]:
        nonlocal fence_count
        fence_count += 1
        async with original_fence(*args, **kwargs) as fence:
            yield fence

    monkeypatch.setattr(coordinator, "fence", counted_fence)
    original_upsert = client.upsert
    batch_sizes: list[int] = []

    async def recording_upsert(*, collection_name: str, points: Any, wait: bool) -> Any:
        batch_sizes.append(len(points))
        return await original_upsert(collection_name=collection_name, points=points, wait=wait)

    monkeypatch.setattr(client, "upsert", recording_upsert)
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]

    await writer.mutate_legacy(
        "upsert", collection, {"points": _points(source_id, 17), "wait": True}
    )

    assert batch_sizes == [8, 8, 1]
    assert fence_count == 1


async def test_legacy_upsert_retries_and_reports_a_failed_batch(
    qdrant: tuple[AsyncQdrantClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, collection = qdrant
    source_id = "vo-rag-corpus-sha256:" + "d" * 64
    coordinator = _MemoryCoordinator()
    fence_count = 0
    original_fence = coordinator.fence

    @asynccontextmanager
    async def counted_fence(*args: Any, **kwargs: Any) -> AsyncGenerator[SourceFence, None]:
        nonlocal fence_count
        fence_count += 1
        async with original_fence(*args, **kwargs) as fence:
            yield fence

    monkeypatch.setattr(coordinator, "fence", counted_fence)
    original_upsert = client.upsert
    batch_sizes: list[int] = []

    async def failing_second_batch(*, collection_name: str, points: Any, wait: bool) -> Any:
        batch_sizes.append(len(points))
        if len(batch_sizes) > 1:
            raise RuntimeError("simulated second-batch failure")
        return await original_upsert(collection_name=collection_name, points=points, wait=wait)

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(client, "upsert", failing_second_batch)
    monkeypatch.setattr("vo_rag.core.storage.source_mutations.asyncio.sleep", no_sleep)
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]

    with pytest.raises(SourceMutationBatchError) as caught:
        await writer.mutate_legacy(
            "upsert", collection, {"points": _points(source_id, 17), "wait": True}
        )

    assert batch_sizes == [8, 8, 8, 8]
    assert fence_count == 1
    assert caught.value.failed[source_id] == "RuntimeError"


async def test_legacy_compatibility_write_refuses_managed_source_without_retargeting(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    source_id = "vo-rag-corpus-sha256:" + "b" * 64
    await _seed(client, collection, source_id)
    coordinator = _MemoryCoordinator()
    writer = SourceMutationWriter(client, coordinator)  # type: ignore[arg-type]
    await writer.mutate(
        "set_payload",
        collection,
        {"points": [_point_id(source_id, 0)], "payload": {"revisioned": True}},
    )

    with pytest.raises(SourceMutationBatchError) as caught:
        await writer.mutate_legacy(
            "set_payload",
            collection,
            {"points": [_point_id(source_id, 0)], "payload": {"revisioned": False}},
        )
    assert caught.value.failed[source_id] == "ValueError"

    visible = await _active_rows(client, collection, source_id, 1)
    assert (
        next(row for row in visible if row.payload["chunk_index"] == 0).payload["revisioned"]
        is True
    )


async def test_legacy_compatibility_rejects_graph_without_complete_book_closure(
    qdrant: tuple[AsyncQdrantClient, str],
) -> None:
    client, collection = qdrant
    point = _points("node-id", 1)[0].model_copy(
        update={"payload": {"source_id": "node-id", "label": "Unknown book graph"}}
    )
    writer = SourceMutationWriter(client, _MemoryCoordinator())  # type: ignore[arg-type]

    with pytest.raises(SourceMutationBatchError) as caught:
        await writer.mutate_legacy("upsert", "vo_rag_books_graph_nodes_v1", {"points": [point]})
    assert caught.value.failed["node-id"] == "ValueError"


def test_protected_content_and_identity_fields_cannot_be_changed_by_metadata_patches() -> None:
    point = _points("book-protected", 1)[0]
    with pytest.raises(ValueError, match="identity"):
        SourceMutationWriter._mutate_payload(point, "set_payload", {"source_id": "other-source"})
    with pytest.raises(ValueError, match="identity"):
        SourceMutationWriter._mutate_payload(point, "set_payload", {"content": "rewritten"})


def test_filter_selector_scope_is_explicit_and_does_not_use_unbounded_page_truncation() -> None:
    filter_value = Filter(
        must=[FieldCondition(key="source_id", match=MatchValue(value="book-scoped"))]
    )
    actions = SourceMutationWriter._actions("delete", {"points_selector": filter_value})
    assert actions == [("delete", filter_value, None)]


def test_upsert_rejects_duplicate_logical_ids() -> None:
    point = _points("book-duplicate", 1)[0]
    with pytest.raises(ValueError, match="duplicate point IDs"):
        SourceMutationWriter._actions("upsert", {"points": [point, point]})


def test_update_vectors_rejects_malformed_entries_instead_of_succeeding_empty() -> None:
    with pytest.raises(ValueError, match="bound producer/source/input recipe"):
        SourceMutationWriter._actions("update_vectors", {"points": [object()]})


def test_vector_replacement_cannot_drop_or_add_configured_modalities() -> None:
    point = _points("book-modalities", 1)[0]
    with pytest.raises(ValueError, match="bound producer/source/input recipe"):
        SourceMutationWriter._update_vectors(point, {"dense": [0.2, 0.3]})
    with pytest.raises(ValueError, match="delete_vectors is disabled"):
        SourceMutationWriter._delete_vectors(point, ["sparse"])


def test_batch_rejects_mixed_upsert_and_mutation_ordering() -> None:
    from qdrant_client.models import DeleteOperation, UpsertOperation

    operations = [
        UpsertOperation(upsert=PointsList(points=_points("book-batch"))),
        DeleteOperation(delete=PointIdsList(points=[_point_id("book-batch", 0)])),
    ]
    with pytest.raises(ValueError, match="cannot be mixed"):
        SourceMutationWriter._actions("batch_update_points", {"update_operations": operations})
