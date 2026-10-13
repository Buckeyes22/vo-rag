"""Real-Postgres regressions for QdrantStore admission retries and dependencies."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from qdrant_client.models import FieldCondition, IsEmptyCondition, PointStruct, Record

from tests.core.db.test_source_publication_integration import (
    _cleanup,
    _verified_checkpoint,
)
from tests.core.db.test_source_publication_integration import pool as pool  # noqa: F401
from vo_rag.core.ingest.source_publication import PublicationState, SourceCoordinator
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter

pytestmark = pytest.mark.integration


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def _cleanup_sources(pool: Any, corpus: str, source_ids: list[str]) -> None:
    """Delete only test-owned source rows when using the canonical Books corpus."""
    async with pool.acquire() as connection:
        await connection.execute(
            "DELETE FROM source_publication_jobs WHERE corpus = $1 AND source_id = ANY($2::text[])",
            corpus,
            source_ids,
        )
        await connection.execute(
            "DELETE FROM source_publications WHERE corpus = $1 AND source_id = ANY($2::text[])",
            corpus,
            source_ids,
        )
        await connection.execute(
            "DELETE FROM source_publication_sequences "
            "WHERE corpus = $1 AND source_id = ANY($2::text[])",
            corpus,
            source_ids,
        )


def _store_with_coordinator(
    coordinator: SourceCoordinator, monkeypatch: pytest.MonkeyPatch
) -> QdrantStore:
    store = QdrantStore.__new__(QdrantStore)
    store._publication_enabled = True  # noqa: SLF001
    store._publication_writes_enabled = True  # noqa: SLF001

    async def coordinator_for(_collection: str) -> SourceCoordinator:
        return coordinator

    monkeypatch.setattr(store, "_source_coordinator", coordinator_for)
    return store


class _PausingQdrant:
    def __init__(self) -> None:
        self.points: dict[str, PointStruct] = {}
        self.upsert_entered = asyncio.Event()
        self.release_upsert = asyncio.Event()

    @staticmethod
    def _matches(point: PointStruct, selected: Any) -> bool:
        payload = point.payload or {}
        for condition in getattr(selected, "must", []) or []:
            if isinstance(condition, FieldCondition):
                key = condition.key
                match = condition.match
                expected = getattr(match, "value", None)
                if payload.get(key) != expected:
                    return False
            elif isinstance(condition, IsEmptyCondition):
                key = condition.is_empty.key
                if payload.get(key) not in (None, [], {}):
                    return False
        return True

    async def count(self, *, count_filter: Any, **_: Any) -> Any:
        return SimpleNamespace(
            count=sum(self._matches(point, count_filter) for point in self.points.values())
        )

    async def scroll(self, *, scroll_filter: Any, **_: Any) -> tuple[list[Record], None]:
        rows: list[Record] = []
        for point in self.points.values():
            if not self._matches(point, scroll_filter):
                continue
            if point.vector is None:
                raise AssertionError("the test Qdrant received a point without vectors")
            rows.append(
                Record.model_construct(
                    id=point.id,
                    payload=point.payload,
                    vector=cast(Any, point.vector),
                )
            )
        return rows, None

    async def upsert(self, *, points: list[PointStruct], **_: Any) -> Any:
        self.upsert_entered.set()
        await self.release_upsert.wait()
        self.points.update({str(point.id): point for point in points})
        return SimpleNamespace(status="completed")


@pytest.mark.asyncio
async def test_qdrant_admission_retries_published_job_with_original_expected_pointer(
    pool: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = f"itest-admit-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    store = _store_with_coordinator(coordinator, monkeypatch)
    source_id = "canonical-source"
    collection = "vo_rag_books_raw_v1"
    job_id = "published-job"
    source_sha = _digest("immutable source")
    try:
        admitted = await store.admit_source_revision(collection, source_id, job_id, source_sha)
        assert admitted is not None
        assert admitted.expected_active_revision is None

        async with coordinator.fence(source_id, job_id=job_id) as fence:
            await coordinator.checkpoint(fence, job_id, PublicationState.STAGED, {"staged": True})
            await coordinator.checkpoint(
                fence, job_id, PublicationState.VERIFIED, _verified_checkpoint()
            )
            published = await coordinator.publish(fence, job_id)
        assert published.revision == admitted.revision

        replay = await store.admit_source_revision(collection, source_id, job_id, source_sha)
        assert replay is not None
        assert replay.state is PublicationState.PUBLISHED
        assert replay.revision == admitted.revision
        assert replay.expected_active_revision is None
    finally:
        await _cleanup(pool, corpus)


@pytest.mark.asyncio
async def test_qdrant_admission_reuses_failed_no_write_job_for_same_source_envelope(
    pool: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = f"itest-admit-retry-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    store = _store_with_coordinator(coordinator, monkeypatch)
    source_id = "canonical-source"
    collection = "vo_rag_books_raw_v1"
    job_id = "retryable-job"
    source_sha = _digest("same verified source")
    try:
        first = await store.admit_source_revision(collection, source_id, job_id, source_sha)
        assert first is not None
        await store.abort_source_revision(first, "provider unavailable before vector write")
        durable = await coordinator.read_job(job_id)
        assert durable is not None
        assert durable.state is PublicationState.FAILED_RETRYABLE

        retry = await store.admit_source_revision(collection, source_id, job_id, source_sha)
        assert retry is not None
        assert retry.revision == first.revision
        assert retry.expected_active_revision is first.expected_active_revision
        assert retry.state is PublicationState.FAILED_RETRYABLE
    finally:
        await _cleanup(pool, corpus)


@pytest.mark.asyncio
async def test_dependency_fences_are_ordered_and_serialize_shared_source(
    pool: Any,  # noqa: F811
) -> None:
    corpus = f"itest-dependency-locks-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    second_entered = asyncio.Event()

    async def first_writer() -> None:
        async with coordinator.fence("derived-a", dependency_sources=("raw-b", "raw-a")):
            first_entered.set()
            await release_first.wait()

    async def second_writer() -> None:
        second_started.set()
        async with coordinator.fence("derived-b", dependency_sources=("raw-a", "raw-b")):
            second_entered.set()

    first = asyncio.create_task(first_writer())
    second = asyncio.create_task(second_writer())
    try:
        await asyncio.wait_for(first_entered.wait(), timeout=4)
        await asyncio.wait_for(second_started.wait(), timeout=4)
        await asyncio.sleep(0.1)
        assert not second_entered.is_set()
        release_first.set()
        await asyncio.wait_for(second_entered.wait(), timeout=4)
        await asyncio.wait_for(asyncio.gather(first, second), timeout=4)
    finally:
        release_first.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await _cleanup(pool, corpus)


@pytest.mark.asyncio
async def test_opposite_multi_source_lock_orders_complete_without_deadlock(
    pool: Any,  # noqa: F811
) -> None:
    corpus = f"itest-reverse-dependency-locks-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    entered: list[str] = []
    both_entered = asyncio.Event()

    async def writer(source_id: str, dependencies: tuple[str, str]) -> None:
        async with coordinator.fence(source_id, dependency_sources=dependencies):
            entered.append(source_id)
            if len(entered) == 2:
                both_entered.set()

    try:
        await asyncio.wait_for(
            asyncio.gather(
                writer("derived-a", ("raw-a", "raw-b")),
                writer("derived-b", ("raw-b", "raw-a")),
            ),
            timeout=5,
        )
        assert set(entered) == {"derived-a", "derived-b"}
        assert both_entered.is_set()
    finally:
        await _cleanup(pool, corpus)


@pytest.mark.asyncio
async def test_raw_pointer_advance_waits_until_dependent_vectors_are_verified_and_published(
    pool: Any,  # noqa: F811
) -> None:
    corpus = "books"
    token = uuid4().hex
    coordinator = SourceCoordinator(pool, corpus=corpus)
    raw_source = f"itest-raw-{token}"
    source_id = f"itest-derived-{token}"
    raw_collection = "vo_rag_books_raw_v1"
    raw_job_id = f"itest-raw-revision-one-{token}"
    raw_sha = _digest("raw source one")
    dependent = {raw_source: 1}
    raw_qdrant = _PausingQdrant()
    writer = SourceRevisionWriter(raw_qdrant, coordinator)
    try:
        async with coordinator.fence(raw_source, job_id=raw_job_id) as fence:
            raw_job = await coordinator.begin(
                fence,
                raw_collection,
                raw_source,
                raw_job_id,
                raw_sha,
                None,
                {"writer": "test-raw-v1"},
            )
            await coordinator.checkpoint(
                fence, raw_job_id, PublicationState.STAGED, {"staged": True}
            )
            await coordinator.checkpoint(
                fence, raw_job_id, PublicationState.VERIFIED, _verified_checkpoint()
            )
            await coordinator.publish(fence, raw_job_id)
        assert raw_job.revision == 1

        point = PointStruct(
            id="derived-point-base",
            vector={"dense": [0.6, 0.8]},
            payload={"source_id": source_id, "dependency_revisions": dependent},
        )
        derived_job = asyncio.create_task(
            writer.replace(
                "vo_rag_books_dsi_v1",
                source_id,
                [point],
                job_id=f"itest-derived-job-{token}",
                source_sha256=_digest("derived input"),
                expected_active_revision=None,
                recipe={"writer": "derived-test-v1"},
                expected_dependencies=dependent,
            )
        )
        try:
            await asyncio.wait_for(raw_qdrant.upsert_entered.wait(), timeout=5)
        except TimeoutError:
            await derived_job
            raise

        raw_advance_started = asyncio.Event()
        raw_advance_entered = asyncio.Event()

        async def advance_raw() -> None:
            raw_advance_started.set()
            raw_advance_job_id = f"itest-raw-revision-two-{token}"
            async with coordinator.fence(raw_source, job_id=raw_advance_job_id) as fence:
                job = await coordinator.begin(
                    fence,
                    raw_collection,
                    raw_source,
                    raw_advance_job_id,
                    _digest("raw source two"),
                    1,
                    {"writer": "test-raw-v1"},
                )
                await coordinator.checkpoint(
                    fence, job.job_id, PublicationState.STAGED, {"staged": True}
                )
                await coordinator.checkpoint(
                    fence, job.job_id, PublicationState.VERIFIED, _verified_checkpoint()
                )
                await coordinator.publish(fence, job.job_id)
                raw_advance_entered.set()

        raw_advance = asyncio.create_task(advance_raw())
        await asyncio.wait_for(raw_advance_started.wait(), timeout=3)
        await asyncio.sleep(0.1)
        assert not raw_advance_entered.is_set()

        raw_qdrant.release_upsert.set()
        point_ids = await asyncio.wait_for(derived_job, timeout=10)
        await asyncio.wait_for(raw_advance_entered.wait(), timeout=5)
        await raw_advance

        assert len(point_ids) == 1
        derived_pointer = await coordinator.read_active("vo_rag_books_dsi_v1", source_id)
        assert derived_pointer is not None and derived_pointer.revision == 1
        raw_pointer = await coordinator.read_active(raw_collection, raw_source)
        assert raw_pointer is not None and raw_pointer.revision == 2
    finally:
        raw_qdrant.release_upsert.set()
        await _cleanup_sources(pool, corpus, [raw_source, source_id])


async def test_durable_dependency_sampler_distinguishes_legacy_from_stale_revision(
    pool: Any,
) -> None:
    from vo_rag.api.operational_sampler import sample_source_publications
    from vo_rag.core.observability.metrics import source_publication_dependency_mismatches

    raw_source, derived_source, job_id = (f"sampler-{uuid4().hex}" for _ in range(3))
    await sample_source_publications(pool)
    baseline = source_publication_dependency_mismatches.labels(corpus="books")._value.get()  # noqa: SLF001
    try:
        await pool.execute(
            "INSERT INTO source_publication_jobs "
            "(job_id,corpus,collection,source_id,revision,source_sha256,recipe,state,"
            "published_at,retention_disposition) VALUES "
            "($1,'books','vo_rag_books_summary_v1',$2,1,$3,$4,'published',NOW(),'retained_old_revision')",
            job_id,
            derived_source,
            "a" * 64,
            {"expected_dependencies": {raw_source: None}},
        )
        await pool.execute(
            "INSERT INTO source_publications "
            "(corpus,collection,source_id,active_revision,active_job_id) VALUES "
            "('books','vo_rag_books_summary_v1',$1,1,$2)",
            derived_source,
            job_id,
        )
        await sample_source_publications(pool)
        assert (
            source_publication_dependency_mismatches.labels(corpus="books")._value.get() == baseline
        )  # noqa: SLF001
        await pool.execute(
            "INSERT INTO source_publications (corpus,collection,source_id,active_revision) "
            "VALUES ('books','vo_rag_books_raw_v1',$1,2)",
            raw_source,
        )
        await sample_source_publications(pool)
        assert (
            source_publication_dependency_mismatches.labels(corpus="books")._value.get()
            == baseline + 1
        )  # noqa: SLF001
        await pool.execute(
            "UPDATE source_publication_jobs SET recipe=$2 WHERE job_id=$1",
            job_id,
            {"expected_dependencies": {raw_source: 2}},
        )
        await sample_source_publications(pool)
        assert (
            source_publication_dependency_mismatches.labels(corpus="books")._value.get() == baseline
        )  # noqa: SLF001
    finally:
        await _cleanup_sources(pool, "books", [raw_source, derived_source])
