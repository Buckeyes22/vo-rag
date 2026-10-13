"""Postgres integration tests for durable source publication coordination."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import asyncpg  # type: ignore[import-untyped]
import pytest

from vo_rag.core.db.postgres import _register_codecs, run_migrations
from vo_rag.core.ingest.representation_receipt import canonical_sha256
from vo_rag.core.ingest.source_publication import (
    PublicationBindingError,
    PublicationState,
    PublicationTransitionError,
    SourceCoordinator,
    SourceFence,
    SourcePublicationPending,
    StalePublicationError,
)

pytestmark = pytest.mark.integration

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "vo_rag" / "core" / "db" / "migrations"
)


@pytest.fixture
async def pool() -> Any:
    dsn = os.environ.get("VORAG_TEST_DATABASE_URL")
    if not dsn:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail("VORAG_TEST_DATABASE_URL is required by the testinfra lane")
        pytest.skip("VORAG_TEST_DATABASE_URL unset — skipping source publication integration test")
    value = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=6,
        statement_cache_size=0,
        init=_register_codecs,
    )
    await run_migrations(value, MIGRATIONS_DIR)
    try:
        yield value
    finally:
        await value.close()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _verified_checkpoint() -> dict[str, object]:
    point_id = "staged-1"
    receipt = {
        "verification": "payload_exact_modalities_complete_numeric_transform_verified",
        "writer": {
            "point_id": point_id,
            "payload_sha256": _digest("payload"),
            "content_sha256": _digest("content"),
            "embedding_provenance_sha256": _digest("embedding"),
            "tag_binding_sha256": _digest("tags"),
            "vectors": {"dense": _digest("dense")},
        },
        "stored": {
            "point_id": point_id,
            "payload_sha256": _digest("payload"),
            "content_sha256": _digest("content"),
            "embedding_provenance_sha256": _digest("embedding"),
            "tag_binding_sha256": _digest("tags"),
            "vectors": {"dense": _digest("stored dense")},
        },
    }
    receipts = {point_id: receipt}
    manifest = canonical_sha256(receipts)
    return {
        "verification": {
            "expected_point_ids": [point_id],
            "readback_point_ids": [point_id],
            "expected_receipts": receipts,
            "readback_receipts": receipts,
            "expected_manifest_sha256": manifest,
            "readback_manifest_sha256": manifest,
        }
    }


async def _cleanup(pool: Any, corpus: str) -> None:
    async with pool.acquire() as connection:
        await connection.execute("DELETE FROM source_publication_jobs WHERE corpus = $1", corpus)
        await connection.execute("DELETE FROM source_publications WHERE corpus = $1", corpus)
        await connection.execute(
            "DELETE FROM source_publication_sequences WHERE corpus = $1", corpus
        )


async def _partial_retirement_fixture(
    coordinator: SourceCoordinator, fence: SourceFence
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a genuinely written partial intent, with different stored payload."""
    original = _digest("retained original")
    recipe = {"writer": "immutable-original-producer"}
    await coordinator.begin(
        fence, "books_raw", fence.source_id, f"failed-{coordinator.corpus}", original, None, recipe
    )

    def receipt(point_id: str, payload: str) -> dict[str, Any]:
        return {
            "point_id": point_id,
            "payload_sha256": _digest(payload),
            "content_sha256": _digest("content"),
            "embedding_provenance_sha256": _digest("embedding"),
            "tag_binding_sha256": _digest("tags"),
            "vectors": {"dense": _digest("vector")},
        }

    before = {"legacy": receipt("legacy", "old")}
    intent = {"staged": receipt("staged", "wanted")}
    checkpoint = {
        "before_receipts": before,
        "writer_intent": intent,
        "expected_point_ids": ["staged"],
        "acknowledged_batches": [["staged"]],
        "write_attempts": 1,
    }
    await coordinator.checkpoint(
        fence, f"failed-{coordinator.corpus}", PublicationState.PARTIAL_PRESERVED, checkpoint
    )
    retained = {
        "corpus": coordinator.corpus,
        "collection": "books_raw",
        "source_id": fence.source_id,
        "source_sha256": original,
        "failed_revision": 1,
        "active_revision": None,
        "before_receipts": before,
        "staged_receipts": {"staged": receipt("staged", "actually stored")},
    }
    return checkpoint, {
        "failed_job_id": f"failed-{coordinator.corpus}",
        "successor_job_id": f"successor-{coordinator.corpus}",
        "collection": "books_raw",
        "source_id": fence.source_id,
        "source_sha256": original,
        "expected_checkpoint_sha256": canonical_sha256(checkpoint),
        "expected_recipe_sha256": canonical_sha256(recipe),
        "retention_evidence": retained,
        "retention_evidence_sha256": canonical_sha256(retained),
        "successor_recipe": {"writer": "lossless-recovery-producer"},
        "expected_successor_revision": 2,
    }


async def test_atomic_unpublished_successor_retains_intent_pointer_and_exact_retry(
    pool: Any,
) -> None:
    corpus = f"itest-retirement-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    try:
        async with coordinator.fence(
            "source-retirement", job_id=f"failed-{coordinator.corpus}"
        ) as fence:
            old_checkpoint, arguments = await _partial_retirement_fixture(coordinator, fence)
            successor = await coordinator.supersede_unpublished_and_begin(fence, **arguments)
            assert successor.revision == 2 and successor.state is PublicationState.PREPARING
            failed = await coordinator.read_job(f"failed-{coordinator.corpus}")
            assert failed is not None and failed.state is PublicationState.SUPERSEDED
            assert failed.checkpoint == old_checkpoint
            assert failed.recipe == {"writer": "immutable-original-producer"}
            pointer = await coordinator.read_active("books_raw", fence.source_id)
            assert pointer is not None and pointer.revision is None and pointer.job_id is None
            assert (
                await coordinator.supersede_unpublished_and_begin(fence, **arguments) == successor
            )
        # The new pending job preserves source quarantine and owns retries.
        with pytest.raises(SourcePublicationPending):
            async with coordinator.fence("source-retirement", job_id="unrelated"):
                pytest.fail("successor admission exposed an unowned source")
        async with coordinator.fence(
            "source-retirement", job_id=f"successor-{coordinator.corpus}"
        ) as fence:
            assert (
                await coordinator.supersede_unpublished_and_begin(fence, **arguments) == successor
            )
            with pytest.raises(PublicationBindingError):
                await coordinator.supersede_unpublished_and_begin(
                    fence, **{**arguments, "successor_recipe": {"writer": "changed"}}
                )
    finally:
        await _cleanup(pool, corpus)


@pytest.mark.parametrize("change", ["checkpoint", "recipe", "retention", "pointer", "state"])
async def test_unpublished_successor_rejects_drift_without_retiring_or_allocating(
    pool: Any, change: str
) -> None:
    corpus = f"itest-retirement-reject-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    try:
        async with coordinator.fence(
            "source-reject", job_id=f"failed-{coordinator.corpus}"
        ) as fence:
            _, arguments = await _partial_retirement_fixture(coordinator, fence)
            if change == "checkpoint":
                arguments["expected_checkpoint_sha256"] = _digest("different checkpoint")
            elif change == "recipe":
                arguments["expected_recipe_sha256"] = _digest("different recipe")
            elif change == "retention":
                retained = {**arguments["retention_evidence"], "staged_receipts": {}}
                arguments.update(
                    retention_evidence=retained,
                    retention_evidence_sha256=canonical_sha256(retained),
                )
            elif change == "pointer":
                await pool.execute(
                    "UPDATE source_publications SET active_revision=1, active_job_id=$3 "
                    "WHERE corpus=$1 AND source_id=$2",
                    corpus,
                    fence.source_id,
                    arguments["failed_job_id"],
                )
            else:
                await coordinator.checkpoint(
                    fence,
                    f"failed-{coordinator.corpus}",
                    PublicationState.STAGED,
                    {
                        **arguments["retention_evidence"],
                        "writer_intent": arguments["retention_evidence"]["staged_receipts"],
                        "expected_point_ids": ["staged"],
                    },
                )
                failed = await coordinator.read_job(f"failed-{coordinator.corpus}")
                assert failed is not None
                arguments["expected_checkpoint_sha256"] = canonical_sha256(failed.checkpoint)
            with pytest.raises(
                (PublicationBindingError, PublicationTransitionError, StalePublicationError)
            ):
                await coordinator.supersede_unpublished_and_begin(fence, **arguments)
            assert await coordinator.read_job(f"successor-{coordinator.corpus}") is None
            assert (
                await pool.fetchval(
                    "SELECT last_revision FROM source_publication_sequences WHERE corpus=$1 "
                    "AND source_id=$2",
                    corpus,
                    fence.source_id,
                )
                == 1
            )
            failed = await coordinator.read_job(f"failed-{coordinator.corpus}")
            assert failed is not None and failed.state is not PublicationState.SUPERSEDED
    finally:
        await _cleanup(pool, corpus)


async def test_successor_insert_failure_rolls_back_retirement_and_sequence(pool: Any) -> None:
    corpus = f"itest-retirement-rollback-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)

    class FailingConnection:
        def __init__(self, connection: Any) -> None:
            self.connection = connection

        def transaction(self) -> Any:
            return self.connection.transaction()

        async def fetchrow(self, query: str, *args: Any) -> Any:
            if query.startswith("INSERT INTO source_publication_jobs"):
                raise RuntimeError("injected successor insert failure")
            return await self.connection.fetchrow(query, *args)

        async def fetchval(self, query: str, *args: Any) -> Any:
            return await self.connection.fetchval(query, *args)

        async def execute(self, query: str, *args: Any) -> Any:
            return await self.connection.execute(query, *args)

    class FailingPool:
        @asynccontextmanager
        async def acquire(self) -> AsyncIterator[FailingConnection]:
            async with pool.acquire() as connection:
                yield FailingConnection(connection)

    try:
        async with coordinator.fence(
            "source-rollback", job_id=f"failed-{coordinator.corpus}"
        ) as fence:
            old_checkpoint, arguments = await _partial_retirement_fixture(coordinator, fence)
            coordinator.pool = FailingPool()
            try:
                with pytest.raises(RuntimeError, match="injected"):
                    await coordinator.supersede_unpublished_and_begin(fence, **arguments)
            finally:
                coordinator.pool = pool
            failed = await coordinator.read_job(f"failed-{coordinator.corpus}")
            assert failed is not None and failed.state is PublicationState.PARTIAL_PRESERVED
            assert failed.checkpoint == old_checkpoint
            assert await coordinator.read_job(f"successor-{coordinator.corpus}") is None
            assert (
                await pool.fetchval(
                    "SELECT last_revision FROM source_publication_sequences WHERE corpus=$1 "
                    "AND source_id=$2",
                    corpus,
                    fence.source_id,
                )
                == 1
            )
    finally:
        await _cleanup(pool, corpus)


async def test_revision_checkpoint_publish_and_stale_expected_pointer(pool: Any) -> None:
    corpus = f"itest-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    source_id = "source-a"
    source_sha = _digest("revision one")
    try:
        async with coordinator.fence(source_id, job_id="job-one") as fence:
            job = await coordinator.begin(
                fence,
                "books_raw",
                source_id,
                "job-one",
                source_sha,
                None,
                {"writer": "test-v1"},
            )
            assert job.revision == 1
            assert job.state is PublicationState.PREPARING

            job = await coordinator.checkpoint(
                fence,
                "job-one",
                PublicationState.STAGED,
                {"point_ids": ["staged-1"], "acknowledged": 1},
            )
            durable = await pool.fetchrow(
                "SELECT state, checkpoint FROM source_publication_jobs WHERE job_id = $1",
                "job-one",
            )
            assert durable["state"] == "staged"
            assert durable["checkpoint"] == {"point_ids": ["staged-1"], "acknowledged": 1}

            await coordinator.checkpoint(
                fence, "job-one", PublicationState.VERIFIED, _verified_checkpoint()
            )
            published = await coordinator.publish(fence, "job-one")
            assert published.revision == 1

        assert await coordinator.read_active("books_raw", source_id) == published
        assert await coordinator.list_active("books_raw") == {source_id: 1}

        async with coordinator.fence(source_id, job_id="job-stale") as fence:
            with pytest.raises(StalePublicationError):
                await coordinator.begin(
                    fence,
                    "books_raw",
                    source_id,
                    "job-stale",
                    _digest("revision two"),
                    None,
                    {"writer": "test-v1"},
                )
    finally:
        await _cleanup(pool, corpus)


async def test_partial_job_blocks_new_job_and_resumes_with_durable_revision(pool: Any) -> None:
    corpus = f"itest-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    source_id = "source-b"
    source_sha = _digest("partial source")
    try:
        async with coordinator.fence(source_id, job_id="repair-one") as fence:
            first = await coordinator.begin(
                fence, "books_raw", source_id, "repair-one", source_sha, None, {}
            )
            await coordinator.checkpoint(
                fence,
                "repair-one",
                PublicationState.PARTIAL_PRESERVED,
                {"acknowledgement": "unknown", "old_revision_visible": True},
            )

        with pytest.raises(SourcePublicationPending):
            async with coordinator.fence(source_id, job_id="normal-ingest"):
                pytest.fail("a different job entered while a partial source job was pending")

        async with coordinator.fence(source_id, job_id="repair-one") as fence:
            resumed = await coordinator.begin(
                fence, "books_raw", source_id, "repair-one", source_sha, None, {}
            )
            assert resumed.revision == first.revision
            assert resumed.state is PublicationState.PARTIAL_PRESERVED
            staged = await coordinator.checkpoint(
                fence, "repair-one", PublicationState.STAGED, {"resumed": True}
            )
            assert staged.revision == first.revision
    finally:
        await _cleanup(pool, corpus)


async def test_legacy_write_guard_requires_complete_locked_dependency_closure(pool: Any) -> None:
    corpus = f"itest-legacy-closure-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    try:
        async with coordinator.fence("derived-source") as fence:
            with pytest.raises(ValueError, match="does not cover every source dependency"):
                await coordinator.assert_legacy_write_allowed(
                    fence, "derived-source", ("raw-source",)
                )

        async with coordinator.fence("derived-source", dependency_sources=("raw-source",)) as fence:
            await coordinator.assert_legacy_write_allowed(fence, "derived-source", ("raw-source",))
    finally:
        await _cleanup(pool, corpus)


async def test_legacy_write_guard_rejects_pending_intent_and_active_pointer_any_collection(
    pool: Any,
) -> None:
    corpus = f"itest-legacy-reject-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    pending_source = "pending-source"
    active_source = "active-source"
    try:
        async with coordinator.fence(pending_source) as fence:
            await coordinator.begin(
                fence,
                "vo_rag_books_raw_v1",
                pending_source,
                "pending-job",
                _digest("pending input"),
                None,
                {"writer": "test"},
            )
            with pytest.raises(SourcePublicationPending, match="publication job may have written"):
                await coordinator.assert_legacy_write_allowed(fence, pending_source)

        async with coordinator.fence(active_source, job_id="active-job") as fence:
            await coordinator.begin(
                fence,
                "vo_rag_books_raw_v1",
                active_source,
                "active-job",
                _digest("active input"),
                None,
                {"writer": "test"},
            )
            await coordinator.checkpoint(fence, "active-job", PublicationState.STAGED, {})
            await coordinator.checkpoint(
                fence, "active-job", PublicationState.VERIFIED, _verified_checkpoint()
            )
            await coordinator.publish(fence, "active-job")

        # The raw pointer is in another collection; the shared source fence must
        # still prevent an in-place compatibility write from any Books writer.
        async with coordinator.fence(active_source) as fence:
            with pytest.raises(SourcePublicationPending, match="active publication revision"):
                await coordinator.assert_legacy_write_allowed(fence, active_source)
    finally:
        await _cleanup(pool, corpus)


async def test_fences_serialize_one_source_but_allow_another_source(pool: Any) -> None:
    corpus = f"itest-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    second_entered = asyncio.Event()
    independent_entered = asyncio.Event()

    async def first_writer() -> None:
        async with coordinator.fence("same-source"):
            first_entered.set()
            await release_first.wait()

    async def second_writer() -> None:
        second_started.set()
        async with coordinator.fence("same-source"):
            second_entered.set()

    async def independent_writer() -> None:
        async with coordinator.fence("other-source"):
            independent_entered.set()

    first = asyncio.create_task(first_writer())
    second = asyncio.create_task(second_writer())
    independent = asyncio.create_task(independent_writer())
    try:
        await asyncio.wait_for(first_entered.wait(), timeout=3)
        await asyncio.wait_for(second_started.wait(), timeout=3)
        await asyncio.wait_for(independent_entered.wait(), timeout=3)
        assert not second_entered.is_set()
        release_first.set()
        await asyncio.wait_for(second_entered.wait(), timeout=3)
        await asyncio.gather(first, second, independent)
    finally:
        release_first.set()
        await asyncio.gather(first, second, independent, return_exceptions=True)
        await _cleanup(pool, corpus)


async def test_published_job_retry_is_immutable_and_superseded_publish_is_rejected(
    pool: Any,
) -> None:
    corpus = f"itest-{uuid4().hex}"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    source_id = "source-terminal-retry"
    first_sha = _digest("first immutable input")
    second_sha = _digest("second immutable input")
    try:
        async with coordinator.fence(source_id, job_id="first-job") as fence:
            first = await coordinator.begin(
                fence, "books_raw", source_id, "first-job", first_sha, None, {"writer": "v1"}
            )
            await coordinator.checkpoint(
                fence, "first-job", PublicationState.STAGED, {"staged": True}
            )
            await coordinator.checkpoint(
                fence, "first-job", PublicationState.VERIFIED, _verified_checkpoint()
            )
            first_pointer = await coordinator.publish(fence, "first-job")
            assert first_pointer.revision == first.revision

        async with coordinator.fence(source_id, job_id="first-job") as fence:
            repeated = await coordinator.begin(
                fence, "books_raw", source_id, "first-job", first_sha, None, {"writer": "v1"}
            )
            assert repeated.state is PublicationState.PUBLISHED
            assert repeated.revision == first.revision
            assert await coordinator.read_job("first-job") == repeated
            with pytest.raises(PublicationBindingError):
                await coordinator.begin(
                    fence,
                    "books_raw",
                    source_id,
                    "first-job",
                    second_sha,
                    None,
                    {"writer": "v1"},
                )
            assert await coordinator.publish(fence, "first-job") == first_pointer

        async with coordinator.fence(source_id, job_id="second-job") as fence:
            second = await coordinator.begin(
                fence,
                "books_raw",
                source_id,
                "second-job",
                second_sha,
                first.revision,
                {"writer": "v1"},
            )
            await coordinator.checkpoint(
                fence, "second-job", PublicationState.STAGED, {"staged": True}
            )
            await coordinator.checkpoint(
                fence,
                "second-job",
                PublicationState.VERIFIED,
                _verified_checkpoint(),
            )
            second_pointer = await coordinator.publish(fence, "second-job")
            assert second_pointer.revision == second.revision

        async with coordinator.fence(source_id, job_id="first-job") as fence:
            with pytest.raises(StalePublicationError):
                await coordinator.publish(fence, "first-job")
    finally:
        await _cleanup(pool, corpus)


async def test_distinct_source_fences_reserve_checkpoint_connections_across_coordinators(
    pool: Any,
) -> None:
    dsn = os.environ["VORAG_TEST_DATABASE_URL"]
    small_pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=2,
        max_size=2,
        statement_cache_size=0,
        init=_register_codecs,
    )
    corpus = f"itest-{uuid4().hex}"
    first = SourceCoordinator(small_pool, corpus=corpus, fence_timeout_s=5)
    second = SourceCoordinator(small_pool, corpus=corpus, fence_timeout_s=5)

    async def writer(index: int) -> None:
        source_id = f"source-{index}"
        job_id = f"job-{index}"
        coordinator = first if index % 2 else second
        async with coordinator.fence(source_id, job_id=job_id) as fence:
            await coordinator.begin(
                fence,
                "books_raw",
                source_id,
                job_id,
                _digest(job_id),
                None,
                {"writer": "two-connection-pool"},
            )
            await coordinator.checkpoint(
                fence, job_id, PublicationState.STAGED, {"source": source_id}
            )

    try:
        await asyncio.wait_for(asyncio.gather(*(writer(index) for index in range(6))), timeout=20)
        assert (
            await small_pool.fetchval(
                """SELECT count(*) FROM source_publication_jobs
                   WHERE corpus = $1 AND state = 'staged'""",
                corpus,
            )
            == 6
        )
    finally:
        await _cleanup(small_pool, corpus)
        await small_pool.close()
