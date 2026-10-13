"""Failure and recovery behavior for complete source revisions on real services."""

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import asyncpg  # type: ignore[import-untyped]
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, SparseVector

from vo_rag.core.db.postgres import _register_codecs, run_migrations
from vo_rag.core.ingest.source_publication import SourceCoordinator
from vo_rag.core.storage import setup_collections
from vo_rag.core.storage.publication_filter import publication_filter
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter, revision_points

pytestmark = pytest.mark.integration

_MIGRATIONS = Path(__file__).resolve().parents[3] / "src" / "vo_rag" / "core" / "db" / "migrations"


@pytest.fixture
def endpoints() -> tuple[str, str | None, str]:
    qdrant_url = os.environ.get("VORAG_TEST_QDRANT_URL")
    dsn = os.environ.get("VORAG_TEST_DATABASE_URL")
    if not qdrant_url or not dsn:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail("testinfra requires both VORAG_TEST_QDRANT_URL and VORAG_TEST_DATABASE_URL")
        pytest.skip("source revision writer integration requires local Qdrant and Postgres")
    return qdrant_url, os.environ.get("VORAG_TEST_QDRANT_API_KEY") or None, dsn


@pytest.fixture
async def services(
    endpoints: tuple[str, str | None, str],
) -> AsyncGenerator[tuple[AsyncQdrantClient, Any, SourceCoordinator, str, str], None]:
    url, api_key, dsn = endpoints
    corpus = f"itest-{uuid4().hex}"
    source_id = f"source-{uuid4().hex}"
    client = AsyncQdrantClient(url=url, api_key=api_key, prefer_grpc=False, timeout=120)
    pool = await asyncpg.create_pool(
        dsn=dsn, min_size=1, max_size=6, statement_cache_size=0, init=_register_codecs
    )
    coordinator = SourceCoordinator(pool, corpus=corpus)
    collection = "vo_rag_books_raw_v1"
    try:
        await run_migrations(pool, _MIGRATIONS)
        await setup_collections(client)
        yield client, pool, coordinator, collection, source_id
    finally:
        await client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
            ),
        )
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM source_publication_jobs WHERE corpus = $1", corpus
            )
            await connection.execute("DELETE FROM source_publications WHERE corpus = $1", corpus)
            await connection.execute(
                "DELETE FROM source_publication_sequences WHERE corpus = $1", corpus
            )
        await pool.close()
        await client.close()


def _points(source_id: str, count: int, *, marker: str) -> list[PointStruct]:
    """Make four-modality writer output with stable legacy/base identities."""
    result: list[PointStruct] = []
    for index in range(count):
        result.append(
            PointStruct(
                id=f"{source_id}-{marker}-{index}",
                vector={
                    "dense": [1.0, 0.01 * (index + 1)] + [0.0] * 1022,
                    "sparse": SparseVector(indices=[1, 2], values=[0.7, 0.2]),
                    "bm25": SparseVector(indices=[3, 4], values=[1.0, 0.4]),
                    "colbert": [[1.0, 0.01 * (index + 1)] + [0.0] * 1022],
                },
                payload={
                    "source_id": source_id,
                    "chunk_index": index,
                    "content": f"{marker} usable text {index}",
                    "embedding_provenance": {"model": "integration-fixture"},
                },
            )
        )
    return result


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _recipe(marker: str) -> dict[str, str]:
    return {"fixture": marker, "writer": "source-revision-integration-v1"}


class _FaultingClient:
    """Proxy real Qdrant and fail one selected batch before or after it applies."""

    def __init__(
        self,
        raw: AsyncQdrantClient,
        *,
        fail_batch: int | None = None,
        after_apply: bool = False,
        corrupt_after_write: bool = False,
    ) -> None:
        self.raw = raw
        self.fail_batch = fail_batch
        self.after_apply = after_apply
        self.corrupt_after_write = corrupt_after_write
        self.batch_number = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)

    async def upsert(
        self, *, collection_name: str, points: Sequence[PointStruct], wait: bool
    ) -> Any:
        batch = self.batch_number
        self.batch_number += 1
        if batch == self.fail_batch and not self.after_apply:
            raise ConnectionError("injected failure before the Qdrant request")
        response = await self.raw.upsert(
            collection_name=collection_name, points=list(points), wait=wait
        )
        if batch == self.fail_batch and self.after_apply:
            raise ConnectionError("injected lost Qdrant acknowledgement after apply")
        if self.corrupt_after_write and batch == 0:
            point = points[0]
            await self.raw.set_payload(
                collection_name=collection_name,
                payload={"content": "corrupted after acknowledged write"},
                points=[point.id],
            )
        return response


async def _replace(
    writer: SourceRevisionWriter,
    collection: str,
    source_id: str,
    points: list[PointStruct],
    *,
    job_id: str,
    marker: str,
    expected: int | None,
) -> list[str]:
    return await writer.replace(
        collection,
        source_id,
        points,
        job_id=job_id,
        source_sha256=_sha(marker),
        expected_active_revision=expected,
        recipe=_recipe(marker),
    )


async def _visible_ids(
    client: AsyncQdrantClient, collection: str, source_id: str, revision: int | None
) -> set[str]:
    rows, _ = await client.scroll(
        collection_name=collection,
        scroll_filter=publication_filter({source_id: revision}),
        limit=256,
        with_payload=True,
        with_vectors=False,
    )
    return {str(point.id) for point in rows if (point.payload or {}).get("source_id") == source_id}


async def _publish_initial(
    client: AsyncQdrantClient,
    coordinator: SourceCoordinator,
    collection: str,
    source_id: str,
    *,
    count: int = 10,
) -> set[str]:
    ids = await _replace(
        SourceRevisionWriter(client, coordinator),
        collection,
        source_id,
        _points(source_id, count, marker="old"),
        job_id="initial-" + uuid4().hex,
        marker="old",
        expected=None,
    )
    return set(ids)


@pytest.mark.parametrize("fail_batch", [0, 1])
@pytest.mark.parametrize("after_apply", [False, True])
async def test_batch_failure_preserves_old_pointer_hides_staged_revision_and_resumes(
    services: tuple[AsyncQdrantClient, Any, SourceCoordinator, str, str],
    fail_batch: int,
    after_apply: bool,
) -> None:
    client, _pool, coordinator, collection, source_id = services
    old_ids = await _publish_initial(client, coordinator, collection, source_id)
    points = _points(source_id, 10, marker=f"new-{fail_batch}-{after_apply}")
    job_id = f"retry-{fail_batch}-{after_apply}-{uuid4().hex}"
    writer = SourceRevisionWriter(
        _FaultingClient(client, fail_batch=fail_batch, after_apply=after_apply), coordinator
    )

    with pytest.raises(ConnectionError):
        await _replace(
            writer,
            collection,
            source_id,
            points,
            job_id=job_id,
            marker=f"new-{fail_batch}-{after_apply}",
            expected=1,
        )

    assert await coordinator.list_active(collection) == {source_id: 1}
    assert await _visible_ids(client, collection, source_id, 1) == old_ids
    source_rows = await SourceRevisionWriter(client, coordinator).census(collection, source_id, 2)
    # Depending on the failure point, zero, one, or two batches may have reached Qdrant.
    expected_staged = 0 if fail_batch == 0 and not after_apply else 8
    if fail_batch == 1 and after_apply:
        expected_staged = 10
    assert len(source_rows) == expected_staged
    assert not (await _visible_ids(client, collection, source_id, 1)) & {
        str(point.id) for point in source_rows
    }

    resumed = await _replace(
        SourceRevisionWriter(client, coordinator),
        collection,
        source_id,
        points,
        job_id=job_id,
        marker=f"new-{fail_batch}-{after_apply}",
        expected=1,
    )
    assert set(resumed) == {
        str(point.id)
        for point in revision_points(
            points,
            corpus=coordinator.corpus,
            collection=collection,
            source_id=source_id,
            revision=2,
        )
    }
    assert await coordinator.list_active(collection) == {source_id: 2}
    assert await _visible_ids(client, collection, source_id, 2) == set(resumed)
    assert (
        await _replace(
            SourceRevisionWriter(client, coordinator),
            collection,
            source_id,
            points,
            job_id=job_id,
            marker=f"new-{fail_batch}-{after_apply}",
            expected=1,
        )
        == resumed
    )


async def test_pointer_acknowledgement_loss_is_resolved_as_published_idempotently(
    services: tuple[AsyncQdrantClient, Any, SourceCoordinator, str, str],
) -> None:
    client, _pool, coordinator, collection, source_id = services
    await _publish_initial(client, coordinator, collection, source_id, count=2)
    points = _points(source_id, 3, marker="cas-ackloss")
    original_publish = coordinator.publish
    lost = False

    async def publish_then_lose_ack(fence: Any, job_id: str) -> Any:
        nonlocal lost
        result = await original_publish(fence, job_id)
        if not lost:
            lost = True
            raise ConnectionError("injected lost pointer commit acknowledgement")
        return result

    coordinator.publish = publish_then_lose_ack  # type: ignore[method-assign]
    writer = SourceRevisionWriter(client, coordinator)
    job_id = "cas-ackloss-" + uuid4().hex
    returned = await _replace(
        writer,
        collection,
        source_id,
        points,
        job_id=job_id,
        marker="cas-ackloss",
        expected=1,
    )
    assert await coordinator.list_active(collection) == {source_id: 2}
    visible = await _visible_ids(client, collection, source_id, 2)
    assert len(visible) == 3

    # The writer reconciles the authoritative pointer after the simulated lost
    # acknowledgement and returns the already-published identities.
    assert set(returned) == visible
    assert (
        set(
            await _replace(
                writer,
                collection,
                source_id,
                points,
                job_id=job_id,
                marker="cas-ackloss",
                expected=1,
            )
        )
        == visible
    )


async def test_smaller_roster_hides_stale_tail_and_keeps_old_points_without_cleanup(
    services: tuple[AsyncQdrantClient, Any, SourceCoordinator, str, str],
) -> None:
    client, _pool, coordinator, collection, source_id = services
    old_ids = await _publish_initial(client, coordinator, collection, source_id, count=10)
    fault = _FaultingClient(client, fail_batch=0, after_apply=False)
    job_id = "short-failed-" + uuid4().hex
    with pytest.raises(ConnectionError):
        await _replace(
            SourceRevisionWriter(fault, coordinator),
            collection,
            source_id,
            _points(source_id, 3, marker="short-new"),
            job_id=job_id,
            marker="short-new",
            expected=1,
        )
    all_rows = await SourceRevisionWriter(client, coordinator).census(collection, source_id, 1)
    assert {str(row.id) for row in all_rows} == old_ids
    assert await _visible_ids(client, collection, source_id, 1) == old_ids

    new_ids = await _replace(
        SourceRevisionWriter(client, coordinator),
        collection,
        source_id,
        _points(source_id, 3, marker="short-new"),
        job_id=job_id,
        marker="short-new",
        expected=1,
    )
    assert len(new_ids) == 3
    assert await _visible_ids(client, collection, source_id, 2) == set(new_ids)
    # Full old revision is retained; its stale seven-point tail is invisible by
    # the publication predicate and has not been destructively pruned.
    writer = SourceRevisionWriter(client, coordinator)
    rows_after = await writer.census(collection, source_id, 1)
    rows_after.extend(await writer.census(collection, source_id, 2))
    assert len(rows_after) == len(old_ids) + len(new_ids)
    assert old_ids.isdisjoint(new_ids)


async def test_writer_rejects_scope_broadening_duplicate_identity_and_representation_corruption(
    services: tuple[AsyncQdrantClient, Any, SourceCoordinator, str, str],
) -> None:
    client, _pool, coordinator, collection, source_id = services
    old_ids = await _publish_initial(client, coordinator, collection, source_id, count=2)
    duplicate = _points(source_id, 2, marker="duplicate")
    duplicate[1] = duplicate[1].model_copy(update={"id": duplicate[0].id})
    with pytest.raises(ValueError, match="duplicate"):
        revision_points(
            duplicate, corpus="books", collection=collection, source_id=source_id, revision=2
        )

    wrong_scope = _points("other-source", 1, marker="wrong")
    with pytest.raises(ValueError, match="source scope"):
        revision_points(
            wrong_scope, corpus="books", collection=collection, source_id=source_id, revision=2
        )

    job_id = "corrupt-" + uuid4().hex
    with pytest.raises(ValueError, match="payload"):
        await _replace(
            SourceRevisionWriter(_FaultingClient(client, corrupt_after_write=True), coordinator),
            collection,
            source_id,
            _points(source_id, 1, marker="corrupt"),
            job_id=job_id,
            marker="corrupt",
            expected=1,
        )
    assert await coordinator.list_active(collection) == {source_id: 1}
    assert await _visible_ids(client, collection, source_id, 1) == old_ids
    # The failed candidate remains quarantined, and no delete/cleanup ran.
    job = await _pool.fetchrow(
        "SELECT state FROM source_publication_jobs WHERE job_id = $1", job_id
    )
    assert job is not None and str(job["state"]) == "partial_preserved"
    assert (
        len(await SourceRevisionWriter(client, coordinator).census(collection, source_id, 2)) == 1
    )


async def test_raw_quality_warning_copy_forward_is_books_scoped_and_idempotent(
    endpoints: tuple[str, str | None, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw warning COW advances once without adding a self dependency or touching neighbors."""
    url, api_key, dsn = endpoints
    monkeypatch.setenv("VORAG_SOURCE_PUBLICATION_ENABLED", "true")
    monkeypatch.setenv("VORAG_SOURCE_PUBLICATION_WRITES_ENABLED", "true")
    store = QdrantStore(url=url, api_key=api_key, prefer_grpc=False)
    client = store._raw_client
    pool = await asyncpg.create_pool(
        dsn=dsn, min_size=1, max_size=6, statement_cache_size=0, init=_register_codecs
    )
    collection = "vo_rag_books_raw_v1"
    source_id = f"source-quality-{uuid4().hex}"
    corpus = "books"
    coordinator = SourceCoordinator(pool, corpus=corpus)
    outside_id: str | None = None

    async def use_test_coordinator(_collection: str) -> SourceCoordinator:
        return coordinator

    monkeypatch.setattr(store, "_source_coordinator", use_test_coordinator)
    try:
        await run_migrations(pool, _MIGRATIONS)
        await setup_collections(client)
        points = _points(source_id, 2, marker="quality-before")
        points = [
            point.model_copy(
                update={
                    "vector": {
                        "dense": [1.0] + [0.0] * 1023,
                        "sparse": SparseVector(indices=[3, 9], values=[0.25, 0.75]),
                        "bm25": SparseVector(indices=[4, 12], values=[0.5, 1.25]),
                        "colbert": [[1.0] + [0.0] * 1023],
                    },
                    "payload": {
                        **(point.payload or {}),
                        "tf_event_family": ["relay"],
                        "citation_id": f"citation-{(point.payload or {})['chunk_index']}",
                    },
                }
            )
            for point in points
        ]
        outside = _points(f"outside-{uuid4().hex}", 1, marker="outside")[0]
        outside_id = str(uuid4())
        outside = outside.model_copy(
            update={
                "id": outside_id,
                "vector": {
                    "dense": [1.0] + [0.0] * 1023,
                    "sparse": SparseVector(indices=[7], values=[0.7]),
                    "bm25": SparseVector(indices=[8], values=[0.8]),
                    "colbert": [[1.0] + [0.0] * 1023],
                },
            }
        )
        await client.upsert(collection_name=collection, points=[outside], wait=True)
        writer = SourceRevisionWriter(client, coordinator)
        await _replace(
            writer,
            collection,
            source_id,
            points,
            job_id="quality-initial-" + uuid4().hex,
            marker="quality-before",
            expected=None,
        )
        before = await writer.census(collection, source_id, 1)
        before_by_chunk = {(row.payload or {})["chunk_index"]: row for row in before}
        outside_before = (await client.retrieve(collection, [outside.id], with_vectors=True))[0]
        warnings = [{"stage": "book_quality", "code": "low_text_density"}]

        await store.set_source_quality_warnings(collection, source_id, warnings)

        assert await coordinator.list_active(collection) == {source_id: 2}
        after = await writer.census(collection, source_id, 2)
        assert len(after) == 2
        for row in after:
            payload = row.payload or {}
            chunk_index = payload["chunk_index"]
            previous = before_by_chunk[chunk_index]
            assert row.vector == previous.vector
            expected_payload = dict(previous.payload or {})
            expected_payload["source_revision"] = 2
            expected_payload["quality_warnings"] = warnings
            expected_payload["quality_disposition"] = "warning"
            assert payload == expected_payload
            assert "dependency_revisions" not in payload
            assert "dependency_revision_keys" not in payload
            assert payload["content"] == (previous.payload or {})["content"]
            assert payload["citation_id"] == (previous.payload or {})["citation_id"]
            assert payload["tf_event_family"] == ["relay"]

        retained = await writer.census(collection, source_id, 1)
        retained_by_id = {str(row.id): row for row in retained}
        assert set(retained_by_id) == {str(row.id) for row in before}
        for previous in before_by_chunk.values():
            retained_row = retained_by_id[str(previous.id)]
            assert retained_row.payload == previous.payload
            assert retained_row.vector == previous.vector
        outside_after = (await client.retrieve(collection, [outside_id], with_vectors=True))[0]
        assert outside_after.payload == outside_before.payload
        assert outside_after.vector == outside_before.vector

        physical_count = (await client.count(collection, exact=True)).count
        await store.set_source_quality_warnings(collection, source_id, warnings)
        assert await coordinator.list_active(collection) == {source_id: 2}
        assert (await client.count(collection, exact=True)).count == physical_count
    finally:
        await client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
            ),
        )
        if outside_id is not None:
            await client.delete(collection_name=collection, points_selector=[outside_id])
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM source_publication_jobs WHERE corpus = $1 AND source_id = $2",
                corpus,
                source_id,
            )
            await connection.execute(
                "DELETE FROM source_publications WHERE corpus = $1 AND source_id = $2",
                corpus,
                source_id,
            )
            await connection.execute(
                "DELETE FROM source_publication_sequences WHERE corpus = $1 AND source_id = $2",
                corpus,
                source_id,
            )
        await pool.close()
        await client.close()


async def test_mechanical_probe_resumes_after_qdrant_ack_is_lost(
    services: tuple[AsyncQdrantClient, Any, SourceCoordinator, str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real writer partial stays hidden, then the same zero-model job publishes fully."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from vo_rag.core.ingest.representation_receipt import canonical_sha256, representation_receipt
    from vo_rag.ops import publication_probe as probe

    client, pool, _fixture_coordinator, collection, fixture_source_id = services
    source_id = f"mechanical-probe-{uuid4().hex}"
    envelope_id = f"mechanical-envelope-{uuid4().hex}"
    artifact_id = f"mechanical-artifact-{uuid4().hex}"
    source_bytes = b"mechanical publication probe source bytes"
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    parse_bytes = b"accepted parsed representation fixture"
    parse_sha = hashlib.sha256(parse_bytes).hexdigest()
    source_path = tmp_path / "source.md"
    parse_path = tmp_path / "parsed.json"
    state_dir = tmp_path / "state"
    source_path.write_bytes(source_bytes)
    parse_path.write_bytes(parse_bytes)
    source_path.chmod(0o600)
    parse_path.chmod(0o600)
    coordinator = SourceCoordinator(pool, corpus="books")
    monkeypatch.setattr(probe, "_SOURCE_ID", source_id)
    monkeypatch.setattr(probe, "_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(probe, "_ENVELOPE_ID", envelope_id)
    monkeypatch.setattr(probe, "_ARTIFACT_ID", artifact_id)
    monkeypatch.setattr(probe, "_RAW_COLLECTION", collection)
    monkeypatch.setattr(probe, "_RAW_REVISION_ONE_POINT_ID", "unused-fixture-id")
    monkeypatch.setattr(probe, "_PARSE_KEY", f"parsed/{envelope_id}/parsed.json")
    monkeypatch.setattr(probe, "_DERIVED_COLLECTIONS", ())

    writer = SourceRevisionWriter(client, coordinator)
    initial = _points(source_id, 1, marker="mechanical-copy")
    await writer.replace(
        collection,
        source_id,
        initial,
        job_id=f"probe-seed-1-{uuid4().hex}",
        source_sha256=source_sha,
        expected_active_revision=None,
        recipe={"seed": 1},
    )
    second_id = (
        await writer.replace(
            collection,
            source_id,
            initial,
            job_id=f"probe-seed-2-{uuid4().hex}",
            source_sha256=source_sha,
            expected_active_revision=1,
            recipe={"seed": 2},
        )
    )[0]
    garage_key = f"verified/sha256/{source_sha}"
    await pool.execute(
        "INSERT INTO intake_envelopes "
        "(id, source_type, source_data, raw_content, metadata, status, target_pipeline) "
        "VALUES ($1, 'file_drop', $2, $3, '{}'::jsonb, 'completed', 'book_rag')",
        envelope_id,
        {"garage_key": garage_key},
        {
            "filename": "technical-canary.md",
            "file_hash": source_sha,
            "size_bytes": len(source_bytes),
            "garage_key": garage_key,
            "object_verification": {
                "status": "verified",
                "sha256": source_sha,
                "size_bytes": len(source_bytes),
                "authoritative_key": garage_key,
            },
        },
    )
    await pool.execute(
        "INSERT INTO knowledge_artifacts "
        "(id, envelope_id, pipeline, artifact_type, title, content, tags, priority, status, "
        "license_class, qdrant_collection, qdrant_point_ids, source_provenance) "
        "VALUES ($1,$2,'book_rag','extraction','mechanical fixture',$3,$4,'low','active',"
        "'unknown',$5,$6,$7)",
        artifact_id,
        envelope_id,
        {"source_id": source_id},
        ["fixture"],
        collection,
        [second_id],
        [{"source_id": source_id, "source_type": "file_drop", "contribution": 1.0}],
    )

    settings = SimpleNamespace(model_dump=lambda mode="json": {"integration": "mechanical-probe"})

    class FaultOnce:
        def __init__(self) -> None:
            self.armed = True

        def __getattr__(self, name: str) -> Any:
            return getattr(client, name)

        async def upsert(self, **kwargs: Any) -> Any:
            result = await client.upsert(**kwargs)
            if self.armed:
                self.armed = False
                raise ConnectionError("simulated response loss after Qdrant ACK")
            return result

    faulting = FaultOnce()

    class Store:
        source_visibility_enabled = True
        source_publication_enabled = True
        _raw_client = faulting

        async def _source_coordinator(self, _collection: str) -> SourceCoordinator:
            return coordinator

    store = Store()

    async def load_evidence(
        _source_path: Path,
        _parse_path: Path,
        _store: Any,
        _pool: Any,
        _garage: Any,
        _settings: Any,
        *,
        expected_active_revision: int = 2,
        allow_staged_revision_three: bool = False,
    ) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
        pointer = await coordinator.read_active(collection, source_id)
        assert pointer is not None and pointer.revision == expected_active_revision
        rows, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
            ),
            limit=10,
            with_payload=True,
            with_vectors=True,
        )
        by_revision = {int((row.payload or {})["source_revision"]): row for row in rows}
        assert set(by_revision) in ({1, 2}, {1, 2, 3})
        if not allow_staged_revision_three:
            assert set(by_revision) == {1, 2}
        artifact_row = await _pool.fetchrow(
            "SELECT id, envelope_id, pipeline, artifact_type, title, tags, priority, status, "
            "license_class, content, source_provenance, qdrant_collection, qdrant_point_ids "
            "FROM knowledge_artifacts WHERE id = $1",
            artifact_id,
        )
        assert artifact_row is not None
        artifact = dict(artifact_row)
        active_row = by_revision[expected_active_revision]
        before = {str(row.id): representation_receipt(row) for row in rows}
        census = await probe._derived_census(client)
        canonical_snapshot = {"resolved": False, "source": None}
        envelope = {"id": envelope_id, "source_sha256": source_sha, "parse_sha256": parse_sha}
        envelope_evidence = {"envelope_sha256": canonical_sha256(envelope)}
        evidence = {
            "source_sha256": source_sha,
            "accepted_parse_sha256": parse_sha,
            "source_path": str(source_path.resolve()),
            "accepted_parse_path": str(parse_path.resolve()),
            "envelope": envelope,
            "envelope_evidence": envelope_evidence,
            "canonical_source_snapshot": canonical_snapshot,
            "canonical_source_snapshot_sha256": canonical_sha256(canonical_snapshot),
            "artifact": artifact,
            "artifact_sha256": canonical_sha256(artifact),
            "publication_pointer": {
                "corpus": pointer.corpus,
                "collection": pointer.collection,
                "source_id": pointer.source_id,
                "revision": pointer.revision,
                "source_sha256": pointer.source_sha256,
                "job_id": pointer.job_id,
            },
            "configuration_sha256": canonical_sha256(settings.model_dump(mode="json")),
            "active_revision": expected_active_revision,
            "allocated_revision": expected_active_revision + 1,
            "current_point_id": str(active_row.id),
            "before_receipts": before,
            "before_roster_sha256": canonical_sha256(before),
            "derived_census": census,
            "source_sequence": int(
                await _pool.fetchval(
                    "SELECT last_revision FROM source_publication_sequences "
                    "WHERE corpus='books' AND source_id=$1",
                    source_id,
                )
            ),
            "physical_revision_ids": {
                str(revision): str(row.id) for revision, row in by_revision.items()
            },
        }
        return evidence, list(rows), artifact

    @asynccontextmanager
    async def resources() -> Any:
        yield settings, store, pool, None

    monkeypatch.setattr(probe, "_load_fixture_evidence", load_evidence)
    monkeypatch.setattr(probe, "_resources", resources)
    checkpoint = await probe.prepare_mechanical_probe(source_path, parse_path, state_dir)
    state_path = Path(checkpoint.state_path)

    try:
        with pytest.raises(ConnectionError, match="simulated response loss"):
            await probe.execute_mechanical_probe(state_path)
        partial = probe._load_checkpoint(state_path)
        assert partial.status == "partial_preserved"
        assert (await coordinator.read_active(collection, source_id)).revision == 2  # type: ignore[union-attr]
        assert len(await writer.census(collection, source_id, 3)) == 1
        assert await _visible_ids(client, collection, source_id, 2) == {second_id}

        result = await probe.resume_mechanical_probe(state_path)
        assert result.status == "completed"
        assert result.source_revision == 3
        assert result.inference_requests == 0
        assert result.old_revisions_retained == [1, 2]
        assert (await coordinator.read_active(collection, source_id)).revision == 3  # type: ignore[union-attr]
        assert len(await writer.census(collection, source_id, 1)) == 1
        assert len(await writer.census(collection, source_id, 2)) == 1
        assert len(await writer.census(collection, source_id, 3)) == 1
        current_artifact = await pool.fetchrow(
            "SELECT qdrant_point_ids FROM knowledge_artifacts WHERE id=$1", artifact_id
        )
        assert current_artifact is not None
        assert current_artifact["qdrant_point_ids"] == [result.active_point_id]
        assert probe._load_checkpoint(state_path).status == "completed"
    finally:
        await client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
            ),
        )
        await pool.execute("DELETE FROM knowledge_artifacts WHERE id=$1", artifact_id)
        await pool.execute("DELETE FROM intake_envelopes WHERE id=$1", envelope_id)
        await pool.execute(
            "DELETE FROM source_publication_jobs WHERE corpus='books' AND source_id=$1", source_id
        )
        await pool.execute(
            "DELETE FROM source_publications WHERE corpus='books' AND source_id=$1", source_id
        )
        await pool.execute(
            "DELETE FROM source_publication_sequences WHERE corpus='books' AND source_id=$1",
            source_id,
        )


async def test_repair_extraction_scope_and_fenced_reference_cas_on_real_database(
    services: tuple[AsyncQdrantClient, Any, SourceCoordinator, str, str],
) -> None:
    from types import SimpleNamespace
    from typing import cast

    from vo_rag.core.ingest.repair_evidence import PointDisposition, RepairCorpus, RepairEvidence
    from vo_rag.core.ingest.representation_receipt import (
        canonical_sha256,
        representation_receipt,
    )
    from vo_rag.core.storage.publication_client import PublicationClient
    from vo_rag.core.storage.qdrant import QdrantStore
    from vo_rag.ops.corpus_repair import _verify_live
    from vo_rag.ops.repair_artifact_references import (
        ArtifactReferenceBlocked,
        _rows,
        check_artifact_references,
        reconcile_artifact_references,
    )

    client, pool, _fixture_coordinator, collection, source_id = services
    coordinator = SourceCoordinator(pool, corpus="books")
    writer = SourceRevisionWriter(client, coordinator)
    source_sha = _sha(source_id)
    job_id = f"repair-reference-{uuid4().hex}"
    envelope_id = f"repair-envelope-{uuid4().hex}"
    duplicate_envelope = f"repair-duplicate-{uuid4().hex}"
    artifact_id = f"repair-artifact-{uuid4().hex}"
    duplicate_artifact = f"repair-extra-{uuid4().hex}"
    before_ids = await writer.replace(
        collection,
        source_id,
        _points(source_id, 2, marker="before"),
        job_id=f"repair-seed-{uuid4().hex}",
        source_sha256=source_sha,
        expected_active_revision=None,
        recipe=_recipe("before"),
    )
    output = _points(source_id, 3, marker="after")
    staged = revision_points(
        output, corpus="books", collection=collection, source_id=source_id, revision=2
    )
    after_ids = [str(point.id) for point in staged]
    writer_receipts = [representation_receipt(point) for point in staged]
    evidence = cast(
        RepairEvidence,
        SimpleNamespace(
            corpus=RepairCorpus.BOOKS,
            source_id=source_id,
            target_collection=collection,
            source_envelope_id=envelope_id,
            original=SimpleNamespace(sha256=source_sha),
            replacement_revision_sequence=2,
            before_points=tuple(SimpleNamespace(point_id=point_id) for point_id in before_ids),
            after_points=tuple(
                SimpleNamespace(point_id=point_id, disposition=PointDisposition.REPLACE)
                for point_id in after_ids
            ),
            writer_recipe=SimpleNamespace(
                execution_receipt_sha256=canonical_sha256(writer_receipts)
            ),
        ),
    )

    async def pointers(_corpus: str, _collection: str) -> dict[str, int | None]:
        return {source_id: 2}

    class VerifyStore:
        _raw_client = client
        _client = PublicationClient(client, pointers)

        async def _source_coordinator(self, _collection: str) -> SourceCoordinator:
            return coordinator

        async def search_single_collection(self, *args: Any, **kwargs: Any) -> Any:
            return await QdrantStore.search_single_collection(
                cast(QdrantStore, self), *args, **kwargs
            )

    store = VerifyStore()
    try:
        for owned_envelope in (envelope_id, duplicate_envelope):
            await pool.execute(
                "INSERT INTO intake_envelopes "
                "(id,source_type,source_data,raw_content,metadata,status,target_pipeline) "
                "VALUES($1,'file_drop','{}'::jsonb,'{}'::jsonb,'{}'::jsonb,"
                "'completed','book_rag')",
                owned_envelope,
            )

        async def add_artifact(owned_id: str, owned_envelope: str) -> None:
            await pool.execute(
                "INSERT INTO knowledge_artifacts "
                "(id,envelope_id,pipeline,artifact_type,title,content,tags,priority,status,"
                "license_class,qdrant_collection,qdrant_point_ids,source_provenance) "
                "VALUES($1,$2,'book_rag','extraction','repair reference fixture',$3,"
                "'{}'::text[],'low','active','unknown',$4,$5,$6)",
                owned_id,
                owned_envelope,
                {"source_id": source_id},
                collection,
                before_ids,
                [{"source_id": source_id}],
            )

        await add_artifact(artifact_id, envelope_id)
        async with pool.acquire() as connection:
            snapshots = await _rows(connection, evidence)
        evidence.owned_artifact_snapshots = tuple(snapshots)
        evidence.owned_artifact_snapshots_sha256 = canonical_sha256(snapshots)
        await check_artifact_references(evidence, pool)
        await add_artifact(duplicate_artifact, duplicate_envelope)
        with pytest.raises(ArtifactReferenceBlocked, match="snapshot_changed_before_admission"):
            await check_artifact_references(evidence, pool)
        await pool.execute("DELETE FROM knowledge_artifacts WHERE id=$1", duplicate_artifact)
        published = await writer.replace(
            collection,
            source_id,
            output,
            job_id=job_id,
            source_sha256=source_sha,
            expected_active_revision=1,
            recipe=_recipe("after"),
        )
        assert published == after_ids
        published_rows = await writer.census(collection, source_id, 2)
        stored_receipts = {str(row.id): representation_receipt(row) for row in published_rows}
        assert (
            stored_receipts[after_ids[0]]["vectors"]["dense"]
            != writer_receipts[0]["vectors"]["dense"]
        )
        verification = await _verify_live(
            evidence,
            store,
            expected_job_id=job_id,
            expected_ids=set(after_ids),
        )
        assert verification["active_revision"] == 2
        assert verification["source_point_count"] == 3
        assert (
            await reconcile_artifact_references(evidence, store, job_id=job_id, after_ids=after_ids)
            == "updated"
        )
        assert (
            await reconcile_artifact_references(
                evidence, store, job_id=job_id, after_ids=after_ids, write=False
            )
            == "already_current"
        )
        await pool.execute(
            "UPDATE knowledge_artifacts SET tags=ARRAY['concurrent change']::text[] WHERE id=$1",
            artifact_id,
        )
        with pytest.raises(ArtifactReferenceBlocked, match="exact_snapshot_changed_after_publish"):
            await reconcile_artifact_references(evidence, store, job_id=job_id, after_ids=after_ids)
    finally:
        await pool.execute(
            "DELETE FROM knowledge_artifacts WHERE id=ANY($1::text[])",
            [artifact_id, duplicate_artifact],
        )
        await pool.execute(
            "DELETE FROM intake_envelopes WHERE id=ANY($1::text[])",
            [envelope_id, duplicate_envelope],
        )
        await pool.execute(
            "DELETE FROM source_publication_jobs WHERE corpus='books' AND source_id=$1", source_id
        )
        await pool.execute(
            "DELETE FROM source_publications WHERE corpus='books' AND source_id=$1", source_id
        )
        await pool.execute(
            "DELETE FROM source_publication_sequences WHERE corpus='books' AND source_id=$1",
            source_id,
        )


async def _ready(value: Any) -> Any:
    return value
