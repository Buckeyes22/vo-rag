"""Exercise quality-only dependency copy-forward against local Qdrant."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from vo_rag.core.ingest.representation_receipt import canonical_sha256
from vo_rag.core.ingest.source_publication import (
    ActivePublication,
    PublicationState,
    SourceFence,
    SourcePublicationJob,
    StalePublicationError,
)
from vo_rag.core.storage.publication_filter import dependencies_visible
from vo_rag.core.storage.source_quality_dependencies import copy_forward_quality_dependencies
from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter, revision_points

_RAW = "vo_rag_books_raw_v1"
_STERILE = "vo_rag_books_sterile_v1"
_SUMMARY = "vo_rag_books_summary_tree_v1"
_GRAPH_NODES = "vo_rag_books_graph_nodes_v1"
_COLLECTIONS = (
    _RAW,
    _STERILE,
    _SUMMARY,
    "vo_rag_books_raptor_v1",
    "vo_rag_books_contextualized_v1",
    "vo_rag_books_dsi_v1",
    "vo_rag_books_graph_nodes_v1",
    "vo_rag_books_graph_edges_v1",
)


class _Coordinator:
    """Durable-writer protocol fake backed by local Qdrant point data."""

    corpus = "books"

    def __init__(self) -> None:
        self.active: dict[tuple[str, str], ActivePublication] = {}
        self.jobs: dict[str, SourcePublicationJob] = {}
        self.next_revision: dict[str, int] = {}
        self.advance_raw_when_listing_derived = False
        self.raw_source = ""
        self.advance_raw_after_raw_reads = 0
        self.raw_read_count = 0
        self.list_active_calls: list[str] = []

    @asynccontextmanager
    async def fence(
        self,
        source_id: str,
        *,
        job_id: str | None = None,
        dependency_sources: Sequence[str] = (),
    ) -> AsyncGenerator[SourceFence, None]:
        yield SourceFence(
            None, self.corpus, source_id, job_id, frozenset({source_id, *dependency_sources})
        )

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
        current = self.active.get((collection, source_id))
        if (current.revision if current else None) != expected_active_revision:
            raise StalePublicationError("active pointer changed")
        if job_id in self.jobs:
            return self.jobs[job_id]
        revision = self.next_revision.get(source_id, 0) + 1
        self.next_revision[source_id] = revision
        job = SourcePublicationJob(
            job_id,
            self.corpus,
            collection,
            source_id,
            revision,
            expected_active_revision,
            source_sha256,
            PublicationState.PREPARING,
            {},
            None,
            dict(recipe),
        )
        self.jobs[job_id] = job
        return job

    async def checkpoint(
        self, _fence: SourceFence, job_id: str, state: PublicationState, data: Mapping[str, Any]
    ) -> SourcePublicationJob:
        job = replace(self.jobs[job_id], state=state, checkpoint=dict(data))
        self.jobs[job_id] = job
        return job

    async def publish(self, _fence: SourceFence, job_id: str) -> ActivePublication:
        job = self.jobs[job_id]
        pointer = ActivePublication(
            self.corpus, job.collection, job.source_id, job.revision, job.source_sha256, job.job_id
        )
        self.active[(job.collection, job.source_id)] = pointer
        self.jobs[job_id] = replace(job, state=PublicationState.PUBLISHED)
        return pointer

    async def read_active(self, collection: str, source_id: str) -> ActivePublication | None:
        if collection == _RAW and source_id == self.raw_source and self.advance_raw_after_raw_reads:
            self.raw_read_count += 1
            if self.raw_read_count == self.advance_raw_after_raw_reads:
                pointer = self.active[(collection, source_id)]
                assert pointer.revision is not None
                self.active[(collection, source_id)] = replace(
                    pointer, revision=pointer.revision + 1
                )
        return self.active.get((collection, source_id))

    async def list_active(self, collection: str) -> dict[str, int | None]:
        self.list_active_calls.append(collection)
        if collection != _RAW and self.advance_raw_when_listing_derived:
            pointer = self.active[(_RAW, self.raw_source)]
            assert pointer.revision is not None
            self.active[(_RAW, self.raw_source)] = replace(pointer, revision=pointer.revision + 1)
            self.advance_raw_when_listing_derived = False
        return {
            source: pointer.revision
            for (name, source), pointer in self.active.items()
            if name == collection
        }


class _CancelAfterFirstUpsert:
    """Apply one Qdrant batch, then cancel to model a timeout after acknowledgement."""

    def __init__(self, raw: AsyncQdrantClient, collection: str) -> None:
        self.raw = raw
        self.collection = collection
        self.cancel_next = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)

    async def upsert(self, *, collection_name: str, **kwargs: Any) -> Any:
        result = await self.raw.upsert(collection_name=collection_name, **kwargs)
        if collection_name == self.collection and self.cancel_next:
            self.cancel_next = False
            raise asyncio.CancelledError
        return result


class _MutationGuard:
    """Fail-visible sentinel proving identical warning writes stay no-op."""

    def __init__(self) -> None:
        self.calls = 0

    async def set_payload(self, **_kwargs: Any) -> None:
        self.calls += 1
        raise AssertionError("identical quality payload should not create another revision")


class _ScrollRecorder:
    """Record scroll filters while keeping reads and writes on local Qdrant."""

    def __init__(self, raw: AsyncQdrantClient) -> None:
        self.raw = raw
        self.scrolls: list[tuple[str, str]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)

    async def scroll(self, *, collection_name: str, **kwargs: Any) -> Any:
        condition = kwargs.get("scroll_filter")
        serialized = str(condition.model_dump(mode="json")) if condition is not None else ""
        self.scrolls.append((collection_name, serialized))
        return await self.raw.scroll(collection_name=collection_name, **kwargs)


class _OversizedCollectionClient(_ScrollRecorder):
    """Return an exact over-limit count for one collection before any scroll."""

    def __init__(self, raw: AsyncQdrantClient, collection: str) -> None:
        super().__init__(raw)
        self.collection = collection

    async def count(
        self,
        *,
        collection_name: str,
        count_filter: Any = None,
        exact: bool = True,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> Any:
        if collection_name == self.collection and count_filter is None:
            assert exact is True
            assert timeout == 2
            return SimpleNamespace(count=100_001)
        return await self.raw.count(
            collection_name=collection_name,
            count_filter=count_filter,
            exact=exact,
            timeout=timeout,
            **kwargs,
        )


@pytest.fixture
async def local_qdrant() -> AsyncGenerator[tuple[AsyncQdrantClient, _Coordinator], None]:
    client = AsyncQdrantClient(location=":memory:")
    coordinator = _Coordinator()
    for collection in _COLLECTIONS:
        await client.create_collection(
            collection_name=collection,
            vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
    try:
        yield client, coordinator
    finally:
        await client.close()


def _points(source_id: str, dependency: Mapping[str, int], *, content: str) -> list[PointStruct]:
    return [
        PointStruct(
            id=f"{source_id}-base-{index}",
            vector={
                "dense": [1.0, float(index + 1)],
                "sparse": SparseVector(indices=[1, 2], values=[0.5, 0.25]),
            },
            payload={
                "source_id": source_id,
                "publication_base_id": f"{source_id}-base-{index}",
                "chunk_index": index,
                "content": content,
                "domain_tags": ["fixture-tag"],
                "embedding_provenance": {"model": "fixture"},
                "dependency_revisions": dict(dependency),
                "dependency_revision_keys": {
                    hashlib.sha256(key.encode()).hexdigest(): value
                    for key, value in dependency.items()
                },
            },
        )
        for index in range(2)
    ]


async def _seed(
    client: AsyncQdrantClient,
    coordinator: _Coordinator,
    collection: str,
    source_id: str,
    points: list[PointStruct],
    revision: int,
) -> None:
    staged = revision_points(
        points, corpus="books", collection=collection, source_id=source_id, revision=revision
    )
    await client.upsert(collection_name=collection, points=staged, wait=True)
    coordinator.active[(collection, source_id)] = ActivePublication(
        "books",
        collection,
        source_id,
        revision,
        canonical_sha256(source_id),
        f"seed-{collection}-{source_id}",
    )
    coordinator.next_revision[source_id] = max(
        coordinator.next_revision.get(source_id, 0), revision
    )


async def _quality_raw_revision(
    client: AsyncQdrantClient,
    coordinator: _Coordinator,
    source_id: str,
    *,
    content: str = "raw usable text",
    vector_changed: bool = False,
    tag_changed: bool = False,
) -> None:
    rows, _ = await client.scroll(
        collection_name=_RAW,
        scroll_filter=None,
        limit=100,
        with_payload=True,
        with_vectors=True,
    )
    old = [row for row in rows if (row.payload or {}).get("source_id") == source_id]
    current = coordinator.active[(_RAW, source_id)]
    assert current.revision is not None
    points: list[PointStruct] = []
    for row in old:
        payload = dict(row.payload or {})
        payload.pop("source_revision", None)
        payload["content"] = content
        payload["quality_warnings"] = [{"stage": "parse", "code": "coverage"}]
        payload["quality_disposition"] = "warning"
        if tag_changed:
            payload["domain_tags"] = ["changed-tag"]
        vector = dict(cast(Any, row.vector or {}))
        if vector_changed:
            vector["dense"] = [0.0, 1.0]
        points.append(PointStruct(id=str(row.id), vector=cast(Any, vector), payload=payload))
    revision = current.revision + 1
    staged = revision_points(
        points, corpus="books", collection=_RAW, source_id=source_id, revision=revision
    )
    await client.upsert(collection_name=_RAW, points=staged, wait=True)
    coordinator.active[(_RAW, source_id)] = replace(current, revision=revision)
    coordinator.next_revision[source_id] = max(coordinator.next_revision[source_id], revision)


async def test_copy_forward_makes_explicit_dependents_current_and_retains_old_points(
    local_qdrant: tuple[AsyncQdrantClient, _Coordinator],
) -> None:
    client, coordinator = local_qdrant
    raw_source = "raw-" + uuid4().hex
    sterile_source = "sterile-" + uuid4().hex
    summary_source = "summary-" + uuid4().hex
    outside_source = "outside-" + uuid4().hex
    unrelated_derived_source = "unrelated-" + uuid4().hex
    unregistered_graph_source = "shared-graph-" + uuid4().hex
    await _seed(
        client, coordinator, _RAW, raw_source, _points(raw_source, {}, content="raw usable text"), 1
    )
    await _seed(
        client, coordinator, _RAW, outside_source, _points(outside_source, {}, content="outside"), 1
    )
    dependency = {raw_source: 1}
    await _seed(
        client,
        coordinator,
        _STERILE,
        sterile_source,
        _points(sterile_source, dependency, content="derived usable text"),
        1,
    )
    graph_points = revision_points(
        _points(unregistered_graph_source, dependency, content="shared graph node"),
        corpus="books",
        collection=_GRAPH_NODES,
        source_id=unregistered_graph_source,
        revision=1,
    )
    await client.upsert(collection_name=_GRAPH_NODES, points=graph_points, wait=True)
    await _seed(
        client,
        coordinator,
        _SUMMARY,
        summary_source,
        _points(summary_source, dependency, content="summary text"),
        1,
    )
    await _seed(
        client,
        coordinator,
        _STERILE,
        unrelated_derived_source,
        _points(unrelated_derived_source, {"other-canonical-source": 1}, content="unrelated"),
        1,
    )
    old_derived = {
        collection: await SourceRevisionWriter(client, cast(Any, coordinator)).census(
            collection, source, 1
        )
        for collection, source in ((_STERILE, sterile_source), (_SUMMARY, summary_source))
    }
    await _quality_raw_revision(client, coordinator, raw_source)
    coordinator.list_active_calls.clear()
    scroll_spy = _ScrollRecorder(client)

    outcome = await copy_forward_quality_dependencies(
        scroll_spy, cast(Any, coordinator), raw_source, 1, 2
    )

    assert outcome == {_STERILE: [sterile_source], _SUMMARY: [summary_source]}
    assert coordinator.list_active_calls == []
    assert coordinator.active[(_STERILE, unrelated_derived_source)].revision == 1
    assert (_GRAPH_NODES, unregistered_graph_source) not in coordinator.active
    assert all(unrelated_derived_source not in query for _, query in scroll_spy.scrolls)
    assert any(
        "dependency_revision_keys" in query
        for collection, query in scroll_spy.scrolls
        if collection == _STERILE
    )
    active_raw = {raw_source: 2}
    for collection, source in ((_STERILE, sterile_source), (_SUMMARY, summary_source)):
        current = coordinator.active[(collection, source)].revision
        current_rows = await SourceRevisionWriter(client, cast(Any, coordinator)).census(
            collection, source, current
        )
        assert len(current_rows) == len(old_derived[collection]) == 2
        assert all(dependencies_visible(row.payload or {}, active_raw) for row in current_rows)
        old_by_base = {
            (row.payload or {})["publication_base_id"]: row for row in old_derived[collection]
        }
        new_by_base = {(row.payload or {})["publication_base_id"]: row for row in current_rows}
        assert set(new_by_base) == set(old_by_base)
        for base_id, old_row in old_by_base.items():
            new_row = new_by_base[base_id]
            old_vectors = cast(dict[str, Any], old_row.vector)
            new_vectors = cast(dict[str, Any], new_row.vector)
            assert set(new_vectors) == set(old_vectors)
            for name, old_vector in old_vectors.items():
                new_vector = new_vectors[name]
                if isinstance(old_vector, SparseVector):
                    assert isinstance(new_vector, SparseVector)
                    assert new_vector.indices == old_vector.indices
                    assert new_vector.values == pytest.approx(old_vector.values)
                else:
                    assert new_vector == pytest.approx(old_vector, abs=2e-6)
            old_payload = dict(old_row.payload or {})
            new_payload = dict(new_row.payload or {})
            assert new_payload["dependency_revisions"] == {raw_source: 2}
            assert (
                new_payload["dependency_revision_keys"] != old_payload["dependency_revision_keys"]
            )
            for payload in (old_payload, new_payload):
                for key in ("source_revision", "dependency_revisions", "dependency_revision_keys"):
                    payload.pop(key, None)
            assert new_payload == old_payload
        assert len(
            await SourceRevisionWriter(client, cast(Any, coordinator)).census(collection, source, 1)
        )
    outside = await SourceRevisionWriter(client, cast(Any, coordinator)).census(
        _RAW, outside_source, 1
    )
    assert len(outside) == 2
    graph_rows, _ = await client.scroll(
        collection_name=_GRAPH_NODES,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="source_id", match=MatchValue(value=unregistered_graph_source))
            ]
        ),
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    assert len(graph_rows) == 2
    assert {row.payload["source_revision"] for row in graph_rows if row.payload} == {1}


@pytest.mark.parametrize("drift", ["content", "vector", "tag"])
async def test_non_quality_raw_drift_rejects_rebase_before_derived_writes(
    local_qdrant: tuple[AsyncQdrantClient, _Coordinator], drift: str
) -> None:
    client, coordinator = local_qdrant
    raw_source = "raw-" + uuid4().hex
    derived_source = "derived-" + uuid4().hex
    await _seed(
        client, coordinator, _RAW, raw_source, _points(raw_source, {}, content="raw usable text"), 1
    )
    await _seed(
        client,
        coordinator,
        _STERILE,
        derived_source,
        _points(derived_source, {raw_source: 1}, content="derived usable text"),
        1,
    )
    await _quality_raw_revision(
        client,
        coordinator,
        raw_source,
        content="changed content" if drift == "content" else "raw usable text",
        vector_changed=drift == "vector",
        tag_changed=drift == "tag",
    )

    with pytest.raises(ValueError, match="raw revisions differ|changed vectors"):
        await copy_forward_quality_dependencies(client, cast(Any, coordinator), raw_source, 1, 2)
    assert coordinator.active[(_STERILE, derived_source)].revision == 1
    assert not coordinator.jobs


async def test_raw_revision_race_rejects_copy_forward_before_dependent_write(
    local_qdrant: tuple[AsyncQdrantClient, _Coordinator],
) -> None:
    client, coordinator = local_qdrant
    raw_source = "raw-" + uuid4().hex
    derived_source = "derived-" + uuid4().hex
    await _seed(
        client, coordinator, _RAW, raw_source, _points(raw_source, {}, content="raw usable text"), 1
    )
    await _seed(
        client,
        coordinator,
        _STERILE,
        derived_source,
        _points(derived_source, {raw_source: 1}, content="derived usable text"),
        1,
    )
    await _quality_raw_revision(client, coordinator, raw_source)
    coordinator.raw_source = raw_source
    coordinator.advance_raw_after_raw_reads = 2

    with pytest.raises(StalePublicationError, match="dependency advanced"):
        await copy_forward_quality_dependencies(client, cast(Any, coordinator), raw_source, 1, 2)
    assert coordinator.active[(_STERILE, derived_source)].revision == 1


async def test_collection_census_ceiling_rejects_before_filtered_scan_or_write(
    local_qdrant: tuple[AsyncQdrantClient, _Coordinator],
) -> None:
    client, coordinator = local_qdrant
    raw_source = "raw-" + uuid4().hex
    derived_source = "unrelated-" + uuid4().hex
    await _seed(
        client,
        coordinator,
        _RAW,
        raw_source,
        _points(raw_source, {}, content="raw usable text"),
        1,
    )
    await _seed(
        client,
        coordinator,
        _STERILE,
        derived_source,
        _points(derived_source, {"different-raw-source": 4}, content="unrelated"),
        1,
    )
    await _quality_raw_revision(client, coordinator, raw_source)
    bounded = _OversizedCollectionClient(client, _STERILE)

    with pytest.raises(ValueError, match="100000-point ceiling"):
        await copy_forward_quality_dependencies(bounded, cast(Any, coordinator), raw_source, 1, 2)

    assert not any(collection == _STERILE for collection, _ in bounded.scrolls)
    assert coordinator.active[(_STERILE, derived_source)].revision == 1
    assert coordinator.jobs == {}


async def test_repeated_warning_resumes_cancelled_copy_forward_from_retained_source(
    local_qdrant: tuple[AsyncQdrantClient, _Coordinator],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, coordinator = local_qdrant
    raw_source = "raw-" + uuid4().hex
    derived_source = "derived-" + uuid4().hex
    await _seed(
        client,
        coordinator,
        _RAW,
        raw_source,
        _points(raw_source, {}, content="raw usable text"),
        1,
    )
    await _seed(
        client,
        coordinator,
        _STERILE,
        derived_source,
        _points(derived_source, {raw_source: 1}, content="derived usable text"),
        1,
    )
    await _quality_raw_revision(client, coordinator, raw_source)
    interrupted_client = _CancelAfterFirstUpsert(client, _STERILE)

    with pytest.raises(asyncio.CancelledError):
        await copy_forward_quality_dependencies(
            interrupted_client, cast(Any, coordinator), raw_source, 1, 2
        )

    job_id, partial_job = next(iter(coordinator.jobs.items()))
    assert partial_job.state is PublicationState.PARTIAL_PRESERVED
    assert partial_job.checkpoint["owner"] == "source_revision_writer"
    assert partial_job.checkpoint["recovery"] == (
        "resume identical job and writer output; preserve active pointer"
    )
    assert partial_job.recipe["owner"] == "source_quality_dependency_copy_forward"
    assert "re-read retained active roster" in partial_job.recipe["producer_recovery"]
    assert "writer_intent" in partial_job.checkpoint
    assert coordinator.active[(_STERILE, derived_source)].revision == 1
    dep_key = hashlib.sha256(raw_source.encode()).hexdigest()
    discovered, _ = await client.scroll(
        collection_name=_STERILE,
        scroll_filter=Filter(
            must=[FieldCondition(key=f"dependency_revision_keys.{dep_key}", range=Range(lt=2))]
        ),
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    assert len(discovered) >= 2

    # The callable operator recovery method finds the retained stale roster,
    # resumes its deterministic writer intent, and then an identical warning is
    # a no-op.
    from vo_rag.core.storage.qdrant import QdrantStore

    monkeypatch.setenv("VORAG_SOURCE_PUBLICATION_ENABLED", "true")
    monkeypatch.setenv("VORAG_SOURCE_PUBLICATION_WRITES_ENABLED", "true")
    store = QdrantStore("http://qdrant.invalid", prefer_grpc=False)
    unused_client = store._raw_client
    guard = _MutationGuard()
    store._raw_client = client
    store._client = guard

    async def coordinator_for(_collection: str) -> _Coordinator:
        return coordinator

    monkeypatch.setattr(store, "_source_coordinator", coordinator_for)
    warnings = [{"stage": "parse", "code": "coverage"}]
    try:
        recovered = await store.resume_source_quality_dependencies(raw_source)
        await store.set_source_quality_warnings(_RAW, raw_source, warnings)
    finally:
        await unused_client.close()

    assert recovered == {_STERILE: [derived_source]}
    assert guard.calls == 0
    assert coordinator.jobs[job_id].state is PublicationState.PUBLISHED
    assert coordinator.active[(_STERILE, derived_source)].revision == 2
    assert coordinator.active[(_RAW, raw_source)].revision == 2
