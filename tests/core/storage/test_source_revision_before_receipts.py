"""Regression tests for full before-revision payload and vector receipts."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, cast
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from vo_rag.core.ingest.representation_receipt import representation_receipt
from vo_rag.core.ingest.source_publication import (
    PublicationState,
    SourceFence,
    SourcePublicationJob,
)
from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter


class _Coordinator:
    corpus = "books"

    def __init__(self) -> None:
        self.job: SourcePublicationJob | None = None

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
        self.job = SourcePublicationJob(
            job_id=job_id,
            corpus=self.corpus,
            collection=collection,
            source_id=source_id,
            revision=1,
            expected_active_revision=expected_active_revision,
            source_sha256=source_sha256,
            state=PublicationState.PREPARING,
            checkpoint={},
            retention_disposition=None,
            recipe=dict(recipe),
        )
        return self.job

    async def checkpoint(
        self,
        _fence: SourceFence,
        _job_id: str,
        state: PublicationState,
        data: Mapping[str, Any],
    ) -> SourcePublicationJob:
        assert self.job is not None
        self.job = replace(self.job, state=state, checkpoint=dict(data))
        return self.job

    async def read_active(self, _collection: str, _source_id: str) -> None:
        return None


class _CountingClient:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self.client = client
        self.upsert_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    async def upsert(self, **kwargs: Any) -> Any:
        self.upsert_calls += 1
        return await self.client.upsert(**kwargs)


@pytest.mark.asyncio
async def test_changed_before_vector_with_same_payload_blocks_before_any_write() -> None:
    client = AsyncQdrantClient(location=":memory:")
    collection = "before_receipts_" + uuid4().hex
    source_id = "source-before-receipt"
    old_id = str(uuid4())
    await client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)},
    )
    initial = PointStruct(
        id=old_id,
        vector={"dense": [1.0, 0.0]},
        payload={"source_id": source_id, "content": "same payload"},
    )
    await client.upsert(collection_name=collection, points=[initial], wait=True)
    captured = representation_receipt(initial)
    raw = _CountingClient(client)
    coordinator = _Coordinator()
    writer = SourceRevisionWriter(raw, coordinator)  # type: ignore[arg-type]
    original_census = writer.census
    changed = False

    async def change_vector_before_first_census(
        target: str, selected_source: str, revision: int | None
    ) -> list[Any]:
        nonlocal changed
        if revision is None and not changed:
            changed = True
            await client.upsert(
                collection_name=collection,
                points=[initial.model_copy(update={"vector": {"dense": [0.0, 1.0]}})],
                wait=True,
            )
        return await original_census(target, selected_source, revision)

    cast(Any, writer).census = change_vector_before_first_census
    replacement = PointStruct(
        id="new-point",
        vector={"dense": [0.5, 0.5]},
        payload={"source_id": source_id, "content": "replacement"},
    )

    with pytest.raises(ValueError, match="modalities differ from the approved before receipts"):
        await writer.replace(
            collection,
            source_id,
            [replacement],
            job_id="before-receipt-test",
            source_sha256="a" * 64,
            expected_active_revision=None,
            recipe={"test": "before-receipt"},
            expected_before={old_id: captured["payload_sha256"]},
            expected_before_receipts={old_id: captured},
        )

    assert changed is True
    assert raw.upsert_calls == 0
    assert coordinator.job is not None
    assert coordinator.job.state is PublicationState.PRESERVED_NO_WRITE
    await client.close()
