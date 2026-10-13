"""Producer outputs resolve current IDs while physical citations stay strict."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from vo_rag.core.storage.publication_client import PublicationClient
from vo_rag.pipelines.book_rag.published_points import current_published_ids
from vo_rag.pipelines.book_rag.workflow import BookRagPipeline

_RAW = "vo_rag_books_raw_v1"


@pytest.mark.asyncio
async def test_retained_physical_ids_resolve_only_explicit_current_logical_roster() -> None:
    raw = AsyncQdrantClient(location=":memory:")
    await raw.create_collection(
        _RAW, vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)}
    )
    old, new, base = (str(uuid4()) for _ in range(3))
    payload = {"source_id": "owned", "publication_base_id": base, "content": "passage"}
    await raw.upsert(
        _RAW,
        points=[
            PointStruct(
                id=old, vector={"dense": [1.0, 0.0]}, payload={**payload, "source_revision": 1}
            ),
            PointStruct(
                id=new, vector={"dense": [1.0, 0.0]}, payload={**payload, "source_revision": 2}
            ),
        ],
    )

    async def pointers(corpus: str, collection: str) -> dict[str, int | None]:
        return {"owned": 2}

    client = PublicationClient(raw, pointers)
    store = SimpleNamespace(_client=client)
    try:
        assert await client.retrieve(collection_name=_RAW, ids=[old]) == []
        assert await current_published_ids(
            store=store, collection=_RAW, source_id="owned", point_ids=[old]
        ) == [new]
        with pytest.raises(ValueError, match="source ownership"):
            await current_published_ids(
                store=store, collection=_RAW, source_id="other", point_ids=[old]
            )
        await raw.delete(_RAW, points_selector=[new])
        with pytest.raises(ValueError, match="completely visible"):
            await current_published_ids(
                store=store, collection=_RAW, source_id="owned", point_ids=[old]
            )
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_warning_delivery_refreshes_context_and_owned_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.pipelines.book_rag.test_quality_availability import _context, _pipeline
    from vo_rag.pipelines.book_rag import workflow

    pipeline = _pipeline()
    ctx = _context(pipeline)
    ctx.sterile_point_ids = ["old-sterile"]
    ctx.quality_warnings = [{"stage": "parse", "code": "low_text_density"}]
    writer = AsyncMock()
    pool = SimpleNamespace(execute=AsyncMock())
    pipeline.pg_pool = pool
    pipeline.qdrant_store = SimpleNamespace(set_source_quality_warnings=writer)

    async def resolve(**kwargs: Any) -> list[str]:
        return ["new-raw" if kwargs["collection"] == _RAW else "new-sterile"]

    monkeypatch.setattr(workflow, "current_published_ids", resolve)
    await pipeline._persist_quality_warnings(ctx)
    assert ctx.raw_point_ids == ["new-raw"]
    assert ctx.sterile_point_ids == ["new-sterile"]
    assert writer.await_count == 2
    artifact_calls = [
        call for call in pool.execute.await_args_list if "knowledge_artifacts" in call.args[0]
    ]
    assert len(artifact_calls) == 1
    assert artifact_calls[0].args[2] == ["new-raw"]
    assert artifact_calls[0].args[3] == str(ctx.envelope.id)


@pytest.mark.asyncio
async def test_failed_refresh_does_not_return_unverified_stale_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.pipelines.book_rag.test_quality_availability import _context, _pipeline
    from vo_rag.pipelines.book_rag import workflow

    pipeline: BookRagPipeline = _pipeline()
    ctx = _context(pipeline)
    monkeypatch.setattr(
        workflow, "current_published_ids", AsyncMock(side_effect=ValueError("stale dependency"))
    )
    await pipeline._persist_quality_warnings(ctx)
    assert ctx.raw_point_ids == []
    assert ctx.quality_warnings[-1]["code"] == "warning_persistence_pending"
