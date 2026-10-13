"""Reader-first rollout keeps visibility enforcement separate from revision writes."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient

from vo_rag.core.storage import qdrant as qdrant_module
from vo_rag.core.storage.client_factory import build_qdrant_client
from vo_rag.core.storage.publication_client import PublicationClient
from vo_rag.core.storage.qdrant import QdrantStore


@pytest.fixture(autouse=True)
def local_raw_client(monkeypatch):  # noqa: ANN001, ANN201
    monkeypatch.setattr(
        qdrant_module, "AsyncQdrantClient", lambda **kwargs: AsyncQdrantClient(location=":memory:")
    )


@pytest.mark.asyncio
async def test_compatibility_mode_filters_reads_and_admits_only_unmanaged_sources(monkeypatch):  # noqa: ANN001, ANN201
    monkeypatch.setenv("VORAG_SOURCE_PUBLICATION_ENABLED", "true")
    monkeypatch.setenv("VORAG_SOURCE_PUBLICATION_WRITES_ENABLED", "false")
    store = QdrantStore(":memory:", prefer_grpc=False)
    assert store.source_visibility_enabled
    assert not store.source_publication_enabled
    assert isinstance(store._client, PublicationClient)  # noqa: SLF001
    coordinator = SimpleNamespace(corpus="books", assert_legacy_write_allowed=AsyncMock())

    @asynccontextmanager
    async def fence(source, **kwargs):  # noqa: ANN001, ANN201
        assert source == "book"
        assert kwargs["dependency_sources"] == ("raw-book",)
        yield "held"

    coordinator.fence = fence
    monkeypatch.setattr(store, "_source_coordinator", AsyncMock(return_value=coordinator))
    try:
        job = await store.admit_source_revision(
            "vo_rag_books_summary_v1",
            "book",
            "job",
            "a" * 64,
            expected_dependencies={"raw-book": None},
        )
        assert job is None
        coordinator.assert_legacy_write_allowed.assert_awaited_once_with(
            "held", "book", ("raw-book",)
        )
        coordinator.assert_legacy_write_allowed.side_effect = ValueError("managed source")
        with pytest.raises(ValueError, match="managed source"):
            await store.admit_source_revision(
                "vo_rag_books_summary_v1",
                "book",
                "job",
                "a" * 64,
                expected_dependencies={"raw-book": None},
            )
    finally:
        await store._raw_client.close()  # noqa: SLF001


@pytest.mark.parametrize("factory", [QdrantStore, build_qdrant_client])
def test_revision_writes_require_compatible_readers(monkeypatch, factory):  # noqa: ANN001, ANN201
    monkeypatch.setenv("VORAG_SOURCE_PUBLICATION_ENABLED", "false")
    monkeypatch.setenv("VORAG_SOURCE_PUBLICATION_WRITES_ENABLED", "true")
    with pytest.raises(ValueError, match="require publication readers"):
        factory(":memory:")


@pytest.mark.asyncio
@pytest.mark.parametrize("dependencies", [{"raw": True}, {"raw": 0}, {"raw": -1}, {"": 1}])
async def test_admission_rejects_malformed_dependencies_before_fence(monkeypatch, dependencies):  # noqa: ANN001, ANN201
    monkeypatch.setenv("VORAG_SOURCE_PUBLICATION_ENABLED", "true")
    monkeypatch.setenv("VORAG_SOURCE_PUBLICATION_WRITES_ENABLED", "true")
    store = QdrantStore(":memory:", prefer_grpc=False)
    coordinator = SimpleNamespace(corpus="books", fence=AsyncMock())
    monkeypatch.setattr(store, "_source_coordinator", AsyncMock(return_value=coordinator))
    try:
        with pytest.raises(ValueError, match="positive integers"):
            await store.admit_source_revision(
                "vo_rag_books_summary_v1",
                "book",
                "job",
                "a" * 64,
                expected_dependencies=dependencies,
            )
        coordinator.fence.assert_not_called()
    finally:
        await store._raw_client.close()  # noqa: SLF001
