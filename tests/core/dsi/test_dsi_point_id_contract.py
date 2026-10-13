"""Regression guard for the DSI point-id contract (VOR-210).

Ensures ``persist_dsi_records`` writes point IDs that match
``facet_record_id_for(book_id, summary_model, prompt_hash)`` and that the id
is versioned (distinct per ``prompt_hash`` and per ``summary_model``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from vo_rag.core.dsi.builder import (
    DsiBuildConfig,
    admit_dsi_source_revision,
    persist_dsi_records,
)
from vo_rag.core.dsi.extraction import ExtractionConfig
from vo_rag.core.dsi.schemas import (
    FacetedBookRecord,
    ProvenanceLinkedEntry,
    facet_record_id_for,
)
from vo_rag.core.ingest.source_publication import PublicationState, SourcePublicationJob
from vo_rag.core.storage.qdrant import deterministic_point_id


def _make_facet_record(
    *,
    book_id: str = "book-001",
    summary_model: str = "qwen3-14b",
    prompt_hash: str = "hash_abc",
) -> FacetedBookRecord:
    """Build a minimal valid ``FacetedBookRecord`` for testing."""
    entry = ProvenanceLinkedEntry(
        value="test concept",
        supporting_chunk_ids=["c1"],
        supporting_section_paths=["[Book > Chapter > Section]"],
    )
    return FacetedBookRecord(
        book_id=book_id,
        title="Test Book",
        author="Test Author",
        category_ids=["cat1"],
        short_abstract="A test book about testing.",
        technical_domains=["testing"],
        key_concepts=[entry],
        key_components=["component1"],
        key_methods=[entry],
        key_equations_or_models=[entry],
        figure_heavy_topics=[entry],
        synonyms_and_aliases=[entry],
        prerequisite_topics=["prereq1"],
        chapter_outline="1. Intro\n2. Body\n3. Conclusion",
        representative_chunk_ids=["c1"],
        source_collection_version="v1",
        summary_model=summary_model,
        prompt_hash=prompt_hash,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_provenance_linked_entry_schema_requires_non_empty_provenance_lists() -> None:
    schema = ProvenanceLinkedEntry.model_json_schema()

    assert schema["properties"]["supporting_chunk_ids"]["minItems"] == 1
    assert schema["properties"]["supporting_section_paths"]["minItems"] == 1


@pytest.mark.parametrize(
    "field_name", ["book_id", "title", "author", "summary_model", "prompt_hash"]
)
def test_faceted_book_record_schema_requires_non_empty_identity_strings(field_name: str) -> None:
    schema = FacetedBookRecord.model_json_schema()

    assert schema["properties"][field_name]["minLength"] == 1


class _FakeStore:
    """Captures ``upsert_vectors`` calls and derives the deterministic point ids."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Any]]] = []
        self.point_ids: list[str] = []

    async def upsert_vectors(self, collection: str, points: list[Any]) -> None:
        self.calls.append((collection, points))
        for point in points:
            point_id = deterministic_point_id(
                point.source_id,
                point.chunk_index,
                point.stage,
                point.version_hash,
            )
            self.point_ids.append(point_id)


class _PublishedFakeStore(_FakeStore):
    source_publication_enabled = True

    def __init__(self) -> None:
        super().__init__()
        self.admission: dict[str, Any] | None = None
        self.write_kwargs: dict[str, Any] | None = None
        self.aborted: list[tuple[Any, str]] = []

    async def admit_source_revision(
        self,
        collection: str,
        source_id: str,
        job_id: str,
        source_sha256: str,
        **kwargs: Any,
    ) -> SourcePublicationJob:
        self.admission = {
            "collection": collection,
            "source_id": source_id,
            "job_id": job_id,
            "source_sha256": source_sha256,
            **kwargs,
        }
        return SourcePublicationJob(
            job_id=job_id,
            corpus="books",
            collection=collection,
            source_id=source_id,
            revision=1,
            expected_active_revision=None,
            source_sha256=source_sha256,
            state=PublicationState.PREPARING,
            checkpoint={},
            retention_disposition=None,
            recipe={
                "expected_dependencies": dict(kwargs["expected_dependencies"]),
                "producer_recipe": dict(kwargs["producer_recipe"]),
            },
        )

    async def upsert_vectors(self, collection: str, points: list[Any], **kwargs: Any) -> None:
        self.write_kwargs = kwargs
        await super().upsert_vectors(collection, points)

    async def abort_source_revision(self, job: Any, reason: str) -> None:
        self.aborted.append((job, reason))


async def _fake_embed(texts: list[str]) -> dict[str, Any]:
    """Return dummy BGE-M3 embeddings matching the expected shape."""
    n = len(texts)
    return {
        "dense": [[1.0] * 1024 for _ in range(n)],
        "sparse": [{0: 1.0} for _ in range(n)],
        "colbert": [[[1.0] * 1024] for _ in range(n)],
    }


@pytest.mark.asyncio
async def test_persist_uses_facet_record_id_for_contract() -> None:
    """REGRESSION (VOR-210): persisted point id must equal
    ``facet_record_id_for(book_id, summary_model, prompt_hash)``.

    This test FAILS when persistence writes a bare ``stage="dsi"`` point
    (pre-VOR-209) and PASSES once ``persist_dsi_records`` preserves the
    ``summary_model`` and ``prompt_hash`` in the point id.
    """
    record = _make_facet_record(
        book_id="book-001",
        summary_model="qwen3-14b",
        prompt_hash="hash_abc",
    )
    expected = facet_record_id_for("book-001", "qwen3-14b", "hash_abc")

    store = _FakeStore()
    await persist_dsi_records(
        records=[record],
        embed=_fake_embed,
        store=store,
        collection="test_dsi",
    )

    assert len(store.calls) == 1
    _collection, points = store.calls[0]
    assert len(points) == 1
    assert store.point_ids[0] == expected


@pytest.mark.asyncio
async def test_persist_dsi_point_id_distinct_per_prompt_hash() -> None:
    """Two records for the same book with different ``prompt_hash`` must get
    distinct point ids (prompt-edit isolation)."""
    record_v1 = _make_facet_record(book_id="book-001", prompt_hash="hash_v1")
    record_v2 = _make_facet_record(book_id="book-001", prompt_hash="hash_v2")

    store = _FakeStore()
    await persist_dsi_records(
        records=[record_v1, record_v2],
        embed=_fake_embed,
        store=store,
        collection="test_dsi",
    )

    assert len(store.point_ids) == 2
    assert store.point_ids[0] != store.point_ids[1]


@pytest.mark.asyncio
async def test_persist_dsi_point_id_distinct_per_summary_model() -> None:
    """Two records for the same book with different ``summary_model`` must get
    distinct point ids (model A/B isolation, no collision)."""
    record_a = _make_facet_record(book_id="book-001", summary_model="model-a")
    record_b = _make_facet_record(book_id="book-001", summary_model="model-b")

    store = _FakeStore()
    await persist_dsi_records(
        records=[record_a, record_b],
        embed=_fake_embed,
        store=store,
        collection="test_dsi",
    )

    assert len(store.point_ids) == 2
    assert store.point_ids[0] != store.point_ids[1]


@pytest.mark.asyncio
async def test_published_dsi_admission_binds_canonical_raw_revision_before_embed() -> None:
    source_sha256 = "a" * 64
    source_id = f"vo-rag-corpus-sha256:{source_sha256}"
    collection = "vo_rag_books_dsi_v1"
    config = DsiBuildConfig(
        extraction_config=ExtractionConfig(
            summary_model="qwen3-14b",
            temperature=0.2,
            max_tokens=1000,
            n_representative_chunks=4,
            prompt_hash="hash_abc",
        )
    )
    dependencies = {source_id: 7}
    store = _PublishedFakeStore()
    job = await admit_dsi_source_revision(
        store=store,
        source_id=source_id,
        collection=collection,
        config=config,
        source_collection_version="vo_rag_books_raw_v1",
        expected_dependencies=dependencies,
    )

    assert job is not None
    assert store.admission is not None
    assert store.admission["source_sha256"] == source_sha256
    assert store.admission["expected_dependencies"] == dependencies
    assert store.admission["producer_recipe"]["summary_model"] == "qwen3-14b"

    record = _make_facet_record(book_id=source_id)
    record = record.model_copy(update={"source_collection_version": "vo_rag_books_raw_v1"})
    await persist_dsi_records(
        records=[record],
        embed=_fake_embed,
        store=store,
        collection=collection,
        publication_job=job,
        expected_dependencies=dependencies,
    )

    assert store.write_kwargs == {
        "publication_job": job,
        "expected_dependencies": dependencies,
    }
    point = store.calls[0][1][0]
    assert point.payload["dependency_revisions"] == dependencies
    assert not store.aborted


@pytest.mark.asyncio
async def test_published_dsi_requires_admission_before_embedding() -> None:
    record = _make_facet_record(book_id=f"vo-rag-corpus-sha256:{'b' * 64}")
    store = _PublishedFakeStore()
    embed_called = False

    async def embed(texts: list[str]) -> dict[str, Any]:
        nonlocal embed_called
        embed_called = True
        return await _fake_embed(texts)

    with pytest.raises(ValueError, match="admission before inference"):
        await persist_dsi_records(records=[record], embed=embed, store=store)
    assert not embed_called


@pytest.mark.asyncio
async def test_reader_first_dsi_admission_checks_legacy_scope_without_revision_job() -> None:
    class _ReaderFirstStore:
        source_publication_enabled = False
        source_visibility_enabled = True

        def __init__(self) -> None:
            self.admission: dict[str, Any] | None = None

        async def admit_source_revision(
            self,
            collection: str,
            source_id: str,
            job_id: str,
            source_sha256: str,
            **kwargs: Any,
        ) -> None:
            self.admission = {
                "collection": collection,
                "source_id": source_id,
                "job_id": job_id,
                "source_sha256": source_sha256,
                **kwargs,
            }
            return None

    source_id = f"vo-rag-corpus-sha256:{'c' * 64}"
    config = DsiBuildConfig(
        extraction_config=ExtractionConfig(
            summary_model="qwen3-14b",
            temperature=0.2,
            max_tokens=1000,
            n_representative_chunks=4,
            prompt_hash="hash_abc",
        )
    )
    store = _ReaderFirstStore()

    job = await admit_dsi_source_revision(
        store=store,
        source_id=source_id,
        collection="vo_rag_books_dsi_v1",
        config=config,
        source_collection_version="vo_rag_books_raw_v1",
        expected_dependencies={source_id: None},
    )

    assert job is None
    assert store.admission is not None
    assert store.admission["expected_dependencies"] == {source_id: None}
