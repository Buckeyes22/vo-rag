"""Derived source admission binds work to captured raw Books revisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from vo_rag.core.ingest.source_publication import (
    ActivePublication,
    PublicationState,
    SourcePublicationJob,
)
from vo_rag.core.storage.derived_publication import admit_derived_source


class _Coordinator:
    corpus = "books"

    def __init__(self, raw_revision: int | None) -> None:
        self.raw_revision = raw_revision

    async def read_active(self, collection: str, source_id: str) -> ActivePublication | None:
        if collection == "vo_rag_books_raw_v1":
            if self.raw_revision is None:
                return None
            return ActivePublication(
                "books", collection, source_id, self.raw_revision, None, "raw-job"
            )
        return None


@dataclass
class _Store:
    coordinator: _Coordinator
    source_publication_enabled: bool = True
    source_visibility_enabled: bool = True
    admitted: dict[str, Any] | None = None

    async def _source_coordinator(self, _collection: str) -> _Coordinator:
        return self.coordinator

    async def admit_source_revision(
        self,
        collection: str,
        source_id: str,
        job_id: str,
        source_sha256: str,
        *,
        expected_dependencies: Mapping[str, int | None],
        producer_recipe: Mapping[str, Any],
    ) -> SourcePublicationJob | None:
        self.admitted = {
            "collection": collection,
            "source_id": source_id,
            "job_id": job_id,
            "source_sha256": source_sha256,
            "expected_dependencies": dict(expected_dependencies),
            "producer_recipe": dict(producer_recipe),
        }
        if not self.source_publication_enabled:
            return None
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
                "expected_dependencies": dict(expected_dependencies),
                "producer_recipe": dict(producer_recipe),
            },
        )


@pytest.mark.asyncio
async def test_admission_captures_raw_pointer_and_honestly_hashes_derived_inputs() -> None:
    store = _Store(_Coordinator(raw_revision=9))
    binding = await admit_derived_source(
        store,  # type: ignore[arg-type]
        collection="vo_rag_books_summary_tree_v1",
        source_id="book-1",
        dependency_sources=["book-1"],
        expected_dependencies={"book-1": 9},
        producer="summary-tree",
        input_manifest={"chunk_ids": ["c1", "c2"], "input_sha256": "a" * 64},
        producer_recipe={"summary_model": "configured-model", "prompt_hash": "b" * 64},
    )

    assert binding is not None
    assert binding.dependencies == {"book-1": 9}
    assert store.admitted is not None
    assert store.admitted["expected_dependencies"] == {"book-1": 9}
    assert store.admitted["source_sha256"] == binding.job.source_sha256
    recipe = store.admitted["producer_recipe"]
    assert recipe["identity_kind"] == "derived_input_recipe_sha256"
    assert recipe["raw_source_byte_lineage"] == "unknown"
    assert recipe["dependency_revisions"] == {"book-1": 9}


@pytest.mark.asyncio
async def test_admission_preserves_explicit_legacy_dependency_null() -> None:
    store = _Store(_Coordinator(raw_revision=None))
    binding = await admit_derived_source(
        store,  # type: ignore[arg-type]
        collection="vo_rag_books_raptor_v1",
        source_id="book-legacy",
        dependency_sources=["book-legacy"],
        expected_dependencies={"book-legacy": None},
        producer="raptor",
        input_manifest={"chunks_sha256": "c" * 64},
        producer_recipe={"embedding_model": "configured-model"},
    )

    assert binding is not None
    assert binding.dependencies == {"book-legacy": None}
    assert store.admitted is not None
    assert store.admitted["expected_dependencies"] == {"book-legacy": None}


@pytest.mark.asyncio
async def test_disabled_publication_returns_without_coordinator_access() -> None:
    store = _Store(
        _Coordinator(raw_revision=1),
        source_publication_enabled=False,
        source_visibility_enabled=False,
    )

    binding = await admit_derived_source(
        store,  # type: ignore[arg-type]
        collection="vo_rag_books_summary_tree_v1",
        source_id="book-1",
        dependency_sources=["book-1"],
        expected_dependencies={"book-1": 1},
        producer="summary-tree",
        input_manifest={"input": "stable"},
        producer_recipe={"prompt": "v1"},
    )

    assert binding is None
    assert store.admitted is None


@pytest.mark.asyncio
async def test_reader_first_admission_runs_legacy_guard_then_returns_no_revision_job() -> None:
    store = _Store(_Coordinator(raw_revision=None), source_publication_enabled=False)

    binding = await admit_derived_source(
        store,  # type: ignore[arg-type]
        collection="vo_rag_books_summary_tree_v1",
        source_id="book-legacy",
        dependency_sources=["book-legacy"],
        expected_dependencies={"book-legacy": None},
        producer="summary-tree",
        input_manifest={"chunks_sha256": "c" * 64},
        producer_recipe={"embedding_model": "configured-model"},
    )

    assert binding is None
    assert store.admitted is not None
    assert store.admitted["expected_dependencies"] == {"book-legacy": None}
