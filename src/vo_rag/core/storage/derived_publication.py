"""Bind derived Qdrant representations to the raw Books revisions they used."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client.models import PointStruct

from vo_rag.core.idempotency import deterministic_uuid
from vo_rag.core.ingest.representation_receipt import canonical_sha256
from vo_rag.core.ingest.source_publication import SourceCoordinator, SourcePublicationJob
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter

RAW_BOOKS_COLLECTION = "vo_rag_books_raw_v1"


@dataclass(frozen=True, slots=True)
class DerivedSourceBinding:
    """Pre-model identity and dependency snapshot for one derived source job."""

    coordinator: SourceCoordinator
    job: SourcePublicationJob
    dependencies: Mapping[str, int | None]
    collection: str
    source_id: str


async def admit_derived_source(
    store: QdrantStore,
    *,
    collection: str,
    source_id: str,
    dependency_sources: Sequence[str],
    expected_dependencies: Mapping[str, int | None],
    producer: str,
    input_manifest: Mapping[str, Any],
    producer_recipe: Mapping[str, Any],
) -> DerivedSourceBinding | None:
    """Capture raw dependency revisions and durably admit derived work pre-model.

    ``input_manifest`` must describe the exact source records and inputs the
    producer will consume. The resulting SHA is a derived-input identity, not
    a claim about original source bytes. The source writer rechecks every raw
    pointer under the same sorted advisory fences immediately before staging.
    """
    if not getattr(store, "source_visibility_enabled", store.source_publication_enabled):
        return None
    if not producer or not source_id or not dependency_sources:
        raise ValueError("derived source admission requires producer, source and dependencies")
    dependencies_requested = tuple(sorted(set(dependency_sources)))
    if len(dependencies_requested) != len(dependency_sources):
        raise ValueError("derived source dependencies must be unique")
    coordinator = await store._source_coordinator(collection)
    if coordinator is None or coordinator.corpus != "books":
        raise ValueError("derived source publication requires the Books coordinator")

    dependencies = dict(expected_dependencies)
    if set(dependencies) != set(dependencies_requested):
        raise ValueError("derived inputs require an exact pre-read dependency revision token")
    if any(
        revision is not None and (type(revision) is not int or revision < 1)
        for revision in dependencies.values()
    ):
        raise ValueError("derived dependency revisions must be positive integers or explicit null")

    recipe: dict[str, Any] = {
        "identity_kind": "derived_input_recipe_sha256",
        "raw_source_byte_lineage": "unknown",
        "producer": producer,
        "input_manifest": dict(input_manifest),
        "input_manifest_sha256": canonical_sha256(dict(input_manifest)),
        "dependency_revisions": dependencies,
        "producer_recipe": dict(producer_recipe),
    }
    source_sha256 = canonical_sha256(recipe)
    job_id = deterministic_uuid(
        "derived-source-publication-v1",
        coordinator.corpus,
        collection,
        source_id,
        source_sha256,
    )
    job = await store.admit_source_revision(
        collection,
        source_id,
        job_id,
        source_sha256,
        expected_dependencies=dependencies,
        producer_recipe=recipe,
    )
    if job is None:
        if not getattr(store, "source_publication_enabled", False):
            # Reader-first release: admission above performs the pre-model
            # source/dependency fence check, while persistence stays on the
            # stable legacy IDs until revision writes are enabled.
            return None
        raise RuntimeError("publication admission unexpectedly returned no job")
    return DerivedSourceBinding(coordinator, job, dependencies, collection, source_id)


async def publish_derived_source(
    store: QdrantStore,
    binding: DerivedSourceBinding,
    points: Sequence[PointStruct],
    *,
    expected_before_receipts: Mapping[str, Any] | None = None,
) -> list[str]:
    """Publish one complete derived source roster with exact dependency tags."""
    if not points:
        raise ValueError("derived source publication requires a complete non-empty roster")
    staged_inputs: list[PointStruct] = []
    base_ids = [
        str((point.payload or {}).get("publication_base_id") or point.id) for point in points
    ]
    if len(set(base_ids)) != len(base_ids):
        raise ValueError("derived source output contains duplicate logical point IDs")
    for point in points:
        payload = dict(point.payload or {})
        if payload.get("source_id") != binding.source_id:
            raise ValueError("derived point source scope differs from its admission")
        if "source_revision" in payload:
            raise ValueError("derived producer cannot choose its own source revision")
        payload["dependency_revisions"] = dict(binding.dependencies)
        staged_inputs.append(point.model_copy(update={"payload": payload}))

    return await SourceRevisionWriter(store._raw_client, binding.coordinator).replace(
        binding.collection,
        binding.source_id,
        staged_inputs,
        job_id=binding.job.job_id,
        source_sha256=binding.job.source_sha256,
        expected_active_revision=binding.job.expected_active_revision,
        recipe=dict(binding.job.recipe),
        expected_before_receipts=expected_before_receipts,
        expected_dependencies=binding.dependencies,
    )


async def abort_derived_source(
    store: QdrantStore, binding: DerivedSourceBinding, reason: str
) -> None:
    """Close only an admission proven to have made no point write."""
    await store.abort_source_revision(binding.job, reason)


__all__ = [
    "DerivedSourceBinding",
    "RAW_BOOKS_COLLECTION",
    "abort_derived_source",
    "admit_derived_source",
    "publish_derived_source",
]
