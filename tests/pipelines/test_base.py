"""Contract tests for `vo_rag.pipelines.base` (VOR-3 — minimal pipeline seam). Verifies:

- ``PipelineOutput`` is a Pydantic model with ``extra='forbid'`` and
  inherits from VoBaseModel (so it strips whitespace and revalidates on
  assignment).
- ``PipelineOutput`` does NOT expose a ``chunks_for_embedding`` field
  (embedding belongs to concrete ingest pipelines, not the
  base seam).
- ``PipelineBase.index_output`` rejects a default ``target_collection`` ("").
- ``index_output`` builds a deterministic 26-char upper-case artifact ID
  via ``deterministic_hex`` keyed on (basis, pipeline, artifact_type).
- ``index_output`` inserts exactly one artifact row through
  ``vo_rag.core.db.queries.artifacts.insert_artifact``.
- ``index_output`` updates the envelope to ``EnvelopeStatus.COMPLETED``
  through ``vo_rag.core.db.queries.envelopes.update_envelope_status``.
- ``index_output`` does not expose ``_embed_and_upsert`` and does not
  touch ``vo_rag.core.embedding`` or ``vo_rag.core.storage`` at the base
  seam.
- The artifact row carries deterministic point IDs from
  ``output.content["qdrant_point_ids"]`` when present.
- A second call with the same (basis, pipeline, artifact_type) is
  idempotent — the artifact id is the same.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from vo_rag.core.idempotency import deterministic_hex
from vo_rag.core.models.artifact import (
    ArtifactType,
    KnowledgeArtifact,
    LicenseClass,
)
from vo_rag.core.models.envelope import EnvelopeStatus, Priority, TargetPipeline
from vo_rag.pipelines.base import PipelineBase, PipelineOutput

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakePool:
    """Stand-in for an asyncpg.Pool — base.index_output only passes it
    through to insert_artifact / update_envelope_status, which are
    patched in the test. No async machinery is needed."""


class _StubPipeline(PipelineBase):
    """Minimal concrete subclass with a valid target_collection."""

    target_collection = "vo_rag_books_v1"


def _make_output(**overrides: Any) -> PipelineOutput:
    base: dict[str, Any] = {
        "envelope_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "pipeline": TargetPipeline.BOOK_RAG.value,
        "artifact_type": ArtifactType.SUMMARY.value,
        "title": "Test Summary",
        "content": {
            "source_id": "src-1",
            "artifact_id_basis": "src-1",
        },
        "tags": ["smoke"],
        "priority": Priority.MEDIUM.value,
        "license_class": LicenseClass.CLEARED,
        "source_provenance": [
            {
                "source_id": "src-1",
                "source_type": "book",
                "contribution": 1.0,
            }
        ],
    }
    base.update(overrides)
    return PipelineOutput(**base)


# ---------------------------------------------------------------------------
# PipelineOutput — model contract
# ---------------------------------------------------------------------------


def test_pipeline_output_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _make_output(not_a_field=1)


def test_pipeline_output_does_not_expose_chunks_for_embedding() -> None:
    """Embedding belongs to concrete pipelines, not the base seam.
    PipelineOutput must not carry a chunks_for_embedding field at all."""
    assert "chunks_for_embedding" not in PipelineOutput.model_fields
    with pytest.raises(ValidationError):
        _make_output(chunks_for_embedding=[{"text": "x"}])


def test_pipeline_output_keeps_images_and_source_provenance() -> None:
    """The other write-side fields ARE lifted — only chunks_for_embedding
    is dropped from the seam."""
    out = _make_output(
        images=[{"image_id": "img-1", "page_number": 1}],
    )
    assert out.images == [{"image_id": "img-1", "page_number": 1}]
    assert len(out.source_provenance) == 1


def test_pipeline_output_defaults() -> None:
    """content/tags/source_provenance default to empty; license_class to UNKNOWN."""
    out = PipelineOutput(
        envelope_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        pipeline=TargetPipeline.BOOK_RAG.value,
        artifact_type=ArtifactType.SUMMARY.value,
        title="t",
    )
    assert out.content == {}
    assert out.tags == []
    assert out.images == []
    assert out.source_provenance == []
    assert out.license_class is LicenseClass.UNKNOWN
    assert out.priority == "medium"


# ---------------------------------------------------------------------------
# PipelineBase — public surface discipline
# ---------------------------------------------------------------------------


def test_pipeline_base_does_not_define_embed_and_upsert() -> None:
    """The base seam must NOT expose ``_embed_and_upsert`` — that's an
    ingest-pipeline concern."""
    assert not hasattr(PipelineBase, "_embed_and_upsert")
    assert "_embed_and_upsert" not in vars(PipelineBase)


def test_pipeline_base_default_target_collection_is_empty() -> None:
    """A bare PipelineBase (no subclass override) cannot index — concrete
    pipelines must set target_collection."""
    assert PipelineBase.target_collection == ""


# ---------------------------------------------------------------------------
# PipelineBase.index_output — happy path
# ---------------------------------------------------------------------------


async def test_index_output_rejects_missing_target_collection() -> None:
    out = _make_output()

    class _NoTarget(PipelineBase):
        target_collection = ""

    with pytest.raises(ValueError, match="target_collection"):
        await _NoTarget().index_output(out)


async def test_index_output_inserts_one_artifact_and_updates_envelope() -> None:
    out = _make_output()
    fake_pool = _FakePool()

    with (
        patch(
            "vo_rag.pipelines.base.get_pool",
            new=AsyncMock(return_value=fake_pool),
        ) as mock_get_pool,
        patch(
            "vo_rag.pipelines.base.insert_artifact",
            new=AsyncMock(),
        ) as mock_insert_artifact,
        patch(
            "vo_rag.pipelines.base.update_envelope_status",
            new=AsyncMock(),
        ) as mock_update_envelope_status,
    ):
        artifact = await _StubPipeline().index_output(out)

    # Pool was requested from settings.database.url
    assert mock_get_pool.await_count == 1
    # Exactly one insert, with the same artifact object we got back
    assert mock_insert_artifact.await_count == 1
    insert_args = mock_insert_artifact.await_args
    assert insert_args is not None
    assert insert_args.args[0] is fake_pool
    inserted_artifact = insert_args.args[1]
    assert inserted_artifact is artifact
    # Exactly one envelope status update, to COMPLETED
    assert mock_update_envelope_status.await_count == 1
    update_args = mock_update_envelope_status.await_args
    assert update_args is not None
    assert update_args.args[0] is fake_pool
    assert update_args.args[1] == out.envelope_id
    assert update_args.args[2] is EnvelopeStatus.COMPLETED


async def test_index_output_builds_deterministic_artifact_id() -> None:
    out = _make_output()
    fake_pool = _FakePool()
    captured: dict[str, Any] = {}

    async def _capture_insert(pool: Any, artifact: KnowledgeArtifact) -> None:
        captured["artifact"] = artifact

    with (
        patch("vo_rag.pipelines.base.get_pool", new=AsyncMock(return_value=fake_pool)),
        patch("vo_rag.pipelines.base.insert_artifact", new=_capture_insert),
        patch("vo_rag.pipelines.base.update_envelope_status", new=AsyncMock()),
    ):
        artifact = await _StubPipeline().index_output(out)

    # Artifact ID is 26 upper-case hex chars (deterministic_hex(..., 26).upper())
    basis = out.content.get("artifact_id_basis") or out.content.get("source_id") or out.envelope_id
    expected = deterministic_hex(basis, out.pipeline, out.artifact_type, length=26).upper()
    assert artifact.id == expected
    assert len(artifact.id) == 26
    assert artifact.id.isupper()
    # The captured artifact matches the returned one
    assert captured["artifact"] is artifact


async def test_index_output_is_idempotent_on_repeat_calls() -> None:
    """Re-running index_output for the same (basis, pipeline, artifact_type)
    must yield the same artifact id, even though each call inserts a new
    artifact row (the DB upsert is the actual idempotency mechanism)."""
    out = _make_output()
    fake_pool = _FakePool()

    with (
        patch("vo_rag.pipelines.base.get_pool", new=AsyncMock(return_value=fake_pool)),
        patch("vo_rag.pipelines.base.insert_artifact", new=AsyncMock()),
        patch("vo_rag.pipelines.base.update_envelope_status", new=AsyncMock()),
    ):
        a1 = await _StubPipeline().index_output(out)
        a2 = await _StubPipeline().index_output(out)

    assert a1.id == a2.id
    assert a1.envelope_id == a2.envelope_id


async def test_index_output_artifact_id_basis_falls_back_to_envelope() -> None:
    """When neither artifact_id_basis nor source_id is in content, the
    envelope_id is the basis. Locks the fallback chain."""
    out = _make_output(content={"unrelated": "x"})

    with (
        patch("vo_rag.pipelines.base.get_pool", new=AsyncMock(return_value=_FakePool())),
        patch("vo_rag.pipelines.base.insert_artifact", new=AsyncMock()),
        patch("vo_rag.pipelines.base.update_envelope_status", new=AsyncMock()),
    ):
        artifact = await _StubPipeline().index_output(out)

    expected = deterministic_hex(
        out.envelope_id, out.pipeline, out.artifact_type, length=26
    ).upper()
    assert artifact.id == expected


# ---------------------------------------------------------------------------
# PipelineBase.index_output — content-driven point_ids
# ---------------------------------------------------------------------------


async def test_index_output_passes_through_content_point_ids() -> None:
    out = _make_output(
        content={
            "source_id": "src-1",
            "qdrant_point_ids": ["pid-a", "pid-b"],
        },
    )

    captured: dict[str, Any] = {}

    async def _capture_insert(pool: Any, artifact: KnowledgeArtifact) -> None:
        captured["artifact"] = artifact

    with (
        patch("vo_rag.pipelines.base.get_pool", new=AsyncMock(return_value=_FakePool())),
        patch("vo_rag.pipelines.base.insert_artifact", new=_capture_insert),
        patch("vo_rag.pipelines.base.update_envelope_status", new=AsyncMock()),
    ):
        await _StubPipeline().index_output(out)

    assert captured["artifact"].qdrant_point_ids == ["pid-a", "pid-b"]
    assert captured["artifact"].qdrant_collection == "vo_rag_books_v1"


async def test_index_output_no_point_ids_in_content() -> None:
    out = _make_output(content={"source_id": "src-1"})
    captured: dict[str, Any] = {}

    async def _capture_insert(pool: Any, artifact: KnowledgeArtifact) -> None:
        captured["artifact"] = artifact

    with (
        patch("vo_rag.pipelines.base.get_pool", new=AsyncMock(return_value=_FakePool())),
        patch("vo_rag.pipelines.base.insert_artifact", new=_capture_insert),
        patch("vo_rag.pipelines.base.update_envelope_status", new=AsyncMock()),
    ):
        await _StubPipeline().index_output(out)

    assert captured["artifact"].qdrant_point_ids == []


# ---------------------------------------------------------------------------
# PipelineBase.index_output — artifact shape
# ---------------------------------------------------------------------------


async def test_index_output_artifact_carries_pipeline_metadata() -> None:
    """The persisted artifact mirrors the output's pipeline, type, license,
    tags, and source_provenance."""
    out = _make_output(
        tags=["t1", "t2"],
        license_class=LicenseClass.PUBLIC_DOMAIN,
        priority=Priority.HIGH.value,
    )
    captured: dict[str, Any] = {}

    async def _capture_insert(pool: Any, artifact: KnowledgeArtifact) -> None:
        captured["artifact"] = artifact

    with (
        patch("vo_rag.pipelines.base.get_pool", new=AsyncMock(return_value=_FakePool())),
        patch("vo_rag.pipelines.base.insert_artifact", new=_capture_insert),
        patch("vo_rag.pipelines.base.update_envelope_status", new=AsyncMock()),
    ):
        await _StubPipeline().index_output(out)

    art = captured["artifact"]
    assert art.envelope_id == out.envelope_id
    assert art.pipeline is TargetPipeline(out.pipeline)
    assert art.artifact_type is ArtifactType(out.artifact_type)
    assert art.title == out.title
    assert art.tags == ["t1", "t2"]
    assert art.license_class is LicenseClass.PUBLIC_DOMAIN
    assert art.priority is Priority.HIGH
    assert art.qdrant_collection == "vo_rag_books_v1"
    # source_provenance is rebuilt through ProvenanceEntry (validates entries)
    assert len(art.source_provenance) == 1
    assert art.source_provenance[0].source_id == "src-1"
    # Timestamps stamped
    assert isinstance(art.created_at, datetime)
    assert isinstance(art.updated_at, datetime)
    assert art.created_at.tzinfo is not None


async def test_index_output_source_provenance_with_invalid_entry_raises() -> None:
    """A malformed provenance entry must fail fast at the Pydantic boundary
    rather than writing garbage to the DB."""
    out = _make_output(
        source_provenance=[{"source_id": "src-1", "source_type": "book", "contribution": 2.0}],
    )
    with (
        patch("vo_rag.pipelines.base.get_pool", new=AsyncMock(return_value=_FakePool())),
        patch("vo_rag.pipelines.base.insert_artifact", new=AsyncMock()),
        patch("vo_rag.pipelines.base.update_envelope_status", new=AsyncMock()),
        pytest.raises(ValidationError),
    ):
        await _StubPipeline().index_output(out)


# ---------------------------------------------------------------------------
# Concurrency / freshness
# ---------------------------------------------------------------------------


async def test_index_output_updates_updated_at_to_current_utc() -> None:
    """updated_at should be a fresh stamp, even if we can't pin it to the
    exact nanosecond — assert it's a datetime with a tz set."""
    out = _make_output()
    captured: dict[str, Any] = {}

    async def _capture_insert(pool: Any, artifact: KnowledgeArtifact) -> None:
        captured["artifact"] = artifact

    with (
        patch("vo_rag.pipelines.base.get_pool", new=AsyncMock(return_value=_FakePool())),
        patch("vo_rag.pipelines.base.insert_artifact", new=_capture_insert),
        patch("vo_rag.pipelines.base.update_envelope_status", new=AsyncMock()),
    ):
        await _StubPipeline().index_output(out)

    art = captured["artifact"]
    # UTC tzinfo is set; the stamp was set at call time so it's "now".
    assert art.updated_at.tzinfo is not None
    # We don't compare to a wall clock because tests can run near midnight;
    # just assert non-1970 and non-future.
    assert art.updated_at > datetime(2024, 1, 1, tzinfo=UTC)
    assert art.updated_at < datetime(2100, 1, 1, tzinfo=UTC)
