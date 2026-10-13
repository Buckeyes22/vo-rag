"""Shared pipeline output contract.

The minimal pipeline seam (VOR-3). It is intentionally narrow:

- ``PipelineOutput`` is the shape every pipeline's final step returns.
- ``PipelineBase.index_output`` records the artifact row and updates
  the envelope to ``COMPLETED``. **Embedding + Qdrant upsert do not
  live here** — they belong to concrete ingest pipelines
  (book_rag, code_rag, etc.). The base seam has no ``_embed_and_upsert``
  method, and ``PipelineOutput`` has no ``chunks_for_embedding`` field.
- The base seam builds deterministic artifact IDs from
  ``content["artifact_id_basis"]`` (falling back to ``content["source_id"]``
  and then ``envelope_id``) using ``vo_rag.core.idempotency``.
- Point IDs already computed by a pipeline are passed through
  ``content["qdrant_point_ids"]`` and stamped onto the artifact row.
  A concrete pipeline that did its own chunking + embedding is
  responsible for producing those IDs first.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import Field

from vo_rag.core.db.postgres import get_pool
from vo_rag.core.db.queries.artifacts import insert_artifact
from vo_rag.core.db.queries.envelopes import update_envelope_status
from vo_rag.core.idempotency import deterministic_hex
from vo_rag.core.models.artifact import (
    ArtifactType,
    KnowledgeArtifact,
    LicenseClass,
    ProvenanceEntry,
)
from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.models.envelope import EnvelopeStatus, Priority, TargetPipeline


class PipelineOutput(VoBaseModel):
    """Every concrete pipeline's final step returns one of these."""

    envelope_id: str
    pipeline: str
    artifact_type: str
    title: str
    content: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    priority: str = "medium"
    license_class: LicenseClass = LicenseClass.UNKNOWN
    images: list[dict[str, Any]] = Field(default_factory=list)
    source_provenance: list[dict[str, Any]] = Field(default_factory=list)


class PipelineBase:
    """Concrete pipelines override ``target_collection`` as needed.

    The base seam records artifacts and updates envelope status. It
    intentionally does NOT do embedding or Qdrant upserts — those are
    the responsibility of concrete ingest pipelines.
    """

    target_collection: str = ""

    async def index_output(self, output: PipelineOutput) -> KnowledgeArtifact:
        if not self.target_collection:
            raise ValueError("target_collection must be configured")

        point_ids = _point_ids_from_content(output.content)
        now = datetime.now(UTC)
        artifact = KnowledgeArtifact(
            id=_artifact_id(output),
            envelope_id=output.envelope_id,
            pipeline=TargetPipeline(output.pipeline),
            artifact_type=ArtifactType(output.artifact_type),
            title=output.title,
            content=output.content,
            tags=output.tags,
            priority=Priority(output.priority),
            license_class=output.license_class,
            qdrant_collection=self.target_collection,
            qdrant_point_ids=point_ids,
            source_provenance=[
                ProvenanceEntry.model_validate(entry) for entry in output.source_provenance
            ],
            created_at=now,
            updated_at=now,
        )

        # Lazy import: avoid pulling asyncpg at module-import time. The
        # base seam only needs the pool at index time.
        pool = await get_pool()
        await insert_artifact(pool, artifact)
        await update_envelope_status(pool, output.envelope_id, EnvelopeStatus.COMPLETED)
        return artifact


def _point_ids_from_content(content: dict[str, Any]) -> list[str]:
    point_ids = content.get("qdrant_point_ids")
    if not isinstance(point_ids, list):
        return []
    return [str(value) for value in point_ids]


def _artifact_id(output: PipelineOutput) -> str:
    artifact_basis = str(
        output.content.get("artifact_id_basis")
        or output.content.get("source_id")
        or output.envelope_id
    )
    return deterministic_hex(
        artifact_basis,
        output.pipeline,
        output.artifact_type,
        length=26,
    ).upper()


__all__ = [
    "PipelineBase",
    "PipelineOutput",
]
