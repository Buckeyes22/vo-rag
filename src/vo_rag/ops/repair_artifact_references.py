"""Reconcile reviewed extraction references after source publication.

The source writer retains old physical points. Its owning extraction must point
at the new roster only after publication succeeds; retries finish this separate
compare-and-swap without repeating inference or point writes.
"""

from __future__ import annotations

from typing import Any

from vo_rag.core.ingest.repair_evidence import RepairCorpus, RepairEvidence
from vo_rag.core.ingest.representation_receipt import canonical_sha256

_ARTIFACT_SELECT = (
    "SELECT id, envelope_id, pipeline, artifact_type, title, tags, priority, status, "
    "license_class, content, source_provenance, qdrant_collection, qdrant_point_ids "
    "FROM knowledge_artifacts WHERE envelope_id = $1 OR content->>'source_id' = $2 "
    "OR source_provenance @> $3::jsonb "
    "OR (qdrant_collection = $4 AND qdrant_point_ids && $5::text[]) "
    "OR envelope_id IN (SELECT id FROM intake_envelopes "
    "WHERE raw_content->>'file_hash' = $6) ORDER BY id LIMIT 2"
)


class ArtifactReferenceBlocked(RuntimeError):
    """A source-owned extraction reference changed or lacks review evidence."""


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def _approved_artifact(evidence: RepairEvidence) -> dict[str, Any]:
    snapshots = evidence.owned_artifact_snapshots
    if not evidence.source_envelope_id or snapshots is None:
        raise ArtifactReferenceBlocked("owned_extraction_snapshot_missing_before_admission")
    if len(snapshots) != 1:
        raise ArtifactReferenceBlocked("owned_extraction_requires_one_reviewed_artifact")
    if canonical_sha256(snapshots) != evidence.owned_artifact_snapshots_sha256:
        raise ArtifactReferenceBlocked("owned_extraction_snapshot_digest_mismatch")
    artifact = dict(snapshots[0])
    before_ids = [str(point.point_id) for point in evidence.before_points]
    point_ids = artifact.get("qdrant_point_ids")
    content = artifact.get("content")
    if (
        artifact.get("envelope_id") != evidence.source_envelope_id
        or artifact.get("pipeline") != "book_rag"
        or artifact.get("artifact_type") != "extraction"
        or artifact.get("status") != "active"
        or artifact.get("qdrant_collection") != evidence.target_collection
        or not isinstance(content, dict)
        or content.get("source_id") != evidence.source_id
        or not isinstance(point_ids, list)
        or len(point_ids) != len(set(point_ids))
        or set(point_ids) != set(before_ids)
    ):
        raise ArtifactReferenceBlocked("owned_extraction_source_or_before_roster_mismatch")
    return artifact


async def _rows(connection: Any, evidence: RepairEvidence) -> list[dict[str, Any]]:
    # Two rows suffice to reject any additional source owner. One row proves
    # the supported single-extraction inventory was captured completely.
    referenced_ids = sorted(
        {str(point.point_id) for point in (*evidence.before_points, *evidence.after_points)}
    )
    rows = await connection.fetch(
        _ARTIFACT_SELECT,
        evidence.source_envelope_id,
        evidence.source_id,
        [{"source_id": evidence.source_id}],
        evidence.target_collection,
        referenced_ids,
        evidence.original.sha256,
    )
    return [_json_value(dict(row)) for row in rows]


async def check_artifact_references(evidence: RepairEvidence, pool: Any) -> None:
    """Check the complete owned extraction snapshot before publication admission.

    Args:
        evidence: Immutable source evidence with the full owned artifact snapshot.
        pool: Existing database connection pool; reads run in a read-only transaction.

    Raises:
        ArtifactReferenceBlocked: Ownership or reviewed metadata differs.
    """
    if evidence.corpus is not RepairCorpus.BOOKS:
        return
    expected = _approved_artifact(evidence)
    async with pool.acquire() as connection, connection.transaction(readonly=True):
        observed = await _rows(connection, evidence)
    if observed != [expected]:
        raise ArtifactReferenceBlocked("owned_extraction_snapshot_changed_before_admission")


async def reconcile_artifact_references(
    evidence: RepairEvidence,
    store: Any,
    *,
    job_id: str,
    after_ids: list[str],
    write: bool = True,
) -> str:
    """Reconcile extraction IDs under the published source job's fence.

    Args:
        evidence: Reviewed complete source and artifact snapshots.
        store: Ordinary source publication store.
        job_id: The exact repair publication job owning the active revision.
        after_ids: Complete published physical point roster, in writer order.
        write: False requires already reconciled IDs and performs no update.

    Returns:
        ``updated``, ``already_current``, or ``not_applicable``.

    Raises:
        ArtifactReferenceBlocked: The active owner, metadata, or roster differs.
    """
    if evidence.corpus is not RepairCorpus.BOOKS:
        return "not_applicable"
    expected = _approved_artifact(evidence)
    approved_ids = [
        str(point.point_id)
        for point in evidence.after_points
        if point.disposition.value == "replace"
    ]
    if (
        not after_ids
        or len(after_ids) != len(set(after_ids))
        or set(after_ids) != set(approved_ids)
    ):
        raise ArtifactReferenceBlocked("owned_extraction_after_roster_invalid")
    coordinator = await store._source_coordinator(evidence.target_collection)
    if coordinator is None:
        raise ArtifactReferenceBlocked("owned_extraction_source_coordinator_missing")
    async with coordinator.fence(evidence.source_id, job_id=job_id) as fence:
        connection = fence.connection
        owner = await connection.fetchrow(
            "SELECT p.active_revision, p.active_job_id, j.revision, j.state, j.source_sha256 "
            "FROM source_publications AS p JOIN source_publication_jobs AS j "
            "ON j.job_id = p.active_job_id AND j.corpus = p.corpus "
            "WHERE p.corpus = $1 AND p.collection = $2 AND p.source_id = $3",
            evidence.corpus.value,
            evidence.target_collection,
            evidence.source_id,
        )
        revision = evidence.replacement_revision_sequence
        if (
            owner is None
            or owner["active_revision"] != revision
            or owner["active_job_id"] != job_id
            or owner["revision"] != revision
            or owner["state"] != "published"
            or owner["source_sha256"] != evidence.original.sha256
        ):
            raise ArtifactReferenceBlocked("owned_extraction_published_source_owner_changed")
        observed = await _rows(connection, evidence)
        intended = {**expected, "qdrant_point_ids": after_ids}
        if observed == [intended]:
            return "already_current"
        if observed != [expected]:
            raise ArtifactReferenceBlocked("owned_extraction_exact_snapshot_changed_after_publish")
        if not write:
            raise ArtifactReferenceBlocked("owned_extraction_references_need_reconciliation")
        changed = await connection.fetchrow(
            "UPDATE knowledge_artifacts SET qdrant_point_ids = $2::text[] "
            "WHERE id = $1 AND envelope_id = $3 AND pipeline = $4 AND artifact_type = $5 "
            "AND status = $6 AND qdrant_collection = $7 AND qdrant_point_ids = $8::text[] "
            "AND title IS NOT DISTINCT FROM $9 AND tags = $10::text[] "
            "AND priority IS NOT DISTINCT FROM $11 "
            "AND license_class IS NOT DISTINCT FROM $12 AND content = $13::jsonb "
            "AND source_provenance = $14::jsonb RETURNING id",
            expected["id"],
            after_ids,
            expected["envelope_id"],
            expected["pipeline"],
            expected["artifact_type"],
            expected["status"],
            expected["qdrant_collection"],
            expected["qdrant_point_ids"],
            expected["title"],
            expected["tags"],
            expected["priority"],
            expected["license_class"],
            expected["content"],
            expected["source_provenance"],
        )
        if changed is None or await _rows(connection, evidence) != [intended]:
            raise ArtifactReferenceBlocked("owned_extraction_reference_cas_or_readback_failed")
        return "updated"
