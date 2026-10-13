"""Stage complete source representations and publish one durable revision decision."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, Record

from vo_rag.core.idempotency import deterministic_uuid
from vo_rag.core.ingest.representation_receipt import (
    canonical_sha256,
    representation_receipt,
    verify_representation,
)
from vo_rag.core.ingest.source_publication import (
    PublicationState,
    SourceCoordinator,
    StalePublicationError,
)

__all__ = ["SourceRevisionWriter", "revision_points"]


def revision_points(
    points: Sequence[PointStruct], *, corpus: str, collection: str, source_id: str, revision: int
) -> list[PointStruct]:
    """Assign collision-safe IDs without reusing legacy chunk ordinal slots."""
    if type(revision) is not int or revision < 1 or not points:
        raise ValueError("a positive revision and complete nonempty source roster are required")
    identifiers = [str(point.id) for point in points]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate writer point identities")
    base_ids = [
        str((point.payload or {}).get("publication_base_id") or point.id) for point in points
    ]
    if len(set(base_ids)) != len(base_ids):
        raise ValueError("duplicate writer logical base identities")
    output: list[PointStruct] = []
    for point in points:
        payload = dict(point.payload or {})
        if payload.get("source_id") != source_id:
            raise ValueError("writer source scope mismatch")
        base_id = str(payload.get("publication_base_id") or point.id)
        point_id = deterministic_uuid(
            "source-publication-v1", corpus, collection, source_id, str(revision), base_id
        )
        payload.update(source_revision=revision, publication_base_id=base_id)
        output.append(point.model_copy(update={"id": point_id, "payload": payload}))
    return output


class SourceRevisionWriter:
    """Publish complete source replacements through the normal Qdrant writer seam.

    The PostgreSQL pointer is the publication event. Qdrant acknowledgement is
    followed by complete payload/vector read-back. Every old point is retained
    with a durable roster; this protects readers holding an earlier pointer.
    Failed staging quarantines only this source and never moves its pointer.
    """

    def __init__(self, raw_client: Any, coordinator: SourceCoordinator) -> None:
        self.raw = raw_client
        self.coordinator = coordinator

    async def census(self, collection: str, source_id: str, revision: int | None) -> list[Record]:
        """Capture a complete selected revision with an independent exact count."""
        from qdrant_client.models import IsEmptyCondition, PayloadField

        selected = Filter(
            must=[
                FieldCondition(key="source_id", match=MatchValue(value=source_id)),
                IsEmptyCondition(is_empty=PayloadField(key="source_revision"))
                if revision is None
                else FieldCondition(key="source_revision", match=MatchValue(value=revision)),
            ]
        )
        count = await self.raw.count(collection_name=collection, count_filter=selected, exact=True)
        rows: list[Record] = []
        offset: Any = None
        while True:
            page, offset = await self.raw.scroll(
                collection_name=collection,
                scroll_filter=selected,
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            rows.extend(page)
            if len(rows) > 100_000:
                raise ValueError("normal source census exceeds its finite 100000-point ceiling")
            if offset is None:
                break
        if len(rows) != count.count or len({str(row.id) for row in rows}) != len(rows):
            raise ValueError("source census incomplete or point identities duplicated")
        return rows

    async def replace(
        self,
        collection: str,
        source_id: str,
        points: list[PointStruct],
        *,
        job_id: str,
        source_sha256: str,
        expected_active_revision: int | None,
        recipe: Mapping[str, Any],
        expected_before: Mapping[str, str] | None = None,
        expected_before_receipts: Mapping[str, Any] | None = None,
        expected_dependencies: Mapping[str, int | None] | None = None,
        tombstone: bool = False,
    ) -> list[str]:
        """Replace one scoped source after fresh binding and full representation verification.

        Args:
            collection: Exact target; no implicit publication or collection copy.
            source_id: Original canonical source identity.
            points: Complete output of the normal representation producer.
            job_id: Durable retry identity bound to these immutable inputs.
            source_sha256: Independently verified source artifact SHA-256.
            expected_active_revision: Pointer observed when this work was registered.
            recipe: Actual producer/runtime contract, with unknown identities explicit.
            expected_before: Repair's approved exact point/payload roster, if applicable.
        """
        dependencies = dict(expected_dependencies or {})
        if tombstone and (points or not expected_before):
            raise ValueError(
                "tombstone requires an explicit nonempty before roster and no after points"
            )
        if not tombstone:
            # Reject invalid complete writer output before allocating a durable
            # pending job. This validation has no storage side effects.
            for point in revision_points(
                points,
                corpus=self.coordinator.corpus,
                collection=collection,
                source_id=source_id,
                revision=1,
            ):
                representation_receipt(point)
                if (
                    dependencies
                    and (point.payload or {}).get("dependency_revisions") != dependencies
                ):
                    raise ValueError("derived representation lacks its complete dependency binding")
        if dependencies and self.coordinator.corpus != "books":
            raise ValueError("raw Books dependencies require the Books corpus")
        for dependency, revision in dependencies.items():
            if not dependency or (
                revision is not None and (type(revision) is not int or revision < 1)
            ):
                raise ValueError("invalid canonical source dependency binding")
        async with self.coordinator.fence(
            source_id, job_id=job_id, dependency_sources=tuple(dependencies)
        ) as fence:
            for dependency, revision in dependencies.items():
                active = await self.coordinator.read_active("vo_rag_books_raw_v1", dependency)
                actual_revision = active.revision if active is not None else None
                if actual_revision != revision:
                    raise StalePublicationError("canonical raw source dependency advanced")
            job = await self.coordinator.begin(
                fence,
                collection,
                source_id,
                job_id,
                source_sha256,
                expected_active_revision,
                recipe,
            )
            staged = (
                []
                if tombstone
                else revision_points(
                    points,
                    corpus=self.coordinator.corpus,
                    collection=collection,
                    source_id=source_id,
                    revision=job.revision,
                )
            )
            for point in staged:
                payload = point.payload or {}
                if dependencies and payload.get("dependency_revisions") != dependencies:
                    raise ValueError("derived representation lacks its complete dependency binding")
                if dependencies:
                    payload["dependency_revision_keys"] = {
                        hashlib.sha256(key.encode()).hexdigest(): revision
                        for key, revision in dependencies.items()
                    }
            ids = [str(point.id) for point in staged]
            intent = {str(point.id): representation_receipt(point) for point in staged}
            prior_intent = job.checkpoint.get("writer_intent")
            if prior_intent is not None and prior_intent != intent:
                raise ValueError("resume producer output differs from its durable writer intent")
            if job.state is PublicationState.PUBLISHED:
                pointer = await self.coordinator.read_active(collection, source_id)
                if pointer is None or pointer.revision != job.revision:
                    raise StalePublicationError("published job was superseded by a newer revision")
                actual = await self.census(collection, source_id, job.revision)
                found = {str(row.id): row for row in actual}
                if set(found) != set(ids):
                    raise ValueError("published source roster changed since its receipt")
                for point in staged:
                    verify_representation(point, found[str(point.id)])
                return ids
            checkpoint: dict[str, Any] = dict(job.checkpoint)
            checkpoint.update(
                {
                    "writer_intent": intent,
                    "expected_point_ids": ids,
                    "old_point_disposition": "retained_old_revision",
                    "source_snapshot_atomic": False,
                    "owner": "source_revision_writer",
                    "recovery": "resume identical job and writer output; preserve active pointer",
                }
            )
            checkpoint.setdefault("acknowledged_batches", [])
            checkpoint.setdefault("write_attempts", 0)
            try:
                before = await self.census(collection, source_id, expected_active_revision)
                before_receipts = {str(row.id): representation_receipt(row) for row in before}
                if (
                    expected_before_receipts is not None
                    and dict(expected_before_receipts) != before_receipts
                ):
                    raise ValueError(
                        "fresh source modalities differ from the approved before receipts"
                    )
                if expected_before is not None and dict(expected_before) != {
                    point_id: receipt["payload_sha256"]
                    for point_id, receipt in before_receipts.items()
                }:
                    raise ValueError(
                        "fresh source census differs from the approved before manifest"
                    )
                repeated = await self.census(collection, source_id, expected_active_revision)
                if before_receipts != {
                    str(row.id): representation_receipt(row) for row in repeated
                }:
                    raise ValueError("source changed during non-atomic census; recapture required")
                prior_before = checkpoint.get("before_receipts")
                if prior_before is not None and prior_before != before_receipts:
                    raise ValueError("before roster drifted since the durable checkpoint")
                checkpoint["before_receipts"] = before_receipts
                if job.state not in (PublicationState.STAGED, PublicationState.VERIFIED):
                    await self.coordinator.checkpoint(
                        fence, job_id, PublicationState.PREPARING, checkpoint
                    )
                    for start in range(0, len(staged), 8):
                        checkpoint["pending_batch"] = ids[start : start + 8]
                        checkpoint["write_attempts"] += 1
                        await self.coordinator.checkpoint(
                            fence, job_id, PublicationState.PREPARING, checkpoint
                        )
                        response = await self.raw.upsert(
                            collection_name=collection, points=staged[start : start + 8], wait=True
                        )
                        if str(response.status) != "completed":
                            raise RuntimeError("Qdrant did not acknowledge completion")
                        checkpoint["acknowledged_batches"].append(ids[start : start + 8])
                        checkpoint.pop("pending_batch", None)
                        await self.coordinator.checkpoint(
                            fence, job_id, PublicationState.PREPARING, checkpoint
                        )
                    await self.coordinator.checkpoint(
                        fence, job_id, PublicationState.STAGED, checkpoint
                    )
                actual = await self.census(collection, source_id, job.revision)
                found = {str(row.id): row for row in actual}
                if set(found) != set(ids):
                    raise ValueError("staged source roster differs from complete writer output")
                receipts = {
                    str(point.id): verify_representation(point, found[str(point.id)])
                    for point in staged
                }
                manifest_sha = canonical_sha256(receipts)
                checkpoint["verification"] = {
                    "expected_point_ids": ids,
                    "readback_point_ids": sorted(found),
                    "point_receipts": receipts,
                    "manifest_sha256": manifest_sha,
                    "empty_source_tombstone": tombstone,
                    "tombstone_before_payloads": dict(expected_before or {}),
                    "tombstone_before_sha256": canonical_sha256(dict(expected_before or {})),
                }
                await self.coordinator.checkpoint(
                    fence, job_id, PublicationState.VERIFIED, checkpoint
                )
                await self.coordinator.publish(fence, job_id)
            except BaseException:
                # The database commit may have succeeded while its reply was
                # lost. Read the authoritative pointer before marking a partial.
                pointer = await self.coordinator.read_active(collection, source_id)
                if pointer is not None and pointer.revision == job.revision:
                    actual = await self.census(collection, source_id, job.revision)
                    found = {str(row.id): row for row in actual}
                    if set(found) != set(ids):
                        raise ValueError(
                            "published source read-back reconciliation failed"
                        ) from None
                    for point in staged:
                        verify_representation(point, found[str(point.id)])
                    return ids
                if checkpoint["write_attempts"] == 0 and not job.checkpoint:
                    checkpoint["no_write"] = {
                        "qdrant_write_attempts": 0,
                        "acknowledgement_unknown": False,
                        "reason": "prewrite source binding or census validation failed",
                    }
                    state = PublicationState.PRESERVED_NO_WRITE
                else:
                    state = PublicationState.PARTIAL_PRESERVED
                # Even if this checkpoint fails, committed PREPARING/STAGED
                # intent remains pending and quarantines only this source.
                await self.coordinator.checkpoint(fence, job_id, state, checkpoint)
                raise
            return ids
