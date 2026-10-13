"""Rebase managed derived Books sources after quality-only raw revisions."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    Record,
)

from vo_rag.core.idempotency import deterministic_uuid
from vo_rag.core.ingest.representation_receipt import (
    canonical_sha256,
    representation_receipt,
    verify_representation,
)
from vo_rag.core.ingest.source_publication import SourceCoordinator
from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter

__all__ = ["copy_forward_quality_dependencies"]

_RAW_COLLECTION = "vo_rag_books_raw_v1"
_DEPENDENT_COLLECTIONS = (
    "vo_rag_books_sterile_v1",
    "vo_rag_books_summary_tree_v1",
    "vo_rag_books_raptor_v1",
    "vo_rag_books_contextualized_v1",
    "vo_rag_books_dsi_v1",
    "vo_rag_books_graph_nodes_v1",
    "vo_rag_books_graph_edges_v1",
)
_COPY_FORWARD_LIMIT = 10_000
_QUALITY_KEYS = frozenset({"quality_warnings", "quality_disposition"})
_DISCOVERY_LIMIT = 100_000
_COLLECTION_CENSUS_LIMIT = 100_000
_QDRANT_READ_TIMEOUT_S = 2


def _record_point(row: Record, payload: Mapping[str, Any]) -> PointStruct:
    """Make a complete revision input from one immutable Qdrant read-back."""
    if not isinstance(row.vector, dict) or not row.vector:
        raise ValueError("quality dependency copy-forward requires every named vector")
    return PointStruct(id=str(row.id), vector=cast(Any, row.vector), payload=dict(payload))


def _quality_only_proof(before: Sequence[Record], after: Sequence[Record]) -> str:
    """Prove raw revisions differ only in quality fields and revision identity."""

    def indexed(rows: Sequence[Record]) -> dict[str, Record]:
        output: dict[str, Record] = {}
        for row in rows:
            payload = row.payload or {}
            base_id = payload.get("publication_base_id")
            if not isinstance(base_id, str) or not base_id or base_id in output:
                raise ValueError("raw quality proof requires unique publication base IDs")
            output[base_id] = row
        return output

    old = indexed(before)
    new = indexed(after)
    if not old or set(old) != set(new):
        raise ValueError("raw quality proof found a changed source roster")
    quality_changed = False
    old_receipts: dict[str, Any] = {}
    new_receipts: dict[str, Any] = {}
    for base_id, old_row in old.items():
        new_row = new[base_id]
        old_payload = dict(old_row.payload or {})
        new_payload = dict(new_row.payload or {})
        for payload in (old_payload, new_payload):
            payload.pop("source_revision", None)
            for key in _QUALITY_KEYS:
                payload.pop(key, None)
        if old_payload != new_payload:
            raise ValueError("raw revisions differ beyond quality and publication metadata")
        old_quality = (
            (old_row.payload or {}).get("quality_warnings"),
            (old_row.payload or {}).get("quality_disposition"),
        )
        new_quality = (
            (new_row.payload or {}).get("quality_warnings"),
            (new_row.payload or {}).get("quality_disposition"),
        )
        quality_changed = quality_changed or old_quality != new_quality
        old_receipt = representation_receipt(old_row)
        new_receipt = representation_receipt(new_row)
        expected_vector = PointStruct(
            id=str(new_row.id),
            vector=cast(Any, old_row.vector),
            payload=dict(new_row.payload or {}),
        )
        try:
            verify_representation(expected_vector, new_row)
        except ValueError as exc:
            raise ValueError("raw quality proof found changed vectors") from exc
        old_receipts[base_id] = old_receipt
        new_receipts[base_id] = new_receipt
    if not quality_changed:
        raise ValueError("raw revisions contain no quality metadata change")
    return canonical_sha256({"before": old_receipts, "after": new_receipts})


async def copy_forward_quality_dependencies(
    raw_client: Any,
    coordinator: SourceCoordinator,
    source_id: str,
    old_revision: int | None = None,
    new_revision: int | None = None,
) -> dict[str, list[str]]:
    """Copy forward managed derived rosters after a proven quality-only raw update.

    The writer binds every replacement to the current complete dependency map,
    records exact before receipts durably, and checks raw pointers again while
    holding the same source fence used by normal publishers.

    Args:
        raw_client: Unwrapped Qdrant client used for exact scans and writes.
        coordinator: Books publication coordinator sharing the writer fences.
        source_id: Canonical raw source whose warnings were persisted.
        old_revision: Exact prior raw revision for an immediate update, or
            ``None`` to discover and recover any older quality-only dependency.
        new_revision: Current raw revision; defaults to its active pointer.

    Returns:
        Successfully published derived source IDs grouped by collection.

    Raises:
        ValueError: If a quality-only proof, complete roster, or copy-forward
            ceiling check fails before or during publication. Each entire
            derived collection must have an exact count at or below 100,000
            points before filtered candidate discovery begins.
        StalePublicationError: If a raw dependency advances during the final
            fenced publication check.
    """
    if coordinator.corpus != "books":
        raise ValueError("quality dependency rebase requires the Books corpus")
    raw_pointer = await coordinator.read_active(_RAW_COLLECTION, source_id)
    current_revision = raw_pointer.revision if raw_pointer is not None else None
    target_revision = new_revision if new_revision is not None else current_revision
    if (
        current_revision is None
        or target_revision != current_revision
        or target_revision < 1
        or (old_revision is not None and (old_revision < 1 or old_revision >= target_revision))
    ):
        raise ValueError("quality dependency rebase requires the current Books raw revision")
    writer = SourceRevisionWriter(raw_client, coordinator)
    rebased: dict[str, list[str]] = {}
    dependency_field = "dependency_revision_keys." + hashlib.sha256(source_id.encode()).hexdigest()

    for collection in _DEPENDENT_COLLECTIONS:
        collection_count = await raw_client.count(
            collection_name=collection,
            exact=True,
            timeout=_QDRANT_READ_TIMEOUT_S,
        )
        if collection_count.count > _COLLECTION_CENSUS_LIMIT:
            raise ValueError(
                f"quality dependency collection census exceeds its "
                f"{_COLLECTION_CENSUS_LIMIT}-point ceiling: {collection}"
            )
        if collection_count.count == 0:
            continue
        selector = Filter(
            must=[
                FieldCondition(
                    key=dependency_field,
                    **(
                        {"match": MatchValue(value=old_revision)}
                        if old_revision is not None
                        else {"range": Range(lt=target_revision)}
                    ),
                )
            ]
        )
        candidate_rows: list[Record] = []
        offset: Any = None
        while True:
            page, offset = await raw_client.scroll(
                collection_name=collection,
                scroll_filter=selector,
                limit=min(128, _DISCOVERY_LIMIT + 1 - len(candidate_rows)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
                timeout=_QDRANT_READ_TIMEOUT_S,
            )
            candidate_rows.extend(page)
            if len(candidate_rows) > _DISCOVERY_LIMIT:
                raise ValueError("stale dependency discovery exceeds its 100000-point ceiling")
            if offset is None:
                break
        candidate_bindings: dict[str, set[int]] = {}
        for row in candidate_rows:
            payload = row.payload or {}
            derived_source_id = payload.get("source_id")
            source_revision = payload.get("source_revision")
            dependency_map = payload.get("dependency_revisions")
            dependency_revision = (
                dependency_map.get(source_id) if isinstance(dependency_map, Mapping) else None
            )
            if (
                not isinstance(derived_source_id, str)
                or type(source_revision) is not int
                or type(dependency_revision) is not int
                or dependency_revision >= target_revision
            ):
                continue
            pointer = await coordinator.read_active(collection, derived_source_id)
            if pointer is not None and pointer.revision == source_revision:
                candidate_bindings.setdefault(derived_source_id, set()).add(dependency_revision)
        for derived_source_id, observed_old_revisions in candidate_bindings.items():
            if old_revision is not None and observed_old_revisions != {old_revision}:
                raise ValueError("active dependent source has mixed raw dependency revisions")
            if old_revision is None and len(observed_old_revisions) != 1:
                raise ValueError("active dependent source has mixed raw dependency revisions")
            dependency_revision = next(iter(observed_old_revisions))
            raw_before = await writer.census(_RAW_COLLECTION, source_id, dependency_revision)
            raw_after = await writer.census(_RAW_COLLECTION, source_id, target_revision)
            try:
                proof_sha256 = _quality_only_proof(raw_before, raw_after)
            except ValueError:
                # Older dependencies may be stale because the raw content changed;
                # those representations must remain hidden until their producer runs.
                if old_revision is None:
                    continue
                raise
            pointer = await coordinator.read_active(collection, derived_source_id)
            active_revision = pointer.revision if pointer is not None else None
            if active_revision is None:
                continue
            rows = await writer.census(collection, derived_source_id, active_revision)
            if not rows:
                continue
            manifests = [(row.payload or {}).get("dependency_revisions") for row in rows]
            references_old_revision = any(
                isinstance(value, Mapping) and value.get(source_id) == dependency_revision
                for value in manifests
            )
            if not references_old_revision:
                continue
            if any(not isinstance(value, Mapping) for value in manifests):
                raise ValueError("dependent source roster has an incomplete dependency map")
            dependencies = [dict(value) for value in manifests if isinstance(value, Mapping)]
            if not dependencies or any(value != dependencies[0] for value in dependencies[1:]):
                raise ValueError("dependent source roster has inconsistent dependency maps")
            current_dependencies = dependencies[0]
            if current_dependencies.get(source_id) != dependency_revision:
                raise ValueError("dependent source does not have one exact old raw binding")
            if len(rows) > _COPY_FORWARD_LIMIT:
                raise ValueError("quality dependency copy-forward exceeds its 10000-point ceiling")

            new_dependencies = dict(current_dependencies)
            new_dependencies[source_id] = target_revision
            points: list[PointStruct] = []
            for row in rows:
                payload = dict(row.payload or {})
                payload["dependency_revisions"] = new_dependencies
                points.append(_record_point(row, payload))
            before_receipts = {str(row.id): representation_receipt(row) for row in rows}
            before_manifest = {
                point_id: receipt["payload_sha256"] for point_id, receipt in before_receipts.items()
            }
            source_pointer = await coordinator.read_active(collection, derived_source_id)
            if source_pointer is None or source_pointer.revision != active_revision:
                raise ValueError("derived source pointer changed during quality dependency census")
            source_sha = source_pointer.source_sha256 or canonical_sha256(
                {
                    "identity_kind": "current_qdrant_representation_manifest",
                    "representation_manifest": before_receipts,
                }
            )
            job_id = deterministic_uuid(
                "quality-dependency-copy-forward-v1",
                coordinator.corpus,
                collection,
                derived_source_id,
                str(active_revision),
                source_id,
                str(dependency_revision),
                str(target_revision),
                proof_sha256,
            )
            await writer.replace(
                collection,
                derived_source_id,
                points,
                job_id=job_id,
                source_sha256=source_sha,
                expected_active_revision=active_revision,
                recipe={
                    "writer": "quality-dependency-copy-forward-v1",
                    "reason": "quality_only_raw_revision",
                    "raw_source_id": source_id,
                    "raw_before_revision": dependency_revision,
                    "raw_after_revision": target_revision,
                    "raw_quality_only_proof_sha256": proof_sha256,
                    "derived_collection": collection,
                    "derived_source_id": derived_source_id,
                    "expected_dependencies_before": current_dependencies,
                    "expected_dependencies_after": new_dependencies,
                    "producer_recovery": (
                        "re-read retained active roster matching before_receipts; rewrite the "
                        "same vectors and payload with dependency revisions only; no model calls"
                    ),
                    "owner": "source_quality_dependency_copy_forward",
                    "selection_ceiling": _COPY_FORWARD_LIMIT,
                },
                expected_before=before_manifest,
                expected_before_receipts=before_receipts,
                expected_dependencies=new_dependencies,
            )
            rebased.setdefault(collection, []).append(derived_source_id)
    return rebased
