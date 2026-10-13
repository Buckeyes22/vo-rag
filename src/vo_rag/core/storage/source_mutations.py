"""Copy-on-write adapters for maintenance mutations of published sources."""

from __future__ import annotations

import asyncio
import copy
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

from qdrant_client.models import (
    ClearPayloadOperation,
    DeleteOperation,
    DeletePayload,
    DeletePayloadOperation,
    DeleteVectorsOperation,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    OverwritePayloadOperation,
    PointIdsList,
    PointsBatch,
    PointsList,
    PointStruct,
    Record,
    SetPayload,
    SetPayloadOperation,
    UpdateResult,
    UpdateStatus,
    UpdateVectorsOperation,
    UpsertOperation,
)

from vo_rag.core.idempotency import deterministic_uuid
from vo_rag.core.ingest.representation_receipt import canonical_sha256, representation_receipt
from vo_rag.core.ingest.source_publication import SourceCoordinator, StalePublicationError
from vo_rag.core.storage.publication_filter import (
    dependencies_visible,
    dependency_filter,
    publication_filter,
)
from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter

__all__ = ["SourceMutationBatchError", "SourceMutationWriter"]

_SELECTION_LIMIT = 10_000
_LEGACY_UPSERT_BATCH = 8
_RAW_BOOKS_COLLECTION = "vo_rag_books_raw_v1"
_IMMUTABLE_PAYLOAD_KEYS = frozenset(
    {
        "source_id",
        "source_revision",
        "publication_base_id",
        "chunk_index",
        "version_hash",
        "source_sha256",
        "canonical_source_id",
        "book_id",
        "book_ids",
        "url",
        "citation_id",
        "citation_key",
        "content",
        "embedding_provenance",
        # Maintenance cannot promote quarantined or unreviewed content into a
        # trusted retrieval tier. Trust changes belong to their owning ingest
        # and safety-review path.
        "trust_level",
    }
)


class SourceMutationBatchError(RuntimeError):
    """A multi-source maintenance operation completed only for some sources."""

    def __init__(self, succeeded: Sequence[str], failed: Mapping[str, str]) -> None:
        self.succeeded = tuple(succeeded)
        self.failed = dict(failed)
        super().__init__(
            "source mutation incomplete; succeeded="
            f"{','.join(self.succeeded) or '(none)'}; failed={','.join(sorted(self.failed))}"
        )


class _SourceTombstoneRequired(ValueError):
    pass


class SourceMutationWriter:
    """Convert maintenance point mutations into complete source revisions.

    The adapter never calls a direct Qdrant mutation. It selects the active
    source roster, applies the requested change in memory, and delegates the
    complete replacement or tombstone to :class:`SourceRevisionWriter`.
    """

    def __init__(self, raw_client: Any, coordinator: SourceCoordinator) -> None:
        self.raw = raw_client
        self.coordinator = coordinator

    async def mutate(self, operation: str, collection: str, kwargs: dict[str, Any]) -> UpdateResult:
        """Apply one supported Qdrant mutation through per-source publication."""
        if not operation or not collection:
            raise ValueError("operation and collection are required")
        actions = self._actions(operation, kwargs)
        selected: dict[str, list[tuple[str, str, Any]]] = defaultdict(list)
        selections: dict[str, list[tuple[Any, dict[str, str]]]] = defaultdict(list)
        failures: dict[str, str] = {}

        for action, selector, value in actions:
            if action == "upsert":
                points = self._as_points(value)
                for point in points:
                    source_id = self._point_source(point)
                    selected[source_id].append((action, str(point.id), point))
                continue
            try:
                rows, selection_failures = await self._select(collection, selector)
                failures.update(selection_failures)
                for row in rows:
                    payload = row.payload or {}
                    candidate_source_id = payload.get("source_id")
                    if not isinstance(candidate_source_id, str) or not candidate_source_id:
                        raise ValueError("selected point has no canonical source_id")
                    selected[candidate_source_id].append((action, str(row.id), value))
                self._record_selections(selections, selector, rows)
            except Exception as exc:  # continue unrelated source groups
                source_hint = str(getattr(exc, "source_id", self._selector_source_hint(selector)))
                failures[source_hint] = type(exc).__name__

        succeeded: list[str] = []
        for source_id, source_actions in selected.items():
            try:
                await self._mutate_source(
                    collection, source_id, operation, source_actions, selections[source_id]
                )
                succeeded.append(source_id)
            except Exception as exc:  # the source writer records durable partial/no-write state
                failures[source_id] = type(exc).__name__

        if failures:
            raise SourceMutationBatchError(succeeded, failures)
        return UpdateResult(status=UpdateStatus.COMPLETED)

    async def mutate_legacy(
        self, operation: str, collection: str, kwargs: dict[str, Any]
    ) -> UpdateResult:
        """Apply a legacy-ID mutation under source/dependency fences.

        This compatibility path is for reader-capable releases while revision
        writes are disabled. It keeps the existing point IDs and refuses every
        source with an active revision or any non-terminal publication job.
        Each selected source's operations are submitted together, so a later
        selector cannot accidentally target IDs made stale by an earlier COW
        publication.
        """
        if not operation or not collection:
            raise ValueError("operation and collection are required")
        actions = self._actions(operation, kwargs)
        selected: dict[str, list[tuple[str, str, Any, Mapping[str, Any]]]] = defaultdict(list)
        selections: dict[str, list[tuple[Any, dict[str, str]]]] = defaultdict(list)
        failures: dict[str, str] = {}

        for action, selector, value in actions:
            if action == "upsert":
                for point in self._as_points(value):
                    source_id = self._point_source(point)
                    selected[source_id].append((action, str(point.id), point, point.payload or {}))
                continue
            try:
                rows = await self._select_legacy(collection, selector)
                for row in rows:
                    payload = row.payload or {}
                    candidate_source_id = payload.get("source_id")
                    if not isinstance(candidate_source_id, str) or not candidate_source_id:
                        raise ValueError("selected point has no canonical source_id")
                    selected[candidate_source_id].append((action, str(row.id), value, payload))
                self._record_selections(selections, selector, rows)
            except Exception as exc:  # continue independent source groups
                hint = self._selector_source_hint(selector)
                failures[hint] = type(exc).__name__

        succeeded: list[str] = []
        for source_id, source_actions in selected.items():
            try:
                await self._mutate_legacy_source(
                    collection, source_id, source_actions, selections[source_id]
                )
                succeeded.append(source_id)
            except Exception as exc:  # report unknown acknowledgements honestly
                failures[source_id] = type(exc).__name__

        if failures:
            raise SourceMutationBatchError(succeeded, failures)
        return UpdateResult(status=UpdateStatus.COMPLETED)

    @staticmethod
    def _record_selections(
        selections: dict[str, list[tuple[Any, dict[str, str]]]],
        selector: Any,
        rows: Sequence[Record],
    ) -> None:
        by_source: dict[str, dict[str, str]] = defaultdict(dict)
        for row in rows:
            source_id = str((row.payload or {}).get("source_id") or "")
            by_source[source_id][str(row.id)] = canonical_sha256(row.payload or {})
        if isinstance(selector, FilterSelector):
            selector = selector.filter
        for source_id, proof in by_source.items():
            scoped_selector: Any = list(proof)
            if isinstance(selector, Filter):
                scoped_selector = Filter(
                    must=[
                        selector,
                        FieldCondition(key="source_id", match=MatchValue(value=source_id)),
                    ]
                )
            selections[source_id].append((scoped_selector, proof))

    async def _validate_selections(
        self,
        collection: str,
        selections: list[tuple[Any, dict[str, str]]],
        *,
        legacy: bool,
    ) -> None:
        for selector, expected in selections:
            if legacy:
                rows = await self._select_legacy(collection, selector)
            else:
                rows, failures = await self._select(collection, selector)
                if failures:
                    raise StalePublicationError("selected source visibility changed before fence")
            actual = {str(row.id): canonical_sha256(row.payload or {}) for row in rows}
            if actual != expected:
                raise StalePublicationError("source mutation selector changed before fence")

    async def _select_legacy(self, collection: str, selector: Any) -> list[Record]:
        """Select the exact legacy point roster without hiding possible revisions."""
        if selector is None:
            raise ValueError("mutation requires an explicit point-ID or filter selector")
        if isinstance(selector, FilterSelector):
            selector = selector.filter
        elif isinstance(selector, PointIdsList):
            selector = list(selector.points)
        if isinstance(selector, Filter):
            count = await self.raw.count(
                collection_name=collection, count_filter=selector, exact=True
            )
            expected = int(count.count)
            if expected > _SELECTION_LIMIT:
                raise ValueError("source mutation selection exceeds its 10000-point ceiling")
            rows: list[Record] = []
            offset: Any = None
            while True:
                page, offset = await self.raw.scroll(
                    collection_name=collection,
                    scroll_filter=selector,
                    limit=min(128, _SELECTION_LIMIT + 1 - len(rows)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                rows.extend(page)
                if len(rows) > _SELECTION_LIMIT:
                    raise ValueError("source mutation selection exceeds its 10000-point ceiling")
                if offset is None:
                    break
            if len(rows) != expected or len({str(row.id) for row in rows}) != len(rows):
                raise ValueError("legacy source selection was incomplete or duplicated")
            return self._require_legacy_rows(rows)
        if isinstance(selector, Sequence) and not isinstance(selector, str | bytes):
            identifiers = list(selector)
            if not identifiers or len(identifiers) > _SELECTION_LIMIT:
                raise ValueError("point-ID selection must contain 1 to 10000 IDs")
            if len(identifiers) != len(set(map(str, identifiers))):
                raise ValueError("point-ID selection contains duplicate IDs")
            rows = await self.raw.retrieve(
                collection_name=collection,
                ids=identifiers,
                with_payload=True,
                with_vectors=False,
            )
            if len(rows) != len(identifiers) or len({str(row.id) for row in rows}) != len(rows):
                raise ValueError("point-ID selection contains missing or duplicate points")
            return self._require_legacy_rows(rows)
        raise ValueError("unsupported Qdrant point selector")

    @staticmethod
    def _require_legacy_rows(rows: Sequence[Record]) -> list[Record]:
        for row in rows:
            payload = row.payload or {}
            if "source_revision" in payload:
                raise StalePublicationError(
                    "legacy maintenance selector includes a revisioned physical point ID"
                )
            source_id = payload.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("selected point has no canonical source_id")
        return list(rows)

    async def _mutate_legacy_source(
        self,
        collection: str,
        source_id: str,
        actions: list[tuple[str, str, Any, Mapping[str, Any]]],
        selections: list[tuple[Any, dict[str, str]]],
    ) -> None:
        dependency_sources = self._legacy_dependencies(collection, actions)
        async with self.coordinator.fence(
            source_id, dependency_sources=tuple(sorted(dependency_sources))
        ) as fence:
            await self.coordinator.assert_legacy_write_allowed(
                fence, source_id, tuple(sorted(dependency_sources))
            )
            await self._validate_selections(collection, selections, legacy=True)
            upserts = [value for action, _, value, _ in actions if action == "upsert"]
            if upserts:
                if len(actions) != len(upserts):
                    raise ValueError("legacy upsert cannot be mixed with other mutations")
                for start in range(0, len(upserts), _LEGACY_UPSERT_BATCH):
                    batch = upserts[start : start + _LEGACY_UPSERT_BATCH]
                    last_error: Exception | None = None
                    for attempt in range(3):
                        try:
                            result = await self.raw.upsert(
                                collection_name=collection, points=batch, wait=True
                            )
                            if result.status is not UpdateStatus.COMPLETED:
                                raise RuntimeError(
                                    "legacy upsert did not receive a completed "
                                    "Qdrant acknowledgement"
                                )
                            last_error = None
                            break
                        except Exception as exc:  # noqa: BLE001 - stable IDs make retry idempotent
                            last_error = exc
                            if attempt < 2:
                                await asyncio.sleep(2 * (attempt + 1))
                    if last_error is not None:
                        raise last_error
                return

            operations: list[Any] = []
            for action, point_id, value, existing_payload in actions:
                ids: Any = [point_id]
                if action == "set_payload":
                    if not isinstance(value, Mapping):
                        raise ValueError("set_payload requires a mapping")
                    self._check_protected_payload(existing_payload, value, source_id=source_id)
                    operations.append(
                        SetPayloadOperation(set_payload=SetPayload(points=ids, payload=dict(value)))
                    )
                elif action == "overwrite_payload":
                    if not isinstance(value, Mapping):
                        raise ValueError("overwrite_payload requires a mapping")
                    self._check_protected_payload(existing_payload, value, source_id=source_id)
                    retained = {
                        key: item
                        for key, item in existing_payload.items()
                        if key in _IMMUTABLE_PAYLOAD_KEYS
                    }
                    operations.append(
                        OverwritePayloadOperation(
                            overwrite_payload=SetPayload(
                                points=ids, payload={**retained, **dict(value)}
                            )
                        )
                    )
                elif action == "delete_payload":
                    keys = list(value)
                    if any(key in _IMMUTABLE_PAYLOAD_KEYS for key in keys):
                        raise ValueError(
                            "cannot delete immutable source or citation identity payload"
                        )
                    operations.append(
                        DeletePayloadOperation(delete_payload=DeletePayload(points=ids, keys=keys))
                    )
                elif action == "clear_payload":
                    retained = {
                        key: item
                        for key, item in existing_payload.items()
                        if key in _IMMUTABLE_PAYLOAD_KEYS
                    }
                    operations.append(
                        OverwritePayloadOperation(
                            overwrite_payload=SetPayload(points=ids, payload=retained)
                        )
                    )
                elif action == "delete":
                    operations.append(DeleteOperation(delete=PointIdsList(points=ids)))
                else:
                    raise ValueError(f"legacy compatibility does not support {action!r}")
            if operations:
                await self.raw.batch_update_points(
                    collection_name=collection, update_operations=operations, wait=True
                )

    def _legacy_dependencies(
        self, collection: str, actions: Sequence[tuple[str, str, Any, Mapping[str, Any]]]
    ) -> set[str]:
        """Derive explicit dependency lock closure from the selected point data."""
        dependencies: set[str] = set()
        payloads = [payload for _, _, _, payload in actions]
        if self.coordinator.corpus != "books" or collection == _RAW_BOOKS_COLLECTION:
            return dependencies
        for payload in payloads:
            dep_map = payload.get("dependency_revisions")
            if dep_map is not None:
                if not isinstance(dep_map, Mapping) or any(
                    not isinstance(key, str) or not key for key in dep_map
                ):
                    raise ValueError("legacy point has invalid canonical dependency mapping")
                dependencies.update(dep_map)
            if payload.get("dependency_revision_keys"):
                raise ValueError("hashed-only legacy dependencies cannot be safely fenced")
            for key in ("book_id",):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    dependencies.add(value)
            value = payload.get("source_id")
            if isinstance(value, str) and value.startswith("vo-rag-corpus-sha256:"):
                dependencies.add(value)
            book_ids = payload.get("book_ids")
            if isinstance(book_ids, list):
                if any(not isinstance(value, str) or not value for value in book_ids):
                    raise ValueError("legacy point has invalid graph book_ids dependency roster")
                dependencies.update(book_ids)
        if not dependencies:
            raise ValueError(
                "legacy derived mutation requires an explicit canonical dependency roster"
            )
        return dependencies

    @staticmethod
    def _actions(operation: str, kwargs: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
        """Normalize supported Qdrant mutations to action/selector/value tuples."""
        if operation == "upsert":
            if kwargs.get("update_filter") is not None or kwargs.get("update_mode") is not None:
                raise ValueError("filtered or conditional upsert semantics are unsupported")
            return [("upsert", None, SourceMutationWriter._as_points(kwargs.get("points")))]
        if operation == "set_payload":
            if kwargs.get("key") is not None:
                raise ValueError("nested set_payload paths are unsupported")
            return [("set_payload", kwargs.get("points"), kwargs.get("payload", {}))]
        if operation == "overwrite_payload":
            return [("overwrite_payload", kwargs.get("points"), kwargs.get("payload", {}))]
        if operation == "delete_payload":
            return [("delete_payload", kwargs.get("points"), kwargs.get("keys", ()))]
        if operation == "delete":
            return [("delete", kwargs.get("points_selector"), None)]
        if operation == "update_vectors":
            raise ValueError(
                "generic vector refresh requires a bound producer/source/input recipe; "
                "use the complete source writer"
            )
        if operation == "delete_vectors":
            return [("delete_vectors", kwargs.get("points"), kwargs.get("vectors", ()))]
        if operation == "batch_update_points":
            return SourceMutationWriter._batch_actions(kwargs.get("update_operations"))
        raise ValueError(f"unsupported source mutation operation {operation!r}")

    @staticmethod
    def _batch_actions(operations: Any) -> list[tuple[str, Any, Any]]:
        if not isinstance(operations, Sequence) or not operations:
            raise ValueError("batch_update_points requires non-empty update_operations")
        result: list[tuple[str, Any, Any]] = []
        for operation in operations:
            if isinstance(operation, UpsertOperation):
                result.append(("upsert", None, operation.upsert))
            elif isinstance(operation, DeleteOperation):
                result.append(("delete", operation.delete, None))
            elif isinstance(operation, SetPayloadOperation):
                value: Any = operation.set_payload
                if value.key is not None:
                    raise ValueError("nested set_payload paths are unsupported")
                result.append(
                    ("set_payload", SourceMutationWriter._model_selector(value), value.payload)
                )
            elif isinstance(operation, OverwritePayloadOperation):
                value = operation.overwrite_payload
                result.append(
                    (
                        "overwrite_payload",
                        SourceMutationWriter._model_selector(value),
                        value.payload,
                    )
                )
            elif isinstance(operation, DeletePayloadOperation):
                value = operation.delete_payload
                result.append(
                    ("delete_payload", SourceMutationWriter._model_selector(value), value.keys)
                )
            elif isinstance(operation, ClearPayloadOperation):
                result.append(
                    (
                        "clear_payload",
                        SourceMutationWriter._model_selector(operation.clear_payload),
                        None,
                    )
                )
            elif isinstance(operation, UpdateVectorsOperation):
                raise ValueError(
                    "batch vector refresh requires a bound producer/source/input recipe; "
                    "use the complete source writer"
                )
            elif isinstance(operation, DeleteVectorsOperation):
                value = operation.delete_vectors
                result.append(
                    ("delete_vectors", SourceMutationWriter._model_selector(value), value.vector)
                )
            else:
                raise ValueError(f"unsupported batch update operation {type(operation).__name__}")
        # Upserts replace complete logical points. Their meaning relative to
        # preceding/following selector mutations is Qdrant-order dependent;
        # the source COW adapter deliberately rejects ambiguous mixed batches.
        upsert_positions = [i for i, item in enumerate(result) if item[0] == "upsert"]
        if upsert_positions and len(result) != len(upsert_positions):
            raise ValueError("batch upsert cannot be mixed with other source mutations")
        if upsert_positions:
            points = [
                point for _, _, value in result for point in SourceMutationWriter._as_points(value)
            ]
            identifiers = [str(point.id) for point in points]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("batch upsert contains duplicate point IDs")
            base_ids = [
                str((point.payload or {}).get("publication_base_id") or point.id)
                for point in points
            ]
            if len(base_ids) != len(set(base_ids)):
                raise ValueError("batch upsert contains duplicate logical base IDs")
        return result

    @staticmethod
    def _model_selector(value: Any) -> Any:
        if getattr(value, "shard_key", None) is not None:
            raise ValueError("shard-specific maintenance selectors are unsupported")
        identifiers = getattr(value, "points", None)
        if identifiers is not None:
            return identifiers
        selected_filter = getattr(value, "filter", None)
        if selected_filter is not None:
            return selected_filter
        raise ValueError("batch mutation requires an explicit point or filter selector")

    @staticmethod
    def _as_points(value: Any) -> list[PointStruct]:
        if isinstance(value, PointsList):
            if value.update_filter is not None or value.update_mode is not None:
                raise ValueError("filtered or conditional upsert semantics are unsupported")
            value = value.points
        elif isinstance(value, PointsBatch):
            if value.update_filter is not None or value.update_mode is not None:
                raise ValueError("filtered or conditional upsert semantics are unsupported")
            batch = value.batch
            payloads: list[dict[str, Any] | None] = (
                list(batch.payloads) if batch.payloads is not None else [None] * len(batch.ids)
            )
            raw_vectors = batch.vectors
            vector_columns = (
                dict(raw_vectors) if isinstance(raw_vectors, Mapping) else {"dense": raw_vectors}
            )
            if len(payloads) != len(batch.ids) or any(
                len(vectors) != len(batch.ids) for vectors in vector_columns.values()
            ):
                raise ValueError(
                    "upsert batch IDs, vectors and payloads have different cardinality"
                )
            value = [
                PointStruct(
                    id=point_id,
                    vector={name: vectors[index] for name, vectors in vector_columns.items()},
                    payload=payloads[index],
                )
                for index, point_id in enumerate(batch.ids)
            ]
        if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
            raise ValueError("upsert requires a non-empty sequence of complete PointStruct values")
        if any(not isinstance(point, PointStruct) for point in value):
            raise ValueError("upsert Batch encoding is unsupported; provide explicit PointStructs")
        points = list(value)
        identifiers = [str(point.id) for point in points]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("upsert contains duplicate point IDs")
        base_ids = [
            str((point.payload or {}).get("publication_base_id") or point.id) for point in points
        ]
        if len(base_ids) != len(set(base_ids)):
            raise ValueError("upsert contains duplicate logical publication_base_id values")
        return points

    @staticmethod
    def _point_source(point: PointStruct) -> str:
        source_id = (point.payload or {}).get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("upsert point has no canonical source_id")
        if "source_revision" in (point.payload or {}):
            raise ValueError("callers cannot assign a source revision")
        return source_id

    async def _select(self, collection: str, selector: Any) -> tuple[list[Record], dict[str, str]]:
        if selector is None:
            raise ValueError("mutation requires an explicit point-ID or filter selector")
        if isinstance(selector, FilterSelector):
            selector = selector.filter
        elif isinstance(selector, PointIdsList):
            selector = list(selector.points)
        book_derived = self.coordinator.corpus == "books" and collection != _RAW_BOOKS_COLLECTION
        if isinstance(selector, Filter):
            active_pointers = await self.coordinator.list_active(collection)
            selected_filter = publication_filter(active_pointers, selector)
            if book_derived:
                active_raw = await self.coordinator.list_active(_RAW_BOOKS_COLLECTION)
                selected_filter = Filter(must=[selected_filter, dependency_filter(active_raw)])
            rows: list[Record] = []
            offset: Any = None
            while True:
                page, offset = await self.raw.scroll(
                    collection_name=collection,
                    scroll_filter=selected_filter,
                    limit=min(128, _SELECTION_LIMIT + 1 - len(rows)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                rows.extend(page)
                if len(rows) > _SELECTION_LIMIT:
                    raise ValueError("source mutation selection exceeds its 10000-point ceiling")
                if offset is None:
                    return rows, {}
        if isinstance(selector, Sequence) and not isinstance(selector, str | bytes):
            identifiers = list(selector)
            if not identifiers or len(identifiers) > _SELECTION_LIMIT:
                raise ValueError("point-ID selection must contain 1 to 10000 IDs")
            if len(identifiers) != len(set(map(str, identifiers))):
                raise ValueError("point-ID selection contains duplicate IDs")
            rows = await self.raw.retrieve(
                collection_name=collection,
                ids=identifiers,
                with_payload=True,
                with_vectors=False,
            )
            if len(rows) != len(set(map(str, identifiers))):
                raise ValueError("point-ID selection contains missing or duplicate points")
            current_rows: list[Record] = []
            failures: dict[str, str] = {}
            active_raw = (
                await self.coordinator.list_active(_RAW_BOOKS_COLLECTION) if book_derived else {}
            )
            for row in rows:
                payload = row.payload or {}
                source_id = payload.get("source_id")
                if not isinstance(source_id, str) or not source_id:
                    raise ValueError("selected point has no canonical source_id")
                pointer = await self.coordinator.read_active(collection, source_id)
                active_revision = pointer.revision if pointer is not None else None
                if payload.get("source_revision") != active_revision:
                    failures[source_id] = "StalePublicationError"
                    continue
                if not dependencies_visible(payload, active_raw):
                    failures[source_id] = "StalePublicationError"
                    continue
                current_rows.append(row)
            return current_rows, failures
        raise ValueError("unsupported Qdrant point selector")

    async def _mutate_source(
        self,
        collection: str,
        source_id: str,
        operation: str,
        actions: list[tuple[str, str, Any]],
        selections: list[tuple[Any, dict[str, str]]],
    ) -> None:
        pointer = await self.coordinator.read_active(collection, source_id)
        expected_revision = pointer.revision if pointer is not None else None
        base_writer = SourceRevisionWriter(self.raw, self.coordinator)
        before = await base_writer.census(collection, source_id, expected_revision)
        if not before and not any(action == "upsert" for action, _, _ in actions):
            raise ValueError("source mutation cannot target an empty or unregistered source")
        before_points = {str(row.id): self._point(row) for row in before}
        dependencies = await self._dependency_manifest(collection, before)
        async with self.coordinator.fence(
            source_id, dependency_sources=tuple(sorted(dependencies))
        ):
            await self._validate_selections(collection, selections, legacy=False)
        original_ids = set(before_points)
        action_by_id: dict[str, list[tuple[str, Any]]] = defaultdict(list)
        upserts: list[PointStruct] = []
        for action, point_id, value in actions:
            if action == "upsert":
                upserts.append(value)
            else:
                action_by_id[point_id].append((action, value))
        selected_ids = set(action_by_id)
        if not selected_ids.issubset(original_ids):
            raise StalePublicationError("selected active point roster changed before source census")

        output = dict(before_points)
        for point_id, point_actions in action_by_id.items():
            point = output[point_id]
            for action, value in point_actions:
                if action in {
                    "set_payload",
                    "overwrite_payload",
                    "delete_payload",
                    "clear_payload",
                }:
                    point = self._mutate_payload(point, action, value, source_id=source_id)
                elif action == "delete":
                    output.pop(point_id, None)
                    break
                elif action == "update_vectors":
                    point = self._update_vectors(point, value)
                elif action == "delete_vectors":
                    point = self._delete_vectors(point, value)
                else:
                    raise ValueError(f"unsupported mutation action {action!r}")
            if point_id in output:
                output[point_id] = point

        by_base_id = {
            str((point.payload or {}).get("publication_base_id") or point.id): point_id
            for point_id, point in output.items()
        }
        incoming_base_ids = [
            str((point.payload or {}).get("publication_base_id") or point.id) for point in upserts
        ]
        if len(incoming_base_ids) != len(set(incoming_base_ids)):
            raise ValueError("upsert contains duplicate logical publication_base_id values")
        for incoming in upserts:
            if self._point_source(incoming) != source_id:
                raise ValueError("upsert batch mixes or broadens source scope")
            payload = dict(incoming.payload or {})
            base_id = str(payload.get("publication_base_id") or incoming.id)
            payload.pop("source_revision", None)
            payload["publication_base_id"] = base_id
            replacement = incoming.model_copy(update={"id": base_id, "payload": payload})
            existing_id = by_base_id.get(base_id)
            if existing_id is not None:
                existing_payload = output[existing_id].payload or {}
                comparison_payload = dict(existing_payload)
                comparison_payload.setdefault("publication_base_id", str(output[existing_id].id))
                self._check_protected_payload(comparison_payload, payload)
                for protected_key in _IMMUTABLE_PAYLOAD_KEYS:
                    if protected_key in existing_payload and protected_key not in payload:
                        payload[protected_key] = existing_payload[protected_key]
                replacement = incoming.model_copy(update={"id": base_id, "payload": payload})
                output[existing_id] = replacement
            else:
                output[base_id] = replacement
                by_base_id[base_id] = base_id

        tombstone = not output
        if dependencies:
            for point_id, point in tuple(output.items()):
                payload = dict(point.payload or {})
                existing_dependencies = payload.get("dependency_revisions")
                if existing_dependencies is not None and existing_dependencies != dependencies:
                    raise ValueError("source roster has inconsistent dependency revisions")
                payload["dependency_revisions"] = dict(dependencies)
                output[point_id] = point.model_copy(update={"payload": payload})
        output_points = list(output.values())
        before_manifest = {
            str(row.id): representation_receipt(row)["payload_sha256"] for row in before
        }
        before_receipts = {str(row.id): representation_receipt(row) for row in before}
        request_manifest = canonical_sha256(
            {
                "operation": operation,
                "actions": [
                    (kind, point_id, self._json_value(value)) for kind, point_id, value in actions
                ],
            }
        )
        source_sha = (
            pointer.source_sha256
            if pointer is not None and pointer.source_sha256 is not None
            else canonical_sha256(
                {
                    "identity_kind": "current_qdrant_representation_manifest",
                    "representation_manifest": {
                        str(row.id): representation_receipt(row) for row in before
                    },
                }
            )
        )
        job_id = deterministic_uuid(
            "source-maintenance-v1",
            self.coordinator.corpus,
            collection,
            source_id,
            str(expected_revision),
            operation,
            request_manifest,
        )
        await base_writer.replace(
            collection,
            source_id,
            output_points,
            job_id=job_id,
            source_sha256=source_sha,
            expected_active_revision=expected_revision,
            recipe={
                "writer": "source-maintenance-cow-v1",
                "operation": operation,
                "request_manifest_sha256": request_manifest,
                "source_identity_kind": (
                    "authoritative_source_sha256"
                    if pointer is not None and pointer.source_sha256 is not None
                    else "current_qdrant_representation_manifest"
                ),
                "raw_source_byte_lineage": (
                    "known"
                    if pointer is not None and pointer.source_sha256 is not None
                    else "unknown"
                ),
                "selection_ceiling": _SELECTION_LIMIT,
                "tombstone": tombstone,
                "expected_dependencies": dependencies,
            },
            expected_before=before_manifest,
            expected_before_receipts=before_receipts,
            expected_dependencies=dependencies,
            tombstone=tombstone,
        )

    async def _dependency_manifest(
        self, collection: str, rows: Sequence[Record]
    ) -> dict[str, int | None]:
        """Capture one exact, consistent dependency set for a whole source roster."""
        if self.coordinator.corpus != "books" or collection == _RAW_BOOKS_COLLECTION:
            return {}
        active_raw = await self.coordinator.list_active(_RAW_BOOKS_COLLECTION)
        manifests: list[dict[str, int | None]] = []
        associations: set[str] = set()
        for row in rows:
            payload = row.payload or {}
            value = payload.get("dependency_revisions")
            if value is not None:
                if not isinstance(value, Mapping) or any(
                    not isinstance(key, str)
                    or (revision is not None and (type(revision) is not int or revision < 1))
                    for key, revision in value.items()
                ):
                    raise ValueError("source point has an invalid dependency revision map")
                manifests.append(dict(value))
            source_id = payload.get("source_id")
            book_id = payload.get("book_id")
            if isinstance(book_id, str):
                associations.add(book_id)
            if isinstance(source_id, str) and (
                source_id in active_raw or source_id.startswith("vo_rag-corpus-sha256:")
            ):
                associations.add(source_id)
            book_ids = payload.get("book_ids")
            if isinstance(book_ids, list):
                associations.update(value for value in book_ids if isinstance(value, str))
        if manifests:
            expected = manifests[0]
            if any(manifest != expected for manifest in manifests[1:]):
                raise ValueError("source roster has inconsistent dependency revision maps")
            if not associations.issubset(expected):
                raise ValueError(
                    "dependency map omits a canonical source associated with this roster"
                )
            return expected
        if any((row.payload or {}).get("dependency_revision_keys") for row in rows):
            raise ValueError("dependency key hashes have no recoverable canonical source map")
        if any(active_raw.get(source_id) is not None for source_id in associations):
            raise StalePublicationError(
                "legacy derived source has no dependency binding for a managed raw source"
            )
        return dict.fromkeys(sorted(associations))

    @staticmethod
    def _point(row: Record) -> PointStruct:
        if not isinstance(row.vector, dict):
            raise ValueError("source mutation requires complete named-vector read-back")
        return PointStruct(
            id=row.id,
            vector=copy.deepcopy(cast(Any, row.vector)),
            payload=copy.deepcopy(row.payload or {}),
        )

    @staticmethod
    def _mutate_payload(
        point: PointStruct, action: str, value: Any, *, source_id: str | None = None
    ) -> PointStruct:
        payload = copy.deepcopy(point.payload or {})
        if action == "set_payload":
            if not isinstance(value, Mapping):
                raise ValueError("set_payload requires a mapping")
            SourceMutationWriter._check_protected_payload(payload, value, source_id=source_id)
            payload.update(copy.deepcopy(dict(value)))
        elif action == "overwrite_payload":
            if not isinstance(value, Mapping):
                raise ValueError("overwrite_payload requires a mapping")
            SourceMutationWriter._check_protected_payload(payload, value, source_id=source_id)
            retained = {
                key: item for key, item in payload.items() if key in _IMMUTABLE_PAYLOAD_KEYS
            }
            payload = {**retained, **copy.deepcopy(dict(value))}
        elif action == "delete_payload":
            keys = list(value)
            if any(key in _IMMUTABLE_PAYLOAD_KEYS for key in keys):
                raise ValueError("cannot delete immutable source or citation identity payload")
            for key in keys:
                payload.pop(key, None)
        elif action == "clear_payload":
            payload = {key: item for key, item in payload.items() if key in _IMMUTABLE_PAYLOAD_KEYS}
        return point.model_copy(update={"payload": payload})

    @staticmethod
    def _check_protected_payload(
        existing: Mapping[str, Any],
        incoming: Mapping[str, Any],
        *,
        source_id: str | None = None,
    ) -> None:
        protected = {
            key
            for key in incoming
            if key in _IMMUTABLE_PAYLOAD_KEYS
            or any(key.startswith(f"{protected_key}.") for protected_key in _IMMUTABLE_PAYLOAD_KEYS)
        }
        if any(existing.get(key) != incoming[key] for key in protected):
            raise ValueError(
                "maintenance payload mutation cannot change source/citation/content identity"
            )
        tag_keys = {key for key in incoming if key.startswith("tf_") or key == "domain_tags"}
        if tag_keys:
            # TF classification outputs carry the content identity that the
            # tagger actually saw. Check it against the selected immutable
            # point inside the source fence; a caller-provided source label by
            # itself is not sufficient provenance.
            from vo_rag.core.corpus.tf_schema import content_sha256

            actual_source = existing.get("source_id")
            actual_content = existing.get("content")
            asserted_source = incoming.get("tf_tag_source_id")
            asserted_content = incoming.get("tf_tag_content_sha256")
            if (
                not isinstance(source_id, str)
                or actual_source != source_id
                or asserted_source != source_id
                or not isinstance(actual_content, str)
                or not actual_content
                or not isinstance(asserted_content, str)
                or asserted_content != content_sha256(actual_content)
            ):
                raise ValueError(
                    "tag mutation requires provenance bound to this source's current content"
                )

    @staticmethod
    def _update_vectors(point: PointStruct, value: Any) -> PointStruct:
        del point, value
        raise ValueError(
            "generic vector refresh requires a bound producer/source/input recipe; "
            "use the complete source writer"
        )

    @staticmethod
    def _delete_vectors(point: PointStruct, names: Any) -> PointStruct:
        del point, names
        raise ValueError(
            "delete_vectors is disabled because every configured collection modality "
            "may serve retrieval"
        )

    @staticmethod
    def _selector_source_hint(selector: Any) -> str:
        if isinstance(selector, FilterSelector):
            selector = selector.filter
        if isinstance(selector, Filter):
            for condition in selector.must or []:
                field = getattr(condition, "key", None)
                match = getattr(condition, "match", None)
                match_value = getattr(match, "value", None)
                if field == "source_id" and match_value is not None:
                    return str(match_value)
        return "(selection)"

    @staticmethod
    def _json_value(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, Mapping):
            return {str(key): SourceMutationWriter._json_value(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [SourceMutationWriter._json_value(item) for item in value]
        return value
