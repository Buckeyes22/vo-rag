"""Coordinate durable source revisions and publication through PostgreSQL."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import weakref
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

from vo_rag.core.ingest.representation_receipt import canonical_sha256

log = structlog.get_logger(__name__)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PENDING_STATES = frozenset(
    {
        "preparing",
        "staged",
        "verified",
        "partial_preserved",
        "failed_retryable",
        "blocked_missing_evidence",
    }
)
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "preparing": frozenset(
        {
            "preparing",
            "staged",
            "partial_preserved",
            "preserved_no_write",
            "blocked_missing_evidence",
            "failed_retryable",
        }
    ),
    "staged": frozenset(
        {
            "staged",
            "verified",
            "partial_preserved",
            "blocked_missing_evidence",
            "failed_retryable",
        }
    ),
    "verified": frozenset({"verified", "published", "partial_preserved", "failed_retryable"}),
    "partial_preserved": frozenset(
        {
            "partial_preserved",
            "preparing",
            "staged",
            "verified",
            "failed_retryable",
            "blocked_missing_evidence",
        }
    ),
    "failed_retryable": frozenset(
        {
            "failed_retryable",
            "preparing",
            "staged",
            "verified",
            "partial_preserved",
            "blocked_missing_evidence",
        }
    ),
    "preserved_no_write": frozenset({"preserved_no_write"}),
    "blocked_missing_evidence": frozenset(
        {
            "blocked_missing_evidence",
            "preparing",
            "staged",
            "verified",
            "partial_preserved",
            "failed_retryable",
            "preserved_no_write",
        }
    ),
    "published": frozenset({"published"}),
    "superseded": frozenset({"superseded"}),
}

_POOL_SEMAPHORES: weakref.WeakKeyDictionary[Any, asyncio.Semaphore] = weakref.WeakKeyDictionary()
_POOL_SEMAPHORES_LOCK = threading.Lock()


def _pool_fence_semaphore(pool: Any) -> asyncio.Semaphore:
    """Reserve checkpoint connections across coordinators sharing one pool."""
    # asyncpg Pool itself cannot be weak-referenced, but its queue is unique to
    # and owned by that pool, and is weak-referenceable. Keying on it avoids
    # keeping closed pools alive in this module-level coordination registry.
    pool_key = getattr(pool, "_queue", pool)
    with _POOL_SEMAPHORES_LOCK:
        semaphore = _POOL_SEMAPHORES.get(pool_key)
        if semaphore is not None:
            return semaphore
        get_max_size = getattr(pool, "get_max_size", None)
        max_size = int(get_max_size()) if callable(get_max_size) else 2
        if max_size < 2:
            raise ValueError(
                "source publication pool needs two connections for fence and checkpoint"
            )
        semaphore = asyncio.Semaphore(max_size // 2)
        _POOL_SEMAPHORES[pool_key] = semaphore
        return semaphore


class PublicationState(StrEnum):
    """Durable states for a single collection-scoped source revision job."""

    PREPARING = "preparing"
    STAGED = "staged"
    VERIFIED = "verified"
    PUBLISHED = "published"
    PARTIAL_PRESERVED = "partial_preserved"
    PRESERVED_NO_WRITE = "preserved_no_write"
    BLOCKED_MISSING_EVIDENCE = "blocked_missing_evidence"
    FAILED_RETRYABLE = "failed_retryable"
    SUPERSEDED = "superseded"


class SourcePublicationError(RuntimeError):
    """Base error for source revision coordination failures."""


class SourcePublicationPending(SourcePublicationError):
    """Raised when another incomplete source job still owns the source fence."""


class SourceFenceTimeout(SourcePublicationError):
    """Raised when the source advisory lock is not acquired within its budget."""


class StalePublicationError(SourcePublicationError):
    """Raised when the active source pointer differs from the job's expectation."""


class PublicationBindingError(SourcePublicationError):
    """Raised when a resumed job ID is presented with different immutable inputs."""


class PublicationTransitionError(SourcePublicationError):
    """Raised when a durable job state transition is not allowed."""


@dataclass(frozen=True, slots=True)
class ActivePublication:
    """A managed collection/source pointer; ``revision=None`` means legacy data."""

    corpus: str
    collection: str
    source_id: str
    revision: int | None
    source_sha256: str | None
    job_id: str | None


@dataclass(frozen=True, slots=True)
class SourcePublicationJob:
    """Identity and durable state for one collection-scoped revision job."""

    job_id: str
    corpus: str
    collection: str
    source_id: str
    revision: int
    expected_active_revision: int | None
    source_sha256: str
    state: PublicationState
    checkpoint: Mapping[str, Any]
    retention_disposition: str | None
    recipe: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceFence:
    """A transaction-scoped advisory lock held for one corpus/source pair.

    ``connection`` remains inside its transaction until the surrounding
    :meth:`SourceCoordinator.fence` context exits. Durable job and checkpoint
    writes use a separate pool connection so they survive loss of this lock
    transaction.
    """

    connection: Any
    corpus: str
    source_id: str
    resume_job_id: str | None
    locked_sources: frozenset[str] = field(default_factory=frozenset)


class SourceCoordinator:
    """Serialize source writers and persist monotonic publication decisions.

    The coordinator is corpus-scoped. Its advisory key covers every collection
    for the same source in that corpus, while publication pointers remain
    collection-specific. A pending or uncertain job blocks a different job for
    that source until the original job is resumed to a terminal disposition.
    """

    def __init__(self, pool: Any, *, corpus: str, fence_timeout_s: float = 30.0) -> None:
        if not corpus or not corpus.strip():
            raise ValueError("corpus must be non-empty")
        if fence_timeout_s <= 0:
            raise ValueError("fence_timeout_s must be positive")
        self.pool = pool
        self.corpus = corpus.strip()
        self.fence_timeout_s = fence_timeout_s
        self._fence_semaphore = _pool_fence_semaphore(pool)

    @asynccontextmanager
    async def fence(
        self,
        source_id: str,
        *,
        job_id: str | None = None,
        dependency_sources: Sequence[str] = (),
    ) -> AsyncIterator[SourceFence]:
        """Hold a transaction advisory lock while this source's writes run.

        The lock is acquired inside the transaction rather than as a session
        lock, so PgBouncer transaction pooling cannot return a locked session
        to another caller. The held connection consumes one pool slot; durable
        checkpoints require a second slot.
        """
        _require_nonempty("source_id", source_id)
        sources = {source_id, *dependency_sources}
        for dependency in sources:
            _require_nonempty("dependency_source_id", dependency)
        locks = sorted({_advisory_key(self.corpus, source) for source in sources})
        try:
            async with asyncio.timeout(self.fence_timeout_s):
                await self._fence_semaphore.acquire()
        except TimeoutError as exc:
            raise SourceFenceTimeout(
                f"source fence capacity exceeded its {self.fence_timeout_s:g}s budget"
            ) from exc
        try:
            async with self.pool.acquire() as connection, connection.transaction():
                try:
                    async with asyncio.timeout(self.fence_timeout_s):
                        for lock_class, lock_object in locks:
                            await connection.execute(
                                "SELECT pg_advisory_xact_lock($1, $2)", lock_class, lock_object
                            )
                except TimeoutError as exc:
                    raise SourceFenceTimeout(
                        f"source fence acquisition exceeded {self.fence_timeout_s:g}s"
                    ) from exc
                fence = SourceFence(connection, self.corpus, source_id, job_id, frozenset(sources))
                pending = await self._pending_job(fence)
                if pending is not None and pending != job_id:
                    raise SourcePublicationPending(
                        f"source has unfinished publication job {pending!r}; resume it first"
                    )
                yield fence
        finally:
            self._fence_semaphore.release()

    async def begin(
        self,
        fence: SourceFence,
        collection: str,
        source_id: str,
        job_id: str,
        source_sha256: str,
        expected_active_revision: int | None,
        recipe: Mapping[str, Any],
    ) -> SourcePublicationJob:
        """Register or resume a revision before any Qdrant point write.

        Revision allocation and the job row commit on a separate connection
        while ``fence`` keeps this source serialized. Repeating the same job ID
        with identical immutable bindings returns the original revision.
        """
        self._validate_fence(fence, source_id)
        _require_nonempty("collection", collection)
        _require_nonempty("job_id", job_id)
        _require_sha256(source_sha256)
        _require_revision(expected_active_revision, allow_none=True)
        recipe_value = _json_object(recipe, "recipe")

        async with self.pool.acquire() as connection, connection.transaction():
            existing = await connection.fetchrow(
                "SELECT * FROM source_publication_jobs WHERE job_id = $1 FOR UPDATE",
                job_id,
            )
            if existing is not None:
                self._assert_job_binding(
                    existing,
                    fence=fence,
                    collection=collection,
                    source_id=source_id,
                    source_sha256=source_sha256,
                    expected_active_revision=expected_active_revision,
                    recipe=recipe_value,
                )
                state = PublicationState(str(existing["state"]))
                if state in {
                    PublicationState.PRESERVED_NO_WRITE,
                    PublicationState.SUPERSEDED,
                }:
                    raise PublicationTransitionError(f"job {job_id!r} is already terminal")
                return _job_from_row(existing)

            active = await connection.fetchrow(
                """SELECT active_revision FROM source_publications
                   WHERE corpus = $1 AND collection = $2 AND source_id = $3 FOR UPDATE""",
                fence.corpus,
                collection,
                source_id,
            )
            active_revision = (
                int(active["active_revision"])
                if active and active["active_revision"] is not None
                else None
            )
            if active_revision != expected_active_revision:
                raise StalePublicationError(
                    "expected active revision "
                    f"{expected_active_revision!r}, found {active_revision!r}"
                )
            await connection.execute(
                """INSERT INTO source_publications
                       (corpus, collection, source_id, active_revision)
                   VALUES ($1, $2, $3, NULL) ON CONFLICT DO NOTHING""",
                fence.corpus,
                collection,
                source_id,
            )
            await connection.execute(
                """INSERT INTO source_publication_sequences (corpus, source_id, last_revision)
                   VALUES ($1, $2, 0) ON CONFLICT DO NOTHING""",
                fence.corpus,
                source_id,
            )
            revision = int(
                await connection.fetchval(
                    """UPDATE source_publication_sequences
                       SET last_revision = last_revision + 1
                       WHERE corpus = $1 AND source_id = $2
                       RETURNING last_revision""",
                    fence.corpus,
                    source_id,
                )
            )
            row = await connection.fetchrow(
                """INSERT INTO source_publication_jobs (
                       job_id, corpus, collection, source_id, revision,
                       expected_active_revision, source_sha256, recipe, state
                   ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'preparing')
                   RETURNING *""",
                job_id,
                fence.corpus,
                collection,
                source_id,
                revision,
                expected_active_revision,
                source_sha256,
                recipe_value,
            )
        if row is None:  # pragma: no cover - INSERT RETURNING always returns a row
            raise RuntimeError("source publication job insert returned no row")
        return _job_from_row(row)

    async def supersede_unpublished_and_begin(
        self,
        fence: SourceFence,
        *,
        failed_job_id: str,
        successor_job_id: str,
        collection: str,
        source_id: str,
        source_sha256: str,
        expected_checkpoint_sha256: str,
        expected_recipe_sha256: str,
        retention_evidence: Mapping[str, Any],
        retention_evidence_sha256: str,
        successor_recipe: Mapping[str, Any],
        expected_successor_revision: int,
    ) -> SourcePublicationJob:
        """Retain an unpublished partial job and atomically admit its successor.

        The caller captures actual before/staged receipts under the failed-job
        source fence. This operation binds that evidence without claiming the
        failed output was verified or never written. The original recipe and
        checkpoint remain unchanged; the successor records the retention proof.

        Args:
            fence: Failed-job fence for admission, or successor fence for retries.
            failed_job_id: Exact unpublished partial job to retain.
            successor_job_id: New immutable recovery identity.
            collection: Same collection as the failed job.
            source_id: Same canonical source as the failed job.
            source_sha256: Unchanged original source digest.
            expected_checkpoint_sha256: Canonical digest of the failed checkpoint.
            expected_recipe_sha256: Canonical digest of its immutable recipe.
            retention_evidence: Fresh scoped before/staged representation receipts.
            retention_evidence_sha256: Canonical digest of that actual evidence.
            successor_recipe: Exact approved immutable successor producer recipe.
            expected_successor_revision: Exact next global source sequence.

        Returns:
            The new job, or that same job on an exactly bound retry.

        Raises:
            PublicationBindingError: Inputs or actual retention scope differ.
            PublicationTransitionError: The predecessor is not unpublished partial.
            StalePublicationError: An active pointer or source sequence has changed.
        """
        self._validate_fence(fence, source_id)
        for name, value in (
            ("failed_job_id", failed_job_id),
            ("successor_job_id", successor_job_id),
            ("collection", collection),
        ):
            _require_nonempty(name, value)
        if failed_job_id == successor_job_id or fence.resume_job_id not in {
            failed_job_id,
            successor_job_id,
        }:
            raise PublicationBindingError("retirement requires the exact failed or successor fence")
        for value in (
            source_sha256,
            expected_checkpoint_sha256,
            expected_recipe_sha256,
            retention_evidence_sha256,
        ):
            _require_sha256(value)
        _require_revision(expected_successor_revision, allow_none=False)
        recipe = _json_object(successor_recipe, "successor recipe")
        retained = _json_object(retention_evidence, "retention evidence")
        if canonical_sha256(retained) != retention_evidence_sha256:
            raise PublicationBindingError("actual retention evidence digest differs")

        async with self.pool.acquire() as connection, connection.transaction():
            failed = await connection.fetchrow(
                "SELECT * FROM source_publication_jobs WHERE job_id = $1 FOR UPDATE",
                failed_job_id,
            )
            if failed is None:
                raise PublicationBindingError("failed publication job is absent")
            old_checkpoint = _decode_json_object(failed["checkpoint"])
            old_recipe = _decode_json_object(failed["recipe"])
            if (
                str(failed["corpus"]) != self.corpus
                or str(failed["source_id"]) != source_id
                or str(failed["collection"]) != collection
                or str(failed["source_sha256"]) != source_sha256
                or failed["expected_active_revision"] is not None
                or canonical_sha256(old_checkpoint) != expected_checkpoint_sha256
                or canonical_sha256(old_recipe) != expected_recipe_sha256
                or int(failed["revision"]) + 1 != expected_successor_revision
            ):
                raise PublicationBindingError("failed job immutable bindings differ")
            before = old_checkpoint.get("before_receipts")
            intent = old_checkpoint.get("writer_intent")
            staged = retained.get("staged_receipts")
            if (
                retained.get("corpus") != self.corpus
                or retained.get("collection") != collection
                or retained.get("source_id") != source_id
                or retained.get("source_sha256") != source_sha256
                or retained.get("failed_revision") != int(failed["revision"])
                or "active_revision" not in retained
                or retained["active_revision"] is not None
                or not isinstance(before, dict)
                or not before
                or retained.get("before_receipts") != before
                or not isinstance(intent, dict)
                or not intent
                or not isinstance(staged, dict)
                or set(staged) != set(intent)
                or set(before) & set(staged)
                or not isinstance(old_checkpoint.get("expected_point_ids"), list)
                or len(old_checkpoint["expected_point_ids"]) != len(intent)
                or set(old_checkpoint["expected_point_ids"]) != set(intent)
            ):
                raise PublicationBindingError("actual retained before/staged roster differs")
            for roster in (before, intent, staged):
                _validate_retained_receipts(roster)
            audit = {
                "failed_job_id": failed_job_id,
                "failed_revision": int(failed["revision"]),
                "failed_checkpoint_sha256": expected_checkpoint_sha256,
                "failed_recipe_sha256": expected_recipe_sha256,
                "actual_retention_evidence_sha256": retention_evidence_sha256,
                "actual_retention_evidence": retained,
                "disposition": "superseded_unpublished_staged_points_retained",
            }
            existing = await connection.fetchrow(
                "SELECT * FROM source_publication_jobs WHERE job_id = $1 FOR UPDATE",
                successor_job_id,
            )
            if existing is not None:
                if (
                    str(failed["state"]) != PublicationState.SUPERSEDED.value
                    or str(existing["corpus"]) != self.corpus
                    or str(existing["source_id"]) != source_id
                    or str(existing["collection"]) != collection
                    or str(existing["source_sha256"]) != source_sha256
                    or existing["expected_active_revision"] is not None
                    or int(existing["revision"]) != expected_successor_revision
                    or _decode_json_object(existing["recipe"]) != recipe
                    or _decode_json_object(existing["checkpoint"]).get(
                        "unpublished_predecessor_retention"
                    )
                    != audit
                ):
                    raise PublicationBindingError("successor retry immutable bindings differ")
                return _job_from_row(existing)
            if (
                str(failed["state"]) != PublicationState.PARTIAL_PRESERVED.value
                or fence.resume_job_id != failed_job_id
                or failed["published_at"] is not None
            ):
                raise PublicationTransitionError(
                    "only an unpublished partial job can be superseded"
                )
            pointer = await connection.fetchrow(
                "SELECT * FROM source_publications WHERE corpus = $1 AND collection = $2 "
                "AND source_id = $3 FOR UPDATE",
                self.corpus,
                collection,
                source_id,
            )
            if (
                pointer is None
                or pointer["active_revision"] is not None
                or pointer["active_job_id"] is not None
            ):
                raise StalePublicationError(
                    "retirement requires the unchanged explicit null pointer"
                )
            sequence = await connection.fetchval(
                "SELECT last_revision FROM source_publication_sequences "
                "WHERE corpus = $1 AND source_id = $2 FOR UPDATE",
                self.corpus,
                source_id,
            )
            if sequence != expected_successor_revision - 1:
                raise StalePublicationError("successor sequence differs from exact approval")
            await connection.execute(
                "UPDATE source_publication_jobs SET state = 'superseded' WHERE job_id = $1",
                failed_job_id,
            )
            await connection.execute(
                "UPDATE source_publication_sequences SET last_revision = $3 "
                "WHERE corpus = $1 AND source_id = $2",
                self.corpus,
                source_id,
                expected_successor_revision,
            )
            row = await connection.fetchrow(
                "INSERT INTO source_publication_jobs (job_id, corpus, collection, source_id, "
                "revision, expected_active_revision, source_sha256, recipe, state, checkpoint) "
                "VALUES ($1, $2, $3, $4, $5, NULL, $6, $7, 'preparing', $8) RETURNING *",
                successor_job_id,
                self.corpus,
                collection,
                source_id,
                expected_successor_revision,
                source_sha256,
                recipe,
                {"unpublished_predecessor_retention": audit},
            )
        if row is None:  # pragma: no cover - INSERT RETURNING always returns a row
            raise RuntimeError("successor publication insert returned no row")
        return _job_from_row(row)

    async def assert_legacy_write_allowed(
        self,
        fence: SourceFence,
        source_id: str,
        dependency_sources: Sequence[str] = (),
    ) -> None:
        """Allow an in-place compatibility write only before publication starts.

        This is the bridge for a reader-capable release with revision writes
        disabled. It must not touch a source after any collection has an active
        revision or while any publication job could have written staged points.
        Every source and dependency is checked under the same transaction-scoped
        advisory locks used by revision writers.
        """
        self._validate_fence(fence, source_id)
        if fence.resume_job_id is not None:
            raise ValueError("legacy writes cannot resume a source publication job")
        required = {source_id, *dependency_sources}
        if not required.issubset(fence.locked_sources):
            raise ValueError("legacy write fence does not cover every source dependency")
        if any(not isinstance(source, str) or not source.strip() for source in required):
            raise ValueError("legacy write source dependencies must be non-empty")

        active = await fence.connection.fetchrow(
            """SELECT source_id, collection, active_revision
                 FROM source_publications
                WHERE corpus = $1 AND source_id = ANY($2::text[])
                  AND active_revision IS NOT NULL
                ORDER BY source_id, collection LIMIT 1""",
            self.corpus,
            sorted(required),
        )
        if active is not None:
            raise SourcePublicationPending(
                "legacy write refused for a source with an active publication revision"
            )

        job = await fence.connection.fetchrow(
            """SELECT source_id, job_id, state
                 FROM source_publication_jobs
                WHERE corpus = $1 AND source_id = ANY($2::text[])
                  AND state <> 'preserved_no_write'
                ORDER BY source_id, updated_at, job_id LIMIT 1""",
            self.corpus,
            sorted(required),
        )
        if job is not None:
            raise SourcePublicationPending(
                "legacy write refused while a source publication job may have written points"
            )

    async def register_revision(
        self,
        fence: SourceFence,
        collection: str,
        source_id: str,
        job_id: str,
        source_sha256: str,
        expected_active_revision: int | None,
        recipe: Mapping[str, Any],
    ) -> SourcePublicationJob:
        """Alias for :meth:`begin` used by writer adapters."""
        return await self.begin(
            fence,
            collection,
            source_id,
            job_id,
            source_sha256,
            expected_active_revision,
            recipe,
        )

    async def checkpoint(
        self,
        fence: SourceFence,
        job_id: str,
        state: PublicationState | str,
        data: Mapping[str, Any],
    ) -> SourcePublicationJob:
        """Commit a durable checkpoint independently of the lock transaction."""
        self._validate_fence(fence, fence.source_id)
        target_state = PublicationState(state)
        checkpoint_value = _json_object(data, "checkpoint")
        if target_state is PublicationState.VERIFIED:
            _validate_verification(checkpoint_value)
        if target_state is PublicationState.PRESERVED_NO_WRITE:
            _validate_no_write_disposition(checkpoint_value)
        async with self.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT * FROM source_publication_jobs WHERE job_id = $1 FOR UPDATE",
                job_id,
            )
            if row is None:
                raise SourcePublicationError(f"unknown source publication job {job_id!r}")
            self._assert_fence_row(row, fence)
            current_state = PublicationState(str(row["state"]))
            if target_state.value not in _ALLOWED_TRANSITIONS[current_state.value]:
                raise PublicationTransitionError(
                    f"job state cannot move from {current_state.value!r} to {target_state.value!r}"
                )
            if target_state is PublicationState.PUBLISHED:
                raise PublicationTransitionError("use publish() to atomically switch the pointer")
            updated = await connection.fetchrow(
                """UPDATE source_publication_jobs
                   SET state = $2, checkpoint = $3,
                       verified_at = CASE
                           WHEN $2 = 'verified' THEN COALESCE(verified_at, NOW())
                           ELSE verified_at
                       END
                   WHERE job_id = $1 RETURNING *""",
                job_id,
                target_state.value,
                checkpoint_value,
            )
        if updated is None:  # pragma: no cover - guarded by the selected row
            raise RuntimeError("source publication checkpoint update returned no row")
        return _job_from_row(updated)

    async def publish(self, fence: SourceFence, job_id: str) -> ActivePublication:
        """CAS the durable pointer after a verified job, retaining old points."""
        self._validate_fence(fence, fence.source_id)
        async with self.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT * FROM source_publication_jobs WHERE job_id = $1 FOR UPDATE",
                job_id,
            )
            if row is None:
                raise SourcePublicationError(f"unknown source publication job {job_id!r}")
            self._assert_fence_row(row, fence)
            state = PublicationState(str(row["state"]))
            if state is PublicationState.PUBLISHED:
                pointer = await connection.fetchrow(
                    """SELECT corpus, collection, source_id, active_revision,
                              source_sha256, active_job_id
                       FROM source_publications
                       WHERE corpus = $1 AND collection = $2 AND source_id = $3""",
                    fence.corpus,
                    row["collection"],
                    fence.source_id,
                )
                if pointer is None:
                    raise SourcePublicationError("published job has no active source pointer")
                if (
                    pointer["active_job_id"] != job_id
                    or pointer["active_revision"] != row["revision"]
                ):
                    raise StalePublicationError(
                        f"published job {job_id!r} was superseded by a newer source revision"
                    )
                return _active_from_row(pointer)
            if state is not PublicationState.VERIFIED:
                raise PublicationTransitionError(
                    f"job {job_id!r} must be verified before publication; state={state.value!r}"
                )
            _validate_verification(_decode_json_object(row["checkpoint"]))
            pointer = await connection.fetchrow(
                """SELECT active_revision FROM source_publications
                   WHERE corpus = $1 AND collection = $2 AND source_id = $3 FOR UPDATE""",
                fence.corpus,
                row["collection"],
                fence.source_id,
            )
            actual = (
                int(pointer["active_revision"])
                if pointer and pointer["active_revision"] is not None
                else None
            )
            expected = (
                int(row["expected_active_revision"])
                if row["expected_active_revision"] is not None
                else None
            )
            if actual != expected:
                raise StalePublicationError(
                    f"publication CAS expected {expected!r}, found {actual!r}"
                )
            await connection.execute(
                """UPDATE source_publications
                   SET active_revision = $4, source_sha256 = $5, active_job_id = $6
                   WHERE corpus = $1 AND collection = $2 AND source_id = $3""",
                fence.corpus,
                row["collection"],
                fence.source_id,
                int(row["revision"]),
                str(row["source_sha256"]),
                job_id,
            )
            await connection.execute(
                """UPDATE source_publication_jobs
                   SET state = 'published', published_at = NOW(),
                       retention_disposition = 'retained_old_revision'
                   WHERE job_id = $1""",
                job_id,
            )
        return ActivePublication(
            fence.corpus,
            str(row["collection"]),
            fence.source_id,
            int(row["revision"]),
            str(row["source_sha256"]),
            job_id,
        )

    async def read_active(self, collection: str, source_id: str) -> ActivePublication | None:
        """Return a managed pointer or ``None`` for an unregistered legacy source."""
        row = await self.pool.fetchrow(
            """SELECT corpus, collection, source_id, active_revision, source_sha256, active_job_id
               FROM source_publications
               WHERE corpus = $1 AND collection = $2 AND source_id = $3""",
            self.corpus,
            collection,
            source_id,
        )
        return _active_from_row(row) if row is not None else None

    async def read_job(self, job_id: str) -> SourcePublicationJob | None:
        """Return one durable job in this corpus for resume or ack-loss recovery."""
        _require_nonempty("job_id", job_id)
        row = await self.pool.fetchrow(
            "SELECT * FROM source_publication_jobs WHERE corpus = $1 AND job_id = $2",
            self.corpus,
            job_id,
        )
        return _job_from_row(row) if row is not None else None

    async def list_active(self, collection: str) -> dict[str, int | None]:
        """Return every managed pointer for a collection without truncation.

        ``None`` denotes a source explicitly registered as legacy-active. An
        unregistered source is absent from the mapping and remains outside the
        pointer filter during rollout.
        """
        rows = await self.pool.fetch(
            """SELECT source_id, active_revision FROM source_publications
               WHERE corpus = $1 AND collection = $2 ORDER BY source_id""",
            self.corpus,
            collection,
        )
        return {
            str(row["source_id"]): (
                int(row["active_revision"]) if row["active_revision"] is not None else None
            )
            for row in rows
        }

    async def _pending_job(self, fence: SourceFence) -> str | None:
        row = await self.pool.fetchrow(
            """SELECT job_id FROM source_publication_jobs
               WHERE corpus = $1 AND source_id = $2 AND state = ANY($3::text[])
               ORDER BY updated_at, job_id LIMIT 1""",
            fence.corpus,
            fence.source_id,
            sorted(_PENDING_STATES),
        )
        return str(row["job_id"]) if row is not None else None

    def _validate_fence(self, fence: SourceFence, source_id: str) -> None:
        if fence.corpus != self.corpus or fence.source_id != source_id:
            raise ValueError("source fence does not match this coordinator and source")

    @staticmethod
    def _assert_fence_row(row: Any, fence: SourceFence) -> None:
        if str(row["corpus"]) != fence.corpus or str(row["source_id"]) != fence.source_id:
            raise ValueError("source publication job does not belong to this fence")
        if fence.resume_job_id is not None and str(row["job_id"]) != fence.resume_job_id:
            raise ValueError("source publication job differs from the fence resume job")

    @classmethod
    def _assert_job_binding(
        cls,
        row: Any,
        *,
        fence: SourceFence,
        collection: str,
        source_id: str,
        source_sha256: str,
        expected_active_revision: int | None,
        recipe: Mapping[str, Any],
    ) -> None:
        cls._assert_fence_row(row, fence)
        bindings_match = (
            str(row["collection"]) == collection
            and str(row["source_id"]) == source_id
            and str(row["source_sha256"]) == source_sha256
            and _optional_int(row["expected_active_revision"]) == expected_active_revision
            and _decode_json_object(row["recipe"]) == dict(recipe)
        )
        if not bindings_match:
            raise PublicationBindingError(
                f"job ID {row['job_id']!r} was already bound to different revision inputs"
            )


def _advisory_key(corpus: str, source_id: str) -> tuple[int, int]:
    """Map a namespaced source to PostgreSQL's two-int transaction lock key."""
    digest = hashlib.blake2b(
        f"vo-rag-source-v1\0{corpus}\0{source_id}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest[:4], "big", signed=True), int.from_bytes(
        digest[4:], "big", signed=True
    )


def _job_from_row(row: Any) -> SourcePublicationJob:
    return SourcePublicationJob(
        job_id=str(row["job_id"]),
        corpus=str(row["corpus"]),
        collection=str(row["collection"]),
        source_id=str(row["source_id"]),
        revision=int(row["revision"]),
        expected_active_revision=_optional_int(row["expected_active_revision"]),
        source_sha256=str(row["source_sha256"]),
        state=PublicationState(str(row["state"])),
        checkpoint=_decode_json_object(row["checkpoint"]),
        recipe=_decode_json_object(row["recipe"]),
        retention_disposition=(
            str(row["retention_disposition"]) if row["retention_disposition"] is not None else None
        ),
    )


def _active_from_row(row: Any) -> ActivePublication:
    return ActivePublication(
        corpus=str(row["corpus"]),
        collection=str(row["collection"]),
        source_id=str(row["source_id"]),
        revision=_optional_int(row["active_revision"]),
        source_sha256=str(row["source_sha256"]) if row["source_sha256"] is not None else None,
        job_id=str(row["active_job_id"]) if row["active_job_id"] is not None else None,
    )


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _decode_json_object(value: Any) -> dict[str, Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, Mapping):
        raise TypeError("stored JSON value must be an object")
    return dict(parsed)


def _json_object(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    # Normalize nested Pydantic/enum values once before comparing resumed jobs.
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    parsed = json.loads(encoded)
    if not isinstance(parsed, dict):  # pragma: no cover - JSON object round-trips as dict
        raise TypeError(f"{name} must be a JSON object")
    return parsed


def _require_nonempty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_sha256(value: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError("source_sha256 must be a lowercase 64-character SHA-256 hex digest")


def _require_revision(value: int | None, *, allow_none: bool) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("revision must be a positive integer")


def _validate_retained_receipts(receipts: Mapping[str, Any]) -> None:
    """Require complete observed fingerprints without asserting writer equality."""
    for point_id, receipt in receipts.items():
        if not isinstance(receipt, dict) or receipt.get("point_id") != point_id:
            raise PublicationBindingError("retained receipt identity differs")
        fingerprints = [
            receipt.get(key)
            for key in (
                "payload_sha256",
                "content_sha256",
                "embedding_provenance_sha256",
                "tag_binding_sha256",
            )
        ]
        vectors = receipt.get("vectors")
        if not isinstance(vectors, dict) or not vectors:
            raise PublicationBindingError("retained receipt lacks observed modalities")
        fingerprints.extend(vectors.values())
        if any(
            not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in fingerprints
        ):
            raise PublicationBindingError("retained receipt contains an invalid fingerprint")


def _validate_no_write_disposition(checkpoint: Mapping[str, Any]) -> None:
    """Require explicit evidence before a no-write job releases its source."""
    proof = checkpoint.get("no_write")
    if not isinstance(proof, Mapping):
        raise PublicationTransitionError("preserved_no_write requires a no_write proof object")
    attempts = proof.get("qdrant_write_attempts")
    if isinstance(attempts, bool) or attempts != 0:
        raise PublicationTransitionError("preserved_no_write requires zero Qdrant write attempts")
    if proof.get("acknowledgement_unknown") is not False:
        raise PublicationTransitionError("preserved_no_write requires known write acknowledgements")
    reason = proof.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise PublicationTransitionError("preserved_no_write requires a named disposition reason")


def _validate_verification(checkpoint: Mapping[str, Any]) -> None:
    """Require an exact point roster and complete writer/read-back receipts."""
    verification = checkpoint.get("verification")
    if not isinstance(verification, Mapping):
        raise PublicationTransitionError("verified checkpoint requires a verification object")

    expected_ids = verification.get("expected_point_ids")
    readback_ids = verification.get("readback_point_ids")
    receipts = verification.get("point_receipts")
    if receipts is None:
        expected_receipts = verification.get("expected_receipts")
        readback_receipts = verification.get("readback_receipts")
        if (
            not isinstance(expected_receipts, Mapping)
            or not isinstance(readback_receipts, Mapping)
            or dict(expected_receipts) != dict(readback_receipts)
        ):
            raise PublicationTransitionError("writer and read-back receipt manifests differ")
        receipts = expected_receipts
    tombstone = verification.get("empty_source_tombstone") is True
    if tombstone:
        before = verification.get("tombstone_before_payloads")
        if (
            not isinstance(before, Mapping)
            or not before
            or any(
                not isinstance(point_id, str)
                or not isinstance(digest, str)
                or not _SHA256_RE.fullmatch(digest)
                for point_id, digest in before.items()
            )
        ):
            raise PublicationTransitionError("tombstone requires the exact before payload roster")
        from vo_rag.core.ingest.representation_receipt import canonical_sha256

        if canonical_sha256(dict(before)) != verification.get("tombstone_before_sha256"):
            raise PublicationTransitionError("tombstone before roster digest mismatch")
        if expected_ids != [] or verification.get("readback_point_ids") != []:
            raise PublicationTransitionError("tombstone must have an empty complete after roster")
    if not isinstance(expected_ids, list) or (not expected_ids and not tombstone):
        raise PublicationTransitionError(
            "verification requires a non-empty, unique complete expected point roster"
        )
    if any(not isinstance(value, str) or not value for value in expected_ids):
        raise PublicationTransitionError("expected point roster contains an invalid point ID")
    expected_set = set(expected_ids)
    if len(expected_set) != len(expected_ids):
        raise PublicationTransitionError("expected point roster contains duplicate IDs")
    if (
        not isinstance(readback_ids, list)
        or any(not isinstance(value, str) or not value for value in readback_ids)
        or len(readback_ids) != len(set(readback_ids))
        or set(readback_ids) != expected_set
    ):
        raise PublicationTransitionError("read-back point roster differs from expected roster")
    if not isinstance(receipts, Mapping) or set(receipts) != expected_set:
        raise PublicationTransitionError(
            "representation receipts do not cover every expected point"
        )

    for point_id in expected_ids:
        receipt = receipts[point_id]
        if not isinstance(receipt, Mapping) or receipt.get("verification") != (
            "payload_exact_modalities_complete_numeric_transform_verified"
        ):
            raise PublicationTransitionError(
                f"point {point_id!r} lacks a complete representation verification"
            )
        writer = receipt.get("writer")
        stored = receipt.get("stored")
        if not isinstance(writer, Mapping) or not isinstance(stored, Mapping):
            raise PublicationTransitionError(
                f"point {point_id!r} lacks writer and stored fingerprints"
            )
        if writer.get("point_id") != point_id or stored.get("point_id") != point_id:
            raise PublicationTransitionError(f"point {point_id!r} receipt identity differs")
        for key in (
            "payload_sha256",
            "content_sha256",
            "embedding_provenance_sha256",
            "tag_binding_sha256",
        ):
            fingerprint = writer.get(key)
            if (
                not isinstance(fingerprint, str)
                or not _SHA256_RE.fullmatch(fingerprint)
                or stored.get(key) != fingerprint
            ):
                raise PublicationTransitionError(
                    f"point {point_id!r} {key} differs between writer and read-back"
                )
        writer_vectors = writer.get("vectors")
        stored_vectors = stored.get("vectors")
        if (
            not isinstance(writer_vectors, Mapping)
            or not writer_vectors
            or not isinstance(stored_vectors, Mapping)
            or set(writer_vectors) != set(stored_vectors)
        ):
            raise PublicationTransitionError(f"point {point_id!r} modality inventory differs")
        for fingerprints in (writer_vectors, stored_vectors):
            if any(
                not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)
                for digest in fingerprints.values()
            ):
                raise PublicationTransitionError(
                    f"point {point_id!r} contains an invalid modality fingerprint"
                )

    manifest_sha256 = verification.get("manifest_sha256")
    if manifest_sha256 is None:
        expected_manifest = verification.get("expected_manifest_sha256")
        readback_manifest = verification.get("readback_manifest_sha256")
        if expected_manifest != readback_manifest:
            raise PublicationTransitionError("expected and read-back manifest digests differ")
        manifest_sha256 = expected_manifest
    if not isinstance(manifest_sha256, str) or not _SHA256_RE.fullmatch(manifest_sha256):
        raise PublicationTransitionError("verification requires a canonical manifest SHA-256")
    from vo_rag.core.ingest.representation_receipt import canonical_sha256

    if canonical_sha256(dict(receipts)) != manifest_sha256:
        raise PublicationTransitionError("verification manifest digest does not cover its receipts")


__all__ = [
    "ActivePublication",
    "PublicationBindingError",
    "PublicationState",
    "PublicationTransitionError",
    "SourceCoordinator",
    "SourceFence",
    "SourceFenceTimeout",
    "SourcePublicationError",
    "SourcePublicationJob",
    "SourcePublicationPending",
    "StalePublicationError",
]
