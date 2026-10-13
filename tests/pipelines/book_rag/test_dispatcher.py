"""Postgres-polling book_rag dispatcher worker tests."""

from __future__ import annotations

import asyncio
import runpy
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import pytest

from vo_rag.core.models.envelope import (
    EnvelopeStatus,
    IntakeEnvelope,
    IntakeSource,
    Priority,
    Sovereignty,
    TargetPipeline,
)
from vo_rag.pipelines.book_rag.workflow import PipelineOutput

if TYPE_CHECKING:
    from vo_rag.core.config.settings import Settings

_ENVELOPE_ID = "01KT6XE8M1MW42M0Y0JJKRZSVP"  # pragma: allowlist secret
_TS = datetime(2026, 1, 1, tzinfo=UTC)


def test_module_entrypoint_invokes_dispatcher_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def _run(coroutine: object) -> None:
        calls.append(coroutine)
        coroutine.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(asyncio, "run", _run)

    runpy.run_module("vo_rag.pipelines.book_rag.dispatcher", run_name="__main__")

    assert len(calls) == 1


def _envelope_row(*, status: EnvelopeStatus = EnvelopeStatus.RECEIVED) -> dict[str, Any]:
    return {
        "id": _ENVELOPE_ID,
        "source_type": IntakeSource.FILE_DROP,
        "source_data": {},
        "raw_content": {
            "garage_key": "drops/book.pdf",
            "file_hash": "a" * 64,
            "object_verification": {
                "status": "verified",
                "sha256": "a" * 64,
                "size_bytes": 42,
            },
        },
        "metadata": {},
        "triage_result": None,
        "status": status,
        "target_pipeline": None,
        "priority": Priority.MEDIUM,
        "error": None,
        "sovereignty": Sovereignty.CLOUD_OK,
        "created_at": _TS,
        "updated_at": _TS,
    }


class _FakeTransaction:
    async def __aenter__(self) -> _FakeTransaction:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


@dataclass
class _FakePool:
    row: dict[str, Any] | None
    claim_queries: list[str] = field(default_factory=list)
    claim_args: list[tuple[Any, ...]] = field(default_factory=list)
    execute_calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    claim_delay: asyncio.Event | None = None

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(_FakeConnection(self))

    async def execute(self, query: str, *args: Any) -> str:
        self.execute_calls.append((query, args))
        if self.row is not None and "UPDATE intake_envelopes" in query:
            if len(args) >= 2:
                self.row["status"] = args[1]
            if len(args) >= 3:
                self.row["error"] = args[2]
        return "UPDATE 1"


@dataclass
class _FakeAcquire:
    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


@dataclass
class _FakeConnection:
    pool: _FakePool

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def fetchrow(self, query: str, *args: Any) -> Mapping[str, Any] | None:
        self.pool.claim_queries.append(query)
        self.pool.claim_args.append(args)
        if self.pool.claim_delay is not None:
            await self.pool.claim_delay.wait()
        if self.pool.row is None or self.pool.row["status"] is not EnvelopeStatus.RECEIVED:
            return None
        self.pool.row["status"] = EnvelopeStatus.PROCESSING
        return dict(self.pool.row)


class _FakeLog:
    def __init__(self) -> None:
        self.errors: list[tuple[str, dict[str, Any]]] = []
        self.infos: list[tuple[str, dict[str, Any]]] = []
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def error(self, event: str, **kwargs: Any) -> None:
        self.errors.append((event, kwargs))

    def info(self, event: str, **kwargs: Any) -> None:
        self.infos.append((event, kwargs))

    def warning(self, event: str, **kwargs: Any) -> None:
        self.warnings.append((event, kwargs))


@pytest.mark.asyncio
async def test_claim_next_envelope_emits_skip_locked_claim_and_only_one_worker_wins() -> None:
    from vo_rag.pipelines.book_rag.dispatcher import claim_next_envelope

    release_claims = asyncio.Event()
    pool = _FakePool(row=_envelope_row(), claim_delay=release_claims)
    conn = _FakeConnection(pool)

    first = asyncio.create_task(claim_next_envelope(conn))
    second = asyncio.create_task(claim_next_envelope(conn))
    await asyncio.sleep(0)
    release_claims.set()

    results = await asyncio.gather(first, second)

    claimed = [result for result in results if result is not None]
    assert len(claimed) == 1
    assert claimed[0].id == _ENVELOPE_ID
    assert claimed[0].status is EnvelopeStatus.PROCESSING
    assert sum(result is None for result in results) == 1
    assert len(pool.claim_queries) == 2
    assert all("FOR UPDATE SKIP LOCKED" in query for query in pool.claim_queries)
    assert all("UPDATE intake_envelopes" in query for query in pool.claim_queries)


@pytest.mark.asyncio
async def test_claim_next_envelope_passes_cutover_boundary_to_postgres() -> None:
    from vo_rag.pipelines.book_rag.dispatcher import claim_next_envelope

    pool = _FakePool(row=_envelope_row())
    conn = _FakeConnection(pool)
    cutover = datetime(2026, 7, 16, tzinfo=UTC)

    claimed = await claim_next_envelope(
        conn,
        stale_timeout=timedelta(hours=2),
        claim_not_before=cutover,
    )

    assert claimed is not None
    assert "created_at >= $4" in pool.claim_queries[0]
    assert "source_type = $5" in pool.claim_queries[0]
    assert "target_pipeline = $6" in pool.claim_queries[0]
    assert "raw_content->>'garage_key'" in pool.claim_queries[0]
    assert "raw_content->>'file_hash'" in pool.claim_queries[0]
    assert "object_verification,status" in pool.claim_queries[0]
    assert "object_verification,sha256" in pool.claim_queries[0]
    assert "SET status = $7" in pool.claim_queries[0]
    assert pool.claim_args[0] == (
        EnvelopeStatus.RECEIVED,
        EnvelopeStatus.PROCESSING,
        7200.0,
        cutover,
        IntakeSource.FILE_DROP,
        TargetPipeline.BOOK_RAG,
        EnvelopeStatus.PROCESSING,
        None,
        600.0,  # live-stream heartbeat bound (VOR-1434)
        None,  # lane_class unset (VOR-1434): claim anything
        21600.0,  # giant fallback age (VOR-1434)
    )


@pytest.mark.asyncio
async def test_claim_next_envelope_observes_stale_reclaim() -> None:
    from vo_rag.core.observability.metrics import dispatcher_claims_total
    from vo_rag.pipelines.book_rag.dispatcher import claim_next_envelope

    row = _envelope_row(status=EnvelopeStatus.PROCESSING)
    row["claimed_from_status"] = EnvelopeStatus.PROCESSING
    before = dispatcher_claims_total.labels(kind="stale_reclaim")._value.get()  # noqa: SLF001
    claimed = await claim_next_envelope(_FakeAuditConnection(row))
    assert claimed is not None
    assert dispatcher_claims_total.labels(kind="stale_reclaim")._value.get() == before + 1  # noqa: SLF001


def test_dispatcher_cutover_timestamp_is_timezone_aware_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import dispatcher as dispatcher_mod

    monkeypatch.setenv("VORAG_BOOK_RAG_DISPATCHER_CLAIM_NOT_BEFORE", "2026-07-16T00:30:00Z")
    assert dispatcher_mod._env_datetime(  # noqa: SLF001
        "VORAG_BOOK_RAG_DISPATCHER_CLAIM_NOT_BEFORE"
    ) == datetime(2026, 7, 16, 0, 30, tzinfo=UTC)

    monkeypatch.setenv("VORAG_BOOK_RAG_DISPATCHER_CLAIM_NOT_BEFORE", "2026-07-16T00:30:00")
    with pytest.raises(ValueError, match="must include a timezone"):
        dispatcher_mod._env_datetime(  # noqa: SLF001
            "VORAG_BOOK_RAG_DISPATCHER_CLAIM_NOT_BEFORE"
        )


@pytest.mark.asyncio
async def test_ingest_exception_marks_envelope_failed_and_logs_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import dispatcher as dispatcher_mod

    pool = _FakePool(row=_envelope_row())
    stop_event = asyncio.Event()
    log = _FakeLog()

    async def _raise_ingest(
        envelope: IntakeEnvelope,
        *,
        settings: object,
        pg_pool: object | None = None,
    ) -> PipelineOutput:
        assert envelope.status is EnvelopeStatus.PROCESSING
        assert pg_pool is pool
        stop_event.set()
        raise RuntimeError("ingest exploded")

    monkeypatch.setattr(dispatcher_mod, "run_book_rag_ingest", _raise_ingest)
    monkeypatch.setattr(dispatcher_mod, "log", log)
    monkeypatch.setattr(dispatcher_mod, "write_dispatcher_heartbeat", lambda state: None)

    await dispatcher_mod.run_dispatcher_loop(
        pool,
        cast("Settings", object()),
        poll_interval=0.01,
        stop_event=stop_event,
        heartbeat_interval=0,
    )

    assert pool.row is not None
    assert pool.row["status"] is EnvelopeStatus.FAILED
    assert pool.row["error"] == "ingest exploded"
    assert log.errors[0][0] == "book_rag_dispatcher_ingest_failed"
    assert log.errors[0][1]["envelope_id"] == _ENVELOPE_ID
    assert log.errors[0][1]["error"] == "ingest exploded"
    assert log.errors[0][1]["exc_info"] is True


@pytest.mark.asyncio
async def test_graceful_stop_waits_for_claimed_envelope_before_loop_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import dispatcher as dispatcher_mod

    pool = _FakePool(row=_envelope_row())
    stop_event = asyncio.Event()
    ingest_started = asyncio.Event()
    release_ingest = asyncio.Event()

    async def _slow_ingest(
        envelope: IntakeEnvelope,
        *,
        settings: object,
        pg_pool: object | None = None,
    ) -> PipelineOutput:
        assert envelope.status is EnvelopeStatus.PROCESSING
        assert pg_pool is pool
        ingest_started.set()
        await release_ingest.wait()
        return PipelineOutput(
            source_id="source-1",
            status=EnvelopeStatus.COMPLETED,
            qdrant_point_ids=[],
            raw_point_ids=[],
            sterile_point_ids=[],
            smoke_decision=None,
            chunk_count=0,
        )

    monkeypatch.setattr(dispatcher_mod, "run_book_rag_ingest", _slow_ingest)
    monkeypatch.setattr(dispatcher_mod, "write_dispatcher_heartbeat", lambda state: None)

    task = asyncio.create_task(
        dispatcher_mod.run_dispatcher_loop(
            pool,
            cast("Settings", object()),
            poll_interval=0.01,
            stop_event=stop_event,
            heartbeat_interval=0,
        )
    )
    await asyncio.wait_for(ingest_started.wait(), timeout=1.0)

    stop_event.set()
    await asyncio.sleep(0)

    assert task.done() is False
    assert pool.row is not None
    assert pool.row["status"] is EnvelopeStatus.PROCESSING

    release_ingest.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert cast("EnvelopeStatus", pool.row["status"]) is EnvelopeStatus.COMPLETED


@pytest.mark.asyncio
async def test_graceful_stop_requeues_claim_when_drain_deadline_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag import dispatcher as dispatcher_mod

    pool = _FakePool(row=_envelope_row())
    stop_event = asyncio.Event()
    ingest_started = asyncio.Event()

    async def _blocked_ingest(*args: Any, **kwargs: Any) -> PipelineOutput:
        ingest_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(dispatcher_mod, "run_book_rag_ingest", _blocked_ingest)
    monkeypatch.setattr(dispatcher_mod, "write_dispatcher_heartbeat", lambda state: None)

    task = asyncio.create_task(
        dispatcher_mod.run_dispatcher_loop(
            pool,
            cast("Settings", object()),
            poll_interval=0.01,
            stop_event=stop_event,
            heartbeat_interval=0,
            shutdown_grace=0,
        )
    )
    await asyncio.wait_for(ingest_started.wait(), timeout=1.0)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert pool.row is not None
    assert pool.row["status"] is EnvelopeStatus.RECEIVED
    assert "retryable" in str(pool.row["error"])


@pytest.mark.asyncio
async def test_cancelled_ingest_is_observable(monkeypatch: pytest.MonkeyPatch) -> None:
    from vo_rag.core.observability.metrics import dispatcher_cancellations_total
    from vo_rag.pipelines.book_rag import dispatcher as dispatcher_mod

    pool = _FakePool(row=_envelope_row(status=EnvelopeStatus.PROCESSING))
    started = asyncio.Event()

    async def _block(*args: Any, **kwargs: Any) -> PipelineOutput:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(dispatcher_mod, "run_book_rag_ingest", _block)
    envelope = IntakeEnvelope.model_validate(pool.row)
    before = dispatcher_cancellations_total._value.get()  # noqa: SLF001
    task = asyncio.create_task(
        dispatcher_mod._run_claimed_envelope(  # noqa: SLF001
            pool,
            cast("Settings", object()),
            envelope,
            heartbeat_interval=0,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert dispatcher_cancellations_total._value.get() == before + 1  # noqa: SLF001
    assert pool.row is not None
    assert pool.row["status"] is EnvelopeStatus.RECEIVED
    assert "retryable" in str(pool.row["error"])


@dataclass
class _FakeAuditConnection:
    row: Mapping[str, Any] | None
    queries: list[str] = field(default_factory=list)
    args: list[tuple[Any, ...]] = field(default_factory=list)

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def fetchrow(self, query: str, *args: Any) -> Mapping[str, Any] | None:
        self.queries.append(query)
        self.args.append(args)
        return self.row


@pytest.mark.asyncio
async def test_audit_claim_cutoff_warns_when_cutoff_excludes_every_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VOR-1402: a cutoff newer than the whole backlog must not stall silently."""
    import vo_rag.pipelines.book_rag.dispatcher as dispatcher_mod

    log = _FakeLog()
    monkeypatch.setattr(dispatcher_mod, "log", log)
    newest = datetime(2026, 6, 28, 17, 35, tzinfo=UTC)
    conn = _FakeAuditConnection(
        row={
            "claimable_with_cutoff": 0,
            "claimable_ignoring_cutoff": 2722,
            "newest_created_at": newest,
        }
    )
    cutoff = datetime(2026, 7, 16, 0, 30, tzinfo=UTC)

    await dispatcher_mod.audit_claim_cutoff(cast("Any", conn), claim_not_before=cutoff)

    assert len(log.warnings) == 1
    event, fields = log.warnings[0]
    assert event == "book_rag_dispatcher_claim_cutoff_excludes_all"
    assert fields["excluded_envelopes"] == 2722
    assert fields["claim_not_before"] == cutoff.isoformat()
    assert fields["newest_envelope_created_at"] == newest.isoformat()
    assert "COUNT(*)" in conn.queries[0]


@pytest.mark.asyncio
async def test_audit_claim_cutoff_stays_quiet_when_cutoff_admits_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vo_rag.pipelines.book_rag.dispatcher as dispatcher_mod

    log = _FakeLog()
    monkeypatch.setattr(dispatcher_mod, "log", log)
    conn = _FakeAuditConnection(
        row={
            "claimable_with_cutoff": 5,
            "claimable_ignoring_cutoff": 2722,
            "newest_created_at": datetime(2026, 8, 1, tzinfo=UTC),
        }
    )

    await dispatcher_mod.audit_claim_cutoff(
        cast("Any", conn), claim_not_before=datetime(2026, 7, 16, tzinfo=UTC)
    )

    assert log.warnings == []


@pytest.mark.asyncio
async def test_audit_claim_cutoff_stays_quiet_when_backlog_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty queue is not a misconfigured cutoff - do not cry wolf."""
    import vo_rag.pipelines.book_rag.dispatcher as dispatcher_mod

    log = _FakeLog()
    monkeypatch.setattr(dispatcher_mod, "log", log)
    conn = _FakeAuditConnection(
        row={
            "claimable_with_cutoff": 0,
            "claimable_ignoring_cutoff": 0,
            "newest_created_at": None,
        }
    )

    await dispatcher_mod.audit_claim_cutoff(
        cast("Any", conn), claim_not_before=datetime(2026, 7, 16, tzinfo=UTC)
    )

    assert log.warnings == []


@pytest.mark.asyncio
async def test_run_dispatcher_loop_audits_cutoff_before_polling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vo_rag.pipelines.book_rag.dispatcher as dispatcher_mod

    calls: list[datetime] = []

    async def _audit(pool: Any, *, stale_timeout: Any, claim_not_before: datetime) -> None:
        calls.append(claim_not_before)

    monkeypatch.setattr(dispatcher_mod, "_audit_cutoff_from_pool", _audit)
    monkeypatch.setattr(dispatcher_mod, "write_dispatcher_heartbeat", lambda state: None)

    stop_event = asyncio.Event()
    stop_event.set()
    cutoff = datetime(2026, 7, 16, tzinfo=UTC)

    await dispatcher_mod.run_dispatcher_loop(
        _FakePool(row=None),
        cast("Settings", object()),
        poll_interval=0.01,
        stop_event=stop_event,
        claim_not_before=cutoff,
    )

    assert calls == [cutoff]


@pytest.mark.asyncio
async def test_run_dispatcher_loop_skips_audit_when_no_cutoff_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vo_rag.pipelines.book_rag.dispatcher as dispatcher_mod

    calls: list[datetime] = []

    async def _audit(pool: Any, *, stale_timeout: Any, claim_not_before: datetime) -> None:
        calls.append(claim_not_before)

    monkeypatch.setattr(dispatcher_mod, "_audit_cutoff_from_pool", _audit)
    monkeypatch.setattr(dispatcher_mod, "write_dispatcher_heartbeat", lambda state: None)

    stop_event = asyncio.Event()
    stop_event.set()

    await dispatcher_mod.run_dispatcher_loop(
        _FakePool(row=None),
        cast("Settings", object()),
        poll_interval=0.01,
        stop_event=stop_event,
    )

    assert calls == []


@pytest.mark.asyncio
async def test_threaded_heartbeat_keeps_beating_while_ingest_blocks_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VOR-1429: a CPU-bound ingest step starved the asyncio heartbeat, so the
    stale reclaim handed the live book to another lane. The threaded heartbeat
    must keep beating while the event loop is blocked."""
    import time

    from vo_rag.pipelines.book_rag import dispatcher as dispatcher_mod

    beats: list[tuple[str, tuple[Any, ...]]] = []

    class _Conn:
        async def execute(self, query: str, *args: Any) -> str:
            beats.append((query, args))
            return "UPDATE 1"

        async def close(self) -> None:
            return None

    async def _connect(dsn: str, **kwargs: Any) -> _Conn:
        assert dsn == "postgresql://heartbeat-test"
        return _Conn()

    async def _cpu_bound_ingest(
        envelope: IntakeEnvelope,
        *,
        settings: object,
        pg_pool: object | None = None,
    ) -> PipelineOutput:
        time.sleep(0.6)  # blocks the event loop, like a long synchronous step
        return PipelineOutput(
            source_id="source-1",
            status=EnvelopeStatus.COMPLETED,
            qdrant_point_ids=[],
            raw_point_ids=[],
            sterile_point_ids=[],
            smoke_decision=None,
            chunk_count=0,
        )

    pool = _FakePool(row=_envelope_row(status=EnvelopeStatus.PROCESSING))
    monkeypatch.setattr("vo_rag.pipelines.book_rag.dispatcher.asyncpg.connect", _connect)
    monkeypatch.setattr(dispatcher_mod, "run_book_rag_ingest", _cpu_bound_ingest)
    monkeypatch.setattr(dispatcher_mod, "write_dispatcher_heartbeat", lambda state: None)

    await dispatcher_mod._run_claimed_envelope(
        pool,
        cast("Settings", object()),
        IntakeEnvelope.model_validate(_envelope_row(status=EnvelopeStatus.PROCESSING)),
        heartbeat_interval=0.1,
        heartbeat_dsn="postgresql://heartbeat-test",
    )

    assert len(beats) >= 3  # ~6 intervals elapsed while the loop was blocked
    assert all(args == (_ENVELOPE_ID, EnvelopeStatus.PROCESSING.value) for _, args in beats)
    assert pool.row is not None and pool.row["status"] is EnvelopeStatus.COMPLETED
    assert not any(
        "SET status = $2\nWHERE id = $1\n  AND status = $2" in q for q, _ in pool.execute_calls
    )


@pytest.mark.asyncio
async def test_claim_can_be_limited_to_source_extensions() -> None:
    """A GPU-free lane claims only ebook/text sources (VOR-1427)."""
    from vo_rag.pipelines.book_rag.dispatcher import claim_next_envelope

    pool = _FakePool(row=_envelope_row())
    conn = _FakeConnection(pool)

    await claim_next_envelope(conn, only_extensions=("epub", "mobi"))

    assert "ANY($8::text[])" in pool.claim_queries[0]
    assert pool.claim_args[0][7] == ["epub", "mobi"]


def test_only_extensions_env_is_parsed_lowercased_and_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.pipelines.book_rag.dispatcher import _env_extensions

    monkeypatch.setenv("VORAG_TEST_EXT", " EPUB, .mobi ,epub,, txt ")
    assert _env_extensions("VORAG_TEST_EXT") == ("epub", "mobi", "txt")
    monkeypatch.setenv("VORAG_TEST_EXT", "")
    assert _env_extensions("VORAG_TEST_EXT") is None
    monkeypatch.delenv("VORAG_TEST_EXT")
    assert _env_extensions("VORAG_TEST_EXT") is None


@pytest.mark.asyncio
async def test_failed_ingest_counts_a_failure_on_the_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VOR-1434: feeds vo_rag_intake_failures_total (derived from Postgres)."""
    from vo_rag.core.observability.metrics import ENVELOPE_FAILURE_METRIC_SQL
    from vo_rag.pipelines.book_rag import dispatcher as dispatcher_mod

    pool = _FakePool(row=_envelope_row())
    stop_event = asyncio.Event()

    async def _raise_ingest(envelope: IntakeEnvelope, **_: object) -> PipelineOutput:
        stop_event.set()
        raise RuntimeError("ingest exploded")

    monkeypatch.setattr(dispatcher_mod, "run_book_rag_ingest", _raise_ingest)
    monkeypatch.setattr(dispatcher_mod, "write_dispatcher_heartbeat", lambda state: None)

    await dispatcher_mod.run_dispatcher_loop(
        pool,
        cast("Settings", object()),
        poll_interval=0.01,
        stop_event=stop_event,
        heartbeat_interval=0,
    )

    assert (ENVELOPE_FAILURE_METRIC_SQL, (_ENVELOPE_ID,)) in pool.execute_calls
    assert pool.row is not None and pool.row["status"] is EnvelopeStatus.FAILED


@pytest.mark.asyncio
async def test_failure_counter_write_error_never_changes_the_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.core.observability.metrics import ENVELOPE_FAILURE_METRIC_SQL
    from vo_rag.pipelines.book_rag import dispatcher as dispatcher_mod

    class _Pool(_FakePool):
        async def execute(self, query: str, *args: Any) -> str:
            if query == ENVELOPE_FAILURE_METRIC_SQL:
                raise RuntimeError("metrics write failed")
            return await super().execute(query, *args)

    pool = _Pool(row=_envelope_row())
    stop_event = asyncio.Event()
    log = _FakeLog()

    async def _raise_ingest(envelope: IntakeEnvelope, **_: object) -> PipelineOutput:
        stop_event.set()
        raise RuntimeError("ingest exploded")

    monkeypatch.setattr(dispatcher_mod, "run_book_rag_ingest", _raise_ingest)
    monkeypatch.setattr(dispatcher_mod, "log", log)
    monkeypatch.setattr(dispatcher_mod, "write_dispatcher_heartbeat", lambda state: None)

    await dispatcher_mod.run_dispatcher_loop(
        pool,
        cast("Settings", object()),
        poll_interval=0.01,
        stop_event=stop_event,
        heartbeat_interval=0,
    )

    assert pool.row is not None and pool.row["status"] is EnvelopeStatus.FAILED
    assert [event for event, _ in log.errors] == ["book_rag_dispatcher_ingest_failed"]


def _failed_output() -> PipelineOutput:
    return PipelineOutput(
        source_id="source-1",
        status=EnvelopeStatus.FAILED,
        qdrant_point_ids=[],
        raw_point_ids=[],
        sterile_point_ids=[],
        smoke_decision=None,
        chunk_count=0,
    )


async def _run_once(
    monkeypatch: pytest.MonkeyPatch,
    pool: _FakePool,
    ingest: Callable[[], Awaitable[PipelineOutput]],
    log: Any | None = None,
) -> None:
    from vo_rag.pipelines.book_rag import dispatcher as dispatcher_mod

    stop_event = asyncio.Event()

    async def _ingest(envelope: IntakeEnvelope, **_: object) -> PipelineOutput:
        stop_event.set()
        return await ingest()

    monkeypatch.setattr(dispatcher_mod, "run_book_rag_ingest", _ingest)
    monkeypatch.setattr(dispatcher_mod, "write_dispatcher_heartbeat", lambda state: None)
    if log is not None:
        monkeypatch.setattr(dispatcher_mod, "log", log)
    await dispatcher_mod.run_dispatcher_loop(
        pool,
        cast("Settings", object()),
        poll_interval=0.01,
        stop_event=stop_event,
        heartbeat_interval=0,
    )


def _failure_records(pool: _FakePool) -> int:
    from vo_rag.core.observability.metrics import ENVELOPE_FAILURE_METRIC_SQL

    return sum(1 for query, _ in pool.execute_calls if query == ENVELOPE_FAILURE_METRIC_SQL)


@pytest.mark.asyncio
async def test_failed_terminal_status_counts_one_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool(row=_envelope_row())

    async def _ingest() -> PipelineOutput:
        return _failed_output()

    await _run_once(monkeypatch, pool, _ingest)
    assert _failure_records(pool) == 1


@pytest.mark.asyncio
async def test_completed_status_counts_no_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _FakePool(row=_envelope_row())

    async def _ingest() -> PipelineOutput:
        return replace(_failed_output(), status=EnvelopeStatus.COMPLETED)

    await _run_once(monkeypatch, pool, _ingest)
    assert _failure_records(pool) == 0


@pytest.mark.asyncio
async def test_a_failure_is_counted_once_even_when_the_outer_handler_also_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FAILED persisted, then something after it raises: the outer handler
    re-fails the envelope, but the failure is still counted exactly once."""

    class _ExplodingLog(_FakeLog):
        def info(self, event: str, **kwargs: Any) -> None:
            if event == "book_rag_dispatcher_ingest_completed":
                raise RuntimeError("log sink broke")
            super().info(event, **kwargs)

    pool = _FakePool(row=_envelope_row())

    async def _ingest() -> PipelineOutput:
        return _failed_output()

    await _run_once(monkeypatch, pool, _ingest, log=_ExplodingLog())
    assert pool.row is not None and pool.row["status"] is EnvelopeStatus.FAILED
    assert _failure_records(pool) == 1


@pytest.mark.asyncio
async def test_a_raised_failure_is_counted_once(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _FakePool(row=_envelope_row())

    async def _ingest() -> PipelineOutput:
        raise RuntimeError("ingest exploded")

    await _run_once(monkeypatch, pool, _ingest)
    assert _failure_records(pool) == 1


def test_claim_sql_records_claim_time_and_stale_reclaims() -> None:
    from vo_rag.pipelines.book_rag.dispatcher import _CLAIM_SQL

    assert "claimed_at_epoch_s" in _CLAIM_SQL
    assert "'stale_reclaims'" in _CLAIM_SQL
    assert "'stale_reclaims_total'" in _CLAIM_SQL
    assert "stream_heartbeat_epoch_s" in _CLAIM_SQL  # live streams aren't claimable
