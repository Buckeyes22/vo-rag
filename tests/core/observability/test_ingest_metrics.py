"""Per-envelope ingest metrics and the stuck/overloaded-ingest gauges (VOR-1434).

Pure unit tests: fake pools stand in for asyncpg. The SQL itself is exercised
against a real Postgres in ``test_ingest_metrics_integration.py``.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest
from prometheus_client import REGISTRY

from vo_rag.core.observability import metrics as m


def _value(name: str, labels: dict[str, str] | None = None) -> float | None:
    return REGISTRY.get_sample_value(name, labels or {})


@pytest.fixture(autouse=True)
def _fresh_totals() -> None:
    # The Postgres-derived counters keep a per-process high-water mark; start
    # every test from "never sampled" so test order doesn't matter.
    m._ingest_totals.clear()  # noqa: SLF001
    m._ingest_totals_sampled_at.clear()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


def test_recording_helpers_are_noops_outside_an_ingest() -> None:
    assert m.current_ingest_recorder() is None
    m.record_embed_call(retries_429=3)
    m.record_qdrant_call(0.5)
    m.record_ingest_stage("parse", 1.0)


def test_recorder_collects_embed_qdrant_and_stage_data() -> None:
    with m.ingest_recording() as recorder:
        assert m.current_ingest_recorder() is recorder
        m.record_embed_call(retries_429=0)
        m.record_embed_call(retries_429=2)
        m.record_qdrant_call(0.1)
        m.record_qdrant_call(0.3)
        m.record_ingest_stage("parse", 12.5)
    assert m.current_ingest_recorder() is None
    assert recorder.embed_calls == 2
    assert recorder.embed_retries_429 == 2
    assert recorder.stages == {"parse": 12.5}
    assert recorder.qdrant_summary() == {"calls": 2, "p95_s": 0.3, "max_s": 0.3}


def test_recorder_is_shared_with_child_tasks() -> None:
    async def child() -> None:
        m.record_embed_call(retries_429=1)

    async def main() -> m.IngestMetricsRecorder:
        with m.ingest_recording() as recorder:
            await asyncio.gather(child(), child())
            await asyncio.to_thread(m.record_qdrant_call, 0.2)
        return recorder

    recorder = asyncio.run(main())
    assert recorder.embed_calls == 2
    assert recorder.embed_retries_429 == 2
    assert recorder.qdrant_summary() is not None


def test_qdrant_p95_is_nearest_rank() -> None:
    recorder = m.IngestMetricsRecorder()
    for i in range(1, 101):  # 0.01 .. 1.00
        recorder.record_qdrant_call(i / 100)
    summary = recorder.qdrant_summary()
    assert summary is not None
    assert summary["calls"] == 100
    assert summary["p95_s"] == pytest.approx(0.95)
    assert summary["max_s"] == pytest.approx(1.0)


def test_empty_recorder_has_no_qdrant_summary() -> None:
    assert m.IngestMetricsRecorder().qdrant_summary() is None


def test_qdrant_latencies_are_a_bounded_reservoir() -> None:
    recorder = m.IngestMetricsRecorder(reservoir_size=100)
    for i in range(10_000):
        recorder.record_qdrant_call(2.5 if i == 5_000 else (i % 100) / 100)
    assert len(recorder.qdrant_latencies) == 100
    summary = recorder.qdrant_summary()
    assert summary is not None
    assert summary["calls"] == 10_000  # exact, not the sample size
    assert summary["max_s"] == 2.5  # exact, even if the sample missed it
    assert 0.8 <= summary["p95_s"] <= 1.0  # true p95 is 0.95


def test_default_reservoir_is_capped() -> None:
    recorder = m.IngestMetricsRecorder()
    for _ in range(m.QDRANT_LATENCY_RESERVOIR + 500):
        recorder.record_qdrant_call(0.1)
    assert len(recorder.qdrant_latencies) == m.QDRANT_LATENCY_RESERVOIR


def test_entering_a_stage_times_the_previous_one() -> None:
    recorder = m.IngestMetricsRecorder()
    recorder.enter_stage("parse", now=100.0)
    recorder.enter_stage("parse", now=101.0)  # same stage again: no-op
    recorder.enter_stage("visual", now=130.0)
    assert recorder.current_stage == "visual"
    recorder.enter_stage("chunk_embed", now=190.0)
    recorder.close_stage(now=200.0)
    recorder.close_stage(now=500.0)  # nothing open: no-op
    assert recorder.stages == {"parse": 30.0, "visual": 60.0, "chunk_embed": 10.0}


def test_enter_ingest_stage_uses_the_current_recorder() -> None:
    m.enter_ingest_stage("parse")  # no recorder: no-op
    with m.ingest_recording() as recorder:
        m.enter_ingest_stage("parse")
        m.enter_ingest_stage("visual")
    assert "parse" in recorder.stages
    assert recorder.current_stage == "visual"


def test_record_ingest_pages_sets_the_current_recorder() -> None:
    m.record_ingest_pages(10)  # no recorder: no-op
    with m.ingest_recording() as recorder:
        m.record_ingest_pages(0)
        assert recorder.pages is None
        m.record_ingest_pages(812)
    assert recorder.pages == 812


# ---------------------------------------------------------------------------
# Persistence (best-effort, merge-only)
# ---------------------------------------------------------------------------


class _RecordingPool:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        if self.error is not None:
            raise self.error
        return "UPDATE 1"


async def test_persist_sends_absolute_attempt_totals() -> None:
    recorder = m.IngestMetricsRecorder()
    recorder.record_stage("parse", 30.0)
    recorder.record_embed_call(retries_429=4)
    recorder.record_embed_call(retries_429=0)
    recorder.record_qdrant_call(0.2)
    recorder.pages = 321
    pool = _RecordingPool()

    await m.persist_ingest_metrics(pool, "env-1", recorder)

    [(query, args)] = pool.executed
    assert query == m.ENVELOPE_METRICS_MERGE_SQL
    envelope_id, overwrite, stages, calls, retries, attempt_id = args
    assert envelope_id == "env-1"
    assert overwrite == {"pages": 321, "qdrant": {"calls": 1, "p95_s": 0.2, "max_s": 0.2}}
    assert stages == {"parse": {"seconds": 30.0}}
    assert (calls, retries) == (2, 4)
    assert attempt_id == recorder.attempt_id

    # Nothing new: nothing sent.
    await m.persist_ingest_metrics(pool, "env-1", recorder)
    assert len(pool.executed) == 1

    # New work: the attempt's running totals, not a delta.
    recorder.record_embed_call(retries_429=1)
    await m.persist_ingest_metrics(pool, "env-1", recorder)
    _, (_, _, stages2, calls2, retries2, _) = pool.executed[-1]
    assert (stages2, calls2, retries2) == ({}, 3, 5)


async def test_persist_never_raises_and_resends_after_a_failure() -> None:
    recorder = m.IngestMetricsRecorder()
    recorder.record_embed_call(retries_429=2)
    recorder.record_stage("parse", 3.0)
    pool = _RecordingPool(error=RuntimeError("db down"))

    await m.persist_ingest_metrics(pool, "env-1", recorder)  # must not raise

    pool.error = None
    await m.persist_ingest_metrics(pool, "env-1", recorder)
    _, (_, _, stages, calls, retries, _) = pool.executed[-1]
    assert (stages, calls, retries) == ({"parse": {"seconds": 3.0}}, 1, 2)


async def test_a_flush_cancelled_after_commit_never_double_counts() -> None:
    """The server may have committed before the cancel reached us: the next
    flush re-sends the same absolute totals, which the SQL applies idempotently
    (see the integration test), never a doubled delta."""

    class _CommitThenCancel(_RecordingPool):
        async def execute(self, query: str, *args: Any) -> str:
            self.executed.append((query, args))
            raise asyncio.CancelledError

    recorder = m.IngestMetricsRecorder()
    recorder.record_embed_call(retries_429=3)
    pool = _CommitThenCancel()
    with pytest.raises(asyncio.CancelledError):
        await m.persist_ingest_metrics(pool, "env-1", recorder)

    ok = _RecordingPool()
    await m.persist_ingest_metrics(ok, "env-1", recorder)
    _, (_, _, _, calls, retries, attempt_id) = ok.executed[-1]
    assert (calls, retries) == (1, 3)
    assert attempt_id == pool.executed[0][1][5]


async def test_persist_without_a_pool_is_a_noop() -> None:
    recorder = m.IngestMetricsRecorder()
    recorder.record_embed_call(retries_429=0)
    await m.persist_ingest_metrics(None, "env-1", recorder)


def test_merge_sql_only_touches_the_metrics_key() -> None:
    sql = m.ENVELOPE_METRICS_MERGE_SQL
    assert "SET metadata = env.metadata || jsonb_build_object('metrics'" in sql
    assert "jsonb_typeof(env.metadata) = 'object'" in sql


async def test_periodic_flush_writes_while_work_is_running() -> None:
    recorder = m.IngestMetricsRecorder()
    pool = _RecordingPool()
    task = m.start_periodic_ingest_flush(pool, "env-p", recorder, interval_s=0.01)
    assert task is not None
    recorder.record_embed_call(retries_429=5)
    for _ in range(100):
        if pool.executed:
            break
        await asyncio.sleep(0.01)
    await m.stop_periodic_ingest_flush(task)
    assert task.done()
    [(_, (_, _, _, calls, retries, _))] = pool.executed
    assert (calls, retries) == (1, 5)


async def test_periodic_flush_survives_write_errors() -> None:
    recorder = m.IngestMetricsRecorder()
    pool = _RecordingPool(error=RuntimeError("db down"))
    recorder.record_embed_call(retries_429=1)
    task = m.start_periodic_ingest_flush(pool, "env-p", recorder, interval_s=0.01)
    await asyncio.sleep(0.05)
    assert task is not None and not task.done()
    await m.stop_periodic_ingest_flush(task)
    assert recorder.embed_calls == 1  # kept for the final flush


async def test_stopping_the_flush_propagates_the_callers_own_cancellation() -> None:
    """``stop_periodic_ingest_flush`` must not swallow a cancel aimed at its caller."""
    release = asyncio.Event()

    class _SlowPool(_RecordingPool):
        async def execute(self, query: str, *args: Any) -> str:
            self.executed.append((query, args))
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()  # a flush that outlives its first cancel
            return "UPDATE 1"

    recorder = m.IngestMetricsRecorder()
    recorder.record_embed_call(retries_429=1)
    pool = _SlowPool()
    task = m.start_periodic_ingest_flush(pool, "env-c", recorder, interval_s=0.001)
    for _ in range(100):
        if pool.executed:
            break
        await asyncio.sleep(0.005)

    stopper = asyncio.create_task(m.stop_periodic_ingest_flush(task))
    await asyncio.sleep(0.01)
    stopper.cancel()  # the caller itself is being cancelled
    with pytest.raises(asyncio.CancelledError):
        await stopper
    release.set()
    await asyncio.sleep(0)


async def test_stop_logs_a_crashed_flush_task_without_raising() -> None:
    async def _boom() -> None:
        raise RuntimeError("flush task crashed")

    task = asyncio.create_task(_boom())
    await asyncio.sleep(0)
    await m.stop_periodic_ingest_flush(task)  # must not raise


async def test_stream_flush_beats_a_heartbeat_each_tick() -> None:
    recorder = m.IngestMetricsRecorder()
    pool = _RecordingPool()
    task = m.start_periodic_ingest_flush(pool, "env-h", recorder, interval_s=0.005, heartbeat=True)
    for _ in range(100):
        if sum(1 for q, _ in pool.executed if q == m.ENVELOPE_STREAM_HEARTBEAT_SQL) >= 2:
            break
        await asyncio.sleep(0.005)
    await m.stop_periodic_ingest_flush(task)
    beats = [a for q, a in pool.executed if q == m.ENVELOPE_STREAM_HEARTBEAT_SQL]
    assert len(beats) >= 2 and beats[0] == ("env-h",)


def test_periodic_flush_needs_a_pool() -> None:
    assert m.start_periodic_ingest_flush(None, "e", m.IngestMetricsRecorder()) is None


async def test_record_stream_start_is_best_effort() -> None:
    pool = _RecordingPool()
    await m.record_stream_start(pool, "env-s")
    assert pool.executed == [(m.ENVELOPE_STREAM_START_SQL, ("env-s",))]
    await m.record_stream_start(_RecordingPool(error=RuntimeError("boom")), "env-s")


async def test_record_envelope_failure_is_best_effort() -> None:
    pool = _RecordingPool()
    await m.record_envelope_failure(pool, "env-9")
    assert pool.executed == [(m.ENVELOPE_FAILURE_METRIC_SQL, ("env-9",))]

    await m.record_envelope_failure(_RecordingPool(error=RuntimeError("boom")), "env-9")


# ---------------------------------------------------------------------------
# Scrape-time gauges from Postgres
# ---------------------------------------------------------------------------


class _Conn:
    def __init__(self, rows: dict[str, dict[str, Any] | None], error: Exception | None = None):
        self.rows = rows
        self.error = error
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.calls.append((query, args))
        if self.error is not None:
            raise self.error
        if query == m.INGEST_LIVE_SQL:
            return self.rows["live"]
        assert query == m.INGEST_TOTALS_SQL
        return self.rows["totals"]


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    @asynccontextmanager
    async def acquire(self):  # noqa: ANN201
        yield self.conn


def _conn(*, totals: dict[str, Any] | None = None, **live: Any) -> _Conn:
    live_row: dict[str, Any] = {
        "processing_seconds": None,
        "budget_seconds": None,
        "reclaims_max": 0,
        "failures_last_hour": 0,
    }
    live_row.update(live)
    totals_row = {"embed_calls_total": 0, "embed_retries_429_total": 0}
    totals_row.update(totals or {})
    return _Conn({"live": live_row, "totals": totals_row})


async def test_kelly_shape_is_over_budget() -> None:
    """One envelope processing for 8 h with a fresh heartbeat: the gauges
    must show it over its budget (the rule compares these two)."""
    conn = _conn(processing_seconds=8 * 3600.0, budget_seconds=900.0)
    await m.sample_ingest_health(_Pool(conn))

    live_calls = [args for query, args in conn.calls if query == m.INGEST_LIVE_SQL]
    assert live_calls == [
        (
            m.ENVELOPE_BUDGET_FLOOR_S,
            m.ENVELOPE_BUDGET_PER_PAGE_S,
            m.ENVELOPE_BUDGET_UNKNOWN_PAGES_PARSE_S,
            m.STREAM_HEARTBEAT_STALE_S,
        )
    ]
    processing = _value("vo_rag_envelope_processing_seconds_max")
    budget = _value("vo_rag_envelope_budget_seconds")
    assert processing == 8 * 3600.0
    assert budget == 900.0
    assert processing > budget


async def test_nothing_processing_publishes_sentinels_that_never_alert() -> None:
    await m.sample_ingest_health(_Pool(_conn()))
    assert _value("vo_rag_envelope_processing_seconds_max") == -1
    assert _value("vo_rag_envelope_budget_seconds") == -1
    assert _value("vo_rag_envelope_reclaims_max") == 0
    assert _value("vo_rag_intake_failures_last_hour") == 0


async def test_reclaims_failures_and_embed_totals_are_published() -> None:
    await m.sample_ingest_health(
        _Pool(
            _conn(
                reclaims_max=3,
                failures_last_hour=6,
                totals={"embed_calls_total": 20_000, "embed_retries_429_total": 1_050},
            )
        )
    )
    assert _value("vo_rag_envelope_reclaims_max") == 3
    assert _value("vo_rag_intake_failures_last_hour") == 6
    assert _value("vo_rag_embed_calls_total") == 20_000
    assert _value("vo_rag_embed_retries_total", {"reason": "429"}) == 1_050
    # The failure signal is a windowed gauge now, not a counter.
    assert _value("vo_rag_intake_failures_total") is None


async def test_embed_totals_are_sampled_on_a_slower_cadence_and_cached() -> None:
    first = _conn(totals={"embed_calls_total": 100})
    await m.sample_ingest_health(_Pool(first), now=1000.0)
    assert any(q == m.INGEST_TOTALS_SQL for q, _ in first.calls)

    soon = _conn(totals={"embed_calls_total": 999})
    await m.sample_ingest_health(_Pool(soon), now=1000.0 + m.INGEST_TOTALS_INTERVAL_S - 1)
    assert [q for q, _ in soon.calls] == [m.INGEST_LIVE_SQL]
    assert _value("vo_rag_embed_calls_total") == 100  # cached

    later = _conn(totals={"embed_calls_total": 150})
    await m.sample_ingest_health(_Pool(later), now=1000.0 + m.INGEST_TOTALS_INTERVAL_S)
    assert _value("vo_rag_embed_calls_total") == 150


async def test_embed_totals_never_go_backwards_within_a_process() -> None:
    """A deleted envelope shrinks the Postgres sum; exposing that drop would
    look like a counter reset and make ``increase()`` invent a burst."""
    await m.sample_ingest_health(_Pool(_conn(totals={"embed_calls_total": 40})), now=0.0)
    await m.sample_ingest_health(
        _Pool(_conn(totals={"embed_calls_total": 30})), now=m.INGEST_TOTALS_INTERVAL_S
    )
    assert _value("vo_rag_embed_calls_total") == 40


async def test_sample_failure_fails_closed_and_is_counted() -> None:
    before = _value("vo_rag_operational_sampler_errors_total", {"component": "ingest_health"}) or 0
    await m.sample_ingest_health(_Pool(_Conn({}, error=RuntimeError("db down"))))
    assert _value("vo_rag_envelope_processing_seconds_max") == -1
    assert _value("vo_rag_envelope_budget_seconds") == -1
    assert _value("vo_rag_envelope_reclaims_max") == -1
    assert _value("vo_rag_intake_failures_last_hour") == -1
    after = _value("vo_rag_operational_sampler_errors_total", {"component": "ingest_health"})
    assert after == before + 1


async def test_sample_without_a_pool_publishes_unknown() -> None:
    await m.sample_ingest_health(None)
    assert _value("vo_rag_envelope_reclaims_max") == -1


def test_envelope_budget_matches_the_canary_formula() -> None:
    from vo_rag.core.corpus import canary

    assert m.ENVELOPE_BUDGET_FLOOR_S == canary.STAGE_BUDGET_FLOOR_S
    assert m.ENVELOPE_BUDGET_PER_PAGE_S == canary.STAGE_BUDGET_PER_PAGE_S


async def test_the_api_operational_sampler_samples_ingest_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gauges are derived on the API's sampling cadence (dispatchers are
    separate processes and never scraped)."""
    from tests.api.conftest import make_services

    from vo_rag.api import operational_sampler as sampler
    from vo_rag.api.routes import system

    async def _ok(*_: Any) -> dict[str, bool]:
        return {"ok": True}

    monkeypatch.setattr(system, "_db_pool_health", _ok)
    monkeypatch.setattr(system, "_qdrant_health", _ok)
    monkeypatch.setattr(system, "_embedding_health", _ok)
    monkeypatch.setattr(system, "_reranker_health", _ok)
    monkeypatch.setattr(
        sampler, "dispatcher_health", lambda *, now: {"ok": True, "age_seconds": 1.0}
    )
    seen: list[Any] = []

    async def _sample(pool: Any) -> None:
        seen.append(pool)

    monkeypatch.setattr(sampler, "sample_ingest_health", _sample)
    services = make_services(db_pool=None)
    await sampler.sample_operational_metrics(services, object())
    assert seen == [None]


async def test_an_unexpected_row_shape_fails_closed_instead_of_raising() -> None:
    conn = _Conn({"live": {"something": 1}, "totals": {"other": 2}})
    await m.sample_ingest_health(_Pool(conn))  # must not raise
    assert _value("vo_rag_envelope_processing_seconds_max") == -1
    assert _value("vo_rag_embed_calls_total") is None
