"""The ingest-metrics SQL against a real Postgres (VOR-1434).

Gated on ``VORAG_TEST_DATABASE_URL`` (``make up`` starts a throwaway local
container; never production). Covers the claim SQL's ``metrics`` bookkeeping,
the merge/failure writes (never clobbering other metadata keys), and the
scrape-time aggregate, including a long-running envelope: one processing for
8 h while its heartbeat keeps ``updated_at`` fresh.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from datetime import timedelta
from pathlib import Path
from typing import Any

import asyncpg  # type: ignore[import-untyped]
import pytest

from vo_rag.core.db.postgres import _register_codecs, run_migrations
from vo_rag.core.observability import metrics as m
from vo_rag.pipelines.book_rag.dispatcher import claim_next_envelope

pytestmark = pytest.mark.integration

_MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "vo_rag" / "core" / "db" / "migrations"
)
_HASH = "c" * 64


def _require_dsn() -> str:
    dsn = os.environ.get("VORAG_TEST_DATABASE_URL")
    if not dsn:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail("VORAG_TEST_DATABASE_URL is required by the testinfra lane")
        pytest.skip("VORAG_TEST_DATABASE_URL not set -- skipping integration test")
    return dsn


@pytest.fixture
async def pool() -> AsyncGenerator[asyncpg.Pool, None]:
    p = await asyncpg.create_pool(
        dsn=_require_dsn(),
        min_size=1,
        max_size=2,
        statement_cache_size=0,
        init=_register_codecs,
    )
    try:
        async with p.acquire() as conn:
            await conn.execute("DROP SCHEMA IF EXISTS vce CASCADE")
            await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
            await conn.execute("CREATE SCHEMA public")
        await run_migrations(p, _MIGRATIONS_DIR)
        yield p
    finally:
        await p.close()


async def _insert(
    pool: asyncpg.Pool,
    envelope_id: str,
    *,
    status: str = "received",
    metadata: dict[str, Any] | None = None,
    updated_ago: timedelta = timedelta(0),
) -> None:
    await pool.execute(
        """
        INSERT INTO intake_envelopes
            (id, source_type, raw_content, metadata, status, target_pipeline,
             created_at, updated_at)
        VALUES ($1, 'file_drop', $2, $3, $4, 'book_rag', NOW() - INTERVAL '1 day', NOW())
        """,
        envelope_id,
        {
            "garage_key": f"drops/{envelope_id}.pdf",
            "file_hash": _HASH,
            "object_verification": {"status": "verified", "sha256": _HASH},
        },
        metadata or {},
        status,
    )
    if updated_ago:
        await _backdate(pool, envelope_id, updated_ago)


async def _backdate(pool: asyncpg.Pool, envelope_id: str, ago: timedelta) -> None:
    # updated_at is trigger-maintained on UPDATE; bypass the trigger to age it.
    await pool.execute("ALTER TABLE intake_envelopes DISABLE TRIGGER USER")
    await pool.execute(
        "UPDATE intake_envelopes SET updated_at = NOW() - $2::interval WHERE id = $1",
        envelope_id,
        ago,
    )
    await pool.execute("ALTER TABLE intake_envelopes ENABLE TRIGGER USER")


async def _metadata(pool: asyncpg.Pool, envelope_id: str) -> dict[str, Any]:
    value = await pool.fetchval("SELECT metadata FROM intake_envelopes WHERE id = $1", envelope_id)
    return value if isinstance(value, dict) else json.loads(value)


async def _claim(pool: asyncpg.Pool) -> Any:
    async with pool.acquire() as conn, conn.transaction():
        return await claim_next_envelope(conn)


# ---------------------------------------------------------------------------
# Claim bookkeeping
# ---------------------------------------------------------------------------


async def test_fresh_claim_stamps_claim_time_and_keeps_other_keys(pool: asyncpg.Pool) -> None:
    await _insert(pool, "01KT6XE8M1MW42M0Y0JJKRZS01", metadata={"quality": {"findings": []}})
    claimed = await _claim(pool)
    assert claimed is not None and str(claimed.id) == "01KT6XE8M1MW42M0Y0JJKRZS01"

    metadata = await _metadata(pool, "01KT6XE8M1MW42M0Y0JJKRZS01")
    assert metadata["quality"] == {"findings": []}
    assert metadata["metrics"]["stale_reclaims"] == 0
    now = await pool.fetchval("SELECT EXTRACT(EPOCH FROM NOW())::double precision")
    assert abs(metadata["metrics"]["claimed_at_epoch_s"] - now) < 60


async def test_each_stale_reclaim_counts_and_a_fresh_claim_resets(pool: asyncpg.Pool) -> None:
    await _insert(
        pool,
        "01KT6XE8M1MW42M0Y0JJKRZS02",
        status="processing",
        metadata={"metrics": {"stale_reclaims": 2, "failures": 1}, "quality": {"x": 1}},
        updated_ago=timedelta(hours=1),
    )
    assert await _claim(pool) is not None
    metadata = await _metadata(pool, "01KT6XE8M1MW42M0Y0JJKRZS02")
    assert metadata["metrics"]["stale_reclaims"] == 3
    assert metadata["metrics"]["failures"] == 1
    assert metadata["quality"] == {"x": 1}

    await pool.execute(
        "UPDATE intake_envelopes SET status = 'received' WHERE id = $1",
        "01KT6XE8M1MW42M0Y0JJKRZS02",
    )
    assert await _claim(pool) is not None
    metrics = (await _metadata(pool, "01KT6XE8M1MW42M0Y0JJKRZS02"))["metrics"]
    assert metrics["stale_reclaims"] == 0
    assert metrics["stale_reclaims_total"] == 1  # never reset


async def test_crash_requeue_crash_still_reaches_the_reclaim_threshold(
    pool: asyncpg.Pool,
) -> None:
    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZS04"
    await _insert(pool, envelope_id)
    assert await _claim(pool) is not None  # fresh claim
    for _ in range(3):
        # crash: the lane dies, the heartbeat stops, another lane reclaims ...
        await _backdate(pool, envelope_id, timedelta(hours=1))
        assert await _claim(pool) is not None
        # ... then a shutdown requeue and a fresh claim in between.
        await pool.execute(
            "UPDATE intake_envelopes SET status = 'received' WHERE id = $1", envelope_id
        )
        assert await _claim(pool) is not None
    metrics = (await _metadata(pool, envelope_id))["metrics"]
    assert metrics["stale_reclaims"] == 0
    assert metrics["stale_reclaims_total"] == 3
    assert (await _live(pool))["reclaims_max"] == 3


async def test_claim_survives_a_malformed_metrics_value(pool: asyncpg.Pool) -> None:
    await _insert(pool, "01KT6XE8M1MW42M0Y0JJKRZS03", metadata={"metrics": "not-an-object"})
    claimed = await _claim(pool)
    assert claimed is not None
    assert (await _metadata(pool, "01KT6XE8M1MW42M0Y0JJKRZS03"))["metrics"]["stale_reclaims"] == 0


# ---------------------------------------------------------------------------
# Merge / failure writes
# ---------------------------------------------------------------------------


async def test_merge_accumulates_embed_and_overwrites_stages(pool: asyncpg.Pool) -> None:
    await _insert(
        pool,
        "env-m",
        metadata={"quality": {"findings": ["x"]}, "metrics": {"stale_reclaims": 1}},
    )
    recorder = m.IngestMetricsRecorder()
    recorder.record_stage("parse", 10.0)
    recorder.record_embed_call(retries_429=3)
    recorder.record_qdrant_call(0.25)
    recorder.pages = 200
    await m.persist_ingest_metrics(pool, "env-m", recorder)

    recorder.record_stage("parse", 12.0)
    recorder.record_stage("chunk_embed", 40.0)
    recorder.record_embed_call(retries_429=0)
    await m.persist_ingest_metrics(pool, "env-m", recorder)

    metadata = await _metadata(pool, "env-m")
    assert metadata["quality"] == {"findings": ["x"]}
    metrics = metadata["metrics"]
    assert metrics["stale_reclaims"] == 1
    assert metrics["pages"] == 200
    assert metrics["stages"] == {"parse": {"seconds": 12.0}, "chunk_embed": {"seconds": 40.0}}
    assert (metrics["embed"]["calls"], metrics["embed"]["retries_429"]) == (2, 3)
    assert metrics["embed"]["attempts"] == {recorder.attempt_id: {"calls": 2, "retries_429": 3}}
    assert metrics["stages_attempt"] == recorder.attempt_id
    assert metrics["qdrant"] == {"calls": 1, "p95_s": 0.25, "max_s": 0.25}


async def test_failure_counter_increments_without_clobbering(pool: asyncpg.Pool) -> None:
    await _insert(pool, "env-f", metadata={"quality": {"q": 1}, "metrics": {"pages": 9}})
    await m.record_envelope_failure(pool, "env-f")
    await m.record_envelope_failure(pool, "env-f")
    metadata = await _metadata(pool, "env-f")
    assert metadata["quality"] == {"q": 1}
    assert {k: v for k, v in metadata["metrics"].items() if k != "failed_at_epoch_s"} == {
        "pages": 9,
        "failures": 2,
    }


# ---------------------------------------------------------------------------
# Scrape-time aggregates
# ---------------------------------------------------------------------------


async def _live(pool: asyncpg.Pool) -> dict[str, Any]:
    row = await pool.fetchrow(
        m.INGEST_LIVE_SQL,
        m.ENVELOPE_BUDGET_FLOOR_S,
        m.ENVELOPE_BUDGET_PER_PAGE_S,
        m.ENVELOPE_BUDGET_UNKNOWN_PAGES_PARSE_S,
        m.STREAM_HEARTBEAT_STALE_S,
    )
    return dict(row)


async def _ago(pool: asyncpg.Pool, interval: str) -> float:
    return float(
        await pool.fetchval(
            "SELECT EXTRACT(EPOCH FROM NOW() - $1::text::interval)::double precision", interval
        )
    )


async def test_kelly_shape_is_over_budget_despite_a_fresh_heartbeat(pool: asyncpg.Pool) -> None:
    """8 h since claim, heartbeat (updated_at) fresh, other lanes completing."""
    await _insert(
        pool,
        "env-kelly",
        status="processing",
        metadata={
            "metrics": {
                "claimed_at_epoch_s": await _ago(pool, "8 hours"),
                "pages": 120,
                "current_stage": "visual",
            }
        },
    )
    await _insert(
        pool,
        "env-fine",
        status="processing",
        metadata={"metrics": {"claimed_at_epoch_s": await _ago(pool, "2 minutes"), "pages": 900}},
    )
    await _insert(pool, "env-done", status="completed")

    live = await _live(pool)
    assert live["processing_seconds"] == pytest.approx(8 * 3600, abs=60)
    assert live["budget_seconds"] == 900.0  # max(15 min, 120 x 6 s)
    assert live["processing_seconds"] > live["budget_seconds"]

    await m.sample_ingest_health(pool)
    assert m.envelope_processing_seconds_max._value.get() > m.envelope_budget_seconds._value.get()  # noqa: SLF001


async def test_big_book_within_its_page_budget_is_not_over(pool: asyncpg.Pool) -> None:
    await _insert(
        pool,
        "env-giant",
        status="processing",
        metadata={"metrics": {"claimed_at_epoch_s": await _ago(pool, "1 hour"), "pages": 1200}},
    )
    live = await _live(pool)
    assert live["budget_seconds"] == 7200.0
    assert live["processing_seconds"] < live["budget_seconds"]


@pytest.mark.parametrize("preflight_pages", [2000, "2000"])
async def test_preflight_pages_set_the_budget_before_parse_knows(
    pool: asyncpg.Pool, preflight_pages: Any
) -> None:
    await _insert(
        pool,
        "env-pre",
        status="processing",
        metadata={
            "preflight": {"pages": preflight_pages},
            "metrics": {"claimed_at_epoch_s": await _ago(pool, "40 minutes")},
        },
    )
    live = await _live(pool)
    assert live["budget_seconds"] == 12000.0
    assert live["processing_seconds"] < live["budget_seconds"]


async def test_parsed_pages_win_over_preflight(pool: asyncpg.Pool) -> None:
    await _insert(
        pool,
        "env-both",
        status="processing",
        metadata={
            "preflight": {"pages": 2000},
            "metrics": {"claimed_at_epoch_s": await _ago(pool, "1 minute"), "pages": 300},
        },
    )
    assert (await _live(pool))["budget_seconds"] == 1800.0


@pytest.mark.parametrize("stage", [None, "register_source", "parse"])
async def test_unknown_pages_while_parsing_get_the_hour_floor(
    pool: asyncpg.Pool, stage: str | None
) -> None:
    """A giant's 30-minute parse with no page count yet is not over budget."""
    metrics: dict[str, Any] = {"claimed_at_epoch_s": await _ago(pool, "30 minutes")}
    if stage is not None:
        metrics["current_stage"] = stage
    await _insert(
        pool, "env-parse", status="processing", metadata={"metrics": metrics, "preflight": {}}
    )
    live = await _live(pool)
    assert live["budget_seconds"] == 3600.0
    assert live["processing_seconds"] < live["budget_seconds"]


async def test_unknown_pages_after_parse_get_the_15_minute_floor(pool: asyncpg.Pool) -> None:
    await _insert(
        pool,
        "env-late",
        status="processing",
        metadata={
            "preflight": {"pages": "unknown"},
            "metrics": {
                "claimed_at_epoch_s": await _ago(pool, "30 minutes"),
                "current_stage": "chunk_embed",
            },
        },
    )
    live = await _live(pool)
    assert live["budget_seconds"] == 900.0
    assert live["processing_seconds"] > live["budget_seconds"]


async def test_an_active_stream_on_a_received_envelope_is_in_flight(pool: asyncpg.Pool) -> None:
    started = await _ago(pool, "2 hours")
    await _insert(
        pool,
        "env-stream",
        metadata={
            "metrics": {
                "claimed_at_epoch_s": started,
                "active_stream": True,
                "stream_heartbeat_epoch_s": await _ago(pool, "30 seconds"),
                "pages": 10,
            }
        },
    )
    # A requeued envelope (received, old claim time, no active stream) is not.
    await _insert(
        pool,
        "env-requeued",
        metadata={"metrics": {"claimed_at_epoch_s": await _ago(pool, "9 hours")}},
    )
    live = await _live(pool)
    assert live["processing_seconds"] == pytest.approx(7200, abs=60)


async def test_stream_start_stamps_and_the_final_flush_clears(pool: asyncpg.Pool) -> None:
    await _insert(pool, "env-s", metadata={"quality": {"q": 1}})
    await m.record_stream_start(pool, "env-s")
    metrics = (await _metadata(pool, "env-s"))["metrics"]
    assert metrics["active_stream"] is True
    assert isinstance(metrics["claimed_at_epoch_s"], float | int)
    assert (await _live(pool))["processing_seconds"] is not None

    recorder = m.IngestMetricsRecorder()
    recorder.active_stream = False
    await m.persist_ingest_metrics(pool, "env-s", recorder)
    metadata = await _metadata(pool, "env-s")
    assert metadata["metrics"]["active_stream"] is False
    assert metadata["quality"] == {"q": 1}
    assert (await _live(pool))["processing_seconds"] is None


async def test_reclaims_gauge_uses_the_cumulative_count(pool: asyncpg.Pool) -> None:
    await _insert(
        pool,
        "env-a",
        status="processing",
        metadata={"metrics": {"stale_reclaims": 0, "stale_reclaims_total": 4}},
    )
    # A completed envelope's old reclaim count must not keep the alert firing.
    await _insert(
        pool, "env-b", status="completed", metadata={"metrics": {"stale_reclaims_total": 9}}
    )
    assert (await _live(pool))["reclaims_max"] == 4


async def test_reclaims_gauge_ignores_a_requeued_received_envelope(pool: asyncpg.Pool) -> None:
    """A reclaimed envelope that was requeued to 'received'
    and now waits in the queue is not in a reclaim loop; counting it kept
    VoRagEnvelopeReclaimLoop firing until the envelope ran again."""
    await _insert(
        pool,
        "env-q",
        status="received",
        metadata={"metrics": {"stale_reclaims": 0, "stale_reclaims_total": 5}},
    )
    await _insert(
        pool,
        "env-p",
        status="processing",
        metadata={"metrics": {"stale_reclaims": 1, "stale_reclaims_total": 1}},
    )
    assert (await _live(pool))["reclaims_max"] == 1


async def test_failures_last_hour_keys_on_the_failure_time_not_updated_at(
    pool: asyncpg.Pool,
) -> None:
    for i in range(3):
        await _insert(pool, f"env-f{i}", status="failed")
        await m.record_envelope_failure(pool, f"env-f{i}")
    # Failed 2 h ago, but a later write bumps updated_at to now: not re-counted.
    await _insert(
        pool,
        "env-old",
        status="failed",
        metadata={"metrics": {"failures": 1, "failed_at_epoch_s": await _ago(pool, "2 hours")}},
    )
    await pool.execute(
        """UPDATE intake_envelopes SET metadata = metadata || '{"quality": {}}' WHERE id = $1""",
        "env-old",
    )
    # Failed recently but already requeued: not failed any more, not counted.
    await _insert(pool, "env-requeued", status="received")
    await m.record_envelope_failure(pool, "env-requeued")
    await _insert(pool, "env-ok", status="completed")
    assert (await _live(pool))["failures_last_hour"] == 3


async def test_record_envelope_failure_stamps_the_failure_time(pool: asyncpg.Pool) -> None:
    await _insert(pool, "env-t", status="failed", metadata={"quality": {"q": 1}})
    await m.record_envelope_failure(pool, "env-t")
    metadata = await _metadata(pool, "env-t")
    now = await _ago(pool, "0 seconds")
    assert abs(metadata["metrics"]["failed_at_epoch_s"] - now) < 60
    assert metadata["quality"] == {"q": 1}


async def test_a_stream_without_a_recent_heartbeat_is_not_in_flight(pool: asyncpg.Pool) -> None:
    """An API crash mid-stream leaves active_stream=true; once its heartbeat is
    older than the bound it stops reading as in flight."""
    await _insert(
        pool,
        "env-crashed",
        metadata={
            "metrics": {
                "claimed_at_epoch_s": await _ago(pool, "3 hours"),
                "active_stream": True,
                "stream_heartbeat_epoch_s": await _ago(pool, "20 minutes"),
            }
        },
    )
    await _insert(
        pool,
        "env-no-beat",
        metadata={
            "metrics": {"claimed_at_epoch_s": await _ago(pool, "3 hours"), "active_stream": True}
        },
    )
    assert (await _live(pool))["processing_seconds"] is None


async def test_stream_heartbeat_refreshes_the_bound(pool: asyncpg.Pool) -> None:
    await _insert(pool, "env-beat")
    await m.record_stream_start(pool, "env-beat")
    await pool.execute(m.ENVELOPE_STREAM_HEARTBEAT_SQL, "env-beat")
    metrics = (await _metadata(pool, "env-beat"))["metrics"]
    assert metrics["active_stream"] is True
    assert isinstance(metrics["stream_heartbeat_epoch_s"], float | int)
    assert (await _live(pool))["processing_seconds"] is not None


async def test_a_claim_clears_a_crashed_streams_flag(pool: asyncpg.Pool) -> None:
    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZS05"
    await _insert(
        pool,
        envelope_id,
        metadata={"metrics": {"active_stream": True, "stream_heartbeat_epoch_s": 1.0}},
    )
    assert await _claim(pool) is not None
    assert (await _metadata(pool, envelope_id))["metrics"]["active_stream"] is False


async def test_merge_is_idempotent_per_attempt(pool: asyncpg.Pool) -> None:
    """Re-sending an attempt's absolute totals (e.g. after a flush was
    cancelled post-commit) never double-counts; a new attempt adds on top."""
    await _insert(pool, "env-i", metadata={"metrics": {"embed": {"calls": 10, "retries_429": 1}}})
    args: tuple[Any, ...] = ("env-i", {}, {}, 5, 2, "attempt-a")
    await pool.execute(m.ENVELOPE_METRICS_MERGE_SQL, *args)
    await pool.execute(m.ENVELOPE_METRICS_MERGE_SQL, *args)  # replay
    embed = (await _metadata(pool, "env-i"))["metrics"]["embed"]
    assert (embed["calls"], embed["retries_429"]) == (15, 3)

    await pool.execute(m.ENVELOPE_METRICS_MERGE_SQL, "env-i", {}, {}, 7, 2, "attempt-a")
    embed = (await _metadata(pool, "env-i"))["metrics"]["embed"]
    assert (embed["calls"], embed["retries_429"]) == (17, 3)

    await pool.execute(m.ENVELOPE_METRICS_MERGE_SQL, "env-i", {}, {}, 4, 4, "attempt-b")
    embed = (await _metadata(pool, "env-i"))["metrics"]["embed"]
    assert (embed["calls"], embed["retries_429"]) == (21, 7)


async def test_totals_sum_embed_counts_and_ignore_junk(pool: asyncpg.Pool) -> None:
    await _insert(
        pool,
        "env-b",
        status="completed",
        metadata={"metrics": {"embed": {"calls": 100, "retries_429": 7}}},
    )
    await _insert(
        pool,
        "env-c",
        status="failed",
        metadata={"metrics": {"embed": {"calls": 50, "retries_429": "junk"}}},
    )
    await _insert(pool, "env-d", metadata={"metrics": "junk"})
    row = dict(await pool.fetchrow(m.INGEST_TOTALS_SQL))
    assert row["embed_calls_total"] == 150
    assert row["embed_retries_429_total"] == 7


async def test_empty_table_aggregates(pool: asyncpg.Pool) -> None:
    live = await _live(pool)
    assert live["processing_seconds"] is None
    assert live["reclaims_max"] == 0
    assert live["failures_last_hour"] == 0


async def test_live_query_can_use_the_status_index(pool: asyncpg.Pool) -> None:
    """``idx_envelopes_status`` serves the in-flight scan (seqscan disabled so
    the tiny test table doesn't make the planner prefer a full scan)."""
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("SET LOCAL enable_seqscan = off")
        plan = await conn.fetch(
            "EXPLAIN "
            + m.INGEST_LIVE_SQL.replace("$1", "900")
            .replace("$2", "6")
            .replace("$3", "3600")
            .replace("$4", "600")
        )
    text = "\n".join(r[0] for r in plan)
    print(text)
    assert "idx_envelopes_status" in text
    assert "Seq Scan on intake_envelopes" not in text


# ---------------------------------------------------------------------------
# Interleaved attempts, per-attempt stages, live-stream claims
# ---------------------------------------------------------------------------


async def _embed(pool: asyncpg.Pool, envelope_id: str) -> tuple[Any, Any]:
    embed = (await _metadata(pool, envelope_id))["metrics"]["embed"]
    return embed["calls"], embed["retries_429"]


async def test_interleaved_attempts_never_double_count(pool: asyncpg.Pool) -> None:
    """A flushes, B flushes, A flushes again (a dispatcher claimed a live
    stream's envelope, or a zombie worker after a stale reclaim): each
    attempt's totals count once."""
    await _insert(pool, "env-ab")
    merge = m.ENVELOPE_METRICS_MERGE_SQL
    await pool.execute(merge, "env-ab", {}, {}, 5, 1, "attempt-a")
    await pool.execute(merge, "env-ab", {}, {}, 3, 0, "attempt-b")
    await pool.execute(merge, "env-ab", {}, {}, 7, 2, "attempt-a")
    await pool.execute(merge, "env-ab", {}, {}, 7, 2, "attempt-a")  # replay
    await pool.execute(merge, "env-ab", {}, {}, 4, 1, "attempt-b")
    assert await _embed(pool, "env-ab") == (11, 3)  # A=7/2 + B=4/1
    attempts = (await _metadata(pool, "env-ab"))["metrics"]["embed"]["attempts"]
    assert attempts == {
        "attempt-a": {"calls": 7, "retries_429": 2},
        "attempt-b": {"calls": 4, "retries_429": 1},
    }

    # The readers see the per-attempt sum.
    row = dict(await pool.fetchrow(m.INGEST_TOTALS_SQL))
    assert (row["embed_calls_total"], row["embed_retries_429_total"]) == (11, 3)


async def test_a_new_attempt_replaces_the_previous_attempts_stages(pool: asyncpg.Pool) -> None:
    await _insert(pool, "env-st")
    merge = m.ENVELOPE_METRICS_MERGE_SQL
    first = {"parse": {"seconds": 10.0}, "visual": {"seconds": 20.0}, "chunk_embed": {"seconds": 5}}
    await pool.execute(merge, "env-st", {}, first, 0, 0, "attempt-a")
    # attempt b only gets as far as parse: a/visual and a/chunk_embed must not linger
    await pool.execute(merge, "env-st", {}, {"parse": {"seconds": 12.0}}, 0, 0, "attempt-b")
    await pool.execute(merge, "env-st", {}, {"visual": {"seconds": 30.0}}, 0, 0, "attempt-b")
    metrics = (await _metadata(pool, "env-st"))["metrics"]
    assert metrics["stages"] == {"parse": {"seconds": 12.0}, "visual": {"seconds": 30.0}}
    assert metrics["stages_attempt"] == "attempt-b"


async def test_a_live_stream_is_not_claimable(pool: asyncpg.Pool) -> None:
    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZS06"
    await _insert(pool, envelope_id)
    await m.record_stream_start(pool, envelope_id)  # fresh heartbeat
    assert await _claim(pool) is None
    status = await pool.fetchval("SELECT status FROM intake_envelopes WHERE id = $1", envelope_id)
    assert status == "received"
    assert (await _metadata(pool, envelope_id))["metrics"]["active_stream"] is True


async def test_a_stale_stream_is_claimable_and_the_claim_clears_it(pool: asyncpg.Pool) -> None:
    envelope_id = "01KT6XE8M1MW42M0Y0JJKRZS07"
    await _insert(
        pool,
        envelope_id,
        metadata={
            "metrics": {
                "active_stream": True,
                "claimed_at_epoch_s": await _ago(pool, "2 hours"),
                "stream_heartbeat_epoch_s": await _ago(pool, "11 minutes"),
            }
        },
    )
    claimed = await _claim(pool)
    assert claimed is not None and str(claimed.id) == envelope_id
    assert (await _metadata(pool, envelope_id))["metrics"]["active_stream"] is False
