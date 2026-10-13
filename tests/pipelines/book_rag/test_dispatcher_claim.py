"""Lane-class claim filtering (VOR-1434): a dedicated giant lane claims
only ``preflight.route == "giant"`` envelopes, a normal lane claims everything
except a giant, and an unset lane claims anything (the default). A
normal lane may still take a starving giant once it has waited past a
configurable age, so giants never wait forever when no giant lane runs.

Fast unit tests below use the same in-memory fakes as ``test_dispatcher.py``
(the SQL text/param-passing contract). The real claim *behaviour* — what a
given lane class actually claims from a real table — is only trustworthy
against real Postgres, so it's covered by the ``pytest.mark.integration``
tests at the bottom (gated on ``VORAG_TEST_DATABASE_URL``, same as
``test_ingest_metrics_integration.py``).
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from vo_rag.core.models.envelope import (
    EnvelopeStatus,
    IntakeSource,
    TargetPipeline,
)

if TYPE_CHECKING:
    from vo_rag.core.config.settings import Settings

_ENVELOPE_ID = "01KT6XE8M1MW42M0Y0JJKRZSVP"  # pragma: allowlist secret
_TS = datetime(2026, 1, 1, tzinfo=UTC)


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
        "status": status,
        "target_pipeline": TargetPipeline.BOOK_RAG,
        "created_at": _TS,
        "updated_at": _TS,
    }


class _FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> None:
        return None


@dataclass
class _FakePool:
    row: dict[str, Any] | None
    claim_queries: list[str] = field(default_factory=list)
    claim_args: list[tuple[Any, ...]] = field(default_factory=list)

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(_FakeConnection(self))

    async def execute(self, query: str, *args: Any) -> str:
        return "UPDATE 0"


@dataclass
class _FakeAcquire:
    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, *exc: object) -> None:
        return None


@dataclass
class _FakeConnection:
    pool: _FakePool

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def fetchrow(self, query: str, *args: Any) -> Mapping[str, Any] | None:
        self.pool.claim_queries.append(query)
        self.pool.claim_args.append(args)
        if self.pool.row is None:
            return None
        self.pool.row["status"] = EnvelopeStatus.PROCESSING
        return dict(self.pool.row)


# ---------------------------------------------------------------------------
# Unit: the lane-class predicate and its parameters (SQL text/param-passing).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_sql_has_a_lane_class_predicate() -> None:
    from vo_rag.pipelines.book_rag.dispatcher import _CLAIM_SQL

    assert "$10" in _CLAIM_SQL
    assert "preflight" in _CLAIM_SQL
    assert "'giant'" in _CLAIM_SQL
    assert "IS DISTINCT FROM 'giant'" in _CLAIM_SQL


@pytest.mark.asyncio
async def test_claim_passes_lane_class_and_fallback_age_positionally() -> None:
    from vo_rag.pipelines.book_rag.dispatcher import claim_next_envelope

    pool = _FakePool(row=_envelope_row())
    conn = _FakeConnection(pool)

    await claim_next_envelope(conn, lane_class="giant", giant_fallback_after_s=3600.0)

    assert pool.claim_args[0][9] == "giant"
    assert pool.claim_args[0][10] == 3600.0


@pytest.mark.asyncio
async def test_claim_defaults_lane_class_to_none_and_fallback_to_six_hours() -> None:
    from vo_rag.pipelines.book_rag.dispatcher import (
        DEFAULT_GIANT_FALLBACK_AFTER_S,
        claim_next_envelope,
    )

    pool = _FakePool(row=_envelope_row())
    conn = _FakeConnection(pool)

    await claim_next_envelope(conn)

    assert pool.claim_args[0][9] is None
    assert pool.claim_args[0][10] == DEFAULT_GIANT_FALLBACK_AFTER_S
    assert DEFAULT_GIANT_FALLBACK_AFTER_S == 6 * 60 * 60.0


@pytest.mark.asyncio
async def test_run_dispatcher_loop_threads_lane_class_into_the_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import vo_rag.pipelines.book_rag.dispatcher as dispatcher_mod

    seen: dict[str, Any] = {}

    async def _fake_claim_from_pool(pool: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        return None

    monkeypatch.setattr(dispatcher_mod, "_claim_from_pool", _fake_claim_from_pool)
    monkeypatch.setattr(dispatcher_mod, "write_dispatcher_heartbeat", lambda state: None)

    stop_event = asyncio.Event()

    async def _stop_after_one_poll() -> None:
        await asyncio.sleep(0)
        stop_event.set()

    asyncio.ensure_future(_stop_after_one_poll())

    await dispatcher_mod.run_dispatcher_loop(
        _FakePool(row=None),
        cast("Settings", object()),
        poll_interval=0.01,
        stop_event=stop_event,
        lane_class="normal",
        giant_fallback_after_s=1234.0,
    )

    assert seen["lane_class"] == "normal"
    assert seen["giant_fallback_after_s"] == 1234.0


# ---------------------------------------------------------------------------
# Unit: env parsing.
# ---------------------------------------------------------------------------


def test_env_lane_class_accepts_giant_and_normal(monkeypatch: pytest.MonkeyPatch) -> None:
    from vo_rag.pipelines.book_rag.dispatcher import _env_lane_class

    monkeypatch.setenv("VORAG_TEST_LANE_CLASS", "giant")
    assert _env_lane_class("VORAG_TEST_LANE_CLASS") == "giant"
    monkeypatch.setenv("VORAG_TEST_LANE_CLASS", " Normal ")
    assert _env_lane_class("VORAG_TEST_LANE_CLASS") == "normal"


def test_env_lane_class_unset_or_invalid_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from vo_rag.pipelines.book_rag.dispatcher import _env_lane_class

    monkeypatch.delenv("VORAG_TEST_LANE_CLASS", raising=False)
    assert _env_lane_class("VORAG_TEST_LANE_CLASS") is None
    monkeypatch.setenv("VORAG_TEST_LANE_CLASS", "")
    assert _env_lane_class("VORAG_TEST_LANE_CLASS") is None
    monkeypatch.setenv("VORAG_TEST_LANE_CLASS", "bogus")
    assert _env_lane_class("VORAG_TEST_LANE_CLASS") is None


# ---------------------------------------------------------------------------
# Integration: real Postgres claim behaviour.
#
# Gated on VORAG_TEST_DATABASE_URL, same convention as
# test_ingest_metrics_integration.py (`make up` starts a throwaway local
# container; never production).
# ---------------------------------------------------------------------------

try:
    import asyncpg  # type: ignore[import-untyped]

    from vo_rag.core.db.postgres import _register_codecs, run_migrations
    from vo_rag.pipelines.book_rag.dispatcher import claim_next_envelope as _claim_next_envelope

    _ASYNCPG_AVAILABLE = True
except ImportError:  # pragma: no cover - asyncpg ships with the runtime extra
    _ASYNCPG_AVAILABLE = False

_MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "vo_rag" / "core" / "db" / "migrations"
)
_HASH = "d" * 64


def _require_dsn() -> str:
    dsn = os.environ.get("VORAG_TEST_DATABASE_URL")
    if not dsn:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail("VORAG_TEST_DATABASE_URL is required by the testinfra lane")
        pytest.skip("VORAG_TEST_DATABASE_URL not set -- skipping integration test")
    return dsn


@pytest.fixture
async def pg_pool() -> AsyncGenerator[Any, None]:
    if not _ASYNCPG_AVAILABLE:
        pytest.skip("asyncpg not installed -- install the runtime extra")
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
    pool: Any,
    envelope_id: str,
    *,
    route: str | None,
    created_ago: timedelta = timedelta(hours=1),
    status: str = "received",
    updated_ago: timedelta = timedelta(0),
) -> None:
    metadata: dict[str, Any] = {}
    if route is not None:
        metadata["preflight"] = {"route": route}
    await pool.execute(
        """
        INSERT INTO intake_envelopes
            (id, source_type, raw_content, metadata, status, target_pipeline,
             created_at, updated_at)
        VALUES ($1, 'file_drop', $2, $3, $5, 'book_rag', NOW() - $4::interval,
                NOW() - $6::interval)
        """,
        envelope_id,
        {
            "garage_key": f"drops/{envelope_id}.pdf",
            "file_hash": _HASH,
            "object_verification": {"status": "verified", "sha256": _HASH},
        },
        metadata,
        created_ago,
        status,
        updated_ago,
    )


async def _claim(pool: Any, **kwargs: Any) -> Any:
    async with pool.acquire() as conn, conn.transaction():
        return await _claim_next_envelope(conn, **kwargs)


@pytest.mark.integration
async def test_unset_lane_claims_a_giant_and_a_normal_envelope(pg_pool: Any) -> None:
    await _insert(pg_pool, "01KT6XE8M1MW42M0Y0JJKRZS10", route="giant")
    claimed = await _claim(pg_pool)
    assert claimed is not None and str(claimed.id) == "01KT6XE8M1MW42M0Y0JJKRZS10"

    await _insert(pg_pool, "01KT6XE8M1MW42M0Y0JJKRZS11", route="native")
    claimed = await _claim(pg_pool)
    assert claimed is not None and str(claimed.id) == "01KT6XE8M1MW42M0Y0JJKRZS11"


@pytest.mark.integration
async def test_giant_lane_claims_only_giant_routed_envelopes(pg_pool: Any) -> None:
    await _insert(pg_pool, "01KT6XE8M1MW42M0Y0JJKRZS20", route="native")
    assert await _claim(pg_pool, lane_class="giant") is None

    await _insert(pg_pool, "01KT6XE8M1MW42M0Y0JJKRZS21", route="giant")
    claimed = await _claim(pg_pool, lane_class="giant")
    assert claimed is not None and str(claimed.id) == "01KT6XE8M1MW42M0Y0JJKRZS21"


@pytest.mark.integration
async def test_normal_lane_never_claims_a_fresh_giant(pg_pool: Any) -> None:
    await _insert(pg_pool, "01KT6XE8M1MW42M0Y0JJKRZS30", route="giant")
    assert await _claim(pg_pool, lane_class="normal") is None


@pytest.mark.integration
async def test_normal_lane_claims_non_giant_and_route_less_envelopes(pg_pool: Any) -> None:
    await _insert(pg_pool, "01KT6XE8M1MW42M0Y0JJKRZS40", route="native")
    claimed = await _claim(pg_pool, lane_class="normal")
    assert claimed is not None and str(claimed.id) == "01KT6XE8M1MW42M0Y0JJKRZS40"

    # An envelope with no preflight metadata at all must stay claimable by a
    # normal lane -- NULL route is DISTINCT FROM 'giant'.
    await _insert(pg_pool, "01KT6XE8M1MW42M0Y0JJKRZS41", route=None)
    claimed = await _claim(pg_pool, lane_class="normal")
    assert claimed is not None and str(claimed.id) == "01KT6XE8M1MW42M0Y0JJKRZS41"


@pytest.mark.integration
async def test_normal_lane_falls_back_to_a_starving_giant_past_the_fallback_age(
    pg_pool: Any,
) -> None:
    await _insert(
        pg_pool,
        "01KT6XE8M1MW42M0Y0JJKRZS50",
        route="giant",
        created_ago=timedelta(hours=7),
    )
    # Still within the (default 6h) fallback window from a shorter age: no fallback.
    assert await _claim(pg_pool, lane_class="normal", giant_fallback_after_s=8 * 3600.0) is None
    claimed = await _claim(pg_pool, lane_class="normal", giant_fallback_after_s=6 * 3600.0)
    assert claimed is not None and str(claimed.id) == "01KT6XE8M1MW42M0Y0JJKRZS50"


@pytest.mark.integration
async def test_giant_lane_ignores_the_fallback_age_it_already_claims_everything_giant(
    pg_pool: Any,
) -> None:
    await _insert(
        pg_pool,
        "01KT6XE8M1MW42M0Y0JJKRZS60",
        route="giant",
        created_ago=timedelta(minutes=1),
    )
    claimed = await _claim(pg_pool, lane_class="giant", giant_fallback_after_s=6 * 3600.0)
    assert claimed is not None and str(claimed.id) == "01KT6XE8M1MW42M0Y0JJKRZS60"


@pytest.mark.integration
async def test_normal_lane_never_falls_back_while_a_giant_lane_is_working(
    pg_pool: Any,
) -> None:
    """In a long bulk run every queued giant eventually passes the fallback
    age. While a giant lane is alive -- it holds a processing giant
    heartbeated within the stale timeout -- a normal lane must still leave
    the queued giants to it."""
    await _insert(
        pg_pool,
        "01KT6XE8M1MW42M0Y0JJKRZS70",
        route="giant",
        created_ago=timedelta(hours=9),
        status="processing",
        updated_ago=timedelta(minutes=1),
    )
    await _insert(
        pg_pool, "01KT6XE8M1MW42M0Y0JJKRZS71", route="giant", created_ago=timedelta(hours=7)
    )

    assert await _claim(pg_pool, lane_class="normal", giant_fallback_after_s=6 * 3600.0) is None


@pytest.mark.integration
async def test_normal_lane_falls_back_once_the_giant_lane_has_gone_stale(
    pg_pool: Any,
) -> None:
    """A giant envelope whose heartbeat is older than the stale timeout means
    no giant lane is running it any more: the fallback applies again."""
    await _insert(
        pg_pool,
        "01KT6XE8M1MW42M0Y0JJKRZS80",
        route="giant",
        created_ago=timedelta(hours=1),
        status="processing",
        updated_ago=timedelta(minutes=20),
    )
    await _insert(
        pg_pool, "01KT6XE8M1MW42M0Y0JJKRZS81", route="giant", created_ago=timedelta(hours=7)
    )

    claimed = await _claim(pg_pool, lane_class="normal", giant_fallback_after_s=6 * 3600.0)
    assert claimed is not None and str(claimed.id) == "01KT6XE8M1MW42M0Y0JJKRZS81"


@pytest.mark.integration
async def test_stale_reclaim_respects_the_lane_class(pg_pool: Any) -> None:
    """A stale processing envelope is reclaimed only by a
    lane of its own class, and the reclaim is counted in its metrics."""
    await _insert(
        pg_pool,
        "01KT6XE8M1MW42M0Y0JJKRZS90",
        route="giant",
        created_ago=timedelta(hours=1),
        status="processing",
        updated_ago=timedelta(minutes=20),
    )
    await _insert(
        pg_pool,
        "01KT6XE8M1MW42M0Y0JJKRZS91",
        route="native",
        created_ago=timedelta(hours=2),
        status="processing",
        updated_ago=timedelta(minutes=20),
    )

    giant = await _claim(pg_pool, lane_class="giant")
    assert giant is not None and str(giant.id) == "01KT6XE8M1MW42M0Y0JJKRZS90"
    assert await _claim(pg_pool, lane_class="giant") is None

    normal = await _claim(pg_pool, lane_class="normal")
    assert normal is not None and str(normal.id) == "01KT6XE8M1MW42M0Y0JJKRZS91"

    rows = await pg_pool.fetch(
        "SELECT id, status, metadata FROM intake_envelopes ORDER BY id",
    )
    for row in rows:
        assert row["status"] == "processing"
        assert row["metadata"]["metrics"]["stale_reclaims"] == 1
        assert row["metadata"]["metrics"]["stale_reclaims_total"] == 1
    assert rows[0]["metadata"]["preflight"] == {"route": "giant"}
    assert rows[1]["metadata"]["preflight"] == {"route": "native"}
