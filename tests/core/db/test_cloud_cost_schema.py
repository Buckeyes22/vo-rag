"""Contract tests for the cloud_cost workload schema.

Static unit test: asserts the migration file (1) exists, (2) drops the legacy
workload CHECK constraint by its deterministic name, and (3) re-creates a CHECK
that admits exactly the static-router workloads: text-gen / vlm / dsi /
embedding-batch / misc. Catches regressions without needing a live database.

Integration test: gated on ``VORAG_TEST_DATABASE_URL``; actually applies the
migrations and asserts the new CHECK accepts the new values and rejects
the legacy-only values. Mirrors the pattern in test_postgres_integration.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
import pytest

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "vo_rag" / "core" / "db" / "migrations"
)
MIGRATION_FILE = MIGRATIONS_DIR / "032_cloud_cost_static_workloads.sql"

ALLOWED_WORKLOADS: tuple[str, ...] = (
    "text-gen",
    "vlm",
    "dsi",
    "embedding-batch",
    "misc",
)
LEGACY_ONLY_WORKLOADS: tuple[str, ...] = (
    "re-embedding",
    "ocr-burst",
    "sft-generation",
)

EXPECTED_CHECK = "workload IN ('text-gen', 'vlm', 'dsi', 'embedding-batch', 'misc')"


# ---------- static unit test (no live DB required) ----------


def test_migration_file_exists() -> None:
    assert MIGRATION_FILE.is_file(), f"missing migration: {MIGRATION_FILE}"


def test_migration_is_non_destructive_no_truncate() -> None:
    sql = MIGRATION_FILE.read_text(encoding="utf-8")
    upper = sql.upper()
    # Must not touch existing rows.
    assert "TRUNCATE" not in upper, "migration must not TRUNCATE cloud_cost"
    assert "DELETE" not in upper, "migration must not DELETE from cloud_cost"
    # Must not drop/recreate the table itself.
    assert "DROP TABLE" not in upper, "migration must not DROP TABLE cloud_cost"


def test_migration_drops_legacy_check_by_deterministic_name() -> None:
    """The migration must drop the known workload constraint idempotently."""
    sql = MIGRATION_FILE.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", sql)
    assert re.search(
        r"DROP CONSTRAINT IF EXISTS cloud_cost_workload_check",
        normalized,
        flags=re.IGNORECASE,
    ), "migration must idempotently drop the deterministic workload constraint"
    assert "pg_get_constraintdef" not in sql


def test_migration_recreates_check_with_static_router_workloads() -> None:
    sql = MIGRATION_FILE.read_text(encoding="utf-8")
    # The new CHECK must allow exactly the static-router workloads, in this order.
    # Normalize whitespace before comparison.
    normalized = re.sub(r"\s+", " ", sql)
    assert EXPECTED_CHECK in normalized, (
        f"migration must contain CHECK {EXPECTED_CHECK!r}; got:\n{sql}"
    )
    for legacy in LEGACY_ONLY_WORKLOADS:
        # The new CHECK must not list legacy-only values verbatim.
        # (Each legacy value must not appear inside a quoted IN-list element.)
        assert not re.search(rf"'\s*{re.escape(legacy)}\s*'", normalized), (
            f"legacy workload {legacy!r} must not be in the new CHECK"
        )


# ---------- live-DB integration test (skipped without a test DB) ----------


DSN_ENV = "VORAG_TEST_DATABASE_URL"


def _require_dsn() -> str:
    dsn = os.environ.get(DSN_ENV)
    if not dsn:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail(f"{DSN_ENV} is required by the testinfra lane")
        pytest.skip(f"{DSN_ENV} not set — skipping integration test")
    return dsn


@pytest.fixture
async def pool() -> asyncpg.Pool:
    dsn = _require_dsn()
    p = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=2,
        statement_cache_size=0,
    )
    try:
        yield p
    finally:
        await p.close()


@pytest.mark.integration
class TestCloudCostSchema:
    async def test_new_workloads_accepted_after_migration(self, pool: asyncpg.Pool) -> None:
        """After 032 runs, every static-router workload inserts cleanly."""
        from vo_rag.core.db.postgres import run_migrations

        async with pool.acquire() as conn:
            await conn.execute("DROP SCHEMA IF EXISTS vce CASCADE")
            await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
            await conn.execute("CREATE SCHEMA public")
        await run_migrations(pool, MIGRATIONS_DIR)

        async with pool.acquire() as conn:
            for workload in ALLOWED_WORKLOADS:
                await conn.execute(
                    """
                    INSERT INTO cloud_cost
                        (job_id, provider, gpu_type, workload,
                         started_at, cost_usd, status)
                    VALUES ($1, 'runpod', 'L4', $2, now(), 0.0, 'running')
                    """,
                    f"job-{workload}-ok",
                    workload,
                )

    async def test_legacy_only_workloads_rejected_after_migration(self, pool: asyncpg.Pool) -> None:
        """After 032 runs, legacy-only values (re-embedding / ocr-burst /
        sft-generation) violate the new CHECK and are refused with a
        CheckViolationError — proving the constraint was actually swapped."""
        from vo_rag.core.db.postgres import run_migrations

        async with pool.acquire() as conn:
            await conn.execute("DROP SCHEMA IF EXISTS vce CASCADE")
            await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
            await conn.execute("CREATE SCHEMA public")
        await run_migrations(pool, MIGRATIONS_DIR)

        async with pool.acquire() as conn:
            for legacy in LEGACY_ONLY_WORKLOADS:
                with pytest.raises(asyncpg.CheckViolationError):
                    await conn.execute(
                        """
                        INSERT INTO cloud_cost
                            (job_id, provider, gpu_type, workload,
                             started_at, cost_usd, status)
                        VALUES ($1, 'runpod', 'L4', $2, now(), 0.0, 'running')
                        """,
                        f"job-{legacy}-reject",
                        legacy,
                    )

    async def test_migration_converges_an_already_correct_schema(self, pool: asyncpg.Pool) -> None:
        """Reapplying 032's SQL to the desired schema stays valid and deterministic."""
        from vo_rag.core.db.postgres import run_migrations

        async with pool.acquire() as conn:
            await conn.execute("DROP SCHEMA IF EXISTS vce CASCADE")
            await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
            await conn.execute("CREATE SCHEMA public")
        await run_migrations(pool, MIGRATIONS_DIR)

        sql = MIGRATION_FILE.read_text(encoding="utf-8")
        async with pool.acquire() as conn:
            await conn.execute(sql)
            definition = await conn.fetchval(
                """
                SELECT pg_get_constraintdef(oid)
                FROM pg_constraint
                WHERE conrelid = 'public.cloud_cost'::regclass
                  AND conname = 'cloud_cost_workload_check'
                """
            )
        assert definition is not None
        assert "embedding-batch" in str(definition)

    async def test_migration_preserves_historical_legacy_rows_but_rejects_new_ones(
        self, pool: asyncpg.Pool
    ) -> None:
        """Production upgrade keeps old accounting facts without admitting new drift."""
        legacy_sql = (MIGRATIONS_DIR / "015_cloud_cost.sql").read_text(encoding="utf-8")
        migration_sql = MIGRATION_FILE.read_text(encoding="utf-8")
        async with pool.acquire() as conn:
            await conn.execute("DROP SCHEMA IF EXISTS vce CASCADE")
            await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
            await conn.execute("CREATE SCHEMA public")
            await conn.execute(legacy_sql)
            await conn.execute(
                """
                INSERT INTO cloud_cost
                    (job_id, provider, gpu_type, workload, started_at, cost_usd, status)
                VALUES ('historical-reembed', 'runpod', 'L4', 're-embedding', now(), 1.0,
                        'completed')
                """
            )

            await conn.execute(migration_sql)

            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM cloud_cost WHERE workload = 're-embedding'"
                )
                == 1
            )
            assert (
                await conn.fetchval(
                    """
                SELECT convalidated FROM pg_constraint
                WHERE conrelid = 'public.cloud_cost'::regclass
                  AND conname = 'cloud_cost_workload_check'
                """
                )
                is False
            )
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO cloud_cost
                        (job_id, provider, gpu_type, workload, started_at, cost_usd, status)
                    VALUES ('new-reembed', 'runpod', 'L4', 're-embedding', now(), 0.0,
                            'running')
                    """
                )
