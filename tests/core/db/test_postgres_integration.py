"""Integration tests for the asyncpg pool + migration runner.

These tests are gated on the ``VORAG_TEST_DATABASE_URL`` env var. When it is
not set, the whole module is skipped — so a developer without a live Postgres
runs a fast unit-only suite.

To run locally::

    docker compose -f docker-compose.testinfra.yml up -d postgres
    VORAG_TEST_DATABASE_URL=postgresql://vo_rag:vo_rag@localhost/vo_rag  # pragma: allowlist secret

To exercise a fresh boot, drop the schema first::

    PGPASSWORD=vo_rag psql -h localhost -p 5434 -U vo_rag -d vo_rag \\
        -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"  # pragma: allowlist secret
"""

from __future__ import annotations

import os
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
import pytest

from vo_rag.core.db.postgres import (
    MigrationChecksumError,
    close_pool,
    get_pool,
    migration_status,
    run_migrations,
)

pytestmark = pytest.mark.integration

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "vo_rag" / "core" / "db" / "migrations"
)
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
    """Yield a clean asyncpg pool connected to the test database.

    We do NOT use the singleton ``get_pool`` here because the global
    ``_pool`` survives across tests in the same process. Each test gets its
    own pool and tears it down at the end.
    """
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


@pytest.fixture
async def clean_schema(pool: asyncpg.Pool) -> asyncpg.Pool:
    """Wipe the public schema so ``run_migrations`` is a clean boot."""
    async with pool.acquire() as conn:
        # The vce schema is created by migration 023 — drop it explicitly so
        # the next ``run_migrations`` call recreates it cleanly.
        await conn.execute("DROP SCHEMA IF EXISTS vce CASCADE")
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    return pool


class TestMigrations:
    async def test_run_migrations_is_idempotent(self, clean_schema: asyncpg.Pool) -> None:
        """First run applies every migration; second run is a no-op."""
        await run_migrations(clean_schema, MIGRATIONS_DIR)

        status_first = await migration_status(clean_schema, MIGRATIONS_DIR)
        assert status_first["ok"] is True
        assert status_first["pending"] == []
        assert status_first["unknown"] == []
        assert status_first["checksum_drifted"] == []
        assert status_first["checksum_untracked"] == []

        # Second run must be a no-op (idempotent).
        await run_migrations(clean_schema, MIGRATIONS_DIR)
        status_second = await migration_status(clean_schema, MIGRATIONS_DIR)
        assert status_second == status_first

    async def test_migration_status_clean_after_run(self, clean_schema: asyncpg.Pool) -> None:
        await run_migrations(clean_schema, MIGRATIONS_DIR)
        status = await migration_status(clean_schema, MIGRATIONS_DIR)
        assert status["ok"] is True
        assert status["expected_count"] == status["applied_count"]
        assert status["pending"] == []
        assert status["unknown"] == []
        assert status["checksum_drifted"] == []
        assert status["checksum_untracked"] == []

    async def test_migration_status_reports_pending_when_empty(
        self, clean_schema: asyncpg.Pool
    ) -> None:
        status = await migration_status(clean_schema, MIGRATIONS_DIR)
        # ``schema_migrations`` doesn't exist yet — migration_status should
        # treat the whole expected set as pending, not raise.
        assert status["ok"] is False
        assert status["applied_count"] == 0
        pending = status["pending"]
        assert isinstance(pending, list)
        assert len(pending) == status["expected_count"]
        assert status["unknown"] == []
        assert status["checksum_drifted"] == []
        assert status["checksum_untracked"] == []

    async def test_schema_migrations_table_tracks_versions(
        self, clean_schema: asyncpg.Pool
    ) -> None:
        await run_migrations(clean_schema, MIGRATIONS_DIR)
        async with clean_schema.acquire() as conn:
            rows = await conn.fetch(
                "SELECT version, checksum FROM schema_migrations ORDER BY version"
            )
        versions = [r["version"] for r in rows]
        # Unique versions — the renumbering of the 014/017 duplicate pairs
        # must mean no version appears twice in schema_migrations.
        assert len(versions) == len(set(versions))
        # Every expected migration file is represented.
        expected = sorted(p.stem for p in MIGRATIONS_DIR.glob("*.sql"))
        assert sorted(versions) == expected
        assert all(len(str(row["checksum"])) == 64 for row in rows)

    async def test_migration_status_reports_checksum_drift(
        self, clean_schema: asyncpg.Pool
    ) -> None:
        await run_migrations(clean_schema, MIGRATIONS_DIR)
        async with clean_schema.acquire() as conn:
            await conn.execute(
                "UPDATE schema_migrations SET checksum = $2 WHERE version = $1",
                "001_initial",
                "0" * 64,
            )

        status = await migration_status(clean_schema, MIGRATIONS_DIR)

        assert status["ok"] is False
        assert status["checksum_drifted"] == ["001_initial"]

        with pytest.raises(MigrationChecksumError, match="001_initial"):
            await run_migrations(clean_schema, MIGRATIONS_DIR)

    async def test_migration_status_reports_historical_untracked_checksum(
        self, clean_schema: asyncpg.Pool
    ) -> None:
        await run_migrations(clean_schema, MIGRATIONS_DIR)
        async with clean_schema.acquire() as conn:
            await conn.execute(
                "UPDATE schema_migrations SET checksum = NULL WHERE version = $1",
                "001_initial",
            )

        status = await migration_status(clean_schema, MIGRATIONS_DIR)

        assert status["ok"] is True
        assert status["checksum_drifted"] == []
        assert status["checksum_untracked"] == ["001_initial"]

    async def test_migration_status_reports_unknown_ledger_row(
        self, clean_schema: asyncpg.Pool
    ) -> None:
        await run_migrations(clean_schema, MIGRATIONS_DIR)
        async with clean_schema.acquire() as conn:
            await conn.execute(
                "INSERT INTO schema_migrations(version, checksum) VALUES ($1, NULL)",
                "999_restored_copy_only",
            )

        status = await migration_status(clean_schema, MIGRATIONS_DIR)

        assert status["ok"] is False
        assert status["unknown"] == ["999_restored_copy_only"]

    async def test_migration_status_accepts_reviewed_legacy_mapping(
        self, clean_schema: asyncpg.Pool
    ) -> None:
        await run_migrations(clean_schema, MIGRATIONS_DIR)
        async with clean_schema.acquire() as conn:
            await conn.execute(
                "INSERT INTO schema_migrations(version, checksum) VALUES ($1, NULL)",
                "999_reviewed_legacy",
            )
            await conn.execute(
                """
                INSERT INTO schema_migration_legacy_mappings (
                    legacy_version, canonical_version, disposition, evidence_review_id
                ) VALUES ($1, NULL, 'retired', 'test-restored-copy')
                """,
                "999_reviewed_legacy",
            )

        status = await migration_status(clean_schema, MIGRATIONS_DIR)

        assert status["ok"] is True
        assert status["unknown"] == []
        assert status["legacy_mapped"] == ["999_reviewed_legacy"]

    async def test_known_tables_exist_after_migrations(self, clean_schema: asyncpg.Pool) -> None:
        await run_migrations(clean_schema, MIGRATIONS_DIR)
        async with clean_schema.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
                """
            )
        public_tables = {r["tablename"] for r in rows}
        # A non-exhaustive spot-check on tables every query helper depends on.
        required = {
            "intake_envelopes",
            "knowledge_artifacts",
            "sources",
            "compliance_log",
            "answer_log",
            "answer_feedback",
            "query_log",
            "query_source_hit",
            "pii_subjects",
            "agent_termination_log",
            "portal_upload_sessions",
            "scraped_sources",
            "cloud_kill_log",
            "schema_migrations",
        }
        missing = required - public_tables
        assert not missing, f"missing tables after migrations: {sorted(missing)}"

    async def test_vce_schema_and_tables_present(self, clean_schema: asyncpg.Pool) -> None:
        await run_migrations(clean_schema, MIGRATIONS_DIR)
        async with clean_schema.acquire() as conn:
            rows = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'vce' ORDER BY tablename"
            )
        vce_tables = {r["tablename"] for r in rows}
        assert "video_manifest" in vce_tables
        assert "coaching_graph_nodes" in vce_tables
        assert "stage_run_log" in vce_tables


class TestPoolLifecycle:
    async def test_get_pool_then_close_pool_round_trip(self) -> None:
        """Smoke-test the module-level singleton API end to end.

        Uses the real ``get_pool`` / ``close_pool`` to catch import-order or
        global-state regressions that the per-test ``asyncpg.create_pool``
        fixture would hide.
        """
        dsn = _require_dsn()
        try:
            pool = await get_pool(dsn, min_size=1, max_size=2)
            async with pool.acquire() as conn:
                value = await conn.fetchval("SELECT 1")
            assert value == 1
        finally:
            await close_pool()
