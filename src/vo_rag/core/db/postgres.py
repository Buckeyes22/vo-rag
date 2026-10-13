"""Postgres connection pool and migration runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
import structlog

_pool: asyncpg.Pool | None = None
_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MigrationFile:
    """Describe one immutable SQL migration discovered on disk."""

    version: str
    path: Path
    checksum: str


class MigrationChecksumError(RuntimeError):
    """Raised when an applied migration no longer matches its recorded bytes."""


async def get_pool(
    dsn: str | None = None, *, min_size: int = 2, max_size: int = 10
) -> asyncpg.Pool:
    """Return singleton asyncpg pool configured for PgBouncer transaction mode."""
    global _pool
    if _pool is None:
        assert dsn, "dsn required on first call"
        _pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=min_size,
            max_size=max_size,
            statement_cache_size=0,
            init=_register_codecs,
        )
    return _pool


def get_existing_pool() -> asyncpg.Pool | None:
    """Return the process pool without creating or acquiring a connection."""
    return _pool


async def close_pool() -> None:
    """Close singleton pool if initialized."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def _register_codecs(conn: asyncpg.Connection) -> None:
    """Register JSONB codec for dict/list round-trip compatibility."""
    await conn.set_type_codec(
        "jsonb",
        schema="pg_catalog",
        encoder=lambda value: json.dumps(value),
        decoder=json.loads,
        format="text",
    )


async def run_migrations(pool: asyncpg.Pool, migrations_dir: str | Path) -> None:
    """Apply unapplied .sql migrations and record versions atomically."""
    migration_files = _migration_files(migrations_dir)

    async with pool.acquire() as conn:
        await _ensure_schema_migrations(conn)
        applied_migrations = await _fetch_applied_migrations(conn)

        drifted = [
            migration.version
            for migration in migration_files
            if (
                (recorded := applied_migrations.get(migration.version)) is not None
                and recorded != migration.checksum
            )
        ]
        if drifted:
            raise MigrationChecksumError(
                "applied migration checksum mismatch: " + ", ".join(sorted(drifted))
            )

        for migration_file in migration_files:
            if migration_file.version in applied_migrations:
                continue

            _log.info(
                "applying_migration",
                version=migration_file.version,
                checksum=migration_file.checksum,
            )
            async with conn.transaction():
                sql = migration_file.path.read_text(encoding="utf-8")
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations(version, checksum) VALUES ($1, $2)",
                    migration_file.version,
                    migration_file.checksum,
                )


def _migration_files(migrations_dir: str | Path) -> list[MigrationFile]:
    directory = Path(migrations_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"migration directory does not exist: {directory}")

    paths = sorted(directory.glob("*.sql"))
    if not paths:
        raise ValueError(f"migration directory contains no .sql files: {directory}")

    migrations = [
        MigrationFile(
            version=path.stem,
            path=path,
            checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in paths
    ]
    versions = [migration.version for migration in migrations]
    if len(versions) != len(set(versions)):
        raise ValueError(f"duplicate migration versions in {directory}")
    return migrations


async def _ensure_schema_migrations(conn: asyncpg.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            checksum TEXT,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        ALTER TABLE schema_migrations
            ADD COLUMN IF NOT EXISTS checksum TEXT
        """
    )


async def _fetch_applied_migrations(conn: asyncpg.Connection) -> dict[str, str | None]:
    has_checksum = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'schema_migrations'
              AND column_name = 'checksum'
        )
        """
    )
    if has_checksum:
        rows = await conn.fetch("SELECT version, checksum FROM schema_migrations")
        return {
            str(row["version"]): str(row["checksum"]) if row["checksum"] is not None else None
            for row in rows
        }

    rows = await conn.fetch("SELECT version FROM schema_migrations")
    return {str(row["version"]): None for row in rows}


async def _fetch_legacy_migration_mappings(
    conn: asyncpg.Connection,
) -> dict[str, str | None]:
    """Return reviewed legacy-ledger classifications, if reconciliation exists."""
    exists = await conn.fetchval(
        "SELECT to_regclass('public.schema_migration_legacy_mappings') IS NOT NULL"
    )
    if not exists:
        return {}
    rows = await conn.fetch(
        """
        SELECT legacy_version, canonical_version
        FROM schema_migration_legacy_mappings
        """
    )
    return {
        str(row["legacy_version"]): (
            str(row["canonical_version"]) if row["canonical_version"] is not None else None
        )
        for row in rows
    }


async def migration_status(pool: asyncpg.Pool, migrations_dir: str | Path) -> dict[str, object]:
    """Return migration drift status without mutating the database."""
    migration_files = _migration_files(migrations_dir)
    expected = [migration.version for migration in migration_files]
    expected_set = set(expected)
    checksums = {migration.version: migration.checksum for migration in migration_files}

    async with pool.acquire() as conn:
        try:
            applied = await _fetch_applied_migrations(conn)
            legacy_mappings = await _fetch_legacy_migration_mappings(conn)
        except asyncpg.UndefinedTableError:
            applied = {}
            legacy_mappings = {}

    effective_applied = set(applied)
    effective_applied.update(
        canonical
        for canonical in legacy_mappings.values()
        if canonical is not None and canonical in expected_set
    )
    pending = [version for version in expected if version not in effective_applied]
    unknown = sorted(set(applied) - expected_set - set(legacy_mappings))
    checksum_drifted = sorted(
        version
        for version, checksum in applied.items()
        if version in checksums and checksum is not None and checksum != checksums[version]
    )
    checksum_untracked = sorted(
        version
        for version, checksum in applied.items()
        if version in checksums and checksum is None
    )
    return {
        "ok": not pending and not unknown and not checksum_drifted,
        "expected_count": len(expected),
        "applied_count": len(effective_applied & expected_set),
        "pending": pending,
        "unknown": unknown,
        "checksum_drifted": checksum_drifted,
        "checksum_untracked": checksum_untracked,
        "legacy_mapped": sorted(legacy_mappings),
    }


__all__ = [
    "MigrationChecksumError",
    "MigrationFile",
    "close_pool",
    "get_pool",
    "migration_status",
    "run_migrations",
]
