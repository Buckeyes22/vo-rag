"""`rekey_chunk_tags` against a real, migrated Postgres (VOR-1450).

Gated on live testinfra (``VORAG_TEST_DATABASE_URL``; ``make up`` or
``scripts/testinfra-ci.sh`` start throwaway local containers). Never points
at production.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
import pytest

from vo_rag.core.corpus.tf_store import rekey_chunk_tags
from vo_rag.core.db.postgres import run_migrations

pytestmark = pytest.mark.integration

_MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "vo_rag" / "core" / "db" / "migrations"
)


def _require_pg_dsn() -> str:
    dsn = os.environ.get("VORAG_TEST_DATABASE_URL")
    if not dsn:
        if os.environ.get("VORAG_REQUIRE_TESTINFRA") == "1":
            pytest.fail("VORAG_TEST_DATABASE_URL is required by the testinfra lane")
        pytest.skip("VORAG_TEST_DATABASE_URL not set -- skipping integration test")
    return dsn


@pytest.fixture
async def pg_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    pool = await asyncpg.create_pool(
        dsn=_require_pg_dsn(), min_size=1, max_size=2, statement_cache_size=0
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute("DROP SCHEMA IF EXISTS vce CASCADE")
            await conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
            await conn.execute("CREATE SCHEMA public")
        await run_migrations(pool, _MIGRATIONS_DIR)
        yield pool
    finally:
        await pool.close()


async def _row(pool: asyncpg.Pool, key: str, tag: str) -> None:
    await pool.execute(
        "INSERT INTO tf_chunk_tags (source_id, content_sha256, tags, tag_version, tagger_model)"
        " VALUES ('s1', $1, $2::jsonb, 1, 'm')",
        key,
        f'{{"t": "{tag}"}}',
    )


async def _keys(pool: asyncpg.Pool) -> dict[str, str]:
    rows = await pool.fetch(
        "SELECT content_sha256, tags->>'t' AS t FROM tf_chunk_tags WHERE source_id = 's1'"
    )
    return {r["content_sha256"]: r["t"] for r in rows}


async def test_rekey_moves_rows_keeps_newer_and_is_idempotent(pg_pool: asyncpg.Pool) -> None:
    await _row(pg_pool, "old1", "a")
    await _row(pg_pool, "old2", "stale")
    await _row(pg_pool, "new2", "newer")
    pairs = [("old1", "new1"), ("old2", "new2"), ("absent", "new3"), ("same", "same")]
    assert await rekey_chunk_tags(pg_pool, "s1", pairs) == 2
    # old1 moved; old2's copy never overwrote the newer new2 row; both legacy rows gone.
    assert await _keys(pg_pool) == {"new1": "a", "new2": "newer"}
    assert await rekey_chunk_tags(pg_pool, "s1", pairs) == 0
    assert await _keys(pg_pool) == {"new1": "a", "new2": "newer"}
