"""``REQUEUE_SQL`` against a real Postgres (VOR-1434).

Requeueing a failed envelope clears ``metadata.metrics.active_stream`` (as the
dispatcher's claim does) and keeps every other metadata key; malformed
metadata is left alone. Gated on ``VORAG_TEST_DATABASE_URL`` (never production).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import asyncpg  # type: ignore[import-untyped]
import pytest

from vo_rag.core.db.postgres import _register_codecs, run_migrations
from vo_rag.ocr_farm.coordinator import REQUEUE_SQL

pytestmark = pytest.mark.integration

_MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "vo_rag" / "core" / "db" / "migrations"
)


@pytest.fixture
async def pool() -> AsyncGenerator[asyncpg.Pool, None]:
    dsn = os.environ.get("VORAG_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("VORAG_TEST_DATABASE_URL not set -- skipping integration test")
    p = await asyncpg.create_pool(
        dsn=dsn, min_size=1, max_size=2, statement_cache_size=0, init=_register_codecs
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


async def _insert(pool: asyncpg.Pool, envelope_id: str, status: str, metadata: Any) -> None:
    await pool.execute(
        """
        INSERT INTO intake_envelopes
            (id, source_type, raw_content, metadata, status, target_pipeline)
        VALUES ($1, 'file_drop', $2, $3, $4, 'book_rag')
        """,
        envelope_id,
        {"garage_key": f"drops/{envelope_id}.pdf"},
        metadata,
        status,
    )


async def _row(pool: asyncpg.Pool, envelope_id: str) -> tuple[str, Any]:
    row = await pool.fetchrow(
        "SELECT status, metadata FROM intake_envelopes WHERE id = $1", envelope_id
    )
    meta = row["metadata"]
    return row["status"], meta if not isinstance(meta, str) else json.loads(meta)


async def test_requeue_clears_active_stream_and_keeps_other_keys(pool: asyncpg.Pool) -> None:
    metadata = {"title": "T", "metrics": {"active_stream": True, "stale_reclaims": 2}}
    await _insert(pool, "E1", "failed", metadata)

    assert (await pool.execute(REQUEUE_SQL, "E1")).endswith(" 1")

    status, meta = await _row(pool, "E1")
    assert status == "received"
    assert meta == {"title": "T", "metrics": {"active_stream": False, "stale_reclaims": 2}}


async def test_requeue_leaves_metadata_without_metrics_alone(pool: asyncpg.Pool) -> None:
    await _insert(pool, "E2", "failed", {"title": "T", "metrics": "garbage"})
    await _insert(pool, "E3", "failed", {"title": "T"})

    await pool.execute(REQUEUE_SQL, "E2")
    await pool.execute(REQUEUE_SQL, "E3")

    assert await _row(pool, "E2") == ("received", {"title": "T", "metrics": "garbage"})
    assert await _row(pool, "E3") == ("received", {"title": "T"})


async def test_requeue_keeps_the_failed_guard(pool: asyncpg.Pool) -> None:
    await _insert(pool, "E4", "processing", {"metrics": {"active_stream": True}})

    assert (await pool.execute(REQUEUE_SQL, "E4")).endswith(" 0")

    assert await _row(pool, "E4") == ("processing", {"metrics": {"active_stream": True}})
