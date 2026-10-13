"""Envelope lookups (`envelope_ingested_at`, `ebook_source_ids`) against a real, migrated Postgres
(VOR-1427).

Gated on live testinfra (``VORAG_TEST_DATABASE_URL``; ``make up`` or
``scripts/testinfra-ci.sh`` start throwaway local containers). Never points
at production.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
import pytest

from vo_rag.core.corpus.duplicates import envelope_ingested_at
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


async def _envelope(
    pool: asyncpg.Pool,
    envelope_id: str,
    *,
    status: str,
    created_at: datetime,
    raw_content: dict[str, str],
    source_data: dict[str, str] | None = None,
) -> None:
    await pool.execute(
        "INSERT INTO intake_envelopes (id, source_type, source_data, raw_content, status, "
        "created_at) VALUES ($1, 'file_drop', $2::jsonb, $3::jsonb, $4, $5)",
        envelope_id,
        json.dumps(source_data or {}),
        json.dumps(raw_content),
        status,
        created_at,
    )


async def test_earliest_completed_envelope_dates_each_book(pg_pool: asyncpg.Pool) -> None:
    corpus_sha, acquired_sha, source_data_sha, missing_sha = "a" * 64, "b" * 64, "c" * 64, "d" * 64
    await _envelope(
        pg_pool,
        "e1",
        status="completed",
        created_at=datetime(2026, 7, 1, tzinfo=UTC),
        raw_content={"garage_key": f"corpus/{corpus_sha}/book.pdf"},
    )
    await _envelope(
        pg_pool,
        "e2",
        status="completed",
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
        raw_content={"garage_key": f"corpus/{corpus_sha}/book.pdf"},
    )
    # An earlier FAILED envelope for the same bytes is not an ingest.
    await _envelope(
        pg_pool,
        "e3",
        status="failed",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        raw_content={"garage_key": f"corpus/{corpus_sha}/book.pdf"},
    )
    # File-drop envelopes: no hash in the key, the sha256 is in file_hash.
    await _envelope(
        pg_pool,
        "e4",
        status="completed",
        created_at=datetime(2026, 8, 2, tzinfo=UTC),
        raw_content={"garage_key": "intake/01ABC/book.epub", "file_hash": acquired_sha},
    )
    await _envelope(
        pg_pool,
        "e5",
        status="completed",
        created_at=datetime(2026, 8, 3, tzinfo=UTC),
        raw_content={"text": "x"},
        source_data={"garage_key": f"books/{source_data_sha}.epub"},
    )

    result = await envelope_ingested_at(
        pg_pool,
        [
            f"vo-rag-corpus-sha256:{corpus_sha}",
            f"vo-rag-corpus-sha256:{acquired_sha}",
            source_data_sha.upper(),
            f"vo-rag-corpus-sha256:{missing_sha}",
            "not-a-hash",
        ],
    )

    assert result == {
        f"vo-rag-corpus-sha256:{corpus_sha}": str(datetime(2026, 5, 1, tzinfo=UTC)),
        f"vo-rag-corpus-sha256:{acquired_sha}": str(datetime(2026, 8, 2, tzinfo=UTC)),
        source_data_sha.upper(): str(datetime(2026, 8, 3, tzinfo=UTC)),
        f"vo-rag-corpus-sha256:{missing_sha}": None,
        "not-a-hash": None,
    }


async def test_ebook_source_ids_reads_the_envelope_garage_key(pg_pool: asyncpg.Pool) -> None:
    from vo_rag.core.corpus.source_format import ebook_source_ids

    epub_sha, mobi_sha, pdf_sha, azw3_sha = "1" * 64, "2" * 64, "3" * 64, "4" * 64
    when = datetime(2026, 7, 1, tzinfo=UTC)
    await _envelope(
        pg_pool,
        "x1",
        status="completed",
        created_at=when,
        raw_content={"garage_key": f"corpus/{epub_sha}/Book.EPUB"},
    )
    # File-drop key: no hash in it, the sha256 is in file_hash.
    await _envelope(
        pg_pool,
        "x2",
        status="completed",
        created_at=when,
        raw_content={"garage_key": "intake/01ABC/book.mobi", "file_hash": mobi_sha},
    )
    await _envelope(
        pg_pool,
        "x3",
        status="completed",
        created_at=when,
        raw_content={"garage_key": f"corpus/{pdf_sha}/legacy.pdf"},
    )
    await _envelope(
        pg_pool,
        "x4",
        status="failed",
        created_at=when,
        raw_content={"text": "x"},
        source_data={"garage_key": f"books/{azw3_sha}.azw3"},
    )

    result = await ebook_source_ids(
        pg_pool,
        [
            f"vo-rag-corpus-sha256:{epub_sha}",
            f"vo-rag-corpus-sha256:{mobi_sha}",
            f"vo-rag-corpus-sha256:{pdf_sha}",
            azw3_sha.upper(),
            "not-a-hash",
        ],
    )

    assert result == {
        f"vo-rag-corpus-sha256:{epub_sha}",
        f"vo-rag-corpus-sha256:{mobi_sha}",
        azw3_sha.upper(),
    }
