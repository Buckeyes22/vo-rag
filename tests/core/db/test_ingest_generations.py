"""Durable ingest-generation query and schema contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from vo_rag.core.db.queries.ingest_generations import (
    activate_generation_record_locked,
    count_active_generation_artifacts,
    create_generation,
    generation_claim_is_current,
    get_active_generation,
    get_superseded_generation_ids,
    record_generation_cleanup,
    refresh_generation_claim,
    release_generation_claim,
    transition_generation,
    update_generation_counts,
)
from vo_rag.core.models.ingest_generation import (
    IngestGenerationState,
    SourceIngestGeneration,
)


@dataclass
class _Pool:
    row: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    execute_result: str = "UPDATE 1"
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append((query, args))
        return self.execute_result

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.calls.append((query, args))
        return self.row

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((query, args))
        return self.rows


def _row(**changes: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "generation_id": "gen-1",
        "source_id": "source-1",
        "raw_collection": "vo_rag_books_raw_v1",
        "envelope_id": "env-1",
        "run_id": None,
        "pipeline": "book_rag",
        "schema_version": "book-rag-points-v1",
        "chunking_version": "default",
        "sterilize_enabled": False,
        "state": "active",
        "expected_raw_points": 2,
        "actual_raw_points": 2,
        "expected_sterile_points": 2,
        "actual_sterile_points": 2,
        "expected_artifacts": 1,
        "actual_artifacts": 1,
        "validation_decision": '{"passed": true}',
        "smoke_decision": "pass",
        "failure_reason": None,
        "created_at": None,
        "updated_at": None,
        "activated_at": None,
        "completed_at": None,
    }
    values.update(changes)
    return values


@pytest.mark.asyncio
async def test_create_generation_atomically_claims_or_takes_over_expired_staging() -> None:
    pool = _Pool()
    generation = SourceIngestGeneration(
        generation_id="gen-1",
        source_id="source-1",
        raw_collection="vo_rag_books_raw_v1",
        envelope_id="env-1",
        schema_version="book-rag-points-v1",
        chunking_version="default",
    )

    inserted = await create_generation(pool, generation)

    assert inserted is True
    query, args = pool.calls[0]
    assert "ON CONFLICT (generation_id) DO UPDATE" in query
    assert "claim_expires_at <= NOW()" in query
    assert args[9] is IngestGenerationState.STAGING


@pytest.mark.asyncio
async def test_generation_claim_release_is_owner_scoped_and_recoverable() -> None:
    pool = _Pool()

    released = await release_generation_claim(pool, "gen-1", "owner-1")

    assert released is True
    query, args = pool.calls[0]
    assert "owner_token = $2" in query
    assert "state IN ('staging', 'validated')" in query
    assert args == ("gen-1", "owner-1")


@pytest.mark.asyncio
async def test_generation_claim_current_requires_unexpired_owner() -> None:
    pool = _Pool(row={"current": False})

    current = await generation_claim_is_current(pool, "gen-1", "owner-old")

    assert current is False
    query, args = pool.calls[0]
    assert "owner_token = $2" in query
    assert "claim_expires_at > NOW()" in query
    assert args == ("gen-1", "owner-old")


@pytest.mark.asyncio
async def test_expired_same_owner_cannot_resurrect_generation_lease() -> None:
    pool = _Pool(execute_result="UPDATE 0")

    refreshed = await refresh_generation_claim(
        pool,
        "gen-1",
        "owner-expired",
        datetime.now(UTC),
    )

    assert refreshed is False
    query, args = pool.calls[0]
    assert "owner_token = $2" in query
    assert "claim_expires_at > NOW()" in query
    assert args[0:2] == ("gen-1", "owner-expired")


@pytest.mark.asyncio
async def test_generation_count_update_is_owner_and_expiry_fenced() -> None:
    pool = _Pool()

    changed = await update_generation_counts(
        pool,
        "gen-1",
        expected_raw_points=2,
        actual_raw_points=2,
        owner_token="owner-1",
    )

    assert changed is True
    query, args = pool.calls[0]
    assert "owner_token = $7" in query
    assert "claim_expires_at > NOW()" in query
    assert args[6] == "owner-1"


@pytest.mark.asyncio
async def test_active_generation_decodes_json_without_pool_codec() -> None:
    pool = _Pool(row=_row())

    generation = await get_active_generation(
        pool,
        "source-1",
        "vo_rag_books_raw_v1",
    )

    assert generation is not None
    assert generation.state is IngestGenerationState.ACTIVE
    assert generation.validation_decision == {"passed": True}
    assert "state = 'active'" in pool.calls[0][0]
    assert pool.calls[0][1] == ("source-1", "vo_rag_books_raw_v1")


@pytest.mark.asyncio
async def test_active_generation_artifact_count_is_collection_scoped() -> None:
    pool = _Pool(row={"count": 1})

    count = await count_active_generation_artifacts(
        pool,
        "source-1",
        "vo_rag_reddit_v1",
        "gen-1",
    )

    assert count == 1
    query, args = pool.calls[0]
    assert "status = 'active'" in query
    assert "content->>'generation_id' = $3" in query
    assert "raw_collection" in query
    assert args == ("source-1", "vo_rag_reddit_v1", "gen-1")


@pytest.mark.asyncio
async def test_transition_is_compare_and_set() -> None:
    pool = _Pool()

    changed = await transition_generation(
        pool,
        "gen-1",
        from_states=(IngestGenerationState.STAGING,),
        to_state=IngestGenerationState.QUARANTINED,
        smoke_decision="quarantine",
    )

    assert changed is True
    query, args = pool.calls[0]
    assert "state = ANY($6::text[])" in query
    assert args[1] == "quarantined"
    assert args[5] == ["staging"]


@pytest.mark.asyncio
async def test_cleanup_failure_is_durably_retryable() -> None:
    pool = _Pool()

    await record_generation_cleanup(
        pool,
        "gen-old",
        required=True,
        error="raw: TimeoutError",
    )

    query, args = pool.calls[0]
    assert "cleanup_required = $2" in query
    assert "state IN ('active', 'superseded')" in query
    assert args == ("gen-old", True, "raw: TimeoutError")


@pytest.mark.asyncio
async def test_superseded_generation_ids_are_cleanup_targets() -> None:
    pool = _Pool(rows=[{"generation_id": "gen-old-1"}, {"generation_id": "gen-old-2"}])

    generation_ids = await get_superseded_generation_ids(
        pool,
        "source-1",
        "vo_rag_reddit_v1",
    )

    assert generation_ids == ["gen-old-1", "gen-old-2"]
    query, args = pool.calls[0]
    assert "state = 'superseded'" in query
    assert args == ("source-1", "vo_rag_reddit_v1")


@pytest.mark.asyncio
async def test_locked_activation_supersedes_old_generation_and_promotes_artifact() -> None:
    conn = _Pool(row={"state": "validated"})

    await activate_generation_record_locked(
        conn,
        "gen-1",
        "source-1",
        "vo_rag_reddit_v1",
    )

    sql = "\n".join(query for query, _args in conn.calls)
    assert "FOR UPDATE" in sql
    assert "SET state = 'superseded'" in sql
    assert "SET status = 'superseded'" in sql
    assert "SET status = 'active'" in sql
    assert "state = 'validated'" in sql
    assert "raw_collection = $3" in sql
    assert all(
        "vo_rag_reddit_v1" in args
        for query, args in conn.calls
        if "source_ingest_generations" in query or "knowledge_artifacts" in query
    )


@pytest.mark.asyncio
async def test_locked_activation_rejects_nonvalidated_generation() -> None:
    conn = _Pool(row={"state": "staging"})
    with pytest.raises(RuntimeError, match="not validated"):
        await activate_generation_record_locked(
            conn,
            "gen-1",
            "source-1",
            "vo_rag_books_raw_v1",
        )


@pytest.mark.asyncio
async def test_locked_activation_is_fenced_to_unexpired_owner() -> None:
    conn = _Pool(row={"state": "validated"})

    await activate_generation_record_locked(
        conn,
        "gen-1",
        "source-1",
        "vo_rag_books_raw_v1",
        "owner-1",
    )

    first_query, first_args = conn.calls[0]
    assert "owner_token = $4" in first_query
    assert "claim_expires_at > NOW()" in first_query
    assert first_args == (
        "gen-1",
        "source-1",
        "vo_rag_books_raw_v1",
        "owner-1",
    )


def test_migration_enforces_one_active_and_reconciled_counts() -> None:
    migration = (
        Path(__file__).parents[3]
        / "src/vo_rag/core/db/migrations/035_source_ingest_generations.sql"
    ).read_text(encoding="utf-8")
    assert "WHERE state = 'active'" in migration
    assert "ON source_ingest_generations (source_id, raw_collection)" in migration
    assert "expected_raw_points = actual_raw_points" in migration
    assert "expected_sterile_points = actual_sterile_points" in migration
    assert "expected_artifacts = actual_artifacts" in migration
    assert "validation_decision->>'passed' = 'true'" in migration
    assert "smoke_decision IN ('pass', 'skipped')" in migration
    assert "WHERE cleanup_required" in migration
