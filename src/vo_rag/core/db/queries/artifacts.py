"""Typed query helpers for knowledge_artifacts."""

from __future__ import annotations

import asyncpg  # type: ignore[import-untyped]
import structlog
from pydantic import ValidationError

from vo_rag.core.db.cursor import decode_cursor, encode_cursor
from vo_rag.core.models.artifact import KnowledgeArtifact, LicenseClass
from vo_rag.core.models.envelope import TargetPipeline

_log = structlog.get_logger(__name__)


def _record_to_artifact(record: asyncpg.Record) -> KnowledgeArtifact:
    return KnowledgeArtifact.model_validate({**record})


async def insert_artifact(pool: asyncpg.Pool, artifact: KnowledgeArtifact) -> None:
    """Insert or refresh a knowledge artifact row."""
    await pool.execute(
        """
        INSERT INTO knowledge_artifacts (
            id, envelope_id, pipeline, artifact_type, title,
            content, tags, priority, status, license_class,
            qdrant_collection, qdrant_point_ids, source_provenance,
            created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9, $10,
            $11, $12, $13,
            $14, $15
        )
        ON CONFLICT (id)
        DO UPDATE SET
            envelope_id = EXCLUDED.envelope_id,
            pipeline = EXCLUDED.pipeline,
            artifact_type = EXCLUDED.artifact_type,
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            tags = EXCLUDED.tags,
            priority = EXCLUDED.priority,
            status = EXCLUDED.status,
            license_class = EXCLUDED.license_class,
            qdrant_collection = EXCLUDED.qdrant_collection,
            qdrant_point_ids = EXCLUDED.qdrant_point_ids,
            source_provenance = EXCLUDED.source_provenance,
            updated_at = EXCLUDED.updated_at
        """,
        artifact.id,
        artifact.envelope_id,
        artifact.pipeline,
        artifact.artifact_type,
        artifact.title,
        artifact.content,
        artifact.tags,
        artifact.priority,
        artifact.status,
        artifact.license_class,
        artifact.qdrant_collection,
        artifact.qdrant_point_ids,
        [entry.model_dump(mode="json") for entry in artifact.source_provenance],
        artifact.created_at,
        artifact.updated_at,
    )


async def get_artifact(pool: asyncpg.Pool, artifact_id: str) -> KnowledgeArtifact | None:
    """Get artifact by ID."""
    row = await pool.fetchrow("SELECT * FROM knowledge_artifacts WHERE id = $1", artifact_id)
    if row is None:
        return None
    return _record_to_artifact(row)


async def list_artifacts(
    pool: asyncpg.Pool,
    *,
    cursor: str | None = None,
    limit: int = 50,
    pipeline: TargetPipeline | str | None = None,
    tags: list[str] | None = None,
    license_class: LicenseClass | str | None = None,
) -> tuple[list[KnowledgeArtifact], str | None]:
    """List artifacts via keyset pagination on (created_at DESC, id DESC).

    Pages consisting entirely of invalid legacy rows are skipped server-side
    (up to a bounded number of hops) so callers never receive an empty page
    with a live cursor while valid rows remain further on.
    """
    items, next_cursor = await _list_artifacts_page(
        pool,
        cursor=cursor,
        limit=limit,
        pipeline=pipeline,
        tags=tags,
        license_class=license_class,
    )
    hops = 0
    while not items and next_cursor is not None and hops < 10:
        items, next_cursor = await _list_artifacts_page(
            pool,
            cursor=next_cursor,
            limit=limit,
            pipeline=pipeline,
            tags=tags,
            license_class=license_class,
        )
        hops += 1
    return items, next_cursor


async def _list_artifacts_page(
    pool: asyncpg.Pool,
    *,
    cursor: str | None,
    limit: int,
    pipeline: TargetPipeline | str | None,
    tags: list[str] | None,
    license_class: LicenseClass | str | None,
) -> tuple[list[KnowledgeArtifact], str | None]:
    where_clauses = ["TRUE"]
    args: list[object] = []

    if pipeline is not None:
        args.append(pipeline)
        where_clauses.append(f"pipeline = ${len(args)}")
    if tags:
        args.append(tags)
        where_clauses.append(f"tags @> ${len(args)}::text[]")
    if license_class is not None:
        args.append(license_class)
        where_clauses.append(f"license_class = ${len(args)}")
    if cursor is not None:
        cursor_created_at, cursor_id = decode_cursor(cursor)
        args.extend([cursor_created_at, cursor_id])
        where_clauses.append(f"(created_at, id) < (${len(args) - 1}, ${len(args)})")

    args.append(limit + 1)
    query = f"""
        SELECT *
        FROM knowledge_artifacts
        WHERE {" AND ".join(where_clauses)}
        ORDER BY created_at DESC, id DESC
        LIMIT ${len(args)}
    """

    rows = await pool.fetch(query, *args)
    has_next = len(rows) > limit
    page_rows = rows[:limit] if has_next else rows
    items: list[KnowledgeArtifact] = []
    for row in page_rows:
        try:
            items.append(_record_to_artifact(row))
        except ValidationError:
            # Legacy rows predating the current schema (older ids/provenance shapes)
            # must not poison the whole page.
            _log.warning("skipping_invalid_artifact_row", artifact_id=str(row["id"]))

    next_cursor: str | None = None
    if has_next and page_rows:
        # Cursor advances from the last RAW row, not the last valid item —
        # otherwise a trailing invalid row repeats forever.
        last_row = page_rows[-1]
        next_cursor = encode_cursor(last_row["created_at"], last_row["id"])
    return items, next_cursor


async def append_qdrant_point_ids(
    pool: asyncpg.Pool, artifact_id: str, point_ids: list[str]
) -> None:
    """Append point IDs to qdrant_point_ids array."""
    await pool.execute(
        """
        UPDATE knowledge_artifacts
        SET qdrant_point_ids = qdrant_point_ids || $2::text[]
        WHERE id = $1
        """,
        artifact_id,
        point_ids,
    )
