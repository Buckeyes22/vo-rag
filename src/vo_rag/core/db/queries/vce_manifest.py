"""Typed query helpers for vce.video_manifest and related vce.* tables."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import asyncpg  # type: ignore[import-untyped]


async def upsert_video_manifest(
    pool: asyncpg.Pool,
    *,
    video_id: str,
    source_platform: str,
    file_sha256: str,
    source_url: str | None = None,
    source_path: str | None = None,
    title: str | None = None,
    channel: str | None = None,
    channel_id: str | None = None,
    duration_seconds: int | None = None,
    uploaded_at: str | None = None,
    license_posture: str = "internal_research",
    coach_persona: str | None = None,
    notes: str | None = None,
) -> bool:
    """Insert a video_manifest row, ignoring conflicts.

    Returns:
        True if the row was newly inserted; False if it already existed.
    """
    uploaded_at_dt: datetime | None = None
    if uploaded_at is not None:
        try:
            uploaded_at_dt = datetime.fromisoformat(uploaded_at)
        except ValueError:
            uploaded_at_dt = None

    result = await pool.execute(
        """
        INSERT INTO vce.video_manifest (
            video_id,
            source_url,
            source_path,
            source_platform,
            file_sha256,
            title,
            channel,
            channel_id,
            duration_seconds,
            uploaded_at,
            license_posture,
            coach_persona,
            notes
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9, $10,
            $11, $12, $13
        )
        ON CONFLICT (video_id) DO NOTHING
        """,
        video_id,
        source_url,
        source_path,
        source_platform,
        file_sha256,
        title,
        channel,
        channel_id,
        duration_seconds,
        uploaded_at_dt,
        license_posture,
        coach_persona,
        notes,
    )
    return bool(result.endswith(" 1"))


async def get_video_manifest(
    pool: asyncpg.Pool,
    video_id: str,
) -> dict[str, Any] | None:
    """Fetch a single video_manifest row as a plain dict, or None."""
    row = await pool.fetchrow(
        "SELECT * FROM vce.video_manifest WHERE video_id = $1",
        video_id,
    )
    if row is None:
        return None
    return dict(row)


async def set_stage_status(
    pool: asyncpg.Pool,
    video_id: str,
    stage: str,
    state: str,
    error: str | None = None,
) -> None:
    """JSONB-patch stage_status for a single stage key.

    The value stored is ``{"state": <state>, "error": <error or null>}``.
    """
    await pool.execute(
        """
        UPDATE vce.video_manifest
        SET stage_status = stage_status || jsonb_build_object(
            $2::text,
            jsonb_build_object('state', $3::text, 'error', $4::text)
        )
        WHERE video_id = $1
        """,
        video_id,
        stage,
        state,
        error,
    )


async def insert_stage_run_log(
    pool: asyncpg.Pool,
    *,
    video_id: str,
    stage_name: str,
    state: str,
    finished_at: bool | datetime = False,
    error: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Append a row to vce.stage_run_log.

    Args:
        finished_at: Pass ``True`` to use NOW() as the finish timestamp, or
            pass an explicit datetime.  Pass ``False`` (default) to leave
            finished_at as NULL (stage still running).
        metrics: Optional JSONB metrics blob.
    """
    finished_at_dt: datetime | None
    if finished_at is True:
        finished_at_dt = datetime.now(tz=UTC)
    elif isinstance(finished_at, datetime):
        finished_at_dt = finished_at
    else:
        finished_at_dt = None

    await pool.execute(
        """
        INSERT INTO vce.stage_run_log (
            video_id,
            stage_name,
            state,
            finished_at,
            error,
            metrics
        ) VALUES ($1, $2, $3, $4, $5, $6)
        """,
        video_id,
        stage_name,
        state,
        finished_at_dt,
        error,
        json.dumps(metrics) if metrics is not None else None,
    )


__all__ = [
    "upsert_video_manifest",
    "get_video_manifest",
    "set_stage_status",
    "insert_stage_run_log",
]
