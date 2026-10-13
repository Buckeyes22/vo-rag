"""Read helpers for the vce schema used by the VCE runner.

The schema is in `core/db/migrations/023_vce_initial.sql`. The helpers
here are intentionally narrow — only what the runner needs:

- :func:`get_all_video_ids` — every ``video_manifest.video_id`` row.
- :func:`get_video_export_inputs` — the per-video fields the runner needs
  to drive a stage DAG (source path, title, duration, sha256).

Write helpers (``upsert_video_manifest``, ``set_stage_status``,
``insert_stage_run_log``) live in :mod:`vo_rag.core.db.queries.vce_manifest`
(not defined here).
"""

from __future__ import annotations

from typing import Any

import asyncpg  # type: ignore[import-untyped]


async def get_all_video_ids(pool: asyncpg.Pool) -> list[str]:
    """Return every ``video_manifest.video_id`` row, ordered by upload time.

    Ordered so re-runs are deterministic.
    """
    rows = await pool.fetch(
        """
        SELECT video_id FROM vce.video_manifest
        ORDER BY uploaded_at NULLS LAST, video_id
        """,
    )
    return [row["video_id"] for row in rows]


async def get_video_export_inputs(pool: asyncpg.Pool, video_id: str) -> dict[str, Any] | None:
    """Return the per-video fields the runner reads.

    The runner does not need the full row — only what the stage DAG asks
    for (source path, title, duration). Returns ``None`` when the video
    is not in the manifest; callers should log and skip.
    """
    row = await pool.fetchrow(
        """
        SELECT video_id, source_path, title, duration_seconds, file_sha256,
               license_posture, stage_status
        FROM vce.video_manifest
        WHERE video_id = $1
        """,
        video_id,
    )
    if row is None:
        return None
    return dict(row)


async def get_videos_with_manifest(pool: asyncpg.Pool) -> list[dict[str, Any]]:
    """Return every manifest row, including stage status, as plain dicts.

    Used by ``vo-rag vce status`` and by tests that want a snapshot of
    the current schema state.
    """
    rows = await pool.fetch(
        "SELECT * FROM vce.video_manifest ORDER BY uploaded_at NULLS LAST, video_id",
    )
    return [dict(row) for row in rows]


async def count_stage_state(pool: asyncpg.Pool, state: str = "completed") -> int:
    """Count ``stage_run_log`` rows in a given state (operator smoke)."""
    return int(
        await pool.fetchval(
            "SELECT COUNT(*) FROM vce.stage_run_log WHERE state = $1",
            state,
        )
    )


__all__ = [
    "count_stage_state",
    "get_all_video_ids",
    "get_video_export_inputs",
    "get_videos_with_manifest",
]
