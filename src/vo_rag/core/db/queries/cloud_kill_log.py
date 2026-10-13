"""cloud_kill_log query helper.

The report argument is a ``dict[str, Any]`` rather than a typed model.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg  # type: ignore[import-untyped]


async def persist(pool: asyncpg.Pool, report: dict[str, Any]) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO cloud_kill_log (
                 triggered_at,
                 reason,
                 tailscale_acl_updated,
                 devices_removed,
                 r2_credentials_rotated,
                 skypilot_clusters_destroyed,
                 total_duration_ms,
                 errors
               )
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)""",
            report["triggered_at"],
            report["reason"],
            report["tailscale_acl_updated"],
            report["devices_removed"],
            report["r2_credentials_rotated"],
            report["skypilot_clusters_destroyed"],
            report["total_duration_ms"],
            json.dumps(report["errors"]),
        )
