"""Write helpers for the vce.coaching_records table (the per-video terminal artifact).

The legacy ``record_signature_points`` rows are orphaned (their Qdrant
collection is not part of this schema) and nothing regenerates them.
Instead, each video's terminal :class:`VcePipelineOutput` lands as a
``coaching_record`` row in ``knowledge_artifacts`` via
:class:`vo_rag.pipelines.base.PipelineBase.index_output` — the path the
runner already takes.

This module is the small bridge: it (a) wraps the artifact write with the
vce-specific provenance metadata and (b) provides a count helper for the
operator's smoke CLI (``vo-rag vce status``).

The legacy ``vce.record_signature_points`` table is intentionally NOT
touched here; whether to keep or drop it is left to the operator.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg  # type: ignore[import-untyped]


async def count_coaching_records(pool: asyncpg.Pool) -> int:
    """Count the ``coaching_record`` rows in ``knowledge_artifacts``.

    Used by ``vo-rag vce status`` and by tests that want to verify the
    per-video artifact write succeeded.
    """
    return int(
        await pool.fetchval(
            """
            SELECT COUNT(*) FROM knowledge_artifacts
            WHERE pipeline = 'vce' AND artifact_type = 'coaching_record'
            """,
        )
    )


async def upsert_coaching_record(
    pool: asyncpg.Pool,
    *,
    artifact_id: str,
    envelope_id: str,
    video_id: str,
    title: str,
    content: dict[str, Any],
    source_provenance: list[dict[str, Any]] | None = None,
    tags: list[str] | None = None,
    license_class: str = "pending",
) -> None:
    """Insert or refresh a coaching_record row for one VCE video.

    This is the vce-flavored write-path the runner uses when it does NOT
    want to go through ``PipelineBase.index_output`` (e.g. for batch
    re-runs that bypass envelope lifecycle). Most calls should use
    ``index_output`` directly.

    Args:
        pool: asyncpg pool.
        artifact_id: The 26-char deterministic hex id.
        envelope_id: Stable envelope id (typically ``urn:uuid:<run>:<video>``).
        video_id: The vce.video_manifest.video_id.
        title: Display title for the artifact row.
        content: Free-form JSON (rows, transcript refs, etc.).
        source_provenance: Per-row provenance entries (downstream consumer display).
        tags: Optional tag list (defaults to ``["vce", video_id]``).
        license_class: License class string (``pending`` is the safe default).
    """
    payload_tags = tags or ["vce", video_id]
    provenance_json = json.dumps(source_provenance or [])
    content_json = json.dumps(content)
    await pool.execute(
        """
        INSERT INTO knowledge_artifacts (
            id, envelope_id, pipeline, artifact_type, title,
            content, tags, license_class, source_provenance
        ) VALUES (
            $1, $2, 'vce', 'coaching_record', $3,
            $4::jsonb, $5, $6::license_class, $7::jsonb
        )
        ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            tags = EXCLUDED.tags,
            source_provenance = EXCLUDED.source_provenance,
            updated_at = NOW()
        """,
        artifact_id,
        envelope_id,
        title,
        content_json,
        payload_tags,
        license_class,
        provenance_json,
    )


__all__ = ["count_coaching_records", "upsert_coaching_record"]
