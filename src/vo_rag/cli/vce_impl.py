"""CLI-side implementation helpers for ``vo-rag vce``.

The Click surface (``cli/vce.py``) defers heavy work to functions in this
module so the Click tree imports stay cheap. Each function returns a
plain dict that the CLI prints as a table or JSON.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import structlog

from vo_rag.core.config import get_settings
from vo_rag.core.db.postgres import close_pool, get_pool
from vo_rag.core.storage.garage import GarageClient, garage_from_env

log = structlog.get_logger(__name__)


async def run_vce_cli(
    *,
    export_dir: Path,
    video_ids: list[str] | None,
    executor_name: str,
    baseline_path: Path | None,
    publish_path: Path | None,
    run_id: str | None,
) -> dict[str, Any]:
    """Drive :func:`vo_rag.pipelines.vce.runner.run_vce_batch` from the CLI.

    Builds the small set of seams the runner needs (Postgres pool,
    Garage client, VceSettings, transcription executor) and renders the
    resulting :class:`RunSummary` as a plain dict for the CLI.
    """
    settings = get_settings()
    vce_settings = settings.vce
    bucket_derived = vce_settings.blob.bucket_derived

    pool = await get_pool(settings.database.url)
    garage = garage_from_env()
    executor = _build_executor(executor_name, bucket=bucket_derived, garage=garage)

    try:
        from vo_rag.pipelines.vce.runner import run_vce_batch

        summary = await run_vce_batch(
            pool=pool,
            garage=garage,
            config=vce_settings,
            bucket_derived=bucket_derived,
            export_dir=export_dir,
            video_ids=video_ids,
            executor=executor,
            baseline_path=baseline_path,
            publish_path=publish_path,
            run_id=run_id,
        )
    finally:
        await close_pool()

    result: dict[str, Any] = {
        "run_id": summary.run_id,
        "videos_processed": summary.videos_processed,
        "videos_failed": summary.videos_failed,
        "rows_written": summary.rows_written,
        "cost_usd": summary.cost_usd,
        "paths": summary.paths,
    }
    if publish_path is not None:
        result["publish_path"] = str(publish_path)
    return result


async def status_vce_cli(*, video_id: str | None) -> dict[str, Any]:
    """Operator smoke status payload.

    Counts ``vce.video_manifest`` rows and ``stage_run_log`` state values.
    Lightweight — does not invoke the runner.
    """
    from vo_rag.core.db.queries.vce_export import count_coaching_records
    from vo_rag.core.db.queries.vce_retrieval import get_videos_with_manifest

    pool = await get_pool(get_settings().database.url)
    try:
        manifest_rows = await get_videos_with_manifest(pool)
        if video_id is not None:
            manifest_rows = [row for row in manifest_rows if row["video_id"] == video_id]
        state_counter: Counter[str] = Counter()
        for row in manifest_rows:
            stage_status = row.get("stage_status") or {}
            if not isinstance(stage_status, dict):
                continue
            for _stage_name, payload in stage_status.items():
                if isinstance(payload, dict):
                    state_counter[payload.get("state", "unknown")] += 1
                else:
                    state_counter[str(payload)] += 1
        coaching_records = await count_coaching_records(pool)
    finally:
        await close_pool()

    return {
        "video_count": len(manifest_rows),
        "coaching_records": coaching_records,
        "stage_states": dict(state_counter),
    }


def _build_executor(executor_name: str, *, bucket: str, garage: GarageClient) -> Any:
    """Resolve the CLI executor flag to a concrete executor instance.

    The CLI default is ``stub`` (no transcription cost, deterministic);
    production runs require ``remote`` with
    ``VORAG_VCE_TRANSCRIPTION_ENDPOINT_URL`` pointing at a GPU host.
    """
    from vo_rag.pipelines.vce.transcription import (
        LocalWhisperXExecutor,
        OpenAIAudioTranscriptionExecutor,
        RemoteWhisperXExecutor,
        StubTranscriptionExecutor,
    )

    if executor_name == "stub":
        fixture_dir = Path(
            os.environ.get(
                "VORAG_VCE_STUB_FIXTURE_DIR",
                "tests/pipelines/vce/fixtures/transcripts",
            )
        ).expanduser()
        return StubTranscriptionExecutor(
            fixture_dir=fixture_dir,
            bucket=bucket,
            garage=garage,
            upload=False,
        )
    if executor_name == "local":
        return LocalWhisperXExecutor(bucket=bucket, garage=garage)
    if executor_name == "remote":
        return RemoteWhisperXExecutor(bucket=bucket, garage=garage)
    if executor_name == "speaches":
        return OpenAIAudioTranscriptionExecutor(bucket=bucket, garage=garage)
    raise ValueError(f"unknown vce executor: {executor_name}")


__all__ = ["run_vce_cli", "status_vce_cli"]
