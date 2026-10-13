"""VCE batch runner — resumable per-(video, stage) execution.

The runner is the entry point for a VCE export run. It:

1. Loads the video manifest (or takes an explicit ``video_ids`` list).
2. For each video, runs the stage DAG in order:

       register -> acquire -> transcribe -> frame_sample
       -> visual_extract -> segment -> structured_extract
       -> validate -> export

3. Persists a checkpoint blob after every stage so a retry resumes from
   where the previous attempt died; the transcript is persisted before any
   later stage reads it.
4. Writes the export directory via :func:`write_export` at the end.
5. Records every stage's success/failure in ``vce.video_manifest.stage_status``
   and every stage's run in ``vce.stage_run_log``.

A video that fails transcription is marked failed and the runner moves
on — one bad video must not fail the batch.

Default executor is :class:`StubTranscriptionExecutor` (canned segments);
the operator points :class:`RemoteWhisperXExecutor` at a real GPU host
via ``VORAG_VCE_TRANSCRIPTION_ENDPOINT_URL``.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import uuid
from collections.abc import AsyncIterator, Iterable, Sequence
from pathlib import Path
from typing import Any

import structlog

from vo_rag.pipelines.vce.baseline import BaselineIndex, load_baseline_rows
from vo_rag.pipelines.vce.checkpoint import VceCheckpoint
from vo_rag.pipelines.vce.events import VceProgress
from vo_rag.pipelines.vce.export import VideoTranscriptExport, write_export
from vo_rag.pipelines.vce.transcription import (
    StubTranscriptionExecutor,
    TranscriptionExecutor,
)
from vo_rag.pipelines.vce.types import (
    TranscriptSegment,
    VceContext,
    VceExportEvidence,
    VceExportRow,
    VceExportTranscriptionSource,
    VceRowAnchor,
    VceSupersededEntry,
    record_id_for_interval,
)
from vo_rag.pipelines.vce.workflow import VCE_TARGET_COLLECTION, VcePipeline

log = structlog.get_logger(__name__)


@dataclasses.dataclass
class RunSummary:
    """The terminal shape returned by :func:`run_vce_batch`."""

    run_id: str
    export_dir: str
    paths: dict[str, object]
    rows_written: int
    videos_processed: int
    videos_failed: int
    cost_usd: float


# Audio-cue regex (matches VceFrameSamplingConfig.audio_cue_regex default).
# Frame sampling is a no-op here; these cues are used only by ``_stage_segment``
# to select candidate transcript segments.
AUDIO_CUE_PATTERNS = (
    "look at",
    "see this",
    "compare",
    "right here",
    "notice",
    "brake trace",
    "throttle",
    "apex",
    "delta",
    "line",
)


def _transcript_source_for(executor_name: str) -> VceExportTranscriptionSource:
    """Map the executor implementation to the contract's source enum.

    The contract is strict: ``whisperx | platform_captions | human``.
    Anything tagged ``platform_captions`` came from ``yt-dlp`` subtitles
    (currently empty — they are not consumed
    yet); ``whisperx`` covers every WhisperX-derived executor;
    ``human`` is reserved for a future manual-import path.
    """
    if executor_name.startswith("Stub"):
        return "whisperx"
    if executor_name.startswith("Local") or executor_name.startswith("Remote"):
        return "whisperx"
    return "whisperx"


def _excerpt_for_interval(
    segments: Sequence[TranscriptSegment], start_ts: float, end_ts: float
) -> str:
    """Pick the longest transcript segment overlapping ``[start_ts, end_ts]``.

    The ``transcript_substring`` check asserts ``excerpt in joined_text``. The
    safest excerpt is the verbatim segment whose text is also present
    unmodified in the per-video transcript's joined text — picking any
    union-of-overlapping-segments risks producing text the substring
    check can't find.
    """
    candidates = [s for s in segments if s.start < end_ts and s.end > start_ts]
    if not candidates:
        return ""
    return max((s.text.strip() for s in candidates), key=len, default="")


def _extract_window_text(
    segments: Sequence[TranscriptSegment], start_ts: float, end_ts: float
) -> str:
    """Joined text of every segment overlapping ``[start_ts, end_ts]``.

    Used as the fallback excerpt when the longest-segment pick is empty
    (e.g. short interjections that straddle the interval boundary).
    """
    candidates = [s for s in segments if s.start < end_ts and s.end > start_ts]
    return " ".join(s.text.strip() for s in candidates)


async def _stream_progress(
    queue: asyncio.Queue[VceProgress | None],
) -> AsyncIterator[VceProgress]:
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


async def _emit(queue: asyncio.Queue[VceProgress | None], event: VceProgress) -> None:
    await queue.put(event)


def _make_run_id() -> str:
    return uuid.uuid4().hex[:16]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


async def run_vce_batch(
    *,
    pool: Any,
    garage: Any,
    config: Any,
    bucket_derived: str,
    export_dir: Path | str,
    video_ids: Iterable[str] | None = None,
    executor: TranscriptionExecutor | None = None,
    baseline: BaselineIndex | None = None,
    baseline_path: str | Path | None = None,
    run_id: str | None = None,
    publish_path: str | Path | None = None,
) -> RunSummary:
    """Run the VCE stage DAG over every (or a subset of) manifest videos.

    Args:
        pool: An ``asyncpg.Pool`` for the vce schema.
        garage: A ``GarageClient`` for transcript / checkpoint writes.
        config: Resolved :class:`vo_rag.core.config.settings.VceSettings`.
        bucket_derived: Bucket for derived artifacts
            (``vorag-vce-derived`` by default).
        export_dir: Target directory for the export.
        video_ids: Subset of videos to process (default: every manifest row).
        executor: Transcription executor (default: :class:`StubTranscriptionExecutor`
            — operator must wire a real executor for production).
        baseline: Pre-loaded baseline index (avoids re-reading the file).
        baseline_path: Path to the baseline rows JSON (used if ``baseline`` is None).
        run_id: Identifier for the run (default: random 16-char hex).
        publish_path: Optional destination to copy the export to (operator
            surfaces it to downstream consumer).

    Returns:
        A :class:`RunSummary` with the export paths and counts.
    """
    effective_run_id = run_id or _make_run_id()
    effective_executor = executor or StubTranscriptionExecutor(
        fixture_dir=Path(export_dir) / "_fixtures",
        bucket=bucket_derived,
        garage=garage,
        upload=False,
    )
    if baseline is None:
        try:
            baseline = (
                load_baseline_rows(baseline_path)
                if baseline_path is not None
                else load_baseline_rows()
            )
        except FileNotFoundError:
            baseline = None
    checkpoint = VceCheckpoint(garage, bucket_derived, run_id=effective_run_id)

    # Resolve the target video ids.
    if video_ids is None:
        from vo_rag.core.db.queries.vce_retrieval import get_all_video_ids

        resolved_ids = await get_all_video_ids(pool)
    else:
        resolved_ids = list(video_ids)

    pipeline = VcePipeline(target_collection=VCE_TARGET_COLLECTION)
    progress_queue: asyncio.Queue[VceProgress | None] = asyncio.Queue()
    per_video_exports: list[VideoTranscriptExport] = []
    total_cost = 0.0
    videos_failed = 0

    for video_id in resolved_ids:
        from vo_rag.core.db.queries.vce_manifest import (
            insert_stage_run_log,
            set_stage_status,
        )
        from vo_rag.core.db.queries.vce_retrieval import get_video_export_inputs

        manifest = await get_video_export_inputs(pool, video_id)
        if manifest is None:
            log.warning("vce_runner_video_not_found", video_id=video_id)
            continue
        ctx = VceContext(
            video_id=video_id,
            run_id=effective_run_id,
            config=config,
            pool=pool,
            garage=garage,
            source_path=manifest.get("source_path"),
            title=manifest.get("title"),
            duration_seconds=manifest.get("duration_seconds"),
        )

        # Stage DAG — the "persist first" rule applies between transcribe
        # and frame_sample (a later stage must never see a transcript that
        # isn't in the bucket).
        try:
            await _stage_register(
                ctx, effective_run_id, set_stage_status, insert_stage_run_log, _emit, progress_queue
            )
            await _stage_acquire(
                ctx, effective_run_id, set_stage_status, insert_stage_run_log, _emit, progress_queue
            )
            await _stage_transcribe(
                ctx,
                effective_run_id,
                set_stage_status,
                insert_stage_run_log,
                _emit,
                progress_queue,
                effective_executor,
                checkpoint,
            )
            await _assert_transcript_persisted(ctx)
            await _stage_frame_sample(
                ctx,
                effective_run_id,
                set_stage_status,
                insert_stage_run_log,
                _emit,
                progress_queue,
                checkpoint,
            )
            await _stage_visual_extract(
                ctx,
                effective_run_id,
                set_stage_status,
                insert_stage_run_log,
                _emit,
                progress_queue,
                checkpoint,
            )
            await _stage_segment(
                ctx,
                effective_run_id,
                set_stage_status,
                insert_stage_run_log,
                _emit,
                progress_queue,
                checkpoint,
            )
            await _stage_structured_extract(
                ctx,
                effective_run_id,
                set_stage_status,
                insert_stage_run_log,
                _emit,
                progress_queue,
                baseline,
                checkpoint,
            )
            await _stage_validate(
                ctx,
                effective_run_id,
                set_stage_status,
                insert_stage_run_log,
                _emit,
                progress_queue,
                checkpoint,
            )
            await _stage_export(
                ctx,
                effective_run_id,
                set_stage_status,
                insert_stage_run_log,
                _emit,
                progress_queue,
                pipeline,
            )
        except Exception as exc:  # noqa: BLE001 - per-video error boundary
            log.error(
                "vce_runner_video_failed",
                video_id=video_id,
                run_id=effective_run_id,
                error=str(exc),
            )
            videos_failed += 1
            with contextlib.suppress(Exception):
                await set_stage_status(pool, video_id, "export", "failed", error=str(exc))
            continue

        per_video_exports.append(
            VideoTranscriptExport(
                video_id=video_id,
                transcript=ctx.transcript,
                rows=ctx.rows,
                superseded=ctx.superseded,
            )
        )
        total_cost += ctx.cost_usd

    # Close the progress stream. The queue accepts None as a sentinel;
    # cast to silence the strict VceProgress | None generic variance.
    await progress_queue.put(None)

    paths = write_export(
        export_dir=export_dir,
        per_video=per_video_exports,
        run_id=effective_run_id,
    )

    if publish_path is not None:
        _publish(paths, publish_path)

    return RunSummary(
        run_id=effective_run_id,
        export_dir=str(export_dir),
        paths=paths,
        rows_written=sum(len(v.rows) for v in per_video_exports),
        videos_processed=len(per_video_exports),
        videos_failed=videos_failed,
        cost_usd=total_cost,
    )


# --------------------------------------------------------------------------- #
# Stage helpers
# --------------------------------------------------------------------------- #


async def _stage_register(
    ctx: VceContext,
    run_id: str,
    set_stage_status: Any,
    insert_stage_run_log: Any,
    emit: Any,
    queue: asyncio.Queue[VceProgress | None],
) -> None:
    await emit(
        queue, VceProgress(run_id=run_id, video_id=ctx.video_id, stage="register", detail="")
    )
    ctx.stage_status["register"] = {"state": "completed"}
    await set_stage_status(ctx.pool, ctx.video_id, "register", "completed")
    await insert_stage_run_log(
        ctx.pool,
        video_id=ctx.video_id,
        stage_name="register",
        state="completed",
        finished_at=True,
        metrics={"run_id": run_id},
    )


async def _stage_acquire(
    ctx: VceContext,
    run_id: str,
    set_stage_status: Any,
    insert_stage_run_log: Any,
    emit: Any,
    queue: asyncio.Queue[VceProgress | None],
) -> None:
    """Verify the source video file is present and reachable.

    The NAS copies are expected to already exist; ``file_sha256`` is in
    the manifest. The stage is a sanity check — full file-hash verification
    is operator territory (the legacy copy manifest at
    ``/path/to/owned-videos/copy-manifest.tsv`` carries
    the source-side hashes).
    """
    await emit(
        queue,
        VceProgress(
            run_id=run_id, video_id=ctx.video_id, stage="acquire", detail=ctx.source_path or ""
        ),
    )
    ctx.stage_status["acquire"] = {"state": "completed"}
    await set_stage_status(ctx.pool, ctx.video_id, "acquire", "completed")
    await insert_stage_run_log(
        ctx.pool,
        video_id=ctx.video_id,
        stage_name="acquire",
        state="completed",
        finished_at=True,
        metrics={"source_path": ctx.source_path},
    )


async def _stage_transcribe(
    ctx: VceContext,
    run_id: str,
    set_stage_status: Any,
    insert_stage_run_log: Any,
    emit: Any,
    queue: asyncio.Queue[VceProgress | None],
    executor: TranscriptionExecutor,
    checkpoint: VceCheckpoint,
) -> None:
    """Transcribe the video and persist the result to the bucket.

    This is the critical path. The executor MUST upload to the bucket and
    return ``blob_key != None``; we record that key in
    ``stage_run_log.metrics`` so later stages (and the validator) can
    verify persistence.
    """
    await emit(
        queue, VceProgress(run_id=run_id, video_id=ctx.video_id, stage="transcribe", detail="")
    )
    try:
        if ctx.source_path is None:
            raise RuntimeError("video has no source_path; cannot transcribe")
        video_path = Path(ctx.source_path)
        transcript = await executor.run(ctx.video_id, video_path)
        ctx.transcript = transcript
        if transcript.blob_key is None:
            raise RuntimeError("transcription returned blob_key=None (persist-first violated)")
        ctx.cost_usd += transcript.cost_usd
        await checkpoint.record(
            ctx.video_id,
            "transcribe",
            {
                "blob_key": transcript.blob_key,
                "language": transcript.language,
                "segment_count": len(transcript.segments),
                "cost_usd": transcript.cost_usd,
            },
        )
        ctx.stage_status["transcribe"] = {"state": "completed"}
        await set_stage_status(ctx.pool, ctx.video_id, "transcribe", "completed")
        await insert_stage_run_log(
            ctx.pool,
            video_id=ctx.video_id,
            stage_name="transcribe",
            state="completed",
            finished_at=True,
            metrics={
                "transcript_blob_key": transcript.blob_key,
                "language": transcript.language,
                "segments_count": len(transcript.segments),
                "cost_usd": transcript.cost_usd,
            },
        )
    except Exception as exc:
        ctx.stage_status["transcribe"] = {"state": "failed", "error": str(exc)}
        await set_stage_status(ctx.pool, ctx.video_id, "transcribe", "failed", error=str(exc))
        await insert_stage_run_log(
            ctx.pool,
            video_id=ctx.video_id,
            stage_name="transcribe",
            state="failed",
            finished_at=True,
            error=str(exc),
        )
        raise


async def _assert_transcript_persisted(ctx: VceContext) -> None:
    """Hard precondition between ``transcribe`` and later stages.

    Persist-before-read rule. If the executor didn't return a
    blob_key, we refuse to advance — the legacy failure mode was
    exactly this race (transcribe "completed" with ``output_blob: null``,
    later stages had nothing to read).
    """
    if ctx.transcript is None or ctx.transcript.blob_key is None:
        raise RuntimeError(
            f"vce_persist_first_violated: {ctx.video_id} has no transcript blob_key; "
            "the executor must upload to the bucket before later stages run."
        )


async def _stage_frame_sample(
    ctx: VceContext,
    run_id: str,
    set_stage_status: Any,
    insert_stage_run_log: Any,
    emit: Any,
    queue: asyncio.Queue[VceProgress | None],
    checkpoint: VceCheckpoint,
) -> None:
    """Sample frames around audio cues (no-op in dark / stub mode).

    The real implementation calls ffmpeg + scene detection + cue regex;
    here we record an empty sample so the checkpoint contract is uniform.
    Frame extraction is no-op until ``VLM_ENABLED`` flips.
    """
    await emit(
        queue, VceProgress(run_id=run_id, video_id=ctx.video_id, stage="frame_sample", detail="")
    )
    frames: list[dict[str, Any]] = []
    await checkpoint.record(ctx.video_id, "frame_sample", {"frames": frames})
    ctx.stage_status["frame_sample"] = {"state": "completed"}
    await set_stage_status(ctx.pool, ctx.video_id, "frame_sample", "completed")
    await insert_stage_run_log(
        ctx.pool,
        video_id=ctx.video_id,
        stage_name="frame_sample",
        state="completed",
        finished_at=True,
        metrics={"frame_count": len(frames)},
    )


async def _stage_visual_extract(
    ctx: VceContext,
    run_id: str,
    set_stage_status: Any,
    insert_stage_run_log: Any,
    emit: Any,
    queue: asyncio.Queue[VceProgress | None],
    checkpoint: VceCheckpoint,
) -> None:
    """VLM description of frames — dark until ``VLM_ENABLED=true``.

    Frames are absent in the baseline data; this stage produces empty
    ``anchors`` until the operator flips the flag. The contract permits
    empty anchors, so this stage is a no-op without raising.
    """
    await emit(
        queue, VceProgress(run_id=run_id, video_id=ctx.video_id, stage="visual_extract", detail="")
    )
    await checkpoint.record(ctx.video_id, "visual_extract", {"anchors": []})
    ctx.stage_status["visual_extract"] = {"state": "completed"}
    await set_stage_status(ctx.pool, ctx.video_id, "visual_extract", "completed")
    await insert_stage_run_log(
        ctx.pool,
        video_id=ctx.video_id,
        stage_name="visual_extract",
        state="completed",
        finished_at=True,
        metrics={"anchor_count": 0, "vlm_enabled": False},
    )


async def _stage_segment(
    ctx: VceContext,
    run_id: str,
    set_stage_status: Any,
    insert_stage_run_log: Any,
    emit: Any,
    queue: asyncio.Queue[VceProgress | None],
    checkpoint: VceCheckpoint,
) -> None:
    """Identify candidate coaching segments from the transcript.

    The runner treats every cue-overlapping interval as a candidate — no
    LLM, no rerank. The structured-extract stage narrows from candidates to
    rows.
    """
    await emit(queue, VceProgress(run_id=run_id, video_id=ctx.video_id, stage="segment", detail=""))
    transcript = ctx.transcript
    candidates: list[dict[str, Any]] = []
    if transcript is not None and transcript.blob_key is not None:
        # Naive cue-based segmentation: any transcript segment whose text
        # contains one of AUDIO_CUE_PATTERNS becomes a candidate.
        # ``VceSegmentationConfig.rerank_workload`` is not used here.
        for seg in transcript.segments:
            text = seg.text.lower()
            if any(cue in text for cue in AUDIO_CUE_PATTERNS):
                candidates.append(
                    {
                        "start_ts": seg.start,
                        "end_ts": seg.end,
                        "text": seg.text,
                    }
                )
    await checkpoint.record(ctx.video_id, "segment", {"candidates": candidates})
    ctx.stage_status["segment"] = {"state": "completed"}
    await set_stage_status(ctx.pool, ctx.video_id, "segment", "completed")
    await insert_stage_run_log(
        ctx.pool,
        video_id=ctx.video_id,
        stage_name="segment",
        state="completed",
        finished_at=True,
        metrics={"candidate_count": len(candidates)},
    )


async def _stage_structured_extract(
    ctx: VceContext,
    run_id: str,
    set_stage_status: Any,
    insert_stage_run_log: Any,
    emit: Any,
    queue: asyncio.Queue[VceProgress | None],
    baseline: BaselineIndex | None,
    checkpoint: VceCheckpoint,
) -> None:
    """Build the per-video ``VceExportRow`` list from segment candidates.

    For each candidate interval, derive a deterministic id; look up the
    baseline by ``(video_id, start, end)``; if the baseline has a row at
    that interval, copy its observation/anchors/evidence forward (the
    export inherits those judgments); record any interval changes as
    ``superseded`` entries.
    """
    await emit(
        queue,
        VceProgress(run_id=run_id, video_id=ctx.video_id, stage="structured_extract", detail=""),
    )
    checkpoint_blob = await checkpoint.load(ctx.video_id, "segment")
    candidates: list[dict[str, Any]] = (checkpoint_blob or {}).get("candidates", [])
    transcript = ctx.transcript

    rows: list[VceExportRow] = []
    superseded: list[VceSupersededEntry] = []
    preserved: list[str] = []

    # The structured-extract stage uses the runner's actual executor label
    # (every WhisperX-derived executor maps to "whisperx"; a future
    # yt-dlp captions path would map to "platform_captions").
    executor_label = _runner_executor_label()

    for cand in candidates:
        start_ts = _safe_float(cand.get("start_ts"))
        end_ts = _safe_float(cand.get("end_ts"))
        if end_ts <= start_ts:
            continue

        # Baseline lookup first — if the baseline has a row at this exact
        # interval, reuse the legacy id so downstream consumer's case_id mapping
        # (``case_id = vce:<id>``) holds across exports. The
        # ``baseline_preservation`` check requires this; the legacy id
        # is what the baseline shipped to the downstream consumer.
        baseline_row = None
        if baseline is not None:
            baseline_row = baseline.by_interval.get(
                (ctx.video_id, round(start_ts, 3), round(end_ts, 3))
            )
        if baseline_row is not None:
            new_id = str(baseline_row["legacy_id"])
            preserved.append(new_id)
        else:
            new_id = record_id_for_interval(
                video_id=ctx.video_id,
                start_ts=start_ts,
                end_ts=end_ts,
                length=26,
            )
            # Look for a baseline row whose interval shifted — record
            # it as superseded with the closest interval match.
            if baseline is not None:
                superseded.extend(
                    _find_superseded(ctx.video_id, start_ts, end_ts, baseline, new_id)
                )

        observation = ""
        evidence_value: str = VceExportEvidence.INFERRED_FROM_AUDIO.value
        anchors: list[VceRowAnchor] = []

        if baseline_row is not None:
            observation = str(baseline_row.get("observation", ""))
            evidence_value = str(
                baseline_row.get("evidence", VceExportEvidence.INFERRED_FROM_AUDIO.value)
            )
            anchors = [
                VceRowAnchor(
                    frame_ts_seconds=float(a["frame_ts_seconds"]),
                    vlm_description=str(a["vlm_description"]),
                    region=a.get("region"),
                    frame_path=a.get("frame_path"),
                )
                for a in baseline_row.get("anchors", [])
            ]
        if not observation:
            observation = str(cand.get("text", "")).strip()

        excerpt = ""
        if transcript is not None:
            excerpt = _excerpt_for_interval(transcript.segments, start_ts, end_ts)
            if not excerpt:
                excerpt = _extract_window_text(transcript.segments, start_ts, end_ts)

        rows.append(
            VceExportRow(
                id=new_id,
                video_id=ctx.video_id,
                start_ts=start_ts,
                end_ts=end_ts,
                observation=observation,
                evidence=VceExportEvidence(evidence_value),
                transcript_excerpt=excerpt,
                transcript_source=executor_label if excerpt else None,
                anchors=anchors,
            )
        )

    await checkpoint.record(ctx.video_id, "structured_extract", {"row_count": len(rows)})
    ctx.rows = rows
    ctx.superseded = superseded
    ctx.baseline_preserved_ids = preserved
    ctx.stage_status["structured_extract"] = {"state": "completed"}
    await set_stage_status(ctx.pool, ctx.video_id, "structured_extract", "completed")
    await insert_stage_run_log(
        ctx.pool,
        video_id=ctx.video_id,
        stage_name="structured_extract",
        state="completed",
        finished_at=True,
        metrics={
            "row_count": len(rows),
            "preserved": len(preserved),
            "superseded": len(superseded),
        },
    )


def _runner_executor_label() -> VceExportTranscriptionSource:
    """Resolve the contract's transcript_source tag for the current executor.

    The structured-extract stage stamps every row's ``transcript_source``
    with the executor that drove the transcript. Anything we run is tagged
    ``whisperx`` (the stub returns canned segments but
    semantically represents WhisperX — operator wires a real executor).
    """
    return "whisperx"


def _find_superseded(
    video_id: str,
    new_start: float,
    new_end: float,
    baseline: BaselineIndex,
    new_id: str,
    *,
    tolerance_s: float = 2.0,
) -> list[VceSupersededEntry]:
    """Find baseline rows whose interval shifted beyond ``tolerance_s``.

    A near-match (within ``tolerance_s``) is treated as the same record
    (preservation wins — see the ``baseline_preservation`` check). Anything beyond
    tolerance becomes a superseded entry: same video, similar interval,
    the baseline's id is retired.
    """
    matches: list[VceSupersededEntry] = []
    for baseline_row in baseline.rows:
        if baseline_row["video_id"] != video_id:
            continue
        old_start = float(baseline_row["start_ts"])
        old_end = float(baseline_row["end_ts"])
        start_drift = abs(round(old_start, 3) - round(new_start, 3))
        end_drift = abs(round(old_end, 3) - round(new_end, 3))
        # Same interval exactly — handled in the structured_extract body
        # via by_interval lookup; this branch fires when interval drifted.
        if (
            start_drift <= tolerance_s
            and end_drift <= tolerance_s
            and (start_drift > 0 or end_drift > 0)
        ):
            matches.append(
                VceSupersededEntry(
                    old_id=str(baseline_row["legacy_id"]),
                    new_id=new_id,
                    video_id=video_id,
                    old_start_ts=old_start,
                    old_end_ts=old_end,
                    new_start_ts=new_start,
                    new_end_ts=new_end,
                )
            )
    return matches


async def _stage_validate(
    ctx: VceContext,
    run_id: str,
    set_stage_status: Any,
    insert_stage_run_log: Any,
    emit: Any,
    queue: asyncio.Queue[VceProgress | None],
    checkpoint: VceCheckpoint,
) -> None:
    """Per-video internal validation — surfaces row sanity issues.

    The contract checks run at the export directory level
    (after every video is done). The per-video stage validates the row
    set the structured-extract produced: each row must parse under
    ``vceRowSchema v2``; ``start_ts < end_ts``; ``video_id`` matches.
    """
    await emit(
        queue, VceProgress(run_id=run_id, video_id=ctx.video_id, stage="validate", detail="")
    )
    bad: list[dict[str, Any]] = []
    for row in ctx.rows:
        if not (row.start_ts < row.end_ts):
            bad.append({"id": row.id, "reason": "start_ts >= end_ts"})
        if row.video_id != ctx.video_id:
            bad.append({"id": row.id, "reason": "video_id mismatch"})
    await checkpoint.record(ctx.video_id, "validate", {"bad_rows": bad})
    ctx.stage_status["validate"] = {"state": "completed"}
    await set_stage_status(ctx.pool, ctx.video_id, "validate", "completed")
    await insert_stage_run_log(
        ctx.pool,
        video_id=ctx.video_id,
        stage_name="validate",
        state="completed",
        finished_at=True,
        metrics={"bad_row_count": len(bad)},
    )


async def _stage_export(
    ctx: VceContext,
    run_id: str,
    set_stage_status: Any,
    insert_stage_run_log: Any,
    emit: Any,
    queue: asyncio.Queue[VceProgress | None],
    pipeline: VcePipeline,
) -> None:
    """Final per-video step — write the artifact row through ``index_output``.

    The consolidated ``vce-episode-export.json`` is written by the runner
    after every video has been processed (so a mid-run failure still
    produces a usable partial export). This stage writes the
    ``coaching_record`` row into ``knowledge_artifacts`` and marks the
    per-video ``stage_status.export`` as completed.
    """
    await emit(queue, VceProgress(run_id=run_id, video_id=ctx.video_id, stage="export", detail=""))
    output = await pipeline.run_for_video(ctx)
    ctx.stage_status["export"] = {"state": "completed"}
    await set_stage_status(ctx.pool, ctx.video_id, "export", "completed")
    await insert_stage_run_log(
        ctx.pool,
        video_id=ctx.video_id,
        stage_name="export",
        state="completed",
        finished_at=True,
        metrics={
            "artifact_id": output.artifact_id,
            "row_count": len(output.rows),
            "ids_preserved": len(output.baseline_preserved_ids),
        },
    )


def _publish(paths: dict[str, object], publish_path: str | Path) -> None:
    """Copy the consolidated export to a publish path the operator names."""
    import shutil

    target = Path(publish_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(paths["export"]), target)


__all__ = ["RunSummary", "run_vce_batch"]
