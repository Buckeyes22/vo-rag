"""JSON contract writer for the VCE export.

The downstream consumer expects a single JSON array of rows under
``vceRowSchema v2`` (contract version ``vce-episode-export@2``). The export
directory also carries per-video ``<video_id>.transcript.json`` files for the
``transcript_substring`` check and a ``superseded.json`` for the cases where
a new id was produced because the interval shifted
(``baseline_preservation``).

Files written by :func:`write_export`:

    <export_dir>/vce-episode-export.json     # the JSON array of rows
    <export_dir>/superseded.json             # old_id -> new_id ledger
    <export_dir>/<video_id>.transcript.json  # per-video full transcript
    <export_dir>/manifest.json               # run metadata (counts, contract version)

Determinism: rows are sorted by ``(video_id, start_ts, end_ts)`` so a re-run
on the same inputs produces a byte-equal ``vce-episode-export.json``.
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from vo_rag.pipelines.vce.types import (
    VCE_CONTRACT_VERSION,
    TranscriptArtifact,
    VceExportRow,
    VceSupersededEntry,
)

VCE_EXPORT_FILENAME = "vce-episode-export.json"
VCE_SUPERSEDED_FILENAME = "superseded.json"
VCE_TRANSCRIPT_SUFFIX = ".transcript.json"
VCE_MANIFEST_FILENAME = "manifest.json"


@dataclasses.dataclass
class VideoTranscriptExport:
    """One video's worth of transcript + rows for the export."""

    video_id: str
    transcript: TranscriptArtifact | None
    rows: list[VceExportRow]
    superseded: list[VceSupersededEntry]


def _sort_key(row: VceExportRow) -> tuple[str, float, float, str]:
    return (row.video_id, round(row.start_ts, 3), round(row.end_ts, 3), row.id)


def write_export(
    *,
    export_dir: Path | str,
    per_video: Iterable[VideoTranscriptExport],
    run_id: str | None = None,
) -> dict[str, object]:
    """Write the export directory atomically.

    Args:
        export_dir: Target directory. Created if missing.
        per_video: One :class:`VideoTranscriptExport` per source video.
        run_id: Optional run identifier to record in ``manifest.json``.

    Returns:
        A dict of written paths: ``{"export": ..., "superseded": ...,
        "manifest": ..., "transcripts": [<path>, ...]}``.
    """
    target = Path(export_dir)
    target.mkdir(parents=True, exist_ok=True)

    all_rows: list[VceExportRow] = []
    all_superseded: list[VceSupersededEntry] = []
    transcript_paths: list[str] = []

    # Group rows by video for per-video transcript files.
    rows_by_video: dict[str, list[VceExportRow]] = defaultdict(list)
    for vtx in per_video:
        all_rows.extend(vtx.rows)
        all_superseded.extend(vtx.superseded)
        rows_by_video[vtx.video_id].extend(vtx.rows)

        if vtx.transcript is not None and vtx.transcript.blob_key is not None:
            transcript_path = target / f"{vtx.video_id}{VCE_TRANSCRIPT_SUFFIX}"
            transcript_payload = {
                "video_id": vtx.video_id,
                "language": vtx.transcript.language,
                "blob_key": vtx.transcript.blob_key,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in vtx.transcript.segments
                ],
            }
            transcript_path.write_text(
                json.dumps(transcript_payload, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            transcript_paths.append(str(transcript_path))

    # Sort rows for byte-deterministic re-runs.
    all_rows.sort(key=_sort_key)
    all_superseded.sort(key=lambda e: (e.video_id, e.old_id))

    # The contract says `superseded.json` lists changed intervals — each entry
    # is one old id with its replacement. Format:
    # [{"old_id": "...", "new_id": "...", "video_id": "...", ...}]
    superseded_payload = [entry.model_dump(mode="json") for entry in all_superseded]
    superseded_path = target / VCE_SUPERSEDED_FILENAME
    superseded_path.write_text(
        json.dumps(superseded_payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    # The main contract file. Pydantic-serialise each row to dict, preserving
    # field order. We use ``model_dump(mode="json")`` so enums become strings
    # and None stays null (the contract's id derivation is case-sensitive).
    rows_payload = [row.model_dump(mode="json") for row in all_rows]
    export_path = target / VCE_EXPORT_FILENAME
    export_path.write_text(
        json.dumps(rows_payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    # Manifest is operator metadata — not consumed by downstream consumer, but useful
    # for debugging failed runs (which video, which stage, what cost).
    manifest_payload = {
        "contract_version": VCE_CONTRACT_VERSION,
        "run_id": run_id,
        "row_count": len(all_rows),
        "superseded_count": len(all_superseded),
        "video_count": len(rows_by_video),
        "video_ids": sorted(rows_by_video.keys()),
    }
    manifest_path = target / VCE_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "export": str(export_path),
        "superseded": str(superseded_path),
        "manifest": str(manifest_path),
        "transcripts": transcript_paths,
    }


def load_export_rows(export_dir: Path | str) -> list[VceExportRow]:
    """Read ``vce-episode-export.json`` back as validated :class:`VceExportRow`s.

    Used by the validator and by tests that want to assert the round-trip.
    """
    target = Path(export_dir)
    payload = json.loads((target / VCE_EXPORT_FILENAME).read_text(encoding="utf-8"))
    return [VceExportRow.model_validate(row) for row in payload]


__all__ = [
    "VCE_EXPORT_FILENAME",
    "VCE_MANIFEST_FILENAME",
    "VCE_SUPERSEDED_FILENAME",
    "VCE_TRANSCRIPT_SUFFIX",
    "VideoTranscriptExport",
    "load_export_rows",
    "write_export",
]
