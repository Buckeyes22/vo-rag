"""Shared types for the VCE pipeline.

Pydantic-first: rows are validated against `vceRowSchema v2` before they leave
the pipeline. :class:`VceContext` is the mutable state threaded through the
stage DAG; :class:`VcePipelineOutput` is the per-video terminal result that
:class:`vo_rag.pipelines.base.PipelineBase.index_output` writes through.

Contract version is pinned here — bumping it without a downstream consumer-side
agreement is a breaking change. See `tests/pipelines/vce/_contract.py` for the
Pydantic mirror of the contract.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Pinned contract version — must match downstream consumer's `vce-episode-export@2`.
# Bumping requires a coordinated change on both sides.
VCE_CONTRACT_VERSION: str = "vce-episode-export@2"


class VceExportEvidence(StrEnum):
    """How the coach's words were captured for the row's interval."""

    INFERRED_FROM_AUDIO = "inferred_from_audio"
    EXPLICIT_NUMERIC = "explicit_numeric"
    EXPLICIT_QUALITATIVE = "explicit_qualitative"


VceExportTranscriptionSource = Literal["whisperx", "platform_captions", "human"]


class VceRowAnchor(BaseModel):
    """One frame anchor for a VCE row.

    The contract permits ``anchors: []`` (an empty array is valid); the
    ``frame_path`` may be ``null`` when the frame was described but not
    stored (e.g. VLM ran without persisting the JPEG).
    """

    model_config = ConfigDict(extra="forbid")

    frame_ts_seconds: float = Field(ge=0.0)
    vlm_description: str
    region: str | None = None
    frame_path: str | None = None


class VceExportRow(BaseModel):
    """One row of the downstream consumer `vceRowSchema v2` contract.

    The ``id`` field is stable per ``(video_id, start_ts, end_ts)`` — re-running
    the export on the same video yields the same id. The deterministic derivation
    is :func:`vo_rag.pipelines.vce.idempotency.record_id_for_interval` (defined
    alongside this module) and is the **only** sanctioned way to mint ids.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=20, max_length=32)
    video_id: str
    start_ts: float = Field(ge=0.0)
    end_ts: float = Field(gt=0.0)
    observation: str
    evidence: VceExportEvidence
    transcript_excerpt: str = ""
    transcript_source: VceExportTranscriptionSource | None = None
    anchors: list[VceRowAnchor] = Field(default_factory=list)


class VceSupersededEntry(BaseModel):
    """One entry in ``superseded.json`` — an old id whose interval changed."""

    model_config = ConfigDict(extra="forbid")

    old_id: str
    new_id: str
    video_id: str
    old_start_ts: float
    old_end_ts: float
    new_start_ts: float
    new_end_ts: float


@dataclasses.dataclass
class TranscriptSegment:
    """One segment from a video's transcript.

    ``text`` is verbatim coach speech (or platform-caption text); the contract's
    substring check joins segments and asserts ``excerpt in joined_text``.
    """

    start: float
    end: float
    text: str


@dataclasses.dataclass
class TranscriptArtifact:
    """A transcription result, with the bucket upload recorded.

    The runner enforces "persist first": the executor MUST upload to
    ``vorag-vce-derived/transcripts/<video_id>.transcript.json`` and stamp
    ``blob_key`` here **before** any later stage is allowed to read the
    transcript. A transcription that returns segments but yields
    ``blob_key=None`` is treated as a stage failure (persist before
    any later stage reads — the legacy 25/36 runs had
    ``output_blob: null`` because persistence was skipped).
    """

    segments: list[TranscriptSegment]
    language: str | None
    blob_key: str | None
    cost_usd: float = 0.0


@dataclasses.dataclass
class VceContext:
    """Mutable state threaded through the VCE stage DAG (per video)."""

    video_id: str
    run_id: str
    config: Any  # vo_rag.core.config.settings.VceSettings
    pool: Any  # asyncpg.Pool
    garage: Any  # vo_rag.core.storage.garage.GarageClient
    source_path: str | None = None
    title: str | None = None
    duration_seconds: int | None = None
    transcript: TranscriptArtifact | None = None
    rows: list[VceExportRow] = dataclasses.field(default_factory=list)
    superseded: list[VceSupersededEntry] = dataclasses.field(default_factory=list)
    baseline_preserved_ids: list[str] = dataclasses.field(default_factory=list)
    cost_usd: float = 0.0
    # Status mirrors vce.video_manifest.stage_status — written via set_stage_status.
    # Keys are VceStage values; values are dict[str, str] for state/error.
    stage_status: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class VcePipelineOutput:
    """The per-video terminal result of a VCE pipeline run.

    This is the shape returned by :meth:`VcePipeline.run_for_video` and
    consumed by :meth:`vo_rag.pipelines.base.PipelineBase.index_output`.
    The ``content`` dict is the contract surface — every key here must be
    a stable, documented field on the per-video knowledge artifact.
    """

    video_id: str
    envelope_id: str
    artifact_id: str
    title: str
    export_path: str
    rows_json_path: str
    transcripts_dir: str
    rows: list[VceExportRow]
    superseded: list[VceSupersededEntry]
    baseline_preserved_ids: list[str]
    stage_status: dict[str, dict[str, str]]
    cost_usd: float

    def to_pipeline_output(self) -> Any:
        """Convert to the base ``PipelineOutput`` shape for index_output."""
        # Lazy import to avoid cycles during module import.
        from vo_rag.core.models.artifact import ArtifactType, LicenseClass
        from vo_rag.core.models.envelope import Priority
        from vo_rag.pipelines.base import PipelineOutput

        return PipelineOutput(
            envelope_id=self.envelope_id,
            pipeline="vce",
            artifact_type=ArtifactType.COACHING_RECORD.value,
            title=self.title or self.video_id,
            content={
                "video_id": self.video_id,
                "export_path": self.export_path,
                "rows_json": self.rows_json_path,
                "transcripts_dir": self.transcripts_dir,
                "row_count": len(self.rows),
                "ids_preserved": self.baseline_preserved_ids,
                "superseded_count": len(self.superseded),
                "stage_status": self.stage_status,
                "cost_usd": self.cost_usd,
                "contract_version": VCE_CONTRACT_VERSION,
            },
            tags=["vce", "video", self.video_id],
            priority=Priority.MEDIUM.value,
            license_class=LicenseClass.PENDING,
        )


def record_id_for_interval(
    *,
    video_id: str,
    start_ts: float,
    end_ts: float,
    length: int = 26,
) -> str:
    """The deterministic record id for a (video, interval) pair.

    Re-running the export on the same video yields the same id; reproducible
    exports rely on this. The format matches the legacy
    baseline's 26-char uppercase ids (see
    ``./examples/vce-baseline.json``)
    so existing case identities are preserved.
    """
    from vo_rag.core.idempotency import deterministic_hex

    if not (1 <= length <= 64):
        raise ValueError("length must be 1..64")
    # Round to millisecond precision — the contract treats start_ts / end_ts
    # as numbers, but our id derivation needs a canonical representation.
    rounded_start = f"{start_ts:.3f}"
    rounded_end = f"{end_ts:.3f}"
    return deterministic_hex(
        f"vce:{video_id}:{rounded_start}:{rounded_end}",
        length=length,
    ).upper()


__all__ = [
    "VCE_CONTRACT_VERSION",
    "TranscriptArtifact",
    "TranscriptSegment",
    "VceContext",
    "VceExportEvidence",
    "VceExportRow",
    "VceExportTranscriptionSource",
    "VcePipelineOutput",
    "VceRowAnchor",
    "VceSupersededEntry",
    "record_id_for_interval",
]
