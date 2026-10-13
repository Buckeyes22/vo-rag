"""Video coaching extraction with caller-supplied videos and export contracts."""

from __future__ import annotations

from vo_rag.pipelines.vce.baseline import BaselineIndex, load_baseline_rows
from vo_rag.pipelines.vce.checkpoint import CHECKPOINT_PREFIX, VceCheckpoint
from vo_rag.pipelines.vce.events import VceProgress, VceStage
from vo_rag.pipelines.vce.export import (
    VCE_EXPORT_FILENAME,
    VCE_SUPERSEDED_FILENAME,
    write_export,
)
from vo_rag.pipelines.vce.runner import RunSummary, run_vce_batch
from vo_rag.pipelines.vce.transcription import (
    LocalWhisperXExecutor,
    RemoteWhisperXExecutor,
    StubTranscriptionExecutor,
    TranscriptArtifact,
    TranscriptionExecutor,
    TranscriptSegment,
)
from vo_rag.pipelines.vce.types import (
    VCE_CONTRACT_VERSION,
    VceContext,
    VceExportEvidence,
    VceExportRow,
    VceExportTranscriptionSource,
    VceRowAnchor,
    VceSupersededEntry,
)
from vo_rag.pipelines.vce.validator import (
    ValidationCheck,
    ValidationReport,
    VceExportValidator,
    validate_export,
)
from vo_rag.pipelines.vce.workflow import VcePipeline, VcePipelineOutput

__all__ = [
    "BaselineIndex",
    "CHECKPOINT_PREFIX",
    "LocalWhisperXExecutor",
    "RemoteWhisperXExecutor",
    "RunSummary",
    "StubTranscriptionExecutor",
    "TranscriptArtifact",
    "TranscriptSegment",
    "TranscriptionExecutor",
    "VCE_CONTRACT_VERSION",
    "VCE_EXPORT_FILENAME",
    "VCE_SUPERSEDED_FILENAME",
    "ValidationCheck",
    "ValidationReport",
    "VceCheckpoint",
    "VceContext",
    "VceExportEvidence",
    "VceExportRow",
    "VceExportTranscriptionSource",
    "VceExportValidator",
    "VcePipeline",
    "VcePipelineOutput",
    "VceProgress",
    "VceRowAnchor",
    "VceStage",
    "VceSupersededEntry",
    "load_baseline_rows",
    "run_vce_batch",
    "validate_export",
    "write_export",
]
