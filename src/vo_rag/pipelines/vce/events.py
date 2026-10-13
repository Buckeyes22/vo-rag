"""Progress events for the VCE pipeline.

Mirrors `book_rag/events.py` so the runner can emit the same shape of
progress events to whichever consumer (CLI, MCP, future API) wants them.
The book_rag version subclasses ``llama_index.core.workflow.Event``; VCE
does not use LlamaIndex, so the events are plain Pydantic models.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VceStage = Literal[
    "register",
    "acquire",
    "transcribe",
    "frame_sample",
    "visual_extract",
    "segment",
    "structured_extract",
    "validate",
    "export",
]
VCE_STAGE_ORDER: tuple[VceStage, ...] = (
    "register",
    "acquire",
    "transcribe",
    "frame_sample",
    "visual_extract",
    "segment",
    "structured_extract",
    "validate",
    "export",
)


class VceProgress(BaseModel):
    """Coarse-grained progress tick for a single VCE stage."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    video_id: str
    stage: VceStage
    detail: str = ""
    pct: float | None = Field(default=None, ge=0.0, le=1.0)


__all__ = ["VCE_STAGE_ORDER", "VceProgress", "VceStage"]
