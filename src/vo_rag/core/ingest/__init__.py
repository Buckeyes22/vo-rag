"""core/ingest: source intake → normalized documents.

This package only handles intake submission normalization to
``IntakeEnvelope``; triage and disposition logic live in the pipelines.
"""

from vo_rag.core.ingest.normalize import (
    IntakeSubmission,
    InvalidPipelineHint,
    normalize,
)

__all__ = [
    "IntakeSubmission",
    "InvalidPipelineHint",
    "normalize",
]
