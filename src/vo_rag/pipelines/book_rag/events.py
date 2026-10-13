"""Streamed progress events for the book_rag ingest workflow.

These events are pushed onto the run's event stream via
``ctx.write_event_to_stream(...)`` and consumed by the API SSE, MCP
progress, and CLI rich adapters. They are intentionally lightweight and
JSON-serialisable — no parsed documents, chunks, or embeddings travel on
this stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from llama_index.core.workflow import Event as _EventBase
else:
    try:
        from llama_index.core.workflow import Event as _EventBase
    except ModuleNotFoundError:
        _EventBase = BaseModel


class _StrictEventBase(_EventBase):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def __init__(self, **params: Any) -> None:
        extra = set(params) - set(self.__class__.model_fields) - set(self.__private_attributes__)
        if extra:
            errors = [
                {"type": "extra_forbidden", "loc": (name,), "input": params[name]}
                for name in sorted(extra)
            ]
            raise ValidationError.from_exception_data(self.__class__.__name__, cast(Any, errors))
        super().__init__(**params)


type IngestStep = Literal[
    "register_source",
    "parse",
    "visual",
    "chunk_embed",
    "sterilize",
    "register_artifact",
    "validate",
    "retrieval_smoke",
]
INGEST_STEP_ORDER: Final[list[IngestStep]] = list(
    cast(tuple[IngestStep, ...], get_args(IngestStep.__value__))
)


class IngestProgress(_StrictEventBase):
    """Coarse-grained progress tick for a single ingest step."""

    envelope_id: str
    step: IngestStep
    detail: str = ""
    pct: float | None = Field(default=None, ge=0.0, le=1.0)


class ImageDescribedEvent(_StrictEventBase):
    """Summary image-progress event emitted to external consumers.

    The workflow's parallel per-image fan-out uses internal
    ``ImageDescriptionResult`` events. This public stream event is emitted at
    visual-step summary/checkpoint points; ``image_index`` is currently a coarse
    progress placeholder rather than the fan-out worker result index.
    """

    envelope_id: str
    image_index: int
    total: int


__all__ = ["IngestProgress", "ImageDescribedEvent", "IngestStep", "INGEST_STEP_ORDER"]
