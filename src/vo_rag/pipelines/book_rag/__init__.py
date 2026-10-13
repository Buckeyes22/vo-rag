"""book_rag pipeline — motor-sport engineering book ingestion + retrieval."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vo_rag.pipelines.book_rag.config import BookRagConfig
    from vo_rag.pipelines.book_rag.workflow import (
        BookRagContext,
        BookRagPipeline,
        PipelineOutput,
    )

_EXPORTS = {
    "BookRagConfig": ("vo_rag.pipelines.book_rag.config", "BookRagConfig"),
    "BookRagContext": ("vo_rag.pipelines.book_rag.workflow", "BookRagContext"),
    "BookRagPipeline": ("vo_rag.pipelines.book_rag.workflow", "BookRagPipeline"),
    "PipelineOutput": ("vo_rag.pipelines.book_rag.workflow", "PipelineOutput"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        return getattr(import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BookRagConfig",
    "BookRagContext",
    "BookRagPipeline",
    "PipelineOutput",
]
