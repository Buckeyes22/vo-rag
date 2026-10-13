"""Core chunking helpers and hierarchical retrieval chunker.

The package root keeps its public re-exports lazy so imports of light
submodules such as ``vo_rag.core.chunking.models`` do not import the
transformers-backed token counter or the full hierarchical chunker stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vo_rag.core.chunking.compliance_sub import ComplianceSubChunk, ComplianceSubChunker
    from vo_rag.core.chunking.composite_emitter import CompositeChunk, CompositeEmitter
    from vo_rag.core.chunking.hierarchical import HierarchicalChunker
    from vo_rag.core.chunking.models import PageDrop, PendingShortText, RetrievalChunk
    from vo_rag.core.chunking.table_linearizer import TableLinearizer
    from vo_rag.core.chunking.token_counter import TokenCounter

_EXPORTS = {
    "ComplianceSubChunk": ("vo_rag.core.chunking.compliance_sub", "ComplianceSubChunk"),
    "ComplianceSubChunker": ("vo_rag.core.chunking.compliance_sub", "ComplianceSubChunker"),
    "CompositeChunk": ("vo_rag.core.chunking.composite_emitter", "CompositeChunk"),
    "CompositeEmitter": ("vo_rag.core.chunking.composite_emitter", "CompositeEmitter"),
    "HierarchicalChunker": ("vo_rag.core.chunking.hierarchical", "HierarchicalChunker"),
    "PageDrop": ("vo_rag.core.chunking.models", "PageDrop"),
    "PendingShortText": ("vo_rag.core.chunking.models", "PendingShortText"),
    "RetrievalChunk": ("vo_rag.core.chunking.models", "RetrievalChunk"),
    "TableLinearizer": ("vo_rag.core.chunking.table_linearizer", "TableLinearizer"),
    "TokenCounter": ("vo_rag.core.chunking.token_counter", "TokenCounter"),
    "default_token_counter": ("vo_rag.core.chunking.token_counter", "default_token_counter"),
    "naive_counter": ("vo_rag.core.chunking.counters", "naive_counter"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        return getattr(import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ComplianceSubChunk",
    "ComplianceSubChunker",
    "CompositeChunk",
    "CompositeEmitter",
    "HierarchicalChunker",
    "PageDrop",
    "PendingShortText",
    "RetrievalChunk",
    "TableLinearizer",
    "TokenCounter",
    "default_token_counter",
    "naive_counter",
]
