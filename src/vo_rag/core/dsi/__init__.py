"""DSI — Document Structure Index (Layer 2 advanced understanding).

Exports the four primitives every caller needs:

- ``FacetedBookRecord`` — the per-book extracted record persisted to Qdrant.
- ``ProvenanceLinkedEntry`` — one provenance-linked facet entry within a record.
- ``facet_record_id_for(book_id, summary_model, prompt_hash)`` — deterministic
  point ID, prompt-hash versioned so a prompt edit produces a distinct record.
- ``DSI_PROMPT_VERSION`` — re-exported from ``prompts`` (single source of truth).

Builder/extraction/validation/routing submodules are imported directly by
callers that need them (``vo_rag.core.dsi.builder``, etc.).
"""

from __future__ import annotations

from vo_rag.core.dsi.prompts import DSI_PROMPT_VERSION
from vo_rag.core.dsi.schemas import (
    FacetedBookRecord,
    ProvenanceLinkedEntry,
    facet_record_id_for,
)

__all__ = [
    "DSI_PROMPT_VERSION",
    "FacetedBookRecord",
    "ProvenanceLinkedEntry",
    "facet_record_id_for",
]
