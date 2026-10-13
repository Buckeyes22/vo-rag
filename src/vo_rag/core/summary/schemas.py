"""Layer 1A summary tree node schemas.

Deterministic ``node_id_for`` hashing runs over SORTED ``source_chunk_ids``:
SHA-256 of newline-joined sorted chunk ids, first 16 hex digits as the
integer chunk index, then delegated to ``deterministic_point_id``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Any

import structlog
from pydantic import Field

from vo_rag.core.models import VoBaseModel
from vo_rag.core.storage.qdrant import deterministic_point_id

log = structlog.get_logger(__name__)


class NodeLevel(StrEnum):
    SECTION = "section"
    CHAPTER = "chapter"
    BOOK = "book"


class EvidenceSpan(VoBaseModel):
    chunk_id: str
    section_path: str
    text: str
    score: float
    selection_method: str


class SummaryNode(VoBaseModel):
    # Normalized structural IDs.
    node_id: str
    book_id: str
    level: NodeLevel
    node_type: str
    parent_node_id: str | None
    child_node_ids: list[str] = Field(default_factory=list)
    canonical_path: str
    display_path: str

    # Source provenance for selective expansion.
    source_chunk_ids: list[str]
    source_section_paths: list[str]

    # Extract-Support evidence used.
    evidence_spans: list[EvidenceSpan]

    # Summary content.
    summary_text: str
    summary_model: str
    prompt_hash: str
    prompt_version: str

    # Embedding metadata. BGE-M3 dense vectors are 1024d (see ``core/embedding/schemas.py``).
    embedding_model: str
    embedding_version: str

    # Audit.
    created_at: datetime
    source_collection_version: str

    metadata: dict[str, Any] = Field(default_factory=dict)


def node_id_for(
    book_id: str,
    level: NodeLevel,
    source_chunk_ids: Iterable[str],
    prompt_hash: str,
) -> str:
    """Return the deterministic Layer 1A summary-node ID.

    The existing Qdrant point-id helper accepts an integer chunk index, so the
    constituent raw chunk IDs are sorted, joined, SHA-256 hashed, and truncated
    to a stable 64-bit integer. Identical summary inputs therefore produce
    byte-identical node IDs across retries and hosts.
    """

    sorted_chunk_ids = sorted(source_chunk_ids)
    if not sorted_chunk_ids:
        raise ValueError("source_chunk_ids must not be empty")

    chunk_set_digest = hashlib.sha256("\n".join(sorted_chunk_ids).encode("utf-8")).hexdigest()
    chunk_index = int(chunk_set_digest[:16], 16)
    return deterministic_point_id(
        source_id=book_id,
        chunk_index=chunk_index,
        stage=f"summary_{level.value}",
        version_hash=prompt_hash,
    )
