"""DSI faceted-record schemas.

``DSI_PROMPT_VERSION`` is deliberately not declared here —
``vo_rag.core.dsi.prompts`` is the single source of truth, so the two cannot
drift apart.

Guarantees:

- ``ProvenanceLinkedEntry.supporting_chunk_ids`` / ``supporting_section_paths``:
  every facet entry is anchored to at least one cited chunk and the
  ground-truth section path derived from that chunk.
- ``facet_record_id_for`` returns the versioned vo-rag DSI point id
  (stage = ``f"dsi_{summary_model}"``, version_hash = ``prompt_hash``)
  so a prompt edit or model swap mints a distinct record.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.storage.qdrant import deterministic_point_id


class ProvenanceLinkedEntry(VoBaseModel):
    value: str = Field(min_length=1)
    supporting_chunk_ids: list[str] = Field(min_length=1)
    supporting_section_paths: list[str] = Field(min_length=1)


class FacetedBookRecord(VoBaseModel):
    book_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    author: str = Field(min_length=1)
    category_ids: list[str]
    short_abstract: str
    technical_domains: list[str]
    key_concepts: list[ProvenanceLinkedEntry]
    key_components: list[str]
    key_methods: list[ProvenanceLinkedEntry]
    key_equations_or_models: list[ProvenanceLinkedEntry]
    figure_heavy_topics: list[ProvenanceLinkedEntry]
    synonyms_and_aliases: list[ProvenanceLinkedEntry]
    prerequisite_topics: list[str]
    chapter_outline: str
    representative_chunk_ids: list[str]
    source_collection_version: str = Field(min_length=1)
    summary_model: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


def facet_record_id_for(book_id: str, summary_model: str, prompt_hash: str) -> str:
    """Return the versioned vo-rag DSI point id for ``book_id``.

    This matches the id written by ``persist_dsi_records``:
    ``deterministic_point_id(
        source_id=book_id, chunk_index=0,
        stage=f"dsi_{summary_model}", version_hash=prompt_hash,
    )``.
    A different ``summary_model`` or ``prompt_hash`` mints a distinct point id
    (model A/B isolation + prompt-edit isolation).
    """
    return deterministic_point_id(
        source_id=book_id,
        chunk_index=0,
        stage=f"dsi_{summary_model}",
        version_hash=prompt_hash,
    )


__all__ = [
    "FacetedBookRecord",
    "ProvenanceLinkedEntry",
    "facet_record_id_for",
]
