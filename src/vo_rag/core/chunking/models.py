from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from vo_rag.core.models import VoBaseModel

DropReason = Literal[
    "empty_after_normalize",
    "qa_quarantined",
    "below_min_words",
    "no_text_extracted",
    # A whole chunk judged OCR noise by pipelines/book_rag/ocr_quality.py
    # (VOR-1452); recorded by the book_rag chunk step, not the chunker.
    "ocr_garbage",
]


class RetrievalChunk(VoBaseModel):
    """One retrievable chunk.

    ``content`` is what is stored, shown and tagged; ``embedding_text`` is what
    is embedded. For text and tables it is ``f"{section_path}\\n\\n{content}"``.
    For ``figure_composite`` chunks the two differ on purpose (VOR-1452):
    ``content`` holds only the figure's own text -- "Caption:", a bounded
    "Page context:" excerpt, the "Visual:" VLM description, or "Figure on page
    N (section)" when it has none -- while ``embedding_text`` keeps the wider
    retrieval window (full page context plus "Preceding:"/"Following:"
    neighbour prose), which the text chunks already store.
    """

    chunk_id: str
    chunk_type: str  # "text" | "figure_composite" | "table" | "figure"
    document_id: str
    source_id: str
    page_number: int | None = None
    page_range: str | None = None
    section_path: str
    section_title: str
    chapter_path: str = ""
    content: str
    embedding_text: str
    token_count: int
    has_image: bool = False
    image_id: str | None = None
    image_ref: str | None = None
    vlm_description: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class PageDrop(VoBaseModel):
    page_number: int
    reason: DropReason


class PendingShortText(VoBaseModel):
    content: str
    pages: list[int] = Field(default_factory=list)
    section_path: str
    section_title: str
    metadata: dict[str, object] = Field(default_factory=dict)


#: Raw-payload key holding a figure chunk's embedding window (VOR-1452). Only
#: ``figure_composite`` chunks carry it, and only when it differs from
#: ``content``: for text and tables the embedding is ``section_path`` plus
#: ``content``, which the payload already has. The sterile payload
#: (``model_dump``) carries its own, sterile, ``embedding_text`` for every chunk.
PAYLOAD_EMBEDDING_TEXT_KEY = "embedding_text"


def figure_payload_embedding_text(chunk: RetrievalChunk) -> str | None:
    """The window to persist in the raw payload for ``chunk``, or None."""
    if chunk.chunk_type != "figure_composite":
        return None
    if not chunk.embedding_text or chunk.embedding_text == chunk.content:
        return None
    return chunk.embedding_text


def payload_embedding_text(payload: Mapping[str, Any]) -> str:
    """Text to re-embed a stored chunk with: the persisted figure window when
    present, else ``content`` (the fallback for chunks written without a window)."""
    stored = payload.get(PAYLOAD_EMBEDDING_TEXT_KEY)
    if isinstance(stored, str) and stored.strip():
        return stored
    return str(payload.get("content", ""))


def hit_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """A stored payload as a query hit may carry it (VOR-1452).

    Drops a raw book point's ``embedding_text`` -- the embedding-only neighbour
    window of a figure chunk, which duplicates the neighbouring chunks' prose --
    so it never reaches API/MCP hits, rerankers or answer builders. Raw book
    points are the ones ``do_upsert_raw`` stamps ``stage="raw"``. The sterile
    collection's ``embedding_text`` is sterilized text and is kept. Every other
    payload is returned as is (same object). Brain governing-context points
    likewise hide their bounded ranking packet; content is the complete evidence.
    """
    if payload is None:
        return {}
    if PAYLOAD_EMBEDDING_TEXT_KEY in payload and (
        payload.get("stage") == "raw"
        or (payload.get("stage") == "brain" and payload.get("governing_context_version") == 1)
    ):
        return {k: v for k, v in payload.items() if k != PAYLOAD_EMBEDDING_TEXT_KEY}
    return payload if isinstance(payload, dict) else dict(payload)


def brain_evidence_issue(payload: Mapping[str, Any], collection: str) -> str | None:
    """Return a source-free diagnostic for malformed inline brain authority."""
    if (
        collection != "vo_rag_brain_v1"
        or type(payload.get("governing_context_version")) is not int
        or payload.get("governing_context_version") != 1
        or payload.get("governing_context_complete") is not True
    ):
        return "unsupported_or_unattested_context"
    source = payload.get("source_id")
    revision = payload.get("source_revision", payload.get("commit"))
    content = payload.get("content")
    segments = payload.get("governing_context")
    if not isinstance(source, str) or not source.strip():
        return "missing_source_identity"
    if not isinstance(revision, str) or not revision.strip():
        return "missing_source_snapshot"
    if (
        "source_revision" in payload
        and "commit" in payload
        and payload["source_revision"] != payload["commit"]
    ):
        return "source_snapshot_alias_mismatch"
    if not isinstance(content, str) or not content.strip():
        return "missing_source_content"
    if not isinstance(segments, list):
        return "missing_context_manifest"
    for segment in segments:
        if not isinstance(segment, dict):
            return "invalid_context_segment"
        if segment.get("source_id") != source or segment.get("collection") != collection:
            return "context_source_identity_mismatch"
        if segment.get("source_revision") != revision:
            return "context_source_snapshot_mismatch"
        occurrence = segment.get("section_occurrence", segment.get("occurrence"))
        if type(occurrence) is not int or occurrence < 0:
            return "invalid_context_occurrence"
        point_id = segment.get("point_id")
        if not isinstance(point_id, str) or not point_id.strip():
            return "missing_context_citation"
        text = segment.get("text")
        if not isinstance(text, str) or not text.strip():
            return "missing_context_content"
        if segment.get("content_hash") != hashlib.sha256(text.encode()).hexdigest():
            return "context_content_hash_mismatch"
        if text not in content:
            return "context_content_missing_from_delivery"
    return None


def brain_evidence_is_complete(payload: Mapping[str, Any], collection: str) -> bool:
    """Admit inline brain authority only with its bound source snapshot and bytes."""
    return brain_evidence_issue(payload, collection) is None
