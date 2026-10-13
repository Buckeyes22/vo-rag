"""Retrieval context helpers for answer synthesis.

Pure helpers for the answer route:

- ``resolve_answer_collections``: maps raw → sterile for STERILE /
  PUBLICATION / PARTNER modes.
- ``parent_ids_for_results``: unique parent/chunk ids in input order
  (used to fetch sub-chunks for the compliance gate).
- ``source_ids``: sorted unique source ids (used in answer_log rows).
- ``format_answer_context``: the per-chunk textual block the LLM sees
  via the protected completion path.

These are pure functions over ``SearchResult`` — no DB pool, no
FastAPI request state. The route layer is responsible for assembling
the inputs and persisting the outputs; this module just formats them.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from vo_rag.core.chunking.models import brain_evidence_is_complete
from vo_rag.core.retrieval.query_mode import (
    RAW_TO_STERILE,
    QueryMode,
    resolve_collections,
)
from vo_rag.core.retrieval.search import SearchResult


class ContextScope(StrEnum):
    STERILE = "sterile"
    RAW = "raw"


def resolve_answer_collections(
    *,
    mode: QueryMode,
    requested: list[str] | None,
    context_scope: ContextScope,
) -> list[str]:
    """Resolve the collections to query, remapping raw→sterile when the
    context scope or the mode demands it.

    The remap is the load-bearing guard that keeps raw, unredacted book
    text from reaching the published-answer or partner paths. The
    STERILE scope is unconditional on top of the mode-based remap; the
    RAW scope only preserves raw collections when the mode itself does
    not force the remap (i.e. INTERNAL).

    The route layer is responsible for the 403 ``allow_raw_hosted_egress``
    gate; this helper just shapes the collection list.
    """
    resolved = resolve_collections(mode, requested)
    if context_scope == ContextScope.STERILE or mode in {
        QueryMode.PUBLICATION,
        QueryMode.PARTNER,
    }:
        resolved = [RAW_TO_STERILE.get(collection, collection) for collection in resolved]

    out: list[str] = []
    for collection in resolved:
        if collection not in out:
            out.append(collection)
    return out


def parent_ids_for_results(results: list[SearchResult]) -> list[str]:
    """Return unique parent/chunk ids in input order.

    Resolution order per result: ``parent_chunk_id`` → ``chunk_id`` →
    Qdrant point id. The SQL-IN clause the sub-chunk query sees is
    sensitive to both ordering and duplicates, so dedup-in-order is the
    canonical shape.
    """
    out: list[str] = []
    for result in results:
        payload = result.payload
        parent_id = payload.get("parent_chunk_id") or payload.get("chunk_id") or result.id
        parent_id = str(parent_id)
        if parent_id not in out:
            out.append(parent_id)
    return out


def source_ids(results: list[SearchResult]) -> list[str]:
    """Return sorted unique source ids across ``results``.

    A result with no ``source_id`` in its payload contributes nothing.
    The compliance log column is ``source_id``-shaped; filling it with
    a Qdrant point id would be meaningless and would pollute the
    audit trail.
    """
    ids = {
        str(result.payload.get("source_id"))
        for result in results
        if result.payload.get("source_id")
    }
    return sorted(ids)


class AnswerEvidenceUnavailable(ValueError):
    """Complete governing brain evidence cannot be safely sent to synthesis."""


MAX_BRAIN_ANSWER_CONTEXT_CHARS = 65_536


def format_answer_context(results: list[SearchResult], *, max_chunks: int = 8) -> str:
    """Format the per-chunk textual block the LLM sees.

    Each chunk is rendered as ``[S{index}]`` followed by ``point_id``,
    ``chunk_id`` (which prefers ``parent_chunk_id`` so sub-chunk
    lookups round-trip), ``source_id``, ``title``, ``page``, and the
    chunk's content. The output is a blank-line-separated series of
    blocks joined by ``\\n\\n---\\n\\n`` so the model sees a clean
    document boundary.
    """
    chunks: list[str] = []
    contains_brain_context = False
    for index, result in enumerate(results[:max_chunks], start=1):
        payload = result.payload
        chunk_id = str(payload.get("chunk_id") or payload.get("parent_chunk_id") or result.id)
        source_id = str(payload.get("source_id") or "")
        title = str(payload.get("source_title") or payload.get("title") or "")
        page = payload.get("page_number")
        content = _extract_content(payload)
        if result.collection == "vo_rag_brain_v1":
            contains_brain_context = True
            if not brain_evidence_is_complete(payload, result.collection):
                raise AnswerEvidenceUnavailable("brain_governing_evidence_incomplete")
        authority: list[str] = []
        if result.collection == "vo_rag_brain_v1":
            fields = {
                "collection": result.collection,
                "source_revision": payload.get("source_revision", payload.get("commit")),
                **{
                    key: payload[key]
                    for key in (
                        "kind",
                        "url",
                        "fetched_at",
                        "path",
                        "page_area",
                        "status",
                        "source_origin",
                        "source_lineage",
                        "updated",
                        "provider",
                        "platform",
                        "service",
                        "offering",
                        "edition",
                        "region",
                        "applicability",
                        "product",
                        "version",
                    )
                    if key in payload
                },
            }
            authority = [
                "brain_provenance: " + json.dumps(fields, ensure_ascii=False, default=str),
                "Authority: indexing does not promote inbox proposals to accepted knowledge "
                "or establish that infrastructure is currently running. Preserve explicit "
                "status, area, applicability and temporal qualifications.",
            ]
        chunks.append(
            "\n".join(
                [
                    f"[S{index}]",
                    f"point_id: {result.id}",
                    f"chunk_id: {chunk_id}",
                    f"source_id: {source_id}",
                    f"title: {title}",
                    f"page: {page if page is not None else ''}",
                    *authority,
                    "content:",
                    content,
                ]
            )
        )
    rendered = "\n\n---\n\n".join(chunks)
    if contains_brain_context and len(rendered) > MAX_BRAIN_ANSWER_CONTEXT_CHARS:
        raise AnswerEvidenceUnavailable("brain_answer_evidence_exceeds_65536_character_budget")
    return rendered


def _extract_content(payload: dict[str, Any]) -> str:
    for key in ("content", "text", "content_text", "summary"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


__all__ = [
    "ContextScope",
    "format_answer_context",
    "parent_ids_for_results",
    "resolve_answer_collections",
    "source_ids",
]
