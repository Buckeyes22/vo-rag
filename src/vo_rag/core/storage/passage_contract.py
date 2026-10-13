"""Normalize passage identity and vector representation metadata across writers."""

from __future__ import annotations

import hashlib
from collections.abc import Collection
from typing import Any

WRITER_CONTRACT_VERSION = 1


def passage_payload(
    payload: dict[str, Any],
    *,
    point_id: str,
    source_id: str,
    dense: list[float],
    colbert: list[list[float]],
    bm25: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Preserve intentional parent identity and stamp missing writer contracts."""
    result = dict(payload)
    if not result.get("source_id"):
        result["source_id"] = source_id
    if not result.get("chunk_type"):
        result["chunk_type"] = "text"
    if not result.get("chunk_id"):
        result["chunk_id"] = point_id
    result["writer_contract_version"] = WRITER_CONTRACT_VERSION
    result["colbert_representation"] = (
        "dense_substitute" if colbert == [dense] else "token_vectors" if colbert else "absent"
    )
    result["bm25_representation"] = "absent_no_tokens" if bm25 == {} else "hashed_tokens_v1"
    if not result.get("content"):
        result["content"] = result.get("summary_text") or result.get("label") or ""
    return result


def passage_identity(payload: dict[str, Any], point_id: str) -> tuple[str, str]:
    """Preserve explicit parent/chunk identity, falling back to the storage point."""
    source = str(payload.get("source_id") or point_id)
    identity = payload.get("parent_chunk_id") or payload.get("chunk_id")
    return source, str(identity) if identity else "point:" + point_id


def evidence_provenance(
    payload: dict[str, Any], point_id: str, collection: str
) -> list[dict[str, Any]]:
    """Return citation identities from a prior query collapse or this point."""
    previous = payload.get("duplicate_provenance")
    if isinstance(previous, list):
        valid = [
            item
            for item in previous
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("collection"), str)
        ]
        if valid:
            return valid
    return [
        {
            "id": point_id,
            "collection": collection,
            **{
                field: payload[field]
                for field in (
                    "source_id",
                    "document_id",
                    "chunk_id",
                    "parent_chunk_id",
                    "section_path",
                    "page_number",
                    "page_range",
                    "path",
                    "commit",
                    "page_id",
                    "heading",
                    "url",
                    "fetched_at",
                    "source_url",
                    "source_path",
                    "title",
                    "section_title",
                    "image_id",
                    "image_ref",
                )
                if field in payload
            },
        }
    ]


def collapse_evidence_payloads(
    rows: list[tuple[dict[str, Any], str, str]],
) -> list[tuple[int, dict[str, Any]]]:
    """Collapse ranked exact evidence/context copies while preserving all citations.

    Inputs are already relevance ranked. Storage identity is never rewritten;
    the best representative remains first, and copies do not fabricate votes.
    Visual copies require the same SHA-256 image identity and exact prose/context;
    different or unverified images remain independently meaningful.
    """
    selected: list[tuple[int, dict[str, Any]]] = []
    seen: dict[tuple[str, ...], int] = {}
    for index, (payload, point_id, collection) in enumerate(rows):
        content = payload.get("content")
        visual = (
            payload.get("chunk_type") in {"figure", "figure_composite", "image", "table_image"}
            or payload.get("has_image")
            or payload.get("image_ref")
        )
        context = str(
            (
                payload.get("title") or "",
                payload.get("section_title") or payload.get("heading") or "",
            )
        )
        key: tuple[str, ...] | None = None
        if isinstance(content, str) and content.strip():
            digest = hashlib.sha256(content.encode()).hexdigest()
            image_id = payload.get("image_id")
            if not visual:
                key = ("text", context, digest)
            elif (
                isinstance(image_id, str)
                and len(image_id) == 64
                and all(character in "0123456789abcdef" for character in image_id)
            ):
                metadata = payload.get("metadata")
                section_path = payload.get("section_path")
                # Section paths bind chapter and owning entity. Nested titles often name
                # physical files; use them conservatively only when no path identifies it.
                entity = (
                    (
                        metadata.get("title") or "",
                        metadata.get("source_book") or "",
                    )
                    if not section_path and isinstance(metadata, dict)
                    else ""
                )
                key = ("visual", image_id, context, str(section_path or ""), str(entity), digest)
        provenance = evidence_provenance(payload, point_id, collection)
        if key is not None and key in seen:
            position = seen[key]
            original_index, original = selected[position]
            original_payload, original_id, original_collection = rows[original_index]
            original_provenance = evidence_provenance(original, original_id, original_collection)
            combined = list(original_provenance)
            identifiers = {(item["collection"], item["id"]) for item in combined}
            combined.extend(
                item for item in provenance if (item["collection"], item["id"]) not in identifiers
            )
            if len(combined) > 1:
                selected[position] = (
                    original_index,
                    {**original, "duplicate_provenance": combined},
                )
        else:
            if key is not None:
                seen[key] = len(selected)
            selected.append((index, payload))
    return selected


def model_payload(payload: dict[str, Any] | None, model_fields: Collection[str]) -> dict[str, Any]:
    """Remove declared storage envelope fields before strict native-model parsing."""
    envelope = {
        "writer_contract_version",
        "colbert_representation",
        "bm25_representation",
        "source_id",
        "chunk_id",
        "chunk_index",
        "chunk_type",
        "version_hash",
        "content",
    }
    return {
        key: value
        for key, value in (payload or {}).items()
        if key in model_fields or key not in envelope
    }


__all__ = [
    "WRITER_CONTRACT_VERSION",
    "collapse_evidence_payloads",
    "model_payload",
    "passage_identity",
    "passage_payload",
]
