"""Select ranked evidence without repeating complete brain ancestor units."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from vo_rag.core.chunking.models import brain_evidence_is_complete
from vo_rag.core.retrieval.search import SearchResult

# Everything else, including unknown future authority/scope fields, must agree.
_PASSAGE_FIELDS = {
    "content",
    "source_section_text",
    "governing_context",
    "heading",
    "section_title",
    "chunk_id",
    "chunk_index",
    "point_id",
    "version_hash",
    "embedding_text",
    "source_revision",
    "commit",
    "source_section_occurrence",
    "section_occurrence",
}


def select_ranked_evidence(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Suppress only evidence fully delivered by an already-selected brain hit.

    This does not infer the physical occurrence of an unattested local unit or
    change corpus points. The surviving passage retains its real citation and
    complete governing context, including every original ancestor qualifier.
    """
    if top_k <= 0:
        return []
    selected: list[SearchResult] = []
    for result in results:
        if not any(_covered_by(result, earlier) for earlier in selected):
            selected.append(result)
        if len(selected) >= top_k:
            break
    return selected


def _revision(payload: dict[str, Any]) -> Any:
    return payload.get("source_revision", payload.get("commit"))


def _scope(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in _PASSAGE_FIELDS}


def _same_unit(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = ("text", "content_hash", "source_id", "collection", "source_revision")
    if any(left.get(field) != right.get(field) for field in fields):
        return False
    return left.get("section_occurrence", left.get("occurrence")) == right.get(
        "section_occurrence", right.get("occurrence")
    )


def _covered_by(candidate: SearchResult, selected: SearchResult) -> bool:
    if candidate.collection != "vo_rag_brain_v1" or selected.collection != candidate.collection:
        return False
    original, delivered = candidate.payload, selected.payload
    if not all(brain_evidence_is_complete(p, candidate.collection) for p in (original, delivered)):
        return False
    if original["source_id"] != delivered["source_id"] or _revision(original) != _revision(
        delivered
    ):
        return False
    if _scope(original) != _scope(delivered):
        return False
    local = original.get("source_section_text")
    if not isinstance(local, str) or not local.strip():
        return False
    ancestors = original["governing_context"]
    governing = delivered["governing_context"]
    # Reject extra/unrepresented candidate content, split-unit fragments and
    # arbitrary substring matches. Only the known source identity wrapper may
    # accompany the complete original local unit and its complete ancestors.
    source_text = "\n\n".join([*(segment["text"] for segment in ancestors), local])
    title = original.get("title") or PurePosixPath(str(original.get("path", ""))).stem
    heading = original.get("heading")
    labels = [title]
    if isinstance(heading, str) and heading:
        labels.append(str(title) + " / " + heading)
    if original["content"] not in [str(label) + "\n\n" + source_text for label in labels]:
        return False
    # A repeated equal unit underneath an additional conditional ancestor is
    # not equivalent. Original governors must be the exact ordered prefix
    # governing the matched unit; later descendant governors remain contextual.
    for index, segment in enumerate(governing):
        if segment["text"] != local or index != len(ancestors):
            continue
        if not all(
            _same_unit(left, right)
            for left, right in zip(ancestors, governing[:index], strict=True)
        ):
            continue
        identities = [
            original[field]
            for field in ("source_section_occurrence", "section_occurrence")
            if field in original
        ]
        if any(
            type(occurrence) is not int
            or occurrence < 0
            or occurrence != segment.get("section_occurrence", segment.get("occurrence"))
            for occurrence in identities
        ):
            continue
        return True
    return False
