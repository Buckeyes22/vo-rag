"""Small-to-big auto-merge retrieval post-processing."""

from __future__ import annotations

from collections.abc import Callable

from vo_rag.core.retrieval.search import SearchResult

FetchParentFn = Callable[[str], SearchResult | None]


def auto_merge(
    hits: list[SearchResult],
    fetch_parent_fn: FetchParentFn,
    *,
    threshold: float = 0.5,
) -> list[SearchResult]:
    """Replace retrieved child groups with their parent when enough siblings appear."""
    groups: dict[str, list[SearchResult]] = {}
    for hit in hits:
        parent_id = hit.payload.get("parent_id")
        if isinstance(parent_id, str):
            groups.setdefault(parent_id, []).append(hit)

    replacements: dict[str, SearchResult] = {}
    for parent_id, group in groups.items():
        total = group[0].payload.get("parent_child_count")
        if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
            continue
        if len(group) / total < threshold:
            continue

        parent = fetch_parent_fn(parent_id)
        if parent is None:
            continue

        replacements[parent_id] = SearchResult(
            id=parent.id,
            score=max(child.score for child in group),
            collection=parent.collection,
            payload=parent.payload,
        )

    emitted_parents: set[str] = set()
    merged: list[SearchResult] = []
    for hit in hits:
        parent_id = hit.payload.get("parent_id")
        if not isinstance(parent_id, str) or parent_id not in replacements:
            merged.append(hit)
            continue
        if parent_id in emitted_parents:
            continue
        merged.append(replacements[parent_id])
        emitted_parents.add(parent_id)

    return merged
