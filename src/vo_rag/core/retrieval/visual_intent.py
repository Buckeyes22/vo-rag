"""Visual-intent classification + chunk_type constants for retrieval.

Library-level home for the visual-aware retrieval bits so both the
canonical query path (``HybridRetriever.cross_collection_search``) and
ad-hoc ops scripts share a single source of truth.
"""

from __future__ import annotations

from typing import Any

VISUAL_INTENT_PHRASES: tuple[str, ...] = (
    "show me",
    "show the",
    "look like",
    "looks like",
    "what does",
    "what do",
    "depicted in",
    "shown in",
    "illustrated in",
)

VISUAL_INTENT_TOKENS: frozenset[str] = frozenset(
    {
        "diagram",
        "figure",
        "chart",
        "plot",
        "graph",
        "image",
        "illustration",
        "schematic",
        "drawing",
        "photograph",
        "photo",
        "depicts",
        "depict",
        "illustrate",
        "illustrates",
        "visualize",
        "visualizes",
        "drew",
        "render",
        "rendered",
        "draws",
    }
)

VISUAL_CHUNK_TYPES: tuple[str, ...] = ("figure_composite", "figure", "table")
TEXT_CHUNK_TYPES: tuple[str, ...] = ("text", "body")

# Valid values for the visual_mode parameter exposed on the query API.
VISUAL_MODES: tuple[str, ...] = (
    "intent_aware",  # default — auto-pick text/visual quotas from query intent
    "baseline",  # current behavior — no chunk_type filtering
    "visual_only",  # restrict results to visual chunk_types
    "visual_track",  # fixed 60/40 split (text/visual), always reserve visual slots
)


def has_visual_intent(query: str) -> bool:
    """True if the query reads as visual-intent ("show me the diagram", etc.)."""
    q = query.lower()
    if any(p in q for p in VISUAL_INTENT_PHRASES):
        return True
    tokens = {t.strip(".,?!()'\"") for t in q.split()}
    return bool(tokens & VISUAL_INTENT_TOKENS)


def quotas_for_mode(visual_mode: str, query: str, top_k: int) -> tuple[int, int]:
    """Compute (text_quota, visual_quota) for the chosen visual_mode.

    Returns quotas summing to ``top_k`` (within rounding); caller may
    receive a 70/30 split, 30/70, or 60/40 depending on mode + intent."""
    if visual_mode == "intent_aware":
        if has_visual_intent(query):
            visual = max(1, round(top_k * 0.7))
        else:
            visual = max(1, round(top_k * 0.3))
    elif visual_mode == "visual_track":
        visual = max(1, round(top_k * 0.4))
    else:
        # baseline / visual_only don't use the quota path
        raise ValueError(f"quotas_for_mode unsupported for visual_mode={visual_mode!r}")
    visual = min(visual, max(1, top_k - 1))
    text = top_k - visual
    return text, visual


def merge_chunk_type_filter(
    user_filters: dict[str, Any] | None,
    chunk_types: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Add a chunk_type filter to user_filters, preserving any user override.

    If the user already specified ``chunk_type`` we trust them and don't
    overwrite it (their filter is more specific than the mode default)."""
    merged: dict[str, Any] = dict(user_filters or {})
    merged.setdefault("chunk_type", list(chunk_types))
    return merged


__all__ = [
    "VISUAL_INTENT_PHRASES",
    "VISUAL_INTENT_TOKENS",
    "VISUAL_CHUNK_TYPES",
    "TEXT_CHUNK_TYPES",
    "VISUAL_MODES",
    "has_visual_intent",
    "quotas_for_mode",
    "merge_chunk_type_filter",
]
