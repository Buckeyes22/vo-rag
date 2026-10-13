"""Tests for visual-intent classification + visual-mode constants.

The ``VISUAL_MODES``, ``VISUAL_CHUNK_TYPES``, and ``TEXT_CHUNK_TYPES`` tuples
are part of the public API surface (validators in the query/answer routes
import them). The quota math and ``merge_chunk_type_filter`` semantics are
load-bearing for the cross-collection search dispatch (used by
``HybridRetriever``, not by this leaf module).
"""

from __future__ import annotations

import pytest

from vo_rag.core.retrieval.visual_intent import (
    TEXT_CHUNK_TYPES,
    VISUAL_CHUNK_TYPES,
    VISUAL_INTENT_PHRASES,
    VISUAL_INTENT_TOKENS,
    VISUAL_MODES,
    has_visual_intent,
    merge_chunk_type_filter,
    quotas_for_mode,
)

# --- Constants surface (public API) ---------------------------------------- #


def test_visual_modes_includes_default() -> None:
    assert "intent_aware" in VISUAL_MODES
    assert "baseline" in VISUAL_MODES
    assert "visual_only" in VISUAL_MODES
    assert "visual_track" in VISUAL_MODES


def test_visual_chunk_types_are_distinct_from_text_chunk_types() -> None:
    assert set(VISUAL_CHUNK_TYPES).isdisjoint(set(TEXT_CHUNK_TYPES))


def test_visual_intent_phrase_list_non_empty() -> None:
    assert len(VISUAL_INTENT_PHRASES) > 0


def test_visual_intent_token_set_non_empty() -> None:
    assert len(VISUAL_INTENT_TOKENS) > 0


# --- has_visual_intent ----------------------------------------------------- #


def test_visual_intent_detected_for_visual_keywords() -> None:
    assert has_visual_intent("show me the diagram of slip angle")
    assert has_visual_intent("downforce graph at different speeds")
    assert has_visual_intent("schematic of brake fluid system")
    assert has_visual_intent("what does the racing line look like")


def test_visual_intent_false_for_concept_queries() -> None:
    assert not has_visual_intent("brake fluid DOT specification boiling point")
    assert not has_visual_intent("trail braking corner entry")


# --- quotas_for_mode ------------------------------------------------------- #


def test_quotas_for_intent_aware_visual_query() -> None:
    text, visual = quotas_for_mode("intent_aware", "show me the diagram", 10)
    assert text + visual == 10
    assert visual == 7  # 70/30 split for visual intent


def test_quotas_for_intent_aware_text_query() -> None:
    text, visual = quotas_for_mode("intent_aware", "brake fluid spec", 10)
    assert text + visual == 10
    assert text == 7
    assert visual == 3


def test_quotas_for_visual_track_fixed_split() -> None:
    text, visual = quotas_for_mode("visual_track", "any query", 10)
    assert text + visual == 10
    assert visual == 4
    assert text == 6


def test_quotas_for_unsupported_mode_raises() -> None:
    with pytest.raises(ValueError):
        quotas_for_mode("baseline", "q", 10)
    with pytest.raises(ValueError):
        quotas_for_mode("visual_only", "q", 10)


def test_quotas_min_visual_one_for_small_topk() -> None:
    """``max(1, round(...))`` floor means even a tiny top_k reserves ≥1 visual slot."""
    text, visual = quotas_for_mode("intent_aware", "text query", 2)
    assert visual >= 1
    assert text + visual == 2


# --- merge_chunk_type_filter ---------------------------------------------- #


def test_merge_chunk_type_filter_adds_when_missing() -> None:
    merged = merge_chunk_type_filter({"license_class": "cleared"}, VISUAL_CHUNK_TYPES)
    assert merged["license_class"] == "cleared"
    assert merged["chunk_type"] == list(VISUAL_CHUNK_TYPES)


def test_merge_chunk_type_filter_preserves_user_override() -> None:
    merged = merge_chunk_type_filter({"chunk_type": "text"}, VISUAL_CHUNK_TYPES)
    # User filter wins
    assert merged["chunk_type"] == "text"


def test_merge_chunk_type_filter_none_input() -> None:
    merged = merge_chunk_type_filter(None, TEXT_CHUNK_TYPES)
    assert merged["chunk_type"] == list(TEXT_CHUNK_TYPES)
