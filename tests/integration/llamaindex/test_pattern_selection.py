"""Test the multi-agent pattern-selection decision table.

Every planned pipeline from ``load_pipeline_registry`` must have exactly one
pattern assignment in ``PIPELINE_PATTERN``, and the assignments must match
the documented multi-agent orchestration patterns.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, cast

import pytest

from vo_rag.core.config.pipelines import load_pipeline_registry
from vo_rag.core.models.envelope import TargetPipeline
from vo_rag.integrations.llamaindex.agents.patterns import (
    PIPELINE_PATTERN,
    MultiAgentPattern,
)


def test_every_pipeline_has_exactly_one_pattern() -> None:
    registry = load_pipeline_registry()
    pattern_keys = set(PIPELINE_PATTERN.keys())
    registry_keys = {TargetPipeline(name) for name in registry}

    missing = registry_keys - pattern_keys
    extra = pattern_keys - registry_keys

    assert not missing, f"pipelines missing from PIPELINE_PATTERN: {missing}"
    assert not extra, f"extra keys in PIPELINE_PATTERN not in registry: {extra}"


def test_pattern_table_is_enum_keyed_and_immutable() -> None:
    assert isinstance(PIPELINE_PATTERN, MappingProxyType)
    assert set(PIPELINE_PATTERN) == set(TargetPipeline)
    assert all(isinstance(key, TargetPipeline) for key in PIPELINE_PATTERN)

    mutable_view = cast(Any, PIPELINE_PATTERN)
    with pytest.raises(TypeError):
        mutable_view[TargetPipeline.BOOK_RAG] = MultiAgentPattern.ORCHESTRATOR


def test_specific_pipeline_assignments() -> None:
    assert PIPELINE_PATTERN[TargetPipeline.RESEARCH_AGENT] is MultiAgentPattern.AGENT_WORKFLOW
    assert PIPELINE_PATTERN[TargetPipeline.REPO_ANALYSIS] is MultiAgentPattern.ORCHESTRATOR
    assert PIPELINE_PATTERN[TargetPipeline.TOOL_EVALUATION] is MultiAgentPattern.ORCHESTRATOR
    assert PIPELINE_PATTERN[TargetPipeline.CODE_RAG] is MultiAgentPattern.NONE
    assert PIPELINE_PATTERN[TargetPipeline.ARTICLE_EXTRACTION] is MultiAgentPattern.NONE
    assert PIPELINE_PATTERN[TargetPipeline.STRUCTURED_SCRAPE] is MultiAgentPattern.NONE
    assert PIPELINE_PATTERN[TargetPipeline.IDEA_CAPTURE] is MultiAgentPattern.NONE
    assert PIPELINE_PATTERN[TargetPipeline.URL_SNAPSHOT] is MultiAgentPattern.NONE
    assert PIPELINE_PATTERN[TargetPipeline.VCE] is MultiAgentPattern.CUSTOM_PLANNER


def test_book_rag_is_non_agentic() -> None:
    assert PIPELINE_PATTERN[TargetPipeline.BOOK_RAG] is MultiAgentPattern.NONE
