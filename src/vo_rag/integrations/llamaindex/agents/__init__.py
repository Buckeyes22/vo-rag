"""Agent integration helpers for the LlamaIndex adapter tier."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vo_rag.integrations.llamaindex.agents.output import (
    AgentOutputFold,
    CostLaunchGate,
    assert_agent_llm_substrate,
    fold_agent_output,
    resolve_agent_output_trust_level,
)
from vo_rag.integrations.llamaindex.agents.patterns import (
    PIPELINE_PATTERN,
    MultiAgentPattern,
)

if TYPE_CHECKING:
    from vo_rag.integrations.llamaindex.agents.base import AgentPipelineBase


def __getattr__(name: str) -> Any:
    """Lazily expose agent helpers without importing LlamaIndex at package import."""
    if name == "AgentPipelineBase":
        from vo_rag.integrations.llamaindex.agents.base import AgentPipelineBase

        return AgentPipelineBase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentPipelineBase",
    "AgentOutputFold",
    "CostLaunchGate",
    "MultiAgentPattern",
    "PIPELINE_PATTERN",
    "assert_agent_llm_substrate",
    "fold_agent_output",
    "resolve_agent_output_trust_level",
]
