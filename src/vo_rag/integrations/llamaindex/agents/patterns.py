"""Multi-agent pattern-selection decision table for planned roster pipelines.

This module encodes the mapping from each vo-rag planned pipeline to its
recommended LlamaIndex multi-agent pattern.

Guardrails apply at **every hop**, regardless of the chosen pattern:

- Injection defense: every agent that ingests untrusted content must route its
  LLM call through ``complete_json_protected(...)``
  (``src/vo_rag/core/llm/protected.py:33``).
- Cost gating: meter spend per agent (pattern B's single orchestrator is the
  cleanest chokepoint; patterns A and C must meter at each agent).
- Egress lockdown: the final user-facing output passes through
  ``resolve_answer_collections(...)`` (``src/vo_rag/core/answer/context.py:36``),
  remapping raw→sterile.
- Least-power default: prefer the simplest pattern that satisfies the pipeline.
  Every extra agent adds an injection surface, a cost centre, and a loop risk.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from vo_rag.core.models.envelope import TargetPipeline


class MultiAgentPattern(StrEnum):
    """LlamaIndex multi-agent orchestration patterns.

    Pattern A — AGENT_WORKFLOW (handoff / linear swarm)
        Built-in ``AgentWorkflow`` with ``can_handoff_to``. Fewest lines;
        built-in streaming; least direct control.
    Pattern B — ORCHESTRATOR (sub-agents as tools)
        One top-level orchestrator owns every step. Moderate code; one brain
        sees the whole picture; cost metered at a single chokepoint.
    Pattern C — CUSTOM_PLANNER (DIY prompting and parsing)
        LLM emits a structured XML/JSON plan, parsed and executed imperatively.
        Most lines; most flexibility; deterministic gates between stages.
    NONE — Single agent or no agent
        The pipeline does not benefit from multi-agent composition.
    """

    AGENT_WORKFLOW = "agent_workflow"
    ORCHESTRATOR = "orchestrator"
    CUSTOM_PLANNER = "custom_planner"
    NONE = "none"


PIPELINE_PATTERN: Final[Mapping[TargetPipeline, MultiAgentPattern]] = MappingProxyType(
    {
        # --- Planned pipelines -------------------------------------------------
        # research_agent: Linear research→synthesise→review flow; handoffs are
        # natural. Promote to ORCHESTRATOR once web-fetch cost needs one chokepoint.
        TargetPipeline.RESEARCH_AGENT: MultiAgentPattern.AGENT_WORKFLOW,
        # repo_analysis: One brain must see the whole import graph to decide what
        # to walk next; deterministic list_imports tools sit under the orchestrator.
        TargetPipeline.REPO_ANALYSIS: MultiAgentPattern.ORCHESTRATOR,
        # code_rag: A lone retrieval agent with one QueryEngineTool; multi-agent is
        # over-engineering here.
        TargetPipeline.CODE_RAG: MultiAgentPattern.NONE,
        # tool_evaluation: Fans out fetches across docs/repo/changelog, then
        # reconciles centrally via structured_output_fn. One decision-maker owns
        # the verdict.
        TargetPipeline.TOOL_EVALUATION: MultiAgentPattern.ORCHESTRATOR,
        # article_extraction: Fetch-then-extract is one tool call plus output_cls;
        # no second agent needed.
        TargetPipeline.ARTICLE_EXTRACTION: MultiAgentPattern.NONE,
        # structured_scrape: Same spine as article_extraction; single tool call.
        TargetPipeline.STRUCTURED_SCRAPE: MultiAgentPattern.NONE,
        # vce (video coaching extraction): Multi-stage (transcribe→segment→extract→
        # score) with deterministic gates between stages; the orchestration IS the
        # pipeline.
        TargetPipeline.VCE: MultiAgentPattern.CUSTOM_PLANNER,
        # idea_capture: Single-step capture; no agency required.
        TargetPipeline.IDEA_CAPTURE: MultiAgentPattern.NONE,
        # url_snapshot: Deterministic, single FunctionTool worth of work; no agent
        # loop needed.
        TargetPipeline.URL_SNAPSHOT: MultiAgentPattern.NONE,
        # --- Active pipeline ---------------------------------------------------
        # book_rag: Deterministic ingest pipeline; stays non-agentic.
        TargetPipeline.BOOK_RAG: MultiAgentPattern.NONE,
    }
)

__all__ = [
    "MultiAgentPattern",
    "PIPELINE_PATTERN",
]
