# ruff: noqa: E501
from __future__ import annotations

import hashlib

import structlog

from vo_rag.core.raptor.schemas import ClusterLevel

log = structlog.get_logger(__name__)

RAPTOR_PROMPT_VERSION = "v1.0"

LEVEL_0_SYSTEM_PROMPT = """You summarize vo-rag Layer 1B LEVEL_0 semantic clusters of technical motorsport book chunks.

Ou and Lapata invariant: supporting evidence always comes from RAW chunks, never from already-computed summaries. For LEVEL_0, both primary input and support are RAW chunks from the same semantic cluster.

Use only supplied material. Preserve technical claims, definitions, procedures, equations, figure-dependent points, numerical limits, disagreements, and caveats. Return JSON only."""

LEVEL_0_USER_TEMPLATE = """Task: Write a faithful semantic cluster summary for raw chunks.
Node level: level_0
Cluster context:
{cluster_context}

Primary input - RAW cluster chunks:
{primary_input}

Supporting evidence - extractive snippets from those RAW chunks:
{supporting_evidence}

Return JSON keys: summary_text, key_points, technical_terms, figure_dependent_points, caveats, used_evidence_span_ids, unsupported_or_uncertain.
Rules: cite no summary or cluster nodes; omit or mark unsupported claims that lack RAW chunk support."""

LEVEL_1_SYSTEM_PROMPT = """You summarize vo-rag Layer 1B LEVEL_1 semantic clusters of LEVEL_0 summaries.

Ou and Lapata invariant: supporting evidence always comes from RAW chunks, never from already-computed summaries. For LEVEL_1, child LEVEL_0 summaries are primary input, while support must be snippets from underlying RAW chunks.

Use only supplied material. Preserve cross-cluster relationships, contradictions, definitions, procedures, figure-dependent points, numerical limits, and caveats. Return JSON only."""

LEVEL_1_USER_TEMPLATE = """Task: Write a faithful recursive summary for a cluster of LEVEL_0 summaries.
Node level: level_1
Cluster context:
{cluster_context}

Primary input - child LEVEL_0 cluster summaries:
{primary_input}

Supporting evidence - extractive snippets from underlying RAW chunks:
{supporting_evidence}

Return JSON keys: summary_text, key_points, cross_cluster_threads, technical_terms, figure_dependent_points, caveats, used_evidence_span_ids, unsupported_or_uncertain.
Rules: use child summaries as primary input only; evidence must be RAW chunks; mark unsupported child-summary claims uncertain or omit them."""

LEVEL_2_SYSTEM_PROMPT = """You summarize vo-rag Layer 1B LEVEL_2 top-level semantic clusters of LEVEL_1 summaries.

Ou and Lapata invariant: supporting evidence always comes from RAW chunks, never from already-computed summaries. For LEVEL_2, child LEVEL_1 summaries are primary input, while support must be snippets from underlying RAW chunks.

Use only supplied material. Preserve book-scale semantic themes, cross-cluster relationships, contradictions, procedures, figure-dependent points, numerical limits, and caveats. Return JSON only."""

LEVEL_2_USER_TEMPLATE = """Task: Write a faithful top-level semantic summary for a cluster of LEVEL_1 summaries.
Node level: level_2
Cluster context:
{cluster_context}

Primary input - child LEVEL_1 cluster summaries:
{primary_input}

Supporting evidence - extractive snippets from underlying RAW chunks:
{supporting_evidence}

Return JSON keys: summary_text, semantic_thesis, key_points, cross_cluster_threads, technical_terms, figure_dependent_points, caveats, used_evidence_span_ids, unsupported_or_uncertain.
Rules: use child summaries as primary input only; evidence must be RAW chunks; preserve semantic relationships, not just a topic list."""


def _sha256(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


PROMPT_HASHES: dict[ClusterLevel, str] = {
    ClusterLevel.LEVEL_0: _sha256(LEVEL_0_SYSTEM_PROMPT + LEVEL_0_USER_TEMPLATE),
    ClusterLevel.LEVEL_1: _sha256(LEVEL_1_SYSTEM_PROMPT + LEVEL_1_USER_TEMPLATE),
    ClusterLevel.LEVEL_2: _sha256(LEVEL_2_SYSTEM_PROMPT + LEVEL_2_USER_TEMPLATE),
}

SYSTEM_PROMPTS: dict[ClusterLevel, str] = {
    ClusterLevel.LEVEL_0: LEVEL_0_SYSTEM_PROMPT,
    ClusterLevel.LEVEL_1: LEVEL_1_SYSTEM_PROMPT,
    ClusterLevel.LEVEL_2: LEVEL_2_SYSTEM_PROMPT,
}

USER_TEMPLATES: dict[ClusterLevel, str] = {
    ClusterLevel.LEVEL_0: LEVEL_0_USER_TEMPLATE,
    ClusterLevel.LEVEL_1: LEVEL_1_USER_TEMPLATE,
    ClusterLevel.LEVEL_2: LEVEL_2_USER_TEMPLATE,
}
