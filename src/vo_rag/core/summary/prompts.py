# ruff: noqa: E501
"""Summary prompt templates and version pinning.

Prompt text and the resulting ``PROMPT_HASHES`` are stable (the SHA-256 of
(system + user) template is the ``prompt_hash`` that flows into
``node_id_for``). To change the prose, bump ``SUMMARY_PROMPT_VERSION``
and update the templates in one atomic commit.
"""

from __future__ import annotations

import hashlib

import structlog

from vo_rag.core.summary.schemas import NodeLevel

log = structlog.get_logger(__name__)
SUMMARY_PROMPT_VERSION = "v1.1"
SUMMARY_MODEL_DEFAULT = "Qwen3.6-27B-FP8"
SUMMARY_TEMPERATURE_DEFAULT = 0.2
SUMMARY_MAX_TOKENS_BY_LEVEL: dict[NodeLevel, int] = {
    NodeLevel.SECTION: 800,
    NodeLevel.CHAPTER: 1200,
    NodeLevel.BOOK: 2000,
}

SECTION_SUMMARY_SYSTEM_PROMPT = """You summarize technical motorsport book sections for vo-rag Layer 1A structural retrieval.

Use Ou & Lapata Extract-Support discipline. For SECTION nodes, the primary input is RAW chunks from the section, and the supporting evidence is extractive snippets selected from those same RAW chunks. The recursion invariant is strict: supporting evidence is quoted from RAW chunks, never paraphrased, and never taken from an already-summarized node.

Use only the provided primary input and supporting evidence. Preserve main technical claims, definitions, procedures, equations, figure-dependent points, disagreements, numerical limits, and caveats. Do not add outside knowledge. Do not cite summary nodes.

Target generation defaults: model Qwen3.6-27B-FP8 via RunPod Serverless, temperature 0.2, max_tokens 800.
Return JSON only."""

SECTION_SUMMARY_USER_TEMPLATE = """Task: Write a faithful structural summary for one section.

Node level: section
Section breadcrumb: {section_breadcrumb}

Primary input - RAW chunks from this section:
{primary_input}

Supporting evidence - Extract-Support snippets from the same RAW chunks:
{supporting_evidence}

Output JSON schema:
{{
  "summary_text": "faithful section summary",
  "key_points": ["major point 1", "major point 2"],
  "figure_dependent_points": ["point requiring figure evidence"],
  "technical_terms": ["term"],
  "caveats": ["limitation or warning"],
  "used_evidence_span_ids": ["chunk_id or evidence identifier"],
  "unsupported_or_uncertain": ["claim that could not be supported, if any"]
}}

Rules:
- Primary input is the section's RAW chunks.
- Supporting evidence must be treated as quote-backed support from RAW chunks.
- Do not use already-summarized text as evidence.
- If a claim is not supported by the supplied material, omit it or list it under unsupported_or_uncertain."""

CHAPTER_SUMMARY_SYSTEM_PROMPT = """You summarize technical motorsport book chapters for vo-rag Layer 1A structural retrieval.

Use Ou & Lapata Extract-Support discipline. For CHAPTER nodes, the primary input is child section summaries, but the supporting evidence must be extractive snippets from the underlying RAW chunks. The recursion invariant is strict: recursive summaries organize the input; RAW chunks ground the support. Never use child summaries as evidence, and never quote from already-summarized text.

Use only the provided primary input and supporting evidence. Preserve cross-section relationships, contradictions, definitions, procedures, figure-dependent points, numerical limits, and caveats. Do not add outside knowledge. Do not cite summary nodes.

Target generation defaults: model Qwen3.6-27B-FP8 via RunPod Serverless, temperature 0.2, max_tokens 1200.
Return JSON only."""

CHAPTER_SUMMARY_USER_TEMPLATE = """Task: Write a faithful structural summary for one chapter.

Node level: chapter
Chapter breadcrumb: {section_breadcrumb}

Primary input - child section summaries:
{primary_input}

Supporting evidence - Extract-Support snippets from underlying RAW chunks:
{supporting_evidence}

Output JSON schema:
{{
  "summary_text": "faithful chapter summary",
  "key_points": ["major point 1", "major point 2"],
  "cross_section_threads": ["concept developed across multiple sections"],
  "figure_dependent_points": ["point requiring figure evidence"],
  "technical_terms": ["term"],
  "caveats": ["limitation or warning"],
  "used_evidence_span_ids": ["chunk_id or evidence identifier"],
  "unsupported_or_uncertain": ["claim that could not be supported, if any"]
}}

Rules:
- Child section summaries are the primary recursive input.
- Supporting evidence must come from original RAW chunks, not from section summaries.
- Preserve cross-section relationships and contradictions.
- If a child-summary claim lacks RAW chunk support, mark it uncertain or omit it."""

BOOK_SUMMARY_SYSTEM_PROMPT = """You summarize entire technical motorsport books for vo-rag Layer 1A structural retrieval.

Use Ou & Lapata Extract-Support discipline. For BOOK nodes, the primary input is child chapter summaries, but the supporting evidence must be extractive snippets from the underlying RAW chunks. The recursion invariant is strict: recursive summaries organize the input; RAW chunks ground the support. Never use chapter summaries as evidence, and never quote from already-summarized text.

Use only the provided primary input and supporting evidence. Preserve the book's structure, central technical thesis, chapter progression, cross-chapter relationships, figure-dependent points, numerical limits, disagreements, and caveats. Do not add outside knowledge. Do not cite summary nodes.

Target generation defaults: model Qwen3.6-27B-FP8 via RunPod Serverless, temperature 0.2, max_tokens 2000.
Return JSON only."""

BOOK_SUMMARY_USER_TEMPLATE = """Task: Write a faithful structural summary for one book.

Node level: book
Book breadcrumb: {section_breadcrumb}

Primary input - child chapter summaries:
{primary_input}

Supporting evidence - Extract-Support snippets from underlying RAW chunks:
{supporting_evidence}

Output JSON schema:
{{
  "summary_text": "faithful whole-book summary",
  "book_thesis": "central technical purpose of the book",
  "key_points": ["major point 1", "major point 2"],
  "chapter_progression": ["how the argument or instruction develops"],
  "cross_chapter_threads": ["concept developed across chapters"],
  "figure_dependent_points": ["point requiring figure evidence"],
  "technical_terms": ["term"],
  "caveats": ["limitation or warning"],
  "used_evidence_span_ids": ["chunk_id or evidence identifier"],
  "unsupported_or_uncertain": ["claim that could not be supported, if any"]
}}

Rules:
- Child chapter summaries are the primary recursive input.
- Supporting evidence must come from original RAW chunks, not from chapter summaries.
- Preserve the book's structure, not just a topic list.
- If a chapter-level claim lacks RAW chunk support, mark it uncertain or omit it."""


def _sha256(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


PROMPT_HASHES: dict[NodeLevel, str] = {
    NodeLevel.SECTION: _sha256(SECTION_SUMMARY_SYSTEM_PROMPT + SECTION_SUMMARY_USER_TEMPLATE),
    NodeLevel.CHAPTER: _sha256(CHAPTER_SUMMARY_SYSTEM_PROMPT + CHAPTER_SUMMARY_USER_TEMPLATE),
    NodeLevel.BOOK: _sha256(BOOK_SUMMARY_SYSTEM_PROMPT + BOOK_SUMMARY_USER_TEMPLATE),
}

SYSTEM_PROMPTS: dict[NodeLevel, str] = {
    NodeLevel.SECTION: SECTION_SUMMARY_SYSTEM_PROMPT,
    NodeLevel.CHAPTER: CHAPTER_SUMMARY_SYSTEM_PROMPT,
    NodeLevel.BOOK: BOOK_SUMMARY_SYSTEM_PROMPT,
}

USER_TEMPLATES: dict[NodeLevel, str] = {
    NodeLevel.SECTION: SECTION_SUMMARY_USER_TEMPLATE,
    NodeLevel.CHAPTER: CHAPTER_SUMMARY_USER_TEMPLATE,
    NodeLevel.BOOK: BOOK_SUMMARY_USER_TEMPLATE,
}
