"""LLM-driven synthetic query generation for the 300-query benchmark.

Mirrors the DSI extraction pattern: frozen prompt + injectable llm_call +
per-query validation.  No live LLM calls inside this module — llm_call is
always injected by the caller.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from pydantic import model_validator

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.common import VoBaseModel
from vo_rag.pipelines.book_rag.benchmark_schema import (
    CitationGranularity,
    PilotQuery,
    QueryClass,
)
from vo_rag.pipelines.book_rag.synthetic_query_prompts import (
    SYNTHETIC_QUERY_PROMPT_HASH,
    SYNTHETIC_QUERY_SYSTEM_PROMPT,
    SYNTHETIC_QUERY_USER_TEMPLATE,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Mapping from LLM-output query_type → PilotQuery.query_class_primary
# ---------------------------------------------------------------------------

_QUERY_TYPE_TO_CLASS: dict[str, QueryClass] = {
    "factual": QueryClass.LOCAL_FACTUAL_LOOKUP,
    "conceptual": QueryClass.CHAPTER_SPANNING_SYNTHESIS,
    "comparative": QueryClass.CROSS_BOOK_THEMATIC,
    "procedural": QueryClass.LOCAL_FACTUAL_LOOKUP,
    "summarization": QueryClass.WHOLE_BOOK_SUMMARY,
}

# Chunk content is truncated to this length before insertion into the prompt
# to stay within context budgets.
_CHUNK_CONTENT_MAX_CHARS = 600


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SyntheticQueryConfig(VoBaseModel):
    """Runtime parameters for a synthetic query generation run.

    ``prompt_hash`` must match ``SYNTHETIC_QUERY_PROMPT_HASH``.  Any prompt
    edit changes the hash and surfaces a mismatch here, preventing stale
    prompts from being used silently.
    """

    summary_model: str = "qwen3.6-27b-fp8"
    temperature: float = 0.7  # higher for diversity
    max_tokens: int = 1500
    prompt_hash: str = SYNTHETIC_QUERY_PROMPT_HASH
    chunks_per_query: int = 3  # how many chunks to feed per generation call
    domain: str = "motorsport engineering"

    @model_validator(mode="after")
    def prompt_hash_must_match(self) -> SyntheticQueryConfig:
        if self.prompt_hash != SYNTHETIC_QUERY_PROMPT_HASH:
            raise ValueError(
                f"prompt_hash mismatch: expected {SYNTHETIC_QUERY_PROMPT_HASH!r},"
                f" got {self.prompt_hash!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


class GeneratedQueryRecord(VoBaseModel):
    """Result of one synthetic query generation call."""

    query: PilotQuery | None
    validation_errors: list[str]
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    wall_time_ms: int


# ---------------------------------------------------------------------------
# Chunk selection
# ---------------------------------------------------------------------------


def select_chunks_for_query_generation(
    *,
    all_chunks: list[RetrievalChunk],
    n: int = 3,
    rng: random.Random,
) -> list[RetrievalChunk]:
    """Sample *n* chunks from the corpus with section_path diversity.

    Prefers chunks from distinct section_paths so the generated query is more
    likely to require cross-section reasoning.  Falls back to uniform sampling
    when the corpus is smaller than *n*.

    Parameters
    ----------
    all_chunks:
        Full corpus of available chunks.
    n:
        Number of chunks to select for one generation call.
    rng:
        Seeded ``random.Random`` instance — the caller owns the seed for
        reproducibility.
    """
    if len(all_chunks) <= n:
        return list(all_chunks)

    # Group by section_path to maximise diversity.
    by_section: dict[str, list[RetrievalChunk]] = {}
    for chunk in all_chunks:
        by_section.setdefault(chunk.section_path, []).append(chunk)

    sections = list(by_section.keys())

    # Pick up to n distinct sections, then one chunk from each.
    chosen_sections = rng.sample(sections, min(n, len(sections)))
    selected: list[RetrievalChunk] = []
    for section in chosen_sections:
        selected.append(rng.choice(by_section[section]))
        if len(selected) >= n:
            break

    # If we still need more (fewer distinct sections than n), fill from the
    # remaining pool without repeating already-selected chunk_ids.
    if len(selected) < n:
        selected_ids = {c.chunk_id for c in selected}
        remaining = [c for c in all_chunks if c.chunk_id not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _format_chunk_block(chunks: list[RetrievalChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        content = chunk.content
        if len(content) > _CHUNK_CONTENT_MAX_CHARS:
            content = content[:_CHUNK_CONTENT_MAX_CHARS] + "…"
        parts.append(f"[chunk_id={chunk.chunk_id}] [section_path={chunk.section_path}]\n{content}")
    return "\n\n---\n\n".join(parts)


def build_query_generation_prompt(
    *,
    chunks: list[RetrievalChunk],
    domain: str = "motorsport engineering",
) -> tuple[str, str]:
    """Return *(system, user)* prompt strings for one query generation call.

    Parameters
    ----------
    chunks:
        The chunks selected for this generation call (typically 3).
    domain:
        The subject domain to reference in the prompt.
    """
    chunk_contents = _format_chunk_block(chunks)
    system = SYNTHETIC_QUERY_SYSTEM_PROMPT.format(domain=domain)
    user = SYNTHETIC_QUERY_USER_TEMPLATE.format(chunk_contents=chunk_contents, domain=domain)
    return system, user


# ---------------------------------------------------------------------------
# Response parser / validator
# ---------------------------------------------------------------------------

_REQUIRED_BOOL_FIELDS = (
    "requires_multi_chunk",
    "requires_multi_book",
    "requires_graph",
    "requires_figure",
    "requires_equation",
    "requires_definition",
)

_VALID_QUERY_TYPES = frozenset(
    {"factual", "conceptual", "comparative", "procedural", "summarization"}
)
_VALID_ANSWER_TYPES = frozenset({"short_fact", "paragraph", "list", "formula", "walkthrough"})
_VALID_AMBIGUITY_LEVELS = frozenset({"low", "medium", "high"})


def _coerce_bool(value: Any, field: str, errors: list[str]) -> bool | None:
    """Coerce a JSON value to bool; append to errors and return None on failure."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    errors.append(f"{field}: expected bool, got {type(value).__name__!r} ({value!r})")
    return None


def parse_query_response(
    *,
    llm_response_text: str,
    known_chunk_ids: set[str],
    prompt_hash: str,
) -> GeneratedQueryRecord:
    """Parse and validate a raw LLM JSON response for one synthetic query.

    Validates:
    - All 13 fields present with correct types.
    - ``evidence_chunk_ids`` ⊆ ``known_chunk_ids``.

    On any failure, returns a record with ``query=None`` and the full list of
    validation errors.
    """
    errors: list[str] = []

    # ── JSON parse ──────────────────────────────────────────────────────────
    try:
        data: dict[str, Any] = json.loads(llm_response_text)
    except json.JSONDecodeError as exc:
        return GeneratedQueryRecord(
            query=None,
            validation_errors=[f"JSON decode error: {exc}"],
            prompt_hash=prompt_hash,
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    if not isinstance(data, dict):
        return GeneratedQueryRecord(
            query=None,
            validation_errors=["Response root must be a JSON object"],
            prompt_hash=prompt_hash,
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    # ── Field 1: query_text ─────────────────────────────────────────────────
    query_text = data.get("query_text", "")
    if not isinstance(query_text, str) or not query_text.strip():
        errors.append("query_text: must be a non-empty string")

    # ── Field 2: query_type ─────────────────────────────────────────────────
    query_type = data.get("query_type", "")
    if query_type not in _VALID_QUERY_TYPES:
        errors.append(
            f"query_type: must be one of {sorted(_VALID_QUERY_TYPES)!r}, got {query_type!r}"
        )

    # ── Field 3: expected_answer_type ────────────────────────────────────────
    expected_answer_type = data.get("expected_answer_type", "")
    if expected_answer_type not in _VALID_ANSWER_TYPES:
        errors.append(
            f"expected_answer_type: must be one of"
            f" {sorted(_VALID_ANSWER_TYPES)!r},"
            f" got {expected_answer_type!r}"
        )

    # ── Field 4: ambiguity_level ─────────────────────────────────────────────
    ambiguity_level = data.get("ambiguity_level", "")
    if ambiguity_level not in _VALID_AMBIGUITY_LEVELS:
        errors.append(
            f"ambiguity_level: must be one of"
            f" {sorted(_VALID_AMBIGUITY_LEVELS)!r},"
            f" got {ambiguity_level!r}"
        )

    # ── Fields 5-10: bool flags ──────────────────────────────────────────────
    bool_values: dict[str, bool | None] = {}
    for field in _REQUIRED_BOOL_FIELDS:
        raw = data.get(field)
        if raw is None:
            errors.append(f"{field}: missing required field")
            bool_values[field] = None
        else:
            bool_values[field] = _coerce_bool(raw, field, errors)

    # ── Field 11: domain ────────────────────────────────────────────────────
    domain = data.get("domain", "")
    if not isinstance(domain, str) or not domain.strip():
        errors.append("domain: must be a non-empty string")

    # ── Field 12: difficulty ────────────────────────────────────────────────
    difficulty = data.get("difficulty")
    if not isinstance(difficulty, (int, float)) or not (1 <= float(difficulty) <= 5):
        errors.append(f"difficulty: must be a number in [1, 5], got {difficulty!r}")

    # ── Field 13: evidence_chunk_ids ─────────────────────────────────────────
    evidence_chunk_ids_raw = data.get("evidence_chunk_ids", [])
    if not isinstance(evidence_chunk_ids_raw, list):
        errors.append("evidence_chunk_ids: must be a list")
        evidence_chunk_ids_raw = []

    evidence_chunk_ids: list[str] = []
    for i, cid in enumerate(evidence_chunk_ids_raw):
        if not isinstance(cid, str):
            errors.append(f"evidence_chunk_ids[{i}]: must be a string, got {type(cid).__name__!r}")
        else:
            evidence_chunk_ids.append(cid)

    # ── Guard: evidence_chunk_ids ⊆ known_chunk_ids ──────────────────────────
    unknown = [cid for cid in evidence_chunk_ids if cid not in known_chunk_ids]
    if unknown:
        for bad_id in unknown:
            errors.append(
                f"evidence_chunk_ids: chunk_id {bad_id!r} is not in the"
                " known_chunk_ids for this generation call"
            )

    if errors:
        return GeneratedQueryRecord(
            query=None,
            validation_errors=errors,
            prompt_hash=prompt_hash,
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    # ── Map to PilotQuery ────────────────────────────────────────────────────
    query_class_primary = _QUERY_TYPE_TO_CLASS.get(query_type, QueryClass.LOCAL_FACTUAL_LOOKUP)
    requires_referential = ambiguity_level == "high"
    requires_cross_section = bool(bool_values["requires_multi_chunk"])
    requires_book_routing = bool(bool_values["requires_multi_book"])
    requires_figure = bool(bool_values["requires_figure"])
    requires_entity_rel = False

    # Apply class-flag consistency required by PilotQuery validators.
    if query_class_primary == QueryClass.ENTITY_RELATIONSHIP:
        requires_entity_rel = True
    if query_class_primary == QueryClass.FIGURE_DEPENDENT:
        requires_figure = True
    if query_class_primary == QueryClass.INTRA_SECTION_REFERENTIAL_AMBIGUITY:
        requires_referential = True
    requires_summary = query_class_primary == QueryClass.WHOLE_BOOK_SUMMARY
    if query_class_primary in (
        QueryClass.CHAPTER_SPANNING_SYNTHESIS,
        QueryClass.CROSS_BOOK_THEMATIC,
    ):
        requires_cross_section = True

    try:
        pilot_query = PilotQuery(
            query_id=str(uuid.uuid4()),
            query_text=query_text.strip(),
            query_class_primary=query_class_primary,
            requires_cross_section_concept_aggregation=requires_cross_section,
            requires_book_routing=requires_book_routing,
            requires_figure_evidence=requires_figure,
            requires_entity_relationship_reasoning=requires_entity_rel,
            requires_referential_disambiguation=requires_referential,
            requires_summary_coverage=requires_summary,
            expected_answer=expected_answer_type,
            expected_supporting_chunk_ids=evidence_chunk_ids,
            required_citation_granularity=CitationGranularity.CHUNK,
        )
    except Exception as exc:
        return GeneratedQueryRecord(
            query=None,
            validation_errors=[f"PilotQuery construction error: {exc}"],
            prompt_hash=prompt_hash,
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    return GeneratedQueryRecord(
        query=pilot_query,
        validation_errors=[],
        prompt_hash=prompt_hash,
        input_tokens=0,
        output_tokens=0,
        wall_time_ms=0,
    )


# ---------------------------------------------------------------------------
# Single query generation (async)
# ---------------------------------------------------------------------------


async def generate_query(
    *,
    all_chunks: list[RetrievalChunk],
    config: SyntheticQueryConfig,
    rng: random.Random,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
) -> GeneratedQueryRecord:
    """Generate one synthetic evaluation query.

    Steps:
    1. Select ``config.chunks_per_query`` chunks with section diversity.
    2. Build (system, user) generation prompts.
    3. Invoke *llm_call* (injected — never a live call inside this module).
    4. Parse and validate the LLM response.
    5. Return a ``GeneratedQueryRecord``.
    """
    t0 = time.monotonic()

    chunks = select_chunks_for_query_generation(
        all_chunks=all_chunks,
        n=config.chunks_per_query,
        rng=rng,
    )
    known_chunk_ids = {c.chunk_id for c in all_chunks}
    system, user = build_query_generation_prompt(chunks=chunks, domain=config.domain)

    llm_response: LLMResponse = await llm_call(
        system,
        user,
        config.max_tokens,
        config.temperature,
    )

    wall_time_ms = int((time.monotonic() - t0) * 1000)

    record = parse_query_response(
        llm_response_text=llm_response.text,
        known_chunk_ids=known_chunk_ids,
        prompt_hash=config.prompt_hash,
    )

    # Overlay token + timing accounting from the live llm_call.
    record = GeneratedQueryRecord(
        query=record.query,
        validation_errors=record.validation_errors,
        prompt_hash=record.prompt_hash,
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
        wall_time_ms=wall_time_ms,
    )

    log.info(
        "synthetic_query.generate.complete",
        summary_model=config.summary_model,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        wall_time_ms=record.wall_time_ms,
        validation_ok=record.query is not None,
        n_validation_errors=len(record.validation_errors),
    )

    return record


# ---------------------------------------------------------------------------
# Batch generation (async, bounded concurrency)
# ---------------------------------------------------------------------------


async def generate_query_batch(
    *,
    all_chunks: list[RetrievalChunk],
    n_queries: int,
    config: SyntheticQueryConfig,
    seed: int,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
    max_concurrent: int = 4,
) -> list[GeneratedQueryRecord]:
    """Generate *n_queries* synthetic evaluation queries with bounded concurrency.

    A single seeded ``random.Random`` instance is created for the entire batch
    to ensure reproducibility.  Results are returned in generation-index order.

    Parameters
    ----------
    all_chunks:
        Full corpus of available chunks.
    n_queries:
        Total number of queries to generate.
    config:
        Runtime parameters (model, temperature, prompt_hash, …).
    seed:
        RNG seed for reproducibility.
    llm_call:
        Injectable async callable — the caller is responsible for providing a
        real or mock implementation.
    max_concurrent:
        Maximum number of simultaneous LLM calls.
    """
    rng = random.Random(seed)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _one(index: int) -> tuple[int, GeneratedQueryRecord]:
        async with semaphore:
            record = await generate_query(
                all_chunks=all_chunks,
                config=config,
                rng=rng,
                llm_call=llm_call,
            )
        return index, record

    tasks = [asyncio.create_task(_one(i)) for i in range(n_queries)]
    try:
        indexed_results: list[tuple[int, GeneratedQueryRecord]] = await asyncio.gather(*tasks)
    except (asyncio.CancelledError, Exception) as exc:
        for task in tasks:
            if not task.done():
                task.cancel()
        cleanup_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in cleanup_results:
            if result is exc:
                continue
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                log.warning(
                    "book_rag.synthetic_query_straggler_failed",
                    error=str(result),
                    error_type=type(result).__name__,
                )
        raise

    # Sort by original generation index to preserve order.
    indexed_results.sort(key=lambda pair: pair[0])
    return [record for _, record in indexed_results]


__all__ = [
    "SyntheticQueryConfig",
    "GeneratedQueryRecord",
    "select_chunks_for_query_generation",
    "build_query_generation_prompt",
    "parse_query_response",
    "generate_query",
    "generate_query_batch",
]
