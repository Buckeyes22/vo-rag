"""DSI extraction pipeline (per-book LLM extraction + JSON parse).

Contract:

- ``_OPTIONAL_PROVENANCE_FIELDS`` for ``key_equations_or_models`` and
  ``figure_heavy_topics`` — genre-contingent facets where empty is valid data
  (a circuit atlas legitimately has no equations).
- Malformed-entry drop: one bad optional field is silently dropped, never
  rejects the whole record.
- ``supporting_section_paths`` are derived from cited
  chunks via ``chunk_section_map`` — the LLM-emitted ``section_path`` is
  IGNORED. This removes the section-path hallucination surface entirely.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from typing import Any

import structlog

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.dsi.prompts import DSI_EXTRACTION_SYSTEM_PROMPT, DSI_EXTRACTION_USER_TEMPLATE
from vo_rag.core.dsi.schemas import FacetedBookRecord, ProvenanceLinkedEntry
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.summary.extract_support import select_evidence_snippets

log = structlog.get_logger(__name__)

_CHUNK_CONTENT_MAX_CHARS = 600
_PROVENANCE_LINKED_FIELDS = (
    "key_concepts",
    "key_methods",
    "key_equations_or_models",
    "figure_heavy_topics",
    "synonyms_and_aliases",
)

# Genre-contingent provenance facets: a book can legitimately have NONE of these
# (a circuit atlas / racing biography / rulebook has no equations; a text-only
# book has no figure-heavy topics). For these, an empty list is valid data — NOT
# a rejection. The remaining provenance fields (key_concepts, key_methods,
# synonyms_and_aliases) stay hard-required. Rejecting a circuit guide for empty
# key_equations_or_models + figure_heavy_topics would drop a valid book.
_OPTIONAL_PROVENANCE_FIELDS = frozenset({"key_equations_or_models", "figure_heavy_topics"})


class ExtractionConfig(VoBaseModel):
    summary_model: str = "qwen3.6-27b-fp8"
    temperature: float = 0.2
    max_tokens: int = 4000
    n_representative_chunks: int = 20
    prompt_hash: str


class ExtractionResult(VoBaseModel):
    book_id: str
    record: FacetedBookRecord | None
    validation_errors: list[str]
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    wall_time_ms: int


def select_representative_chunks(
    *,
    chunks: list[RetrievalChunk],
    n: int = 20,
) -> list[RetrievalChunk]:
    """Return the top-n representative chunks for DSI extraction.

    Uses ``select_evidence_snippets`` from ``vo_rag.core.summary.extract_support``
    to score candidates, then de-duplicates by chunk_id and returns the top-n
    source chunks sorted by score descending. If ``len(chunks) <= n``, all chunks
    are returned.
    """
    if len(chunks) <= n:
        return list(chunks)

    spans = select_evidence_snippets(
        chunks,
        topic_or_query="technical concepts methods equations figures key terms",
        k=len(chunks),
        method="hybrid",
    )

    seen: set[str] = set()
    chunk_by_id: dict[str, RetrievalChunk] = {c.chunk_id: c for c in chunks}
    ordered: list[RetrievalChunk] = []
    for span in spans:
        if span.chunk_id not in seen and span.chunk_id in chunk_by_id:
            seen.add(span.chunk_id)
            ordered.append(chunk_by_id[span.chunk_id])
        if len(ordered) >= n:
            break
    return ordered


def _format_chunk_block(chunks: list[RetrievalChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        content = chunk.content
        if len(content) > _CHUNK_CONTENT_MAX_CHARS:
            content = content[:_CHUNK_CONTENT_MAX_CHARS] + "…"
        parts.append(f"[chunk_id={chunk.chunk_id}] [section_path={chunk.section_path}]\n{content}")
    return "\n\n---\n\n".join(parts)


def build_extraction_prompt(
    *,
    book_id: str,
    title: str,
    author: str,
    category_ids: list[str],
    representative_chunks: list[RetrievalChunk],
) -> tuple[str, str]:
    """Return ``(system, user)`` prompt strings for DSI extraction."""
    chunk_contents = _format_chunk_block(representative_chunks)
    user = DSI_EXTRACTION_USER_TEMPLATE.format(
        title=title,
        author=author,
        category_ids=", ".join(category_ids),
        chunk_contents=chunk_contents,
    )
    return DSI_EXTRACTION_SYSTEM_PROMPT, user


def _parse_provenance_linked(
    raw_entries: Any,
    field_name: str,
    errors: list[str],
    chunk_section_map: dict[str, str],
    allow_empty: bool = False,
) -> list[ProvenanceLinkedEntry] | None:
    """Parse provenance-linked entries, deriving section_path from the cited chunk.

    Contract: each entry carries a non-empty ``value`` and a
    ``chunk_id`` copied verbatim from a representative chunk. The section_path is
    NOT taken from the LLM — it is derived from the cited chunk via
    ``chunk_section_map`` (ground truth). This removes the section-path
    hallucination surface entirely (the failure mode that produced 27 violations
    on r-408: the LLM invented section paths while citing valid chunk_ids).

    Resilience: an entry with an empty value, a missing/non-string chunk_id, or a
    chunk_id that doesn't resolve in ``chunk_section_map`` is DROPPED (logged),
    not fatal — a few malformed entries must not reject an otherwise-good record.

    ``allow_empty``: for genre-contingent facets (see ``_OPTIONAL_PROVENANCE_FIELDS``)
    an empty/absent list is VALID data, not a failure — a circuit atlas legitimately
    has no equations, a text-only book no figure-heavy topics. Such fields return ``[]``
    instead of appending a rejection error. Required facets keep ``allow_empty=False``
    and still demand at least one surviving entry.
    """
    if not isinstance(raw_entries, list) or len(raw_entries) == 0:
        if allow_empty:
            return []
        errors.append(f"{field_name}: must be a non-empty list")
        return None

    result: list[ProvenanceLinkedEntry] = []
    dropped = 0
    for entry in raw_entries:
        if not isinstance(entry, dict):
            dropped += 1
            continue
        value = str(entry.get("value", entry.get("name", ""))).strip()
        chunk_id = entry.get("chunk_id")
        if not value or not isinstance(chunk_id, str) or chunk_id not in chunk_section_map:
            dropped += 1
            continue
        result.append(
            ProvenanceLinkedEntry(
                value=value,
                supporting_chunk_ids=[chunk_id],
                supporting_section_paths=[chunk_section_map[chunk_id]],
            )
        )

    if dropped:
        log.info(
            "dsi.parse.dropped_entries",
            field=field_name,
            dropped=dropped,
            kept=len(result),
        )
    if not result:
        if allow_empty:
            return []
        errors.append(
            f"{field_name}: no valid provenance entries (empty values or unresolvable chunk_ids)"
        )
        return None
    return result


def _coerce_str_list(raw: object) -> list[str]:
    """Coerce plain-string list fields while tolerating provenance-like items."""
    if raw is None:
        return []
    if not isinstance(raw, Iterable):
        raise TypeError(f"{type(raw).__name__} object is not iterable")

    result: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            value = item.get("value")
            if value is None:
                continue
            result.append(str(value))
        else:
            result.append(str(item))
    return result


_REQUIRED_SCALAR_FIELDS = (
    "short_abstract",
    "technical_domains",
    "key_components",
    "prerequisite_topics",
    "chapter_outline",
)


def parse_extraction_response(
    *,
    book_id: str,
    title: str,
    author: str,
    category_ids: list[str],
    llm_response_json: str,
    summary_model: str,
    prompt_hash: str,
    source_collection_version: str,
    chunk_section_map: dict[str, str],
    representative_chunk_ids: list[str],
) -> ExtractionResult:
    """Parse and validate a raw JSON response from the extraction LLM.

    Returns ``ExtractionResult`` with ``record=None`` and populated
    ``validation_errors`` on any missing field, type error, or schema violation.
    On success, ``record`` is a fully-populated ``FacetedBookRecord``.
    """
    errors: list[str] = []

    try:
        data: dict[str, Any] = json.loads(llm_response_json)
    except json.JSONDecodeError as exc:
        return ExtractionResult(
            book_id=book_id,
            record=None,
            validation_errors=[f"JSON decode error: {exc}"],
            prompt_hash=prompt_hash,
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    if not isinstance(data, dict):
        return ExtractionResult(
            book_id=book_id,
            record=None,
            validation_errors=["Response root must be a JSON object"],
            prompt_hash=prompt_hash,
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    # Validate required scalar / list fields.
    for field in _REQUIRED_SCALAR_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Validate provenance-linked fields.
    parsed_provenance: dict[str, list[ProvenanceLinkedEntry]] = {}
    for field in _PROVENANCE_LINKED_FIELDS:
        optional = field in _OPTIONAL_PROVENANCE_FIELDS
        if field not in data:
            if optional:
                parsed_provenance[field] = []
                continue
            errors.append(f"Missing required field: {field}")
            continue
        entries = _parse_provenance_linked(
            data[field], field, errors, chunk_section_map, allow_empty=optional
        )
        if entries is not None:
            parsed_provenance[field] = entries

    if errors:
        return ExtractionResult(
            book_id=book_id,
            record=None,
            validation_errors=errors,
            prompt_hash=prompt_hash,
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    # Build FacetedBookRecord.
    try:
        record = FacetedBookRecord(
            book_id=book_id,
            title=title,
            author=author,
            category_ids=category_ids,
            short_abstract=str(data["short_abstract"]),
            technical_domains=_coerce_str_list(data["technical_domains"]),
            key_concepts=parsed_provenance["key_concepts"],
            key_components=_coerce_str_list(data["key_components"]),
            key_methods=parsed_provenance["key_methods"],
            key_equations_or_models=parsed_provenance["key_equations_or_models"],
            figure_heavy_topics=parsed_provenance["figure_heavy_topics"],
            synonyms_and_aliases=parsed_provenance["synonyms_and_aliases"],
            prerequisite_topics=_coerce_str_list(data["prerequisite_topics"]),
            chapter_outline=str(data["chapter_outline"]),
            representative_chunk_ids=list(representative_chunk_ids),
            source_collection_version=source_collection_version,
            summary_model=summary_model,
            prompt_hash=prompt_hash,
            created_at=datetime.now(UTC),
        )
    except Exception as exc:
        return ExtractionResult(
            book_id=book_id,
            record=None,
            validation_errors=[f"FacetedBookRecord construction error: {exc}"],
            prompt_hash=prompt_hash,
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    return ExtractionResult(
        book_id=book_id,
        record=record,
        validation_errors=[],
        prompt_hash=prompt_hash,
        input_tokens=0,
        output_tokens=0,
        wall_time_ms=0,
    )


async def extract_book(
    *,
    book_id: str,
    title: str,
    author: str,
    category_ids: list[str],
    chunks: list[RetrievalChunk],
    config: ExtractionConfig,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
    source_collection_version: str,
) -> ExtractionResult:
    """Run the full per-book DSI extraction pipeline.

    Steps:
      1. Select representative chunks via scoring.
      2. Build (system, user) extraction prompts.
      3. Invoke ``llm_call`` (injectable — never a live call inside this module).
      4. Parse + validate the LLM response.
      5. Log per-book token and wall-time accounting.
    """
    t0 = time.monotonic()

    representative = select_representative_chunks(
        chunks=chunks,
        n=config.n_representative_chunks,
    )

    system, user = build_extraction_prompt(
        book_id=book_id,
        title=title,
        author=author,
        category_ids=category_ids,
        representative_chunks=representative,
    )

    llm_response: LLMResponse = await llm_call(
        system,
        user,
        config.max_tokens,
        config.temperature,
    )

    wall_time_ms = int((time.monotonic() - t0) * 1000)

    # Ground-truth section paths come from the chunks the LLM was shown, keyed by
    # chunk_id — the LLM cites chunk_ids and we derive section_path from these.
    chunk_section_map = {c.chunk_id: c.section_path for c in representative}

    result = parse_extraction_response(
        book_id=book_id,
        title=title,
        author=author,
        category_ids=category_ids,
        llm_response_json=llm_response.text,
        summary_model=config.summary_model,
        prompt_hash=config.prompt_hash,
        source_collection_version=source_collection_version,
        chunk_section_map=chunk_section_map,
        representative_chunk_ids=[c.chunk_id for c in representative],
    )

    result = ExtractionResult(
        book_id=result.book_id,
        record=result.record,
        validation_errors=result.validation_errors,
        prompt_hash=result.prompt_hash,
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
        wall_time_ms=wall_time_ms,
    )

    log.info(
        "dsi.extract_book.complete",
        book_id=book_id,
        summary_model=config.summary_model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        wall_time_ms=result.wall_time_ms,
        validation_ok=result.record is not None,
        n_validation_errors=len(result.validation_errors),
        n_representative_chunks=len(representative),
    )

    return result


__all__ = [
    "ExtractionConfig",
    "ExtractionResult",
    "select_representative_chunks",
    "build_extraction_prompt",
    "parse_extraction_response",
    "extract_book",
]
