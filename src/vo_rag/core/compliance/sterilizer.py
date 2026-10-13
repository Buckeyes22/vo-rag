"""Sterilization service for transforming raw retrieval chunks into sterile chunks.

Sterilization is LLM-only; there is no heuristic fallback mode. Every LLM call flows through
``complete_json_protected`` (F-012 injection trio).
"""

from __future__ import annotations

import time

import structlog
from pydantic import BaseModel, Field

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.compliance.license_filter import PUBLISHABLE
from vo_rag.core.idempotency import deterministic_uuid
from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.protected import complete_json_protected
from vo_rag.core.models.artifact import LicenseClass
from vo_rag.core.parsing.text_normalization import normalize_extracted_text

SYSTEM_PROMPT = """You sterilize copyrighted technical passages. Your output preserves FACTS and
DISCARDS EXPRESSION.

KEEP:
- Factual assertions (physics, engineering, data points)
- Mathematical relationships and equations (verbatim — math is not copyrightable)
- Named technical concepts and their definitions
- Procedural steps (re-expressed in neutral engineering language)

REMOVE:
- Author's prose style, metaphors, analogies
- Pedagogical framing ("as we saw in chapter 2", "imagine you are at a track")
- Narrative voice
- Illustrative examples that reproduce the author's creative expression

Return a JSON object:
{
  "sterile_content": "...",         // the rewritten factual summary
  "facts": ["..."],                  // each factual assertion
  "equations": ["..."],              // each equation (LaTeX where available)
  "concepts": ["..."]                // named concepts defined
}

NEVER reuse distinctive phrases from the input. If a sentence reads like
authorial voice, discard it.
"""


class SterilizerParseError(RuntimeError):
    """Raised when the LLM output cannot be parsed into required JSON."""


class _SterilizedPayload(BaseModel):
    sterile_content: str
    facts: list[str] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    injection_suspected: bool = False
    injection_reason: str | None = None


_MIN_VERBATIM_RUN = 15


def _check_verbatim_budget(original: str, sterile: str) -> tuple[bool, int, float]:
    """Return (ok, longest_run, ratio) for the verbatim budget (F-029).

    longest_run is the length of the longest contiguous substring of
    ``sterile`` that also appears in ``original``.  ratio is the fraction
    of ``sterile`` characters that are part of any such run of at least
    ``_MIN_VERBATIM_RUN`` characters.
    """
    if not sterile:
        return True, 0, 0.0

    n, m = len(original), len(sterile)
    longest_run = 0
    max_at_j: list[int] = [0] * (m + 1)

    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if original[i - 1] == sterile[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > longest_run:
                    longest_run = curr[j]
                if curr[j] > max_at_j[j]:
                    max_at_j[j] = curr[j]
        prev = curr

    verbatim = [False] * m
    for j in range(1, m + 1):
        run_len = max_at_j[j]
        if run_len >= _MIN_VERBATIM_RUN:
            for k in range(j - run_len, j):
                verbatim[k] = True

    total_verbatim = sum(verbatim)
    ratio = total_verbatim / m if m else 0.0
    ok = longest_run <= 100 and ratio <= 0.15
    return ok, longest_run, ratio


log = structlog.get_logger(__name__)


def _build_untrusted_chunk(chunk: RetrievalChunk, passage: str | None = None) -> str:
    normalized = normalize_extracted_text(chunk.content if passage is None else passage)
    return (
        f"source_id: {chunk.source_id}\n"
        f"parent_chunk_id: {chunk.chunk_id}\n"
        f"section_path: {chunk.section_path}\n\n"
        f"Passage:\n{normalized}"
    )


def _figure_window(chunk: RetrievalChunk) -> str | None:
    """A figure's neighbour window (its ``embedding_text`` without the path
    prefix) when it says more than ``content``; None for every other chunk."""
    if chunk.chunk_type != "figure_composite":
        return None
    prefix = f"{chunk.section_path}\n\n"
    window = chunk.embedding_text
    if window.startswith(prefix):
        window = window[len(prefix) :]
    window = window.strip()
    if not window or normalize_extracted_text(window) == normalize_extracted_text(chunk.content):
        return None
    return window


async def _sterile_window(
    chunk: RetrievalChunk, window: str, llm_client: LLMClient
) -> tuple[str | None, bool]:
    """Sterilize a figure's neighbour window for its sterile embedding (VOR-1452).

    The raw window is never stored or embedded on the sterile side: it gets the
    same protected call and verbatim budget as ``content``. Returns the sterile
    window (None on any failure: the caller falls back to embedding the sterile
    content, as before) and the protected call's quarantine verdict, which the
    caller records like the content call's -- the window is
    untrusted input of this chunk whether or not its sterile text is used.
    """
    normalized = normalize_extracted_text(window)
    try:
        parsed, _resp, _validation, should_quarantine = await complete_json_protected(
            llm_client,
            schema=_SterilizedPayload,
            untrusted_content=_build_untrusted_chunk(chunk, window),
            pipeline="book_rag",
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1024,
            allowed_domains=[],
        )
    except Exception as exc:  # noqa: BLE001 - a window failure must not drop the chunk
        log.warning("sterilize.figure_window_failed", chunk_id=chunk.chunk_id, error=str(exc))
        return None, False
    sterile = parsed.sterile_content.strip()
    if not sterile:
        log.warning("sterilize.figure_window_empty", chunk_id=chunk.chunk_id)
        return None, should_quarantine
    ok, longest_run, ratio = _check_verbatim_budget(normalized, sterile)
    if not ok:
        log.warning(
            "sterilize.figure_window_verbatim",
            chunk_id=chunk.chunk_id,
            longest_run=longest_run,
            ratio=round(ratio, 4),
        )
        return None, should_quarantine
    return sterile, should_quarantine


def _license_is_publishable(chunk: RetrievalChunk) -> bool:
    license_value = str(chunk.metadata.get("license_class", LicenseClass.UNKNOWN.value))
    try:
        license_class = LicenseClass(license_value)
    except ValueError:
        license_class = LicenseClass.UNKNOWN
    return license_class in PUBLISHABLE


async def sterilize_chunks(
    chunks: list[RetrievalChunk],
    llm_client: LLMClient,
) -> list[RetrievalChunk]:
    """Sterilize a list of retrieval chunks using the LLM path.

    Returns a list of new ``RetrievalChunk`` objects with redacted content.
    Empty input returns empty output without any LLM call.
    """
    if not chunks:
        return []

    result: list[RetrievalChunk] = []
    for chunk in chunks:
        normalized_content = normalize_extracted_text(chunk.content)
        if not normalized_content.strip():
            continue

        untrusted_chunk = _build_untrusted_chunk(chunk)

        t0 = time.perf_counter()
        parsed, _resp, _validation, should_quarantine = await complete_json_protected(
            llm_client,
            schema=_SterilizedPayload,
            untrusted_content=untrusted_chunk,
            pipeline="book_rag",
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1024,
            allowed_domains=[],
        )
        t1 = time.perf_counter()

        content = parsed.sterile_content.strip()
        if not content:
            raise SterilizerParseError("sterile_content empty")

        ok, longest_run, ratio = _check_verbatim_budget(normalized_content, content)
        if not ok:
            raise SterilizerParseError(
                f"verbatim budget exceeded: longest_run={longest_run}, ratio={ratio:.2%}"
            )

        is_publishable = _license_is_publishable(chunk)
        metadata = dict(chunk.metadata)
        metadata["_sterilization_model"] = llm_client.provider.model
        metadata["_sterilization_duration_ms"] = int((t1 - t0) * 1000)
        metadata["_sterilization_quarantine"] = should_quarantine
        metadata["_sterilization_license_publishable"] = is_publishable
        metadata["facts_extracted"] = parsed.facts
        metadata["equations"] = parsed.equations
        metadata["concepts"] = parsed.concepts

        # A figure embeds its (sterilized) neighbour window, like the raw side
        # does; its stored content stays the sterilized figure text (VOR-1452).
        embedding_body = content
        window = _figure_window(chunk)
        if window is not None:
            sterile_window, window_quarantine = await _sterile_window(chunk, window, llm_client)
            embedding_body = sterile_window or content
            metadata["_sterilization_quarantine"] = should_quarantine or window_quarantine

        sterile_chunk = RetrievalChunk(
            chunk_id=deterministic_uuid(chunk.source_id, chunk.chunk_id, "sterile"),
            chunk_type=chunk.chunk_type,
            document_id=chunk.document_id,
            source_id=chunk.source_id,
            page_number=chunk.page_number,
            page_range=chunk.page_range,
            section_path=chunk.section_path,
            section_title=chunk.section_title,
            chapter_path=chunk.chapter_path,
            content=content,
            embedding_text=f"{chunk.section_path}\n\n{embedding_body}",
            token_count=len(content.split()),
            has_image=chunk.has_image,
            image_id=chunk.image_id,
            image_ref=chunk.image_ref,
            vlm_description=chunk.vlm_description,
            metadata=metadata,
        )
        result.append(sterile_chunk)

    return result


__all__ = [
    "SYSTEM_PROMPT",
    "SterilizerParseError",
    "_check_verbatim_budget",
    "sterilize_chunks",
]
