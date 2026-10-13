from __future__ import annotations

import json

import structlog

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.raptor.prompts import SYSTEM_PROMPTS, USER_TEMPLATES
from vo_rag.core.raptor.schemas import ClusterLevel
from vo_rag.core.summary.extract_support import select_evidence_snippets
from vo_rag.core.summary.schemas import EvidenceSpan

log = structlog.get_logger(__name__)


def select_raw_evidence_for_cluster(
    *,
    cluster_chunks: list[RetrievalChunk],
    topic_or_query: str,
    k: int,
) -> list[dict[str, str | float]]:
    """Select Extract-Support snippets from raw chunks only.

    RAPTOR recursion may summarize summaries as the primary input, but the
    support channel must always go back to the raw RetrievalChunk objects
    ultimately covered by the cluster.
    """

    _validate_raw_chunk_list(cluster_chunks)

    evidence = [
        _span_to_dict(span)
        for span in select_evidence_snippets(
            cluster_chunks,
            topic_or_query=topic_or_query,
            k=k,
        )
    ]
    raw_chunk_ids = {chunk.chunk_id for chunk in cluster_chunks}

    if not verify_raw_evidence_invariant(evidence, raw_chunk_ids):
        raise ValueError("selected RAPTOR evidence must reference only raw cluster chunk IDs")

    log.debug(
        "raptor_raw_evidence_selected",
        raw_chunk_count=len(cluster_chunks),
        evidence_count=len(evidence),
        k=k,
    )
    return evidence


def build_level_extract_support_prompt(
    *,
    level: ClusterLevel,
    primary_input: str,
    supporting_evidence: list[dict[str, str | float]],
    book_id: str,
) -> tuple[str, str]:
    """Return the RAPTOR level-specific Extract-Support prompt pair."""

    system_prompt = SYSTEM_PROMPTS[level]
    user_template = USER_TEMPLATES[level]
    evidence_json = json.dumps(supporting_evidence, indent=2, sort_keys=True)
    cluster_context = json.dumps(
        {
            "book_id": book_id,
            "level": level.value,
        },
        indent=2,
        sort_keys=True,
    )
    user_prompt = user_template.format(
        primary_input=primary_input,
        supporting_evidence=evidence_json,
        book_id=book_id,
        cluster_context=cluster_context,
    )
    return system_prompt, user_prompt


def assemble_recursive_evidence(
    *,
    level: ClusterLevel,
    child_primary_text: str,
    all_underlying_raw_chunks: list[RetrievalChunk],
    k_evidence: int = 10,
) -> tuple[str, list[dict[str, str | float]]]:
    """Return prompt inputs for a RAPTOR recursive summary call.

    This helper does not call an LLM. At LEVEL_0, child_primary_text is raw
    chunk text; at LEVEL_1 and LEVEL_2, it is child summary text. Evidence is
    always selected from all_underlying_raw_chunks.
    """

    evidence = select_raw_evidence_for_cluster(
        cluster_chunks=all_underlying_raw_chunks,
        topic_or_query=child_primary_text,
        k=k_evidence,
    )
    log.debug(
        "raptor_recursive_evidence_assembled",
        level=level.value,
        raw_chunk_count=len(all_underlying_raw_chunks),
        evidence_count=len(evidence),
    )
    return child_primary_text, evidence


def verify_raw_evidence_invariant(
    supporting_evidence: list[dict[str, str | float]],
    raw_chunk_ids: set[str],
) -> bool:
    """Return True iff every evidence span references one of the raw chunk IDs."""

    for span in supporting_evidence:
        chunk_id = span.get("chunk_id")
        if not isinstance(chunk_id, str) or chunk_id not in raw_chunk_ids:
            return False
    return True


def _validate_raw_chunk_list(cluster_chunks: list[RetrievalChunk]) -> None:
    if not isinstance(cluster_chunks, list):
        raise TypeError("cluster_chunks must be a list[RetrievalChunk]")

    if not all(isinstance(chunk, RetrievalChunk) for chunk in cluster_chunks):
        raise TypeError("cluster_chunks must contain only raw RetrievalChunk objects")


def _span_to_dict(span: EvidenceSpan) -> dict[str, str | float]:
    return {
        "chunk_id": span.chunk_id,
        "section_path": span.section_path,
        "text": span.text,
        "score": span.score,
        "selection_method": span.selection_method,
    }


__all__ = [
    "assemble_recursive_evidence",
    "build_level_extract_support_prompt",
    "select_raw_evidence_for_cluster",
    "verify_raw_evidence_invariant",
]
