"""DSI provenance validation.

Verifies every provenance-linked entry in a ``FacetedBookRecord`` references a
valid chunk_id in ``vo_rag_books_raw_v1`` and a real section_path in the book's
structure. Records with unverifiable provenance are rejected.

Public API
----------
ProvenanceViolation
    Data class describing a single failed provenance check.
RejectionDecision
    Accept / re-extract / manual-review verdict for one record.
validate_provenance_offline(record, *, known_chunk_ids, known_section_paths)
    Pure-Python validation against pre-fetched sets.
fetch_book_chunk_ids_and_sections(*, store, book_id, raw_collection)
    Qdrant scroll to build the two sets for offline validation.
validate_provenance_against_qdrant(record, *, store, raw_collection)
    Convenience wrapper: fetch + offline validate.
decide_rejection(violations, attempt, max_attempts)
    Translate violation list into an accept/re-extract/manual-review decision.
"""

from __future__ import annotations

from typing import Literal

import structlog

from vo_rag.core.dsi.schemas import FacetedBookRecord, ProvenanceLinkedEntry
from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.storage.qdrant import QdrantStore

logger = structlog.get_logger(__name__)

# Fields that carry ProvenanceLinkedEntry items — the five DSI facets.
_PROVENANCE_FIELDS: tuple[str, ...] = (
    "key_concepts",
    "key_methods",
    "key_equations_or_models",
    "figure_heavy_topics",
    "synonyms_and_aliases",
)

ViolationType = Literal[
    "unresolvable_chunk_id",
    "unresolvable_section_path",
    "chunk_book_mismatch",
    "chunk_section_mismatch",
]


class ProvenanceViolation(VoBaseModel):
    """A single provenance-check failure within a ``FacetedBookRecord``."""

    book_id: str
    field_name: str
    entry_value: str
    violation_type: ViolationType
    chunk_id: str | None
    section_path: str | None
    detail: str


class RejectionDecision(VoBaseModel):
    """Accept / re-extract / manual-review verdict for one record."""

    record_accepted: bool
    violations: list[ProvenanceViolation]
    action: Literal["accept", "re_extract", "manual_review"]
    attempt_number: int = 1


# ---------------------------------------------------------------------------
# Pure-Python validation (no I/O)
# ---------------------------------------------------------------------------


def validate_provenance_offline(
    record: FacetedBookRecord,
    *,
    known_chunk_ids: set[str],
    known_section_paths: set[str],
) -> list[ProvenanceViolation]:
    """Validate provenance links against pre-fetched chunk and section sets.

    For each of the five provenance-linked fields:

    - Every ``chunk_id`` in ``supporting_chunk_ids`` must appear in
      ``known_chunk_ids``; absent IDs produce an ``unresolvable_chunk_id``
      violation.
    - Every ``section_path`` in ``supporting_section_paths`` must appear in
      ``known_section_paths``; absent paths produce an
      ``unresolvable_section_path`` violation.

    Returns an empty list when the record is clean.
    """
    violations: list[ProvenanceViolation] = []

    for field_name in _PROVENANCE_FIELDS:
        entries: list[ProvenanceLinkedEntry] = getattr(record, field_name)
        for entry in entries:
            for chunk_id in entry.supporting_chunk_ids:
                if chunk_id not in known_chunk_ids:
                    violations.append(
                        ProvenanceViolation(
                            book_id=record.book_id,
                            field_name=field_name,
                            entry_value=entry.value,
                            violation_type="unresolvable_chunk_id",
                            chunk_id=chunk_id,
                            section_path=None,
                            detail=(
                                f"chunk_id {chunk_id!r} not found in"
                                f" vo_rag_books_raw_v1 for book {record.book_id!r}"
                            ),
                        )
                    )

            for section_path in entry.supporting_section_paths:
                if section_path not in known_section_paths:
                    violations.append(
                        ProvenanceViolation(
                            book_id=record.book_id,
                            field_name=field_name,
                            entry_value=entry.value,
                            violation_type="unresolvable_section_path",
                            chunk_id=None,
                            section_path=section_path,
                            detail=(
                                f"section_path {section_path!r} not found in"
                                f" book structure for book {record.book_id!r}"
                            ),
                        )
                    )

    log = logger.bind(
        book_id=record.book_id,
        violation_count=len(violations),
    )
    if violations:
        log.warning(
            "provenance.offline.violations",
            violations=[v.model_dump() for v in violations],
        )
    else:
        log.debug("provenance.offline.clean")

    return violations


# ---------------------------------------------------------------------------
# Qdrant-backed fetch
# ---------------------------------------------------------------------------


async def fetch_book_chunk_ids_and_sections(
    *,
    store: QdrantStore,
    book_id: str,
    raw_collection: str = "vo_rag_books_raw_v1",
) -> tuple[set[str], set[str]]:
    """Scroll ``raw_collection`` and collect all chunk IDs + section paths for *book_id*.

    Uses Qdrant's scroll API to paginate through potentially large books.
    Returns ``(chunk_ids, section_paths)`` for offline validation.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    chunk_ids: set[str] = set()
    section_paths: set[str] = set()

    scroll_filter = Filter(
        must=[
            FieldCondition(
                key="source_book",
                match=MatchValue(value=book_id),
            )
        ]
    )

    log = logger.bind(book_id=book_id, collection=raw_collection)
    log.debug("provenance.fetch.start")

    offset = None
    page = 0
    while True:
        results, next_offset = await store._client.scroll(
            collection_name=raw_collection,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )

        for point in results:
            chunk_ids.add(str(point.id))
            payload = point.payload or {}
            sp = payload.get("section_path") or payload.get("canonical_path")
            if sp:
                section_paths.add(str(sp))

        page += 1
        log.debug(
            "provenance.fetch.page",
            page=page,
            points_in_page=len(results),
            cumulative_chunks=len(chunk_ids),
        )

        if next_offset is None:
            break
        offset = next_offset

    log.info(
        "provenance.fetch.done",
        total_chunks=len(chunk_ids),
        total_sections=len(section_paths),
    )
    return chunk_ids, section_paths


# ---------------------------------------------------------------------------
# Combined async entry point
# ---------------------------------------------------------------------------


async def validate_provenance_against_qdrant(
    record: FacetedBookRecord,
    *,
    store: QdrantStore,
    raw_collection: str = "vo_rag_books_raw_v1",
) -> list[ProvenanceViolation]:
    """Fetch provenance sets from Qdrant, then run offline validation.

    Equivalent to calling :func:`fetch_book_chunk_ids_and_sections` and
    :func:`validate_provenance_offline` in sequence.
    """
    known_chunk_ids, known_section_paths = await fetch_book_chunk_ids_and_sections(
        store=store,
        book_id=record.book_id,
        raw_collection=raw_collection,
    )
    return validate_provenance_offline(
        record,
        known_chunk_ids=known_chunk_ids,
        known_section_paths=known_section_paths,
    )


# ---------------------------------------------------------------------------
# Rejection decision
# ---------------------------------------------------------------------------


def decide_rejection(
    violations: list[ProvenanceViolation],
    attempt: int,
    max_attempts: int = 2,
) -> RejectionDecision:
    """Translate a violation list into an accept/re-extract/manual-review decision.

    - No violations -> ``accept``.
    - Violations and ``attempt < max_attempts`` -> ``re_extract`` (attempt incremented).
    - Violations and ``attempt >= max_attempts`` -> ``manual_review``.
    """
    if not violations:
        return RejectionDecision(
            record_accepted=True,
            violations=[],
            action="accept",
            attempt_number=attempt,
        )

    if attempt < max_attempts:
        return RejectionDecision(
            record_accepted=False,
            violations=violations,
            action="re_extract",
            attempt_number=attempt + 1,
        )

    return RejectionDecision(
        record_accepted=False,
        violations=violations,
        action="manual_review",
        attempt_number=attempt,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ProvenanceViolation",
    "RejectionDecision",
    "validate_provenance_offline",
    "fetch_book_chunk_ids_and_sections",
    "validate_provenance_against_qdrant",
    "decide_rejection",
]
