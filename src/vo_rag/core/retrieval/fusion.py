"""Canonical RRF + dedup fusion seam for cross-collection retrieval.

The two-pass dedup key sequence is the most load-bearing part of retrieval;
do not "simplify" it.

The two-pass dedup is canonical:

  - **Pass 1** (per-source dedup): key = ``(source_id, parent_chunk_id or
    chunk_id or point:ID)`` — collapses raw↔sterile pairs that share a
    parent_chunk_id and prevents the same chunk from counting twice
    within a single input list. Independent ranked tracks still contribute votes.
  - **Pass 2** (cross-variant collapse): key = ``(logical_book_key,
    content_hash)`` — collapses byte-distinct ingest copies of the
    same book; copy contributions retain their best rank rather than fabricating consensus.

RRF math: ``1 / (k_rrf + rank + 1)`` with ``k_rrf = 60`` (classic
Cormack et al.).

**Deliberately not provided:**

  - Convex fusion and score normalizers (``minmax`` / ``zscore`` / ``rank``):
    no A/B result justifies them, so this module does not offer them.
  - A fusion factory: only RRF exists, and the k constant is a single
    keyword argument.

The Qdrant store cross-collection helper shares the same passage identity
contract, preserving independent sections at every supported retrieval seam.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from vo_rag.core.storage.passage_contract import (
    collapse_evidence_payloads,
    evidence_provenance,
    passage_identity,
)

if TYPE_CHECKING:
    from vo_rag.core.retrieval.search import SearchResult

# Classic Cormack et al. RRF k-constant. Changing it changes every merged
# ranking against recorded baselines.
RRF_DEFAULT_K: int = 60


def dedupe_key(result: SearchResult) -> tuple[str, str]:
    """Per-source dedup key: ``(source_id, parent_chunk_id or chunk_id or point:ID)``.

    The ``parent_chunk_id`` wins when present so raw↔sterile pairs sharing
    a parent chunk collapse to one entry. The ``chunk_id`` wins when
    there's no parent. The actual storage point ID handles legacy records
    where neither is set, preserving independent passages on the same page.
    """
    return passage_identity(result.payload or {}, result.id)


def cross_variant_key(result: SearchResult) -> tuple[str, str] | None:
    """Cross-variant collapse key: ``(logical_book_key, content_hash)``.

    Returns ``None`` if either field is missing — in which case the
    per-source dedup key is used unchanged.
    """
    payload = result.payload or {}
    book_key = payload.get("logical_book_key")
    content_hash = payload.get("content_hash")
    if not book_key or not content_hash:
        return None
    return (str(book_key), str(content_hash))


def rrf_merge_and_dedupe(
    lists: list[list[SearchResult]],
    *,
    top_k: int,
    k_rrf: int = RRF_DEFAULT_K,
) -> list[SearchResult]:
    """Merge N ranked ``SearchResult`` lists into a single top-k list.

    Pass 1 dedupes on ``(source_id, chunk_id)`` (with storage-point fallback).
    Pass 2 collapses cross-variant duplicates on
    ``(logical_book_key, content_hash)``, preserving their best RRF contribution.

    ``k_rrf`` is keyword-only by design: an accidental positional
    would silently change the dedup behaviour and break the RRF
    persistence contract.
    """
    # Pass 1: per-source dedup on (source_id, chunk_id) — historical behaviour.
    # Collapses raw↔sterile pairs sharing parent_chunk_id and prevents the
    # same chunk being counted twice within a single input list. Independent
    # ranked tracks still contribute votes.
    scores: dict[tuple[str, str], float] = {}
    picks: dict[tuple[str, str], SearchResult] = {}
    members_by_key: dict[tuple[str, str], list[SearchResult]] = {}
    for hits in lists:
        seen_in_track: set[tuple[str, str]] = set()
        for rank, result in enumerate(hits):
            key = dedupe_key(result)
            members_by_key.setdefault(key, []).append(result)
            picks.setdefault(key, result)
            if key not in seen_in_track:
                scores[key] = scores.get(key, 0.0) + (1.0 / (k_rrf + rank + 1))
                seen_in_track.add(key)

    for key, members in members_by_key.items():
        if len(members) > 1:
            provenance = {
                (item["collection"], item["id"]): item
                for member in members
                for item in evidence_provenance(member.payload or {}, member.id, member.collection)
            }
            if len(provenance) > 1:
                picks[key] = replace(
                    picks[key],
                    payload={
                        **picks[key].payload,
                        "duplicate_provenance": list(provenance.values()),
                    },
                )

    # Pass 2: cross-variant collapse on (logical_book_key, content_hash) so the
    # same passage from N byte-distinct ingest copies of the same book surfaces
    # once. Copies retain their best contribution rather than voting as independent evidence.
    final_scores: dict[tuple[str, str], float] = {}
    final_picks: dict[tuple[str, str], SearchResult] = {}
    final_members: dict[tuple[str, str], list[SearchResult]] = {}
    for key, result in picks.items():
        cv_key = cross_variant_key(result) or key
        if cv_key not in final_scores or scores[key] > final_scores[cv_key]:
            final_picks[cv_key] = result
        final_scores[cv_key] = max(final_scores.get(cv_key, 0.0), scores[key])
        final_members.setdefault(cv_key, []).append(result)

    ranked = sorted(final_picks.items(), key=lambda kv: final_scores[kv[0]], reverse=True)
    ordered = []
    for key, result in ranked:
        members = final_members[key]
        if len(members) > 1:
            provenance = {
                (item["collection"], item["id"]): item
                for member in members
                for item in evidence_provenance(member.payload or {}, member.id, member.collection)
            }
            result = replace(
                result,
                payload={**result.payload, "duplicate_provenance": list(provenance.values())},
            )
        ordered.append(result)
    return dedupe_exact_evidence(ordered)[:top_k]


def dedupe_exact_evidence(results: list[SearchResult]) -> list[SearchResult]:
    """Keep the best exact text/context copy and attach original provenance."""
    return [
        replace(results[index], payload=payload)
        for index, payload in collapse_evidence_payloads(
            [(result.payload or {}, result.id, result.collection) for result in results]
        )
    ]


__all__ = [
    "RRF_DEFAULT_K",
    "cross_variant_key",
    "dedupe_key",
    "rrf_merge_and_dedupe",
]
