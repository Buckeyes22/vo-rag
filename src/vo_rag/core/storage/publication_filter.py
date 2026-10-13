"""Materialize the authoritative source publication decision before Qdrant ranking."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from qdrant_client.models import (
    Condition,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    PayloadField,
)

__all__ = ["publication_filter", "publication_corpus", "dependency_filter", "dependencies_visible"]


def dependency_filter(active_raw: Mapping[str, int | None]) -> Filter:
    """Hide stale derived records when any associated canonical raw source advances.

    Legacy derived points remain visible while their raw source is legacy.
    Unknown historical dependency bindings are never inferred from old payloads.
    Hashed keys preserve arbitrary canonical source identities in Qdrant paths.
    """
    requirements: list[Condition] = []
    for source_id, revision in active_raw.items():
        if revision is None:
            continue
        association = Filter(
            should=[
                FieldCondition(key=key, match=MatchValue(value=source_id))
                for key in ("source_id", "book_id", "book_ids")
            ]
        )
        fingerprint = hashlib.sha256(source_id.encode()).hexdigest()
        requirements.append(
            Filter(
                should=[
                    Filter(must_not=[association]),
                    FieldCondition(
                        key=f"dependency_revision_keys.{fingerprint}",
                        match=MatchValue(value=revision),
                    ),
                ]
            )
        )
    return Filter(must=requirements) if requirements else Filter()


def dependencies_visible(
    payload: Mapping[str, object], active_raw: Mapping[str, int | None]
) -> bool:
    """Apply the same raw dependency policy to ID-based retrieval."""
    associations = {
        value for key in ("source_id", "book_id") if isinstance(value := payload.get(key), str)
    }
    book_ids = payload.get("book_ids")
    if isinstance(book_ids, list):
        associations.update(value for value in book_ids if isinstance(value, str))
    keys = payload.get("dependency_revision_keys")
    observed = keys if isinstance(keys, dict) else {}
    return all(
        revision is None
        or source_id not in associations
        or (
            type(value := observed.get(hashlib.sha256(source_id.encode()).hexdigest())) is int
            and value == revision
        )
        for source_id, revision in active_raw.items()
    )


def publication_corpus(collection: str) -> str | None:
    """Return the owning Books/Reddit corpus, preserving every other boundary."""
    if collection.startswith("vo_rag_books_"):
        return "books"
    if collection == "vo_rag_reddit_v1":
        return "reddit"
    return None


def publication_filter(active: Mapping[str, int | None], existing: Filter | None = None) -> Filter:
    """Restrict candidates to each source's active revision or explicit legacy state.

    The predicate is applied to every search arm before fusion and top-k. It
    never admits an unpublished revision, even if its source pointer is absent.
    A null pointer explicitly retains legacy points during preparation.
    """
    legacy = IsEmptyCondition(is_empty=PayloadField(key="source_revision"))
    unmanaged = Filter(must=[legacy])
    if active:
        unmanaged.must_not = [FieldCondition(key="source_id", match=MatchAny(any=list(active)))]
    alternatives: list[Condition] = [unmanaged]
    for source_id, revision in active.items():
        revision_condition = (
            legacy
            if revision is None
            else FieldCondition(key="source_revision", match=MatchValue(value=revision))
        )
        alternatives.append(
            Filter(
                must=[
                    FieldCondition(key="source_id", match=MatchValue(value=source_id)),
                    revision_condition,
                ]
            )
        )
    visibility = Filter(should=alternatives)
    return Filter(must=[existing, visibility]) if existing is not None else visibility
