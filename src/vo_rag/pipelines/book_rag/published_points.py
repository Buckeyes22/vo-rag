"""Resolve a producer's retained point identities to its current visible roster."""

from __future__ import annotations

from typing import Any

from vo_rag.core.storage.publication_client import PublicationClient
from vo_rag.core.storage.publication_read_view import publication_query

__all__ = ["current_published_ids"]


@publication_query
async def current_published_ids(
    *, store: Any, collection: str, source_id: str, point_ids: list[str]
) -> list[str]:
    """Resolve explicitly verified logical producer IDs after payload revisions.

    Generic physical retrieval remains strict. The producer reads its retained
    immutable points to recover declared base IDs, then resolves those logical
    IDs through the current publication view and checks source ownership.

    Args:
        store: The producer's Qdrant store with its shared publication client.
        collection: Exact owning collection of the producer's points.
        source_id: Canonical source whose point identities may be resolved.
        point_ids: Previously acknowledged, retained physical point IDs.

    Returns:
        Current visible physical IDs in the producer's original logical order.

    Raises:
        ValueError: A retained or visible identity is missing or ambiguous, or
            its source ownership differs from the producer's declared source.
    """
    client = getattr(store, "_client", None)
    if not isinstance(client, PublicationClient) or not point_ids:
        return list(point_ids)
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("producer point roster contains duplicate physical IDs")
    output: list[str] = []
    bases_seen: set[str] = set()
    for offset in range(0, len(point_ids), 1000):
        requested = point_ids[offset : offset + 1000]
        retained = await client.raw.retrieve(
            collection_name=collection, ids=requested, with_payload=True, with_vectors=False
        )
        by_id = {str(row.id): row for row in retained}
        if len(retained) != len(requested) or set(by_id) != set(requested):
            raise ValueError("producer retained point roster is incomplete")
        logical_ids: list[str] = []
        for point_id in requested:
            payload = by_id[point_id].payload or {}
            base_id = payload.get("publication_base_id", point_id)
            if payload.get("source_id") != source_id or not isinstance(base_id, str):
                raise ValueError("producer point identity has invalid source ownership")
            if not base_id or base_id in bases_seen:
                raise ValueError("producer point roster contains ambiguous logical IDs")
            bases_seen.add(base_id)
            logical_ids.append(base_id)
        current = await client.resolve_logical_points(collection, logical_ids)
        if len(current) != len(logical_ids):
            raise ValueError("producer logical point roster is not completely visible")
        for base_id, row in zip(logical_ids, current, strict=True):
            payload = row.payload or {}
            if (
                payload.get("source_id") != source_id
                or payload.get("publication_base_id", str(row.id)) != base_id
            ):
                raise ValueError("producer current point identity has invalid source ownership")
            output.append(str(row.id))
    return output
