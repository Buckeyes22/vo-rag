"""Typed graph/tree node resolution without retargeting physical citation IDs."""

from typing import Any

from vo_rag.core.storage.publication_client import PublicationClient


async def retrieve_logical_points(client: Any, *, collection: str, ids: list[str]) -> list[Any]:
    """Resolve an explicit logical node reference within its owning collection."""
    if isinstance(client, PublicationClient):
        return await client.resolve_logical_points(collection, ids)
    rows: list[Any] = await client.retrieve(collection_name=collection, ids=ids, with_payload=True)
    return rows
