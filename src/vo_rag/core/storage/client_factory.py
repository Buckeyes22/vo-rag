"""Build a Qdrant client with the same source boundary for service and maintenance lanes."""

from __future__ import annotations

import os
from typing import Any

from qdrant_client import AsyncQdrantClient

from vo_rag.core.ingest.source_publication import SourceCoordinator
from vo_rag.core.storage.publication_client import PublicationClient
from vo_rag.core.storage.publication_filter import publication_corpus

__all__ = ["build_qdrant_client"]


async def _coordinator(collection: str) -> SourceCoordinator:
    from vo_rag.core.config.settings import get_settings
    from vo_rag.core.db.postgres import get_pool

    corpus = publication_corpus(collection)
    if corpus is None:
        raise ValueError("source publication requires an explicit Books/Reddit collection")
    return SourceCoordinator(await get_pool(get_settings().database.url), corpus=corpus)


async def _pointers(corpus: str, collection: str) -> dict[str, int | None]:
    coordinator = await _coordinator(collection)
    if coordinator.corpus != corpus:
        raise ValueError("publication corpus mismatch")
    from vo_rag.core.storage.publication_read_view import captured_pointers

    captured = captured_pointers(coordinator.pool, corpus, collection)
    if captured is not None:
        return captured
    return await coordinator.list_active(collection)


async def _mutate(operation: str, collection: str, kwargs: dict[str, Any], raw: Any) -> Any:
    from vo_rag.core.storage.source_mutations import SourceMutationWriter

    writer = SourceMutationWriter(raw, await _coordinator(collection))
    if os.environ.get("VORAG_SOURCE_PUBLICATION_WRITES_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }:
        return await writer.mutate(operation, collection, kwargs)
    return await writer.mutate_legacy(operation, collection, kwargs)


def build_qdrant_client(*args: Any, **kwargs: Any) -> Any:
    """Return a guarded client by default; preserve non-Books/Reddit corpus behavior."""
    if os.environ.get("VORAG_SOURCE_PUBLICATION_WRITES_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    } and os.environ.get("VORAG_SOURCE_PUBLICATION_ENABLED", "true").lower() not in {
        "1",
        "true",
        "yes",
    }:
        raise ValueError("source revision writes require publication readers")
    raw = AsyncQdrantClient(*args, **kwargs)
    if os.environ.get("VORAG_SOURCE_PUBLICATION_ENABLED", "true").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return raw

    async def mutations(operation: str, collection: str, values: dict[str, Any]) -> Any:
        return await _mutate(operation, collection, values, raw)

    return PublicationClient(raw, _pointers, mutations)
