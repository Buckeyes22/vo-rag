"""Enforce Books/Reddit source visibility at every underlying Qdrant read seam."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from qdrant_client.models import FieldCondition, Filter, HasIdCondition, MatchAny, Prefetch

from vo_rag.core.storage.publication_filter import (
    dependencies_visible,
    dependency_filter,
    publication_corpus,
    publication_filter,
)

__all__ = ["PublicationClient"]

PointerReader = Callable[[str, str], Awaitable[dict[str, int | None]]]
MutationWriter = Callable[[str, str, dict[str, Any]], Awaitable[Any]]


class PublicationClient:
    """Wrap all source-backed reads with a fresh authoritative publication predicate.

    Old revisions remain retained for readers that captured a previous pointer.
    Retrieval entry points pin one pointer view across expansion calls; callers
    outside that context read fresh pointers for each call. This does not make
    legacy points or external stores an atomic snapshot.
    """

    def __init__(
        self, raw: Any, pointers: PointerReader, mutations: MutationWriter | None = None
    ) -> None:
        self.raw = raw
        self._pointers = pointers
        self._mutations = mutations

    def __getattr__(self, name: str) -> Any:
        """Delegate schema/control operations to the owning client."""
        if name not in {
            "close",
            "aclose",
            "create_collection",
            "update_collection",
            "get_collection",
            "get_collections",
            "collection_exists",
            "create_payload_index",
            "delete_payload_index",
        }:
            raise AttributeError(f"{name} requires an explicit source publication contract")
        return getattr(self.raw, name)

    async def _filter(self, collection: str, existing: Filter | None) -> Filter | None:
        corpus = publication_corpus(collection)
        if corpus is None:
            return existing
        active = await self._pointers(corpus, collection)
        visible = publication_filter(active, existing)
        if corpus == "books" and collection != "vo_rag_books_raw_v1":
            raw = await self._pointers(corpus, "vo_rag_books_raw_v1")
            return Filter(must=[visible, dependency_filter(raw)])
        return visible

    @staticmethod
    def _prefetch(value: Prefetch, visibility: Filter | None) -> Prefetch:
        nested = value.prefetch
        if isinstance(nested, list):
            nested = [PublicationClient._prefetch(item, visibility) for item in nested]
        elif isinstance(nested, Prefetch):
            nested = PublicationClient._prefetch(nested, visibility)
        combined = (
            Filter(must=[value.filter, visibility])
            if value.filter is not None and visibility is not None
            else value.filter or visibility
        )
        return value.model_copy(update={"filter": combined, "prefetch": nested})

    async def query_points(self, collection_name: str, **kwargs: Any) -> Any:
        """Filter every nested prefetch and final ranking with one pointer snapshot."""
        visibility = await self._filter(collection_name, None)
        existing = kwargs.get("query_filter")
        kwargs["query_filter"] = (
            Filter(must=[existing, visibility])
            if existing is not None and visibility is not None
            else existing or visibility
        )
        prefetch = kwargs.get("prefetch")
        effective_filter = kwargs["query_filter"]
        if isinstance(prefetch, list):
            kwargs["prefetch"] = [self._prefetch(item, effective_filter) for item in prefetch]
        elif isinstance(prefetch, Prefetch):
            kwargs["prefetch"] = self._prefetch(prefetch, effective_filter)
        return await self.raw.query_points(collection_name=collection_name, **kwargs)

    async def scroll(self, collection_name: str, **kwargs: Any) -> Any:
        """Apply source visibility to book, graph, expansion and maintenance reads."""
        kwargs["scroll_filter"] = await self._filter(collection_name, kwargs.get("scroll_filter"))
        return await self.raw.scroll(collection_name=collection_name, **kwargs)

    async def count(self, collection_name: str, **kwargs: Any) -> Any:
        """Count only points admitted by the authoritative source pointer."""
        kwargs["count_filter"] = await self._filter(collection_name, kwargs.get("count_filter"))
        return await self.raw.count(collection_name=collection_name, **kwargs)

    async def retrieve(self, collection_name: str, ids: list[Any], **kwargs: Any) -> Any:
        """Reject stale artifact/graph point IDs using the same publication decision."""
        corpus = publication_corpus(collection_name)
        if corpus is None:
            return await self.raw.retrieve(collection_name=collection_name, ids=ids, **kwargs)
        active = await self._pointers(corpus, collection_name)
        raw_dependencies = (
            await self._pointers(corpus, "vo_rag_books_raw_v1")
            if corpus == "books" and collection_name != "vo_rag_books_raw_v1"
            else {}
        )
        requested_payload = kwargs.get("with_payload", True)
        # Visibility must never depend on whether a caller selected these fields.
        kwargs["with_payload"] = True
        rows = await self.raw.retrieve(collection_name=collection_name, ids=ids, **kwargs)
        visible: list[Any] = []
        for row in rows:
            payload = row.payload or {}
            if not dependencies_visible(payload, raw_dependencies):
                continue
            source_id = payload.get("source_id")
            revision = payload.get("source_revision")
            if revision is not None and type(revision) is not int:
                continue
            if source_id in active:
                if revision != active[source_id]:
                    continue
            elif revision is not None:
                continue
            if requested_payload is False:
                row = row.model_copy(update={"payload": None})
            visible.append(row)
        if visible and requested_payload is not True and requested_payload is not False:
            # Delegate nested include/exclude semantics to Qdrant, after IDs
            # have been checked with complete identity payloads. Published
            # revisions are immutable and retained across pointer changes.
            kwargs["with_payload"] = requested_payload
            return await self.raw.retrieve(
                collection_name=collection_name, ids=[row.id for row in visible], **kwargs
            )
        return visible

    async def facet(self, collection_name: str, **kwargs: Any) -> Any:
        """Exclude unpublished source revisions from catalog and monitoring facets."""
        kwargs["facet_filter"] = await self._filter(collection_name, kwargs.get("facet_filter"))
        return await self.raw.facet(collection_name=collection_name, **kwargs)

    async def resolve_logical_points(self, collection_name: str, ids: list[str]) -> list[Any]:
        """Resolve declared node/base identities through the active revision.

        Typed tree/graph references are logical IDs. Generic ``retrieve`` keeps
        physical point-ID semantics and never redirects stale revision IDs.
        """
        if not ids or len(ids) > 1000 or len(set(ids)) != len(ids):
            raise ValueError("logical node resolution requires 1 to 1000 unique IDs")

        def identity(value: str) -> str:
            try:
                return str(UUID(value))
            except ValueError:
                return value

        requested = {identity(value): value for value in ids}
        if len(requested) != len(ids):
            raise ValueError("logical node roster contains duplicate UUID aliases")
        point_ids: list[Any] = list(ids)
        aliases = sorted(set(ids) | set(requested))
        selection = Filter(
            should=[
                HasIdCondition(has_id=point_ids),
                FieldCondition(key="publication_base_id", match=MatchAny(any=aliases)),
            ]
        )
        visible = await self._filter(collection_name, selection)
        found: dict[str, Any] = {}
        offset: Any = None
        while True:
            rows, offset = await self.raw.scroll(
                collection_name=collection_name,
                scroll_filter=visible,
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for row in rows:
                payload = row.payload or {}
                base = str(payload.get("publication_base_id") or row.id)
                # UUID normalization must not change declared logical identity.
                key = requested.get(identity(base)) or requested.get(identity(str(row.id)))
                if key is None or key in found:
                    raise ValueError(
                        "logical node identity is ambiguous in the active source roster"
                    )
                found[key] = row
            if len(found) > len(ids):
                raise ValueError("logical resolution exceeds its requested identity roster")
            if offset is None:
                return [found[key] for key in ids if key in found]

    async def _mutate(self, operation: str, collection_name: str, kwargs: dict[str, Any]) -> Any:
        if publication_corpus(collection_name) is None:
            return await getattr(self.raw, operation)(collection_name=collection_name, **kwargs)
        if self._mutations is None:
            raise RuntimeError("source mutation requires the shared publication coordinator")
        return await self._mutations(operation, collection_name, kwargs)

    async def upsert(self, collection_name: str, **kwargs: Any) -> Any:
        """Stage source-scoped point mutations through the shared coordinator."""
        return await self._mutate("upsert", collection_name, kwargs)

    async def delete(self, collection_name: str, **kwargs: Any) -> Any:
        """Publish a scoped tombstone or replacement; retain earlier reader revisions."""
        return await self._mutate("delete", collection_name, kwargs)

    async def set_payload(self, collection_name: str, **kwargs: Any) -> Any:
        """Stage additive payload patches before changing source visibility."""
        return await self._mutate("set_payload", collection_name, kwargs)

    async def overwrite_payload(self, collection_name: str, **kwargs: Any) -> Any:
        """Stage explicit payload replacements under the source fence."""
        return await self._mutate("overwrite_payload", collection_name, kwargs)

    async def delete_payload(self, collection_name: str, **kwargs: Any) -> Any:
        """Stage payload-key removals under the source fence."""
        return await self._mutate("delete_payload", collection_name, kwargs)

    async def update_vectors(self, collection_name: str, **kwargs: Any) -> Any:
        """Verify changed vectors before publishing the complete source revision."""
        return await self._mutate("update_vectors", collection_name, kwargs)

    async def delete_vectors(self, collection_name: str, **kwargs: Any) -> Any:
        """Stage named vector removals with complete representation receipts."""
        return await self._mutate("delete_vectors", collection_name, kwargs)

    async def batch_update_points(self, collection_name: str, **kwargs: Any) -> Any:
        """Coordinate bounded mixed vector/payload mutations by source."""
        return await self._mutate("batch_update_points", collection_name, kwargs)
