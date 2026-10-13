"""Pin a query's Books/Reddit source pointers across multi-call expansion."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class PublicationReadView:
    pool_id: int
    pointers: Mapping[tuple[str, str], Mapping[str, int | None]]


_VIEW: ContextVar[PublicationReadView | None] = ContextVar(
    "source_publication_read_view", default=None
)


def captured_pointers(pool: Any, corpus: str, collection: str) -> dict[str, int | None] | None:
    """Use a captured pointer map only for the database that supplied it."""
    view = _VIEW.get()
    if view is None or view.pool_id != id(pool):
        return None
    return dict(view.pointers.get((corpus, collection), {}))


def publication_query[**P, T](
    function: Callable[P, Awaitable[T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    """Capture all managed query pointers in one PostgreSQL statement.

    Old immutable revisions remain retained throughout queries. The snapshot
    pins pointer decisions, not every legacy corpus point or external store.
    Nested retrieval shares the view; independent tasks and databases do not.
    """

    @wraps(function)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        from vo_rag.core.storage.qdrant import QdrantStore

        store: Any = None
        for name in ("store", "qdrant", "retriever"):
            candidate: Any = kwargs.get(name)
            if candidate is not None:
                store = (
                    candidate
                    if isinstance(candidate, QdrantStore)
                    else getattr(candidate, "qdrant", None)
                )
                if isinstance(store, QdrantStore):
                    break
        if store is None and args:
            store = getattr(args[0], "qdrant", None)
        if not isinstance(store, QdrantStore) or not store.source_visibility_enabled:
            return await function(*args, **kwargs)
        requested = kwargs.get("requested_collections")
        if (
            isinstance(requested, list)
            and requested
            and not any(
                isinstance(name, str)
                and (name.startswith("vo_rag_books_") or name == "vo_rag_reddit_v1")
                for name in requested
            )
        ):
            return await function(*args, **kwargs)
        coordinator = await store._source_coordinator("vo_rag_books_raw_v1")
        if coordinator is None:
            raise RuntimeError("source read view requires its publication coordinator")
        pool = coordinator.pool
        existing = _VIEW.get()
        if existing is not None and existing.pool_id == id(pool):
            return await function(*args, **kwargs)
        rows = await pool.fetch(
            "SELECT corpus, collection, source_id, active_revision FROM source_publications "
            "WHERE corpus = ANY($1::text[])",
            ["books", "reddit"],
        )
        pointers: dict[tuple[str, str], dict[str, int | None]] = {}
        for row in rows:
            group = pointers.setdefault((str(row["corpus"]), str(row["collection"])), {})
            group[str(row["source_id"])] = row["active_revision"]
        view = PublicationReadView(
            id(pool),
            MappingProxyType({key: MappingProxyType(value) for key, value in pointers.items()}),
        )
        token = _VIEW.set(view)
        try:
            return await function(*args, **kwargs)
        finally:
            _VIEW.reset(token)

    return wrapped
