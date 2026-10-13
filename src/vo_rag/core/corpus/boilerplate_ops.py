"""Boilerplate-chunk scan/delete over the books-raw collection (VOR-1427).

This is the core logic behind ``vo-rag corpus boilerplate --scan|--delete``
(``vo_rag.cli.corpus_ops``); the CLI module stays a thin click wrapper, the
same layering as ``core/corpus/purge.py``/``core/corpus/health.py``.

``boilerplate_point_ids`` scrolls per book (mirroring
``vo_rag.core.corpus.health._collect_one``'s shape, and reusing its exact
``scroll_book_payloads`` loop / ``all_source_ids`` facet-paging -- there is
exactly one per-book scroll loop and one facet-paging implementation in this
codebase, not two copies with a hardcoded page size) with a ``content``-only
payload and ``with_vectors=False``, and matches each chunk against
``boilerplate_payload_matcher`` -- the same adapter
``vo_rag.cli.corpus_ops``'s ``health`` command feeds into
``run_health``'s ``boilerplate_matcher`` hook, so ``corpus boilerplate`` and
``corpus health``'s ``boilerplate_points`` count can never disagree.

``delete_boilerplate_points`` deletes matched point ids in batches of
``DELETE_BATCH`` (256).

The raw collection is not the only copy of a chunk: ``corpus purge`` also
clears ``vo_rag_books_sterile_v1`` (which publication/partner queries
read) and ``vo_rag_books_contextualized_v1``. ``derived_boilerplate_point_ids``
scans those two with the same matcher, per ``source_id`` (the raw
collection's books, or the caller's ids), so ``--delete`` removes the
boilerplate everywhere a query could still find it. A collection that
doesn't exist on the deployment is reported as ``None`` and skipped.
"""

from __future__ import annotations

from typing import Any

from vo_rag.core.corpus.health import BOOKS_RAW_COLLECTION, all_source_ids, scroll_book_payloads
from vo_rag.core.storage.qdrant import QdrantStore
from vo_rag.pipelines.book_rag.boilerplate import is_boilerplate

#: Batch size for `corpus boilerplate --delete`.
DELETE_BATCH = 256

#: Collections that hold per-``source_id`` copies of the raw chunks (the
#: same set ``core/corpus/purge.py`` clears by ``source_id``).
DERIVED_COLLECTIONS: tuple[str, ...] = (
    "vo_rag_books_sterile_v1",
    "vo_rag_books_contextualized_v1",
)


def boilerplate_payload_matcher(payload: dict[str, Any]) -> bool:
    """``health.BoilerplateMatcher``-shaped adapter over ``is_boilerplate``
    (which takes the chunk text itself, not a whole payload dict). Public:
    ``vo_rag.cli.corpus_ops``'s ``health`` command feeds this straight into
    ``run_health``'s ``boilerplate_matcher`` hook, and ``boilerplate_point_ids``
    below calls it too, so ``corpus health`` and ``corpus boilerplate`` can
    never disagree on what counts as boilerplate."""
    content = payload.get("content")
    return is_boilerplate(str(content if content is not None else ""))


async def _matched_ids(client: Any, source_ids: list[str], collection: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for source_id in source_ids:
        matched: list[str] = []
        async for point in scroll_book_payloads(
            client, source_id, with_payload=["content"], collection=collection
        ):
            if boilerplate_payload_matcher(point.payload or {}):
                matched.append(str(point.id))
        if matched:
            result[source_id] = matched
    return result


async def boilerplate_point_ids(
    store: QdrantStore, *, source_ids: list[str] | None
) -> dict[str, list[str]]:
    """``source_id -> [point id, ...]`` for every boilerplate chunk in the
    books-raw collection.

    Read-only: never writes anything. ``source_ids=None`` scans every book
    (via ``health.all_source_ids``'s facet).
    """
    client = store._client  # noqa: SLF001
    ids = source_ids if source_ids is not None else await all_source_ids(client)
    return await _matched_ids(client, ids, BOOKS_RAW_COLLECTION)


async def derived_boilerplate_point_ids(
    store: QdrantStore, *, source_ids: list[str] | None
) -> dict[str, dict[str, list[str]] | None]:
    """``collection -> source_id -> [point id, ...]`` over
    ``DERIVED_COLLECTIONS``; ``None`` for a collection that doesn't exist.

    Read-only. ``source_ids=None`` scans every book in the books-raw
    collection (``health.all_source_ids``) -- all of them, not only the
    ones with raw boilerplate left, so a copy the raw delete already
    cleared is still found.
    """
    client = store._client  # noqa: SLF001
    existing = {c.name for c in (await client.get_collections()).collections}
    ids = source_ids if source_ids is not None else await all_source_ids(client)
    result: dict[str, dict[str, list[str]] | None] = {}
    for collection in DERIVED_COLLECTIONS:
        result[collection] = (
            await _matched_ids(client, ids, collection) if collection in existing else None
        )
    return result


async def delete_boilerplate_points(
    store: QdrantStore, ids: list[str], *, collection: str = BOOKS_RAW_COLLECTION
) -> int:
    """Delete *ids* from `collection` (the books-raw collection by default)
    in batches of ``DELETE_BATCH``. Returns the count deleted."""
    client = store._client  # noqa: SLF001
    from vo_rag.core.storage.publication_client import PublicationClient

    if isinstance(client, PublicationClient):
        # A publication mutation consumes the selected physical IDs once and
        # groups them by source under separate fences. Reusing an ID in later
        # batches would be stale after that source's first pointer switch.
        if len(ids) > 10_000:
            raise ValueError("boilerplate delete exceeds the 10000-point source mutation ceiling")
        if not ids:
            return 0
        publication_selector: Any = ids
        await client.delete(
            collection_name=collection, points_selector=publication_selector, wait=True
        )
        return len(ids)

    deleted = 0
    for start in range(0, len(ids), DELETE_BATCH):
        batch = ids[start : start + DELETE_BATCH]
        # mypy: `points_selector`'s list overload is `list[int|str|UUID|PointId]`,
        # and lists are invariant, so a plain `list[str]` doesn't satisfy it even
        # though `str` is a union member -- the client accepts it fine at runtime
        # (mirrors the identical workaround in `core/corpus/purge.py`).
        points_selector: Any = batch
        await client.delete(collection_name=collection, points_selector=points_selector, wait=True)
        deleted += len(batch)
    return deleted


__all__ = [
    "DELETE_BATCH",
    "DERIVED_COLLECTIONS",
    "boilerplate_payload_matcher",
    "boilerplate_point_ids",
    "delete_boilerplate_points",
    "derived_boilerplate_point_ids",
]
