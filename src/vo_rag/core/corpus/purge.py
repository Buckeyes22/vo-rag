"""Purge every trace of a book from Qdrant and Postgres (VOR-1427).

The one primitive every later corpus-repair step (duplicates, health, a bad
re-ingest) builds on: given a ``source_id``, remove its points from every
book collection, detach it from any *shared* graph node instead of deleting
that node outright, and delete its matching Postgres rows. Dry run
(``apply=False``) is the default everywhere this is called from — nothing is
written until the caller passes ``apply=True``.

Qdrant collection -> filter key:
    raw / sterile / contextualized  -> ``source_id``
    summary_tree / raptor / dsi     -> ``book_id``
    graph_nodes                     -> ``book_ids`` (list membership)
    graph_edges                     -> ``source_node_id`` / ``target_node_id``

Graph node/edge id form: ``node_id_for``
(``core/graph/schemas.py``) produces an *unhyphenated* 32-hex-char id.
``upsert_node`` writes it as the Qdrant point id (``PointStruct(id=node.
node_id, ...)``) and also stores it verbatim in the node's own ``node_id``
payload field. ``GraphEdge.source_node_id``/``target_node_id`` store that
same unhyphenated string. Qdrant treats a 32-hex string as a UUID: point-id
lookups (``retrieve``, ``delete`` by id list) normalize it internally and
work with either form, but ``scroll``/``retrieve`` *return* ``point.id`` in
canonical **hyphenated** form, and payload *filter* matching
(``FieldCondition``/``MatchAny``) is a literal string comparison with no
normalization. So matching edges against ``str(point.id)`` (hyphenated)
never matches the edges' unhyphenated payload strings — 0 edges, always.
Fixed by reading the node's own ``node_id`` payload field (the exact string
edges reference) and, defensively, both UUID forms of the point id, as the
edge-matching id set — while keeping the actual Qdrant point ids separate
for the node delete itself, since that operation normalizes either form.
(``core/graph/persistence.py:get_neighbors`` matches ids the same way.)

A graph node with more than one book in ``book_ids`` is shared: purging one
of its books detaches that id via ``set_payload`` rather than deleting the
node. Only a node left with an empty ``book_ids`` is actually removed, and
only edges touching an *actually removed* node are deleted with it.

Order of operations: Qdrant is read-only
(preflight: collection existence + counts) until *after* the Postgres
transaction has committed (or, on a dry run, until after its read-only
pass). This means a Postgres failure can never leave a book half-purged
(vectors gone, rows still present) — Postgres either fully commits or fully
rolls back before any Qdrant write is attempted. The inverse (a Qdrant
write failing after Postgres already committed) can still leave a
half-purged book; Qdrant deletes/detaches are each individually idempotent
(re-running against already-deleted points/rows is a no-op), so on that
failure ``purge_source`` raises with exactly which Qdrant collections it
had already finished, and re-running ``purge_source`` for the same
``source_id`` finishes the rest safely.

Postgres, in one transaction:
    - ``source_ingest_generations`` (deleted)
    - ``knowledge_artifacts`` (``pipeline='book_rag' AND content->>'source_id'``, deleted)
    - ``sources`` (deleted)
    - ``ideas`` / ``compliance_log``: **never deleted** (see below),
      only counted as ``ideas_unlinked`` / ``compliance_log_unlinked``
    - ``book_categories``: deleted only when ``keep_categories=False``, and
      only if the table exists (see below); the report key is always
      present regardless

``intake_envelopes`` are kept as history and never touched here.

FK grep (``grep -rn "REFERENCES knowledge_artifacts\\|REFERENCES sources"
src/vo_rag/core/db/migrations``): two hits, both in ``001_initial.sql`` --
``ideas.artifact_id`` and ``compliance_log.artifact_id``, both ``REFERENCES
knowledge_artifacts(id) ON DELETE SET NULL``. No table ``REFERENCES
sources``.

Never delete ``ideas`` or ``compliance_log`` rows. Their FK is ``ON DELETE SET NULL`` precisely so
that deleting the `knowledge_artifacts` row they reference nulls the link
instead of erroring -- deleting the rows themselves would destroy
compliance/audit history. ``_purge_postgres`` only ever runs a read-only
``SELECT count(*)`` against these two tables, in both dry-run and apply
mode, reported as ``ideas_unlinked`` / ``compliance_log_unlinked`` (always
present, 0 included).

``book_categories`` isn't created by any migration in this tree.
``_purge_postgres`` guards it with
``SELECT to_regclass('public.book_categories')`` and reports
``"absent"`` (skipping the delete/count) rather than raising
``UndefinedTableError`` when the table doesn't exist -- in both dry-run and
apply mode.

``postgres["book_categories"]`` is always present and always one of three shapes:
    - ``int``: the row count deleted (apply) or that would be deleted (dry
      run), when ``keep_categories=False`` and the table exists
    - ``"kept"``: ``keep_categories=True`` -- the table is never touched or
      even checked for existence, since the caller asked to leave it alone
    - ``"absent"``: ``keep_categories=False`` and the table doesn't exist
"""

from __future__ import annotations

import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import asyncpg  # type: ignore[import-untyped]
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from vo_rag.core.storage.qdrant import QdrantStore

_BY_SOURCE: tuple[str, ...] = (
    "vo_rag_books_raw_v1",
    "vo_rag_books_sterile_v1",
    "vo_rag_books_contextualized_v1",
)
_BY_BOOK: tuple[str, ...] = (
    "vo_rag_books_summary_tree_v1",
    "vo_rag_books_raptor_v1",
    "vo_rag_books_dsi_v1",
)
_NODES = "vo_rag_books_graph_nodes_v1"
_EDGES = "vo_rag_books_graph_edges_v1"
_ABSENT = "absent"
_KEPT = "kept"


@dataclass
class PurgeReport:
    """What was (or, on a dry run, would be) deleted for one ``source_id``.

    ``qdrant`` and ``postgres`` values are normally ``int`` counts; a
    collection or table that doesn't exist on the target instance reports
    the string ``"absent"`` for that key instead of raising.
    ``postgres["book_categories"]`` is always present and additionally uses
    ``"kept"`` when ``keep_categories=True`` -- see the module docstring.
    """

    source_id: str
    qdrant: dict[str, int | str] = field(default_factory=dict)
    graph_nodes_detached: int = 0
    postgres: dict[str, int | str] = field(default_factory=dict)


def _match(key: str, value: str) -> Filter:
    return Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])


async def _count(client: AsyncQdrantClient, name: str, flt: Filter) -> int:
    result = await client.count(collection_name=name, count_filter=flt, exact=True)
    return int(result.count)


def _node_id_variants(point_id: Any, payload: dict[str, Any]) -> set[str]:
    """Every string form a graph edge might use to reference this node.

    Prefers the node's own ``node_id`` payload field (the exact value
    ``GraphEdge`` stores). Falls back to both UUID forms of the Qdrant point
    id for defensiveness, in case an older/malformed point lacks it.
    """
    variants = {str(point_id)}
    node_id = payload.get("node_id")
    if isinstance(node_id, str):
        variants.add(node_id)
    with suppress(ValueError):
        variants.add(uuid.UUID(str(point_id)).hex)
    return variants


async def purge_source(
    store: QdrantStore,
    pool: asyncpg.Pool,
    source_id: str,
    *,
    apply: bool,
    keep_categories: bool = True,
) -> PurgeReport:
    """Remove every trace of ``source_id``. Reports what it did (or would do).

    ``apply=False`` (the CLI default) only reads: it counts what would be
    deleted/detached without writing anywhere. ``apply=True`` runs the
    Postgres transaction first, then the Qdrant deletes/detaches -- see the
    module docstring's "Order of operations" note.
    """
    if apply and getattr(store, "source_visibility_enabled", False):
        # Publication retains old revisions. A SQL-first erasure would race
        # source admission, and detaching a shared graph book is not a safe
        # metadata-only change. Reject before either store can be modified.
        raise RuntimeError(
            "source purge apply requires a coordinated cross-store erasure owner; "
            "source publication retains old revisions and cannot prove physical erasure. "
            "Dry-run remains available; no PostgreSQL or Qdrant writes were attempted."
        )
    client = store._client  # noqa: SLF001
    report = PurgeReport(source_id=source_id)

    # ---- Preflight (read-only): collection existence + counts ----------
    existing = {c.name for c in (await client.get_collections()).collections}

    point_filters: dict[str, Filter] = {}
    for name, key in [
        *((c, "source_id") for c in _BY_SOURCE),
        *((c, "book_id") for c in _BY_BOOK),
    ]:
        if name not in existing:
            report.qdrant[name] = _ABSENT
            continue
        flt = _match(key, source_id)
        point_filters[name] = flt
        report.qdrant[name] = await _count(client, name, flt)

    deleted_points: list[Any] = []
    detach_updates: list[tuple[Any, list[str]]] = []
    edge_match_ids: set[str] = set()

    if _NODES not in existing:
        report.qdrant[_NODES] = _ABSENT
    else:
        offset: Any = None
        while True:
            points, offset = await client.scroll(
                collection_name=_NODES,
                scroll_filter=_match("book_ids", source_id),
                with_payload=["book_ids", "node_id"],
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                remaining = [str(b) for b in payload.get("book_ids", []) if b != source_id]
                if remaining:
                    detach_updates.append((point.id, remaining))
                else:
                    deleted_points.append(point.id)
                    edge_match_ids |= _node_id_variants(point.id, payload)
            if offset is None:
                break
        report.graph_nodes_detached = len(detach_updates)
        report.qdrant[_NODES] = len(deleted_points)

    edge_filter: Filter | None = None
    if _EDGES not in existing:
        report.qdrant[_EDGES] = _ABSENT
    elif not deleted_points:
        report.qdrant[_EDGES] = 0
    else:
        edge_ids = sorted(edge_match_ids)
        edge_filter = Filter(
            should=[
                FieldCondition(key="source_node_id", match=MatchAny(any=edge_ids)),
                FieldCondition(key="target_node_id", match=MatchAny(any=edge_ids)),
            ]
        )
        report.qdrant[_EDGES] = await _count(client, _EDGES, edge_filter)

    # ---- Postgres transaction (writes, or reads on a dry run) -----------
    # Runs before any Qdrant write so a Postgres failure never leaves points
    # deleted with their rows still present.
    report.postgres = await _purge_postgres(
        pool, source_id, apply=apply, keep_categories=keep_categories
    )

    if not apply:
        return report

    # ---- Qdrant writes ---------------------------------------------------
    # Postgres has already committed. Every delete/detach here is
    # individually idempotent, so on failure we report exactly which
    # collections finished -- re-running purge_source for the same
    # source_id is safe and completes the rest.
    completed: list[str] = []
    try:
        for name, flt in point_filters.items():
            if report.qdrant[name]:
                await client.delete(collection_name=name, points_selector=flt, wait=True)
            completed.append(name)

        for point_id, remaining in detach_updates:
            await client.set_payload(
                collection_name=_NODES,
                payload={"book_ids": remaining},
                points=[point_id],
                wait=True,
            )
        if detach_updates:
            completed.append(f"{_NODES} (detach)")

        if deleted_points:
            if edge_filter is not None:
                await client.delete(collection_name=_EDGES, points_selector=edge_filter, wait=True)
                completed.append(_EDGES)
            # mypy: `points_selector`'s list overload is `list[int|str|UUID|PointId]`,
            # and lists are invariant, so a plain list of point ids doesn't satisfy it
            # even though its elements are union members -- the client accepts it fine.
            node_ids: Any = deleted_points
            await client.delete(collection_name=_NODES, points_selector=node_ids, wait=True)
            completed.append(f"{_NODES} (delete)")
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant purge for {source_id!r} failed after completing {completed!r}; "
            "Postgres rows are already committed -- re-running purge_source for this "
            f"source_id is safe (idempotent) and will finish the rest: {exc}"
        ) from exc

    return report


async def _resolve(
    conn: asyncpg.Connection,
    count_query: str,
    delete_query: str,
    *args: object,
    apply: bool,
) -> int:
    """Run ``count_query`` (dry run) or ``delete_query ... RETURNING`` (apply)."""
    if apply:
        rows = await conn.fetch(delete_query, *args)
        return len(rows)
    value = await conn.fetchval(count_query, *args)
    return int(value)


async def _purge_postgres(
    pool: asyncpg.Pool, source_id: str, *, apply: bool, keep_categories: bool
) -> dict[str, int | str]:
    postgres: dict[str, int | str] = {}
    async with pool.acquire() as conn, conn.transaction():
        artifact_ids: list[str] = [
            row["id"]
            for row in await conn.fetch(
                "SELECT id FROM knowledge_artifacts "
                "WHERE pipeline = 'book_rag' AND content->>'source_id' = $1",
                source_id,
            )
        ]

        # Never delete ideas/compliance_log -- see module docstring.
        # Read-only count in both modes, always present.
        postgres["ideas_unlinked"] = int(
            await conn.fetchval(
                "SELECT count(*) FROM ideas WHERE artifact_id = ANY($1::text[])",
                artifact_ids,
            )
        )
        postgres["compliance_log_unlinked"] = int(
            await conn.fetchval(
                "SELECT count(*) FROM compliance_log WHERE artifact_id = ANY($1::text[])",
                artifact_ids,
            )
        )

        postgres["source_ingest_generations"] = await _resolve(
            conn,
            "SELECT count(*) FROM source_ingest_generations WHERE source_id = $1",
            "DELETE FROM source_ingest_generations WHERE source_id = $1 RETURNING generation_id",
            source_id,
            apply=apply,
        )

        postgres["knowledge_artifacts"] = await _resolve(
            conn,
            "SELECT count(*) FROM knowledge_artifacts "
            "WHERE pipeline = 'book_rag' AND content->>'source_id' = $1",
            "DELETE FROM knowledge_artifacts "
            "WHERE pipeline = 'book_rag' AND content->>'source_id' = $1 RETURNING id",
            source_id,
            apply=apply,
        )

        postgres["sources"] = await _resolve(
            conn,
            "SELECT count(*) FROM sources WHERE id = $1",
            "DELETE FROM sources WHERE id = $1 RETURNING id",
            source_id,
            apply=apply,
        )

        # Always present (#8 follow-up): "kept" when the caller asked to
        # leave it alone (no existence check needed), else an int count or
        # "absent" when the table doesn't exist.
        if keep_categories:
            postgres["book_categories"] = _KEPT
        else:
            table_exists = await conn.fetchval("SELECT to_regclass('public.book_categories')")
            if table_exists is None:
                postgres["book_categories"] = _ABSENT
            else:
                postgres["book_categories"] = await _resolve(
                    conn,
                    "SELECT count(*) FROM book_categories WHERE source_id = $1",
                    "DELETE FROM book_categories WHERE source_id = $1 RETURNING source_id",
                    source_id,
                    apply=apply,
                )

    return postgres


__all__ = ["PurgeReport", "purge_source"]
