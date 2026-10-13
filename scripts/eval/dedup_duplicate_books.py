#!/usr/bin/env python3
"""Find duplicate book ingests that resolve to the same canonical source.

Groups Qdrant chunks by the canonical book identity resolved from the Postgres
``sources`` table (``resolve_canonical_source``) and reports/deletes redundant
source_id ingests of the same file. Dry-run is the default; pass ``--apply`` to
delete redundant chunks.

Source ids use the ``vo-rag-corpus-sha256:`` prefix, and
``--version-hash`` is OPTIONAL (default None / no version filter) because
vo-rag book_rag chunks are unversioned — pass a version_hash only when scanning the
migrated golden corpus that still carries old version stamps. NOTE: this relies on
the ``sources`` table being populated (see ``resolve_canonical_source``); on a
corpus where ``sources`` is empty every source_id reports as unresolved.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import asyncpg  # type: ignore[import-untyped]
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

from vo_rag.core.storage.client_factory import build_qdrant_client
from vo_rag.pipelines.book_rag.source_identity import CanonicalSource, resolve_canonical_source

RAW_COLLECTION = "vo_rag_books_raw_v1"
CANONICAL_PREFIX = "vo-rag-corpus-sha256:"


@dataclass(frozen=True)
class SourceIdStats:
    source_id: str
    chunk_count: int
    source: CanonicalSource


@dataclass(frozen=True)
class DuplicatePlan:
    canonical_key: str
    entries: list[SourceIdStats]
    keep: SourceIdStats
    delete: list[SourceIdStats]


def _scan_filter(source_id: str | None, version_hash: str | None) -> qmodels.Filter | None:
    must: list[Any] = []
    if source_id is not None:
        must.append(
            qmodels.FieldCondition(key="source_id", match=qmodels.MatchValue(value=source_id))
        )
    if version_hash is not None:
        must.append(
            qmodels.FieldCondition(key="version_hash", match=qmodels.MatchValue(value=version_hash))
        )
    return qmodels.Filter(must=must) if must else None


async def _scroll_source_counts(
    client: AsyncQdrantClient,
    *,
    collection: str,
    version_hash: str | None,
    batch_size: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    offset: Any = None
    scroll_filter = _scan_filter(None, version_hash)
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload if isinstance(point.payload, dict) else {}
            source_id = payload.get("source_id")
            if source_id:
                counts[str(source_id)] += 1
        if offset is None:
            return counts


async def _resolve_groups(
    conn: asyncpg.Connection,
    counts: Counter[str],
) -> tuple[dict[str, list[SourceIdStats]], list[str]]:
    groups: dict[str, list[SourceIdStats]] = defaultdict(list)
    unresolved: list[str] = []
    for source_id, chunk_count in sorted(counts.items()):
        source = await resolve_canonical_source(source_id, conn=conn)
        if source is None:
            unresolved.append(source_id)
            continue
        canonical_key = source.file_hash or source.canonical_id
        groups[canonical_key].append(
            SourceIdStats(source_id=source_id, chunk_count=chunk_count, source=source)
        )
    return dict(groups), unresolved


def _is_canonical_source_id(entry: SourceIdStats) -> bool:
    return entry.source_id == entry.source.canonical_id or entry.source_id.startswith(
        CANONICAL_PREFIX
    )


def _choose_keep(entries: list[SourceIdStats]) -> SourceIdStats:
    return max(
        entries,
        key=lambda entry: (_is_canonical_source_id(entry), entry.chunk_count, entry.source_id),
    )


def _build_duplicate_plans(groups: dict[str, list[SourceIdStats]]) -> list[DuplicatePlan]:
    plans: list[DuplicatePlan] = []
    for canonical_key, entries in sorted(groups.items()):
        distinct_entries = sorted(entries, key=lambda entry: entry.source_id)
        if len({entry.source_id for entry in distinct_entries}) <= 1:
            continue
        keep = _choose_keep(distinct_entries)
        delete = [entry for entry in distinct_entries if entry.source_id != keep.source_id]
        plans.append(
            DuplicatePlan(
                canonical_key=canonical_key,
                entries=distinct_entries,
                keep=keep,
                delete=delete,
            )
        )
    return plans


async def _delete_sources(
    client: AsyncQdrantClient,
    *,
    collection: str,
    version_hash: str | None,
    plans: list[DuplicatePlan],
) -> int:
    deleted_chunks = 0
    for plan in plans:
        for entry in plan.delete:
            selector = _scan_filter(entry.source_id, version_hash)
            if selector is None:
                continue
            await client.delete(collection_name=collection, points_selector=selector, wait=True)
            deleted_chunks += entry.chunk_count
    return deleted_chunks


def _print_plan(
    *,
    plans: list[DuplicatePlan],
    unresolved: list[str],
    collection: str,
    version_hash: str | None,
    apply: bool,
) -> None:
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"{mode}: scanned {collection} version_hash={version_hash}")
    if unresolved:
        print(f"unresolved source_ids: {len(unresolved)}")
        for source_id in unresolved:
            print(f"  unresolved: {source_id}")
    if not plans:
        print("duplicate canonical books: 0")
        return
    print(f"duplicate canonical books: {len(plans)}")
    for plan in plans:
        print("")
        print(f"canonical_key: {plan.canonical_key}")
        print("source_ids:")
        for entry in plan.entries:
            marker = "KEEP" if entry.source_id == plan.keep.source_id else "DELETE"
            print(
                f"  [{marker}] {entry.source_id} chunks={entry.chunk_count} "
                f"matched_by={entry.source.matched_by} canonical_id={entry.source.canonical_id}"
            )
        delete_list = ", ".join(entry.source_id for entry in plan.delete)
        action = "delete" if apply else "would delete"
        reason = (
            "canonical source_id" if _is_canonical_source_id(plan.keep) else "highest chunk count"
        )
        print(f"recommendation: keep {plan.keep.source_id} ({reason}); {action} {delete_list}")


async def _amain(args: argparse.Namespace) -> int:
    qdrant_url = args.qdrant_url or os.environ["QDRANT_URL"]
    postgres_url = (
        args.postgres_url or os.environ.get("VORAG_DATABASE_URL") or os.environ.get("DATABASE_URL")
    )
    if not postgres_url:
        raise SystemExit(
            "postgres url required (--postgres-url / VORAG_DATABASE_URL / DATABASE_URL)"
        )
    qdrant = build_qdrant_client(
        url=qdrant_url,
        api_key=(args.qdrant_api_key or os.environ.get("QDRANT_API_KEY")) or None,
        prefer_grpc=False,
        timeout=args.timeout,
    )
    conn = await asyncpg.connect(postgres_url)
    try:
        counts = await _scroll_source_counts(
            qdrant,
            collection=args.collection,
            version_hash=args.version_hash,
            batch_size=args.batch_size,
        )
        groups, unresolved = await _resolve_groups(conn, counts)
        plans = _build_duplicate_plans(groups)
        _print_plan(
            plans=plans,
            unresolved=unresolved,
            collection=args.collection,
            version_hash=args.version_hash,
            apply=args.apply,
        )
        if args.apply:
            deleted = await _delete_sources(
                qdrant, collection=args.collection, version_hash=args.version_hash, plans=plans
            )
            print(f"deleted_chunks: {deleted}")
        else:
            print(f"would_delete_chunks: {sum(e.chunk_count for p in plans for e in p.delete)}")
        return 0
    finally:
        await conn.close()
        await qdrant.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete redundant Qdrant chunks")
    parser.add_argument("--collection", default=RAW_COLLECTION)
    parser.add_argument("--version-hash", default=None, help="optional; only for versioned corpora")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--postgres-url", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    return asyncio.run(_amain(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
