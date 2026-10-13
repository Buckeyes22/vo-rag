"""Persistence for track-and-field tags (VOR-1443): the Postgres per-book
records and chunk classifications keyed by content hash, and the Qdrant side
(source listing, per-book chunk scroll, batched payload apply). asyncpg's jsonb
codec encodes dicts; pass dicts, never json.dumps strings (a string is stored
as a JSON string, not an object)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qdrant_client.models import (
    DeletePayload,
    DeletePayloadOperation,
    FieldCondition,
    Filter,
    MatchValue,
    SetPayload,
    SetPayloadOperation,
)

from vo_rag.core.corpus.health import (
    BOOKS_RAW_COLLECTION,
    all_source_ids,
    scroll_book_payloads,
)
from vo_rag.core.corpus.tf_schema import (
    TF_TAG_VERSION,
    BookTags,
    ChunkTags,
    chunk_payload,
    content_sha256,
)

TF_DOMAIN_TAG = "track-and-field"
_APPLY_BATCH = 256
#: Cleared by apply_payloads on any point whose payload omits tf_pub_year (the
#: book's year was cleared to None), so a stale stamp does not survive a
#: SetPayload that only ever adds/overwrites keys (VOR-1443).
_YEAR_FIELDS = ["tf_pub_year", "tf_pub_decade"]


async def upsert_book(pool: Any, source_id: str, book: BookTags, *, reviewed: bool) -> bool:
    """Upsert one book's record; True when a row was actually written.

    asyncpg answers with a command tag, and the ``ON CONFLICT`` guard means a
    row need not be written: an unreviewed draft meeting a reviewed record
    updates nothing and reports ``"INSERT 0 0"``. Callers report writes, not
    attempts, so the count has to come from the tag's trailing row count.

    Raises ``ValueError`` for an empty or blank ``source_id`` before the pool
    is touched -- a ``""``-keyed row is never a valid record.
    """
    if not source_id or not source_id.strip():
        raise ValueError("source_id is required")
    status = await pool.execute(
        """INSERT INTO tf_book_tags (source_id, tags, reviewed, tag_version, updated_at)
           VALUES ($1, $2, $3, $4, NOW())
           ON CONFLICT (source_id) DO UPDATE
           SET tags = EXCLUDED.tags, reviewed = EXCLUDED.reviewed,
               tag_version = EXCLUDED.tag_version, updated_at = NOW()
           WHERE NOT tf_book_tags.reviewed OR EXCLUDED.reviewed""",
        source_id,
        book.model_dump(),
        reviewed,
        TF_TAG_VERSION,
    )
    count = str(status).rsplit(" ", 1)[-1]
    return count.isdigit() and int(count) > 0


async def book_records(pool: Any) -> dict[str, tuple[BookTags, bool]]:
    rows = await pool.fetch("SELECT source_id, tags, reviewed FROM tf_book_tags")
    return {r["source_id"]: (BookTags.model_validate(r["tags"]), bool(r["reviewed"])) for r in rows}


async def upsert_chunk_tags(
    pool: Any,
    source_id: str,
    rows: list[tuple[str, ChunkTags]],
    *,
    model: str,
    provenance_by_hash: Mapping[str, Mapping[str, object]] | None = None,
) -> int:
    """Upsert chunk tags for one book, returning the number of rows submitted.

    The count is rows submitted, not rows changed: duplicate hashes within one
    call collapse into a single row on conflict.

    Raises ``ValueError`` for an empty or blank ``source_id`` before the pool
    is touched, even when ``rows`` is empty.
    """
    if not source_id or not source_id.strip():
        raise ValueError("source_id is required")
    if not rows:
        return 0
    await pool.executemany(
        """INSERT INTO tf_chunk_tags (source_id, content_sha256, tags, tag_version, tagger_model)
           VALUES ($1, $2, $3, $4, $5)
           ON CONFLICT (source_id, content_sha256) DO UPDATE
           SET tags = EXCLUDED.tags, tag_version = EXCLUDED.tag_version,
               tagger_model = EXCLUDED.tagger_model, created_at = NOW()""",
        [
            (
                source_id,
                h,
                {
                    **tags.model_dump(),
                    "_tag_provenance": dict((provenance_by_hash or {}).get(h, {})),
                },
                TF_TAG_VERSION,
                model,
            )
            for h, tags in rows
        ],
    )
    return len(rows)


async def rekey_chunk_tags(pool: Any, source_id: str, pairs: list[tuple[str, str]]) -> int:
    """Move one book's stored chunk tags from their legacy keys to the current ones.

    ``pairs`` is ``(legacy_hash, current_hash)`` per live chunk. Each legacy row is
    copied under its current key -- never over a row already there, which is the
    newer tag -- and then dropped, so a re-run finds nothing to move (VOR-1450).
    A chunk whose new key is another chunk's stored key (same words, different
    spacing) shares that row: the copy is skipped and its legacy row dropped.
    Returns the number of stored rows moved.
    """
    if not source_id or not source_id.strip():
        raise ValueError("source_id is required")
    wanted = {(old, new) for old, new in pairs if old != new}
    if not wanted:
        return 0
    # Only legacy keys that are actually stored move, so the count is rows moved:
    # a re-run, or a book with no stored row for a chunk, reports 0 for it.
    rows = await pool.fetch(
        "SELECT content_sha256 FROM tf_chunk_tags"
        " WHERE source_id = $1 AND content_sha256 = ANY($2::text[])",
        source_id,
        sorted({old for old, _ in wanted}),
    )
    stored = {r["content_sha256"] for r in rows}
    moves = sorted((old, new) for old, new in wanted if old in stored)
    if not moves:
        return 0
    # One transaction: a failure between the copy and the delete leaves the legacy
    # rows in place rather than half a book moved.
    async with pool.acquire() as conn, conn.transaction():
        await conn.executemany(
            """INSERT INTO tf_chunk_tags
                   (source_id, content_sha256, tags, tag_version, tagger_model, created_at)
               SELECT source_id, $3, tags, tag_version, tagger_model, created_at
                 FROM tf_chunk_tags WHERE source_id = $1 AND content_sha256 = $2
               ON CONFLICT (source_id, content_sha256) DO NOTHING""",
            [(source_id, old, new) for old, new in moves],
        )
        await conn.executemany(
            "DELETE FROM tf_chunk_tags WHERE source_id = $1 AND content_sha256 = $2",
            [(source_id, old) for old, _ in moves],
        )
    return len(moves)


async def chunk_tags_by_hash(
    pool: Any, source_id: str
) -> dict[str, tuple[ChunkTags, str, dict[str, object]]]:
    """Content hash -> tags, recorded model, and optional provenance for current rows.

    Missing provenance is preserved as empty/unknown for legacy rows. The recorded
    model rides along so a restamp never stamps today's default in its place.
    """
    rows = await pool.fetch(
        "SELECT content_sha256, tags, tagger_model FROM tf_chunk_tags"
        " WHERE source_id = $1 AND tag_version = $2",
        source_id,
        TF_TAG_VERSION,
    )
    result: dict[str, tuple[ChunkTags, str, dict[str, object]]] = {}
    for row in rows:
        raw = row["tags"]
        provenance = raw.get("_tag_provenance", {}) if isinstance(raw, Mapping) else {}
        result[row["content_sha256"]] = (
            ChunkTags.model_validate(raw),
            str(row["tagger_model"] or ""),
            dict(provenance) if isinstance(provenance, Mapping) else {},
        )
    return result


async def tf_source_ids(client: Any) -> list[str]:
    """Every source_id of a track-and-field book, sorted.

    Delegates the paging to `health.all_source_ids` rather than faceting with
    a flat limit of its own, so a corpus with more distinct source_ids than
    that limit raises instead of coming back silently short.
    """
    return await all_source_ids(
        client,
        facet_filter=Filter(
            must=[FieldCondition(key="domain_tags", match=MatchValue(value=TF_DOMAIN_TAG))]
        ),
    )


def document_order_key(chunk: Mapping[str, Any]) -> tuple[bool, int, str]:
    """Sort key that puts a book's chunks in document order, page first.

    Works alike on a raw scroll payload and on a ``book_chunks`` record. Ascending
    ``page_number`` -- the only ordering field the payload carries -- with
    anything else (missing, or not an int) treated as unnumbered and sorted
    last, tied by ``chunk_id``. The ``isinstance`` guard is what makes a stray
    non-numeric page number land among the unnumbered chunks instead of raising
    when it is compared against an int. Shared by ``book_chunks`` and
    ``vo-rag corpus tf-chunks sample`` so the review sheet walks the book front
    to back exactly as ``tf-books inputs`` reads it (VOR-1443).
    """
    page_number = chunk.get("page_number")
    page = page_number if isinstance(page_number, int) else None
    return (page is None, page or 0, str(chunk.get("chunk_id") or ""))


async def book_chunks(client: Any, source_id: str) -> list[dict[str, Any]]:
    """One record per chunk of ``source_id``, with its content hash computed, in
    document order.

    Qdrant's scroll cursor has no ordering guarantee, so the result is sorted by
    ``document_order_key`` -- ``tf-books inputs`` reads its front matter and
    first-appearance section titles off this order, and ``tf-chunks sample``
    strides over it (VOR-1443).
    """
    out: list[dict[str, Any]] = []
    async for point in scroll_book_payloads(
        client,
        source_id,
        with_payload=["chunk_id", "section_path", "content", "tf_tag_version", "page_number"],
    ):
        p = point.payload or {}
        text = str(p.get("content") or "")
        page_number = p.get("page_number")
        out.append(
            {
                "point_id": point.id,
                "chunk_id": str(p.get("chunk_id") or point.id),
                "section_path": str(p.get("section_path") or ""),
                "content": text,
                "content_sha256": content_sha256(text),
                "tf_tag_version": p.get("tf_tag_version"),
                "page_number": page_number if isinstance(page_number, int) else None,
            }
        )
    out.sort(key=document_order_key)
    return out


async def apply_payloads(client: Any, items: list[tuple[Any, dict[str, Any]]]) -> int:
    """Write ``(point_id, payload)`` pairs, batched; return how many were sent.

    A payload without ``tf_pub_year`` (the book's publication year was cleared to
    ``None``) also gets a ``DeletePayloadOperation`` for ``tf_pub_year`` /
    ``tf_pub_decade`` on that point -- ``SetPayload`` only ever adds or
    overwrites keys, so a stale year from an earlier stamp would otherwise
    survive forever (VOR-1443, used by both ``tf-chunks load`` and
    ``restamp_source``).
    """
    # Publication changes make the old physical IDs stale after the first
    # source commit. Submit this source's complete patch set in one call so the
    # coordinator can select and publish it atomically. The mutation adapter
    # enforces the same finite ceiling before it writes anything.
    from vo_rag.core.storage.publication_client import PublicationClient

    if isinstance(client, PublicationClient):
        if len(items) > 10_000:
            raise ValueError("tag update exceeds the 10000-point source mutation ceiling")
        # _APPLY_BATCH bounds operations per request: each point has its own payload.
        ops: list[Any] = []
        for pid, payload in items:
            ops.append(SetPayloadOperation(set_payload=SetPayload(payload=payload, points=[pid])))
            if "tf_pub_year" not in payload:
                ops.append(
                    DeletePayloadOperation(
                        delete_payload=DeletePayload(keys=_YEAR_FIELDS, points=[pid])
                    )
                )
        if ops:
            await client.batch_update_points(
                collection_name=BOOKS_RAW_COLLECTION, update_operations=ops, wait=True
            )
    else:
        for start in range(0, len(items), _APPLY_BATCH):
            # _APPLY_BATCH bounds operations per request: each point has its own payload.
            ops = []
            for pid, payload in items[start : start + _APPLY_BATCH]:
                ops.append(
                    SetPayloadOperation(set_payload=SetPayload(payload=payload, points=[pid]))
                )
                if "tf_pub_year" not in payload:
                    ops.append(
                        DeletePayloadOperation(
                            delete_payload=DeletePayload(keys=_YEAR_FIELDS, points=[pid])
                        )
                    )
            await client.batch_update_points(
                collection_name=BOOKS_RAW_COLLECTION, update_operations=ops, wait=True
            )
    return len(items)


async def restamp_source(
    pool: Any, client: Any, source_id: str, book: BookTags, *, reviewed: bool
) -> tuple[int, int]:
    """Write cached tags onto every chunk of ``source_id`` whose content hash is
    known; return (stamped, untagged)."""
    cache = await chunk_tags_by_hash(pool, source_id)
    items: list[tuple[Any, dict[str, Any]]] = []
    untagged = 0
    for chunk in await book_chunks(client, source_id):
        cached = cache.get(chunk["content_sha256"])
        if cached is None:
            untagged += 1
            continue
        tags, model, provenance = cached
        # The cache lookup itself is keyed by (source_id, normalized content
        # hash), so restamping can state that exact identity even for legacy
        # classifier receipts that predate explicit point-level binding.
        bound_provenance = dict(provenance)
        if bound_provenance.get("tf_tag_source_id", source_id) != source_id or (
            bound_provenance.get("tf_tag_content_sha256", chunk["content_sha256"])
            != chunk["content_sha256"]
        ):
            untagged += 1
            continue
        bound_provenance["tf_tag_source_id"] = source_id
        bound_provenance["tf_tag_content_sha256"] = chunk["content_sha256"]
        items.append(
            (
                chunk["point_id"],
                chunk_payload(
                    tags,
                    book,
                    reviewed=reviewed,
                    model=model or "unknown",
                    provenance=bound_provenance,
                ),
            )
        )
    await apply_payloads(client, items)
    return len(items), untagged
