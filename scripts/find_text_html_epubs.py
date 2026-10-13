#!/usr/bin/env python3
"""Scan completed book_rag EPUB envelopes for the ``text/html`` chapter
manifest bug (VOR-1427).

A book whose OPF declares chapter items ``media-type="text/html"`` instead
of the EPUB-required ``application/xhtml+xml`` can still produce a few
chunks (e.g. from a readable nav page) and get marked **completed** while
silently missing its body -- ``EpubParser`` now normalizes those manifests
before parsing (see ``_normalize_epub`` in
``src/vo_rag/core/parsing/parsers/epub.py``), but envelopes ingested before
that fix may still be sitting on an incomplete corpus. This script finds
them.

Read-only: only issues SELECTs against Postgres and GETs against Garage.
It never writes to Postgres, Garage, or the filesystem, and never re-queues
or mutates any envelope.

Settings (Garage endpoint/credentials/bucket, Postgres URL) come from
``vo_rag.core.config.get_settings()`` -- the same config every other
book_rag script/CLI uses (see ``scripts/check_migrations.py``,
``src/vo_rag/cli/corpus_ocr.py``).

Output: one CSV line per completed book_rag envelope whose ``garage_key``
ends in ``.epub``::

    envelope_id,source_id,text_html_items

``text_html_items`` is the count of ``media-type="text/html"`` items across
the EPUB's OPF manifest(s); ``0`` means the EPUB is fine as-is.

Usage:
  uv run python scripts/find_text_html_epubs.py
  uv run python scripts/find_text_html_epubs.py --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from vo_rag.core.parsing.parsers.epub import (
    count_text_html_manifest_items as count_text_html_manifest_items,
)

#: Completed book_rag envelopes whose stored source file is an EPUB.
#: ``source_ingest_generations`` (if any row exists for the envelope) carries
#: the real ``source_id`` used at ingest time; envelopes ingested outside the
#: generation-scoped path have none, so ``file_hash`` is returned as a
#: fallback for the caller to derive ``canonical_source_id`` from.
_ENVELOPES_SQL = """
SELECT e.id AS envelope_id,
       e.raw_content->>'garage_key' AS garage_key,
       e.raw_content->>'file_hash' AS file_hash,
       sig.source_id AS generation_source_id
FROM intake_envelopes e
LEFT JOIN LATERAL (
    SELECT source_id
    FROM source_ingest_generations
    WHERE envelope_id = e.id
    ORDER BY created_at DESC
    LIMIT 1
) sig ON true
WHERE e.status = 'completed'
  AND e.target_pipeline = 'book_rag'
  AND e.raw_content->>'garage_key' LIKE '%.epub'
ORDER BY e.id
"""


async def _resolve_source_id(row: Any, file_hash: str | None) -> str:
    from vo_rag.pipelines.book_rag.source_identity import canonical_source_id

    generation_source_id = row["generation_source_id"]
    if generation_source_id:
        return str(generation_source_id)
    if file_hash:
        return canonical_source_id(file_hash)
    return ""


async def scan(*, limit: int | None = None) -> list[tuple[str, str, int]]:
    """Return ``(envelope_id, source_id, text_html_items)`` for every
    completed book_rag EPUB envelope, read-only."""
    from vo_rag.core.config import get_settings
    from vo_rag.core.db.postgres import close_pool, get_pool
    from vo_rag.core.storage.garage import GarageClient

    settings = get_settings()
    pool = await get_pool(settings.database.url, min_size=1, max_size=4)
    try:
        rows = await pool.fetch(_ENVELOPES_SQL)
    finally:
        await close_pool()

    if limit is not None:
        rows = rows[:limit]

    garage = GarageClient(
        endpoint_url=settings.garage.endpoint,
        access_key=settings.garage.access_key,
        secret_key=settings.garage.secret_key,
        region=settings.garage.region,
    )
    bucket = settings.garage.buckets.artifacts

    results: list[tuple[str, str, int]] = []
    for row in rows:
        garage_key = row["garage_key"]
        if not garage_key:
            continue
        envelope_id = str(row["envelope_id"])
        source_id = await _resolve_source_id(row, row["file_hash"])
        try:
            payload = await garage.get_bytes(bucket, garage_key)
            text_html_items = count_text_html_manifest_items(payload)
        except Exception as exc:  # noqa: BLE001 - report and keep scanning
            print(
                f"# {envelope_id}: failed to fetch/inspect {garage_key!r}: {exc}",
                file=sys.stderr,
            )
            continue
        results.append((envelope_id, source_id, text_html_items))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="scan at most N matching envelopes (default: all)",
    )
    args = parser.parse_args(argv)

    results = asyncio.run(scan(limit=args.limit))
    for envelope_id, source_id, text_html_items in results:
        print(f"{envelope_id},{source_id},{text_html_items}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
