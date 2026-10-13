"""Book identity and source-file format from Postgres (VOR-1427).

A book's ``source_id`` carries the sha256 of its source bytes
(``vo-rag-corpus-sha256:<hex>``, or a legacy bare hex id). The intake
envelope for those bytes is the one whose ``garage_key`` (in
``raw_content`` or ``source_data``) contains the sha256, or whose
``raw_content.file_hash`` is it (file-drop envelopes, keyed
``intake/<ulid>/<name>``, carry no hash in the key). The
chunk payload itself stores no envelope id, so this is the link.

``ebook_source_ids`` answers "was this book's source file an EPUB/MOBI?"
from that envelope's ``garage_key`` extension. `corpus health` asks it only
for books whose payload has no ``parser_used`` (legacy points, and EPUB/MOBI
chunks written before the ebook path stamped one).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Source-file extensions of the reflowable (EPUB/MOBI parser) formats.
EBOOK_EXTENSIONS: tuple[str, ...] = (".epub", ".mobi", ".azw3")

#: Envelope ``e`` holds the bytes whose sha256 is ``k.sha256``.
SHA_ENVELOPE_MATCH = """(
        e.raw_content->>'file_hash' = k.sha256
     OR e.raw_content->>'garage_key' LIKE '%' || k.sha256 || '%'
     OR e.source_data->>'garage_key' LIKE '%' || k.sha256 || '%'
 )"""

#: Every envelope garage key per book, any status: a failed envelope of the
#: same bytes names the same file format. (Plain literals, not an f-string,
#: so the SQL stays static; a test checks both queries embed
#: ``SHA_ENVELOPE_MATCH`` verbatim.)
SOURCE_GARAGE_KEYS_SQL = """
SELECT k.source_id,
       COALESCE(
           NULLIF(e.raw_content->>'garage_key', ''), e.source_data->>'garage_key'
       ) AS garage_key
FROM unnest($1::text[], $2::text[]) AS k(source_id, sha256)
JOIN intake_envelopes e ON (
        e.raw_content->>'file_hash' = k.sha256
     OR e.raw_content->>'garage_key' LIKE '%' || k.sha256 || '%'
     OR e.source_data->>'garage_key' LIKE '%' || k.sha256 || '%'
 )
"""


class FetchPool(Protocol):
    async def fetch(self, query: str, *args: Any) -> Sequence[Any]: ...


def source_sha256(source_id: str) -> str | None:
    """The book's sha256 from its ``source_id``: ``vo-rag-corpus-sha256:<hex>``
    or a legacy bare (any-case) hex id; ``None`` for any other id."""
    candidate = source_id.rsplit(":", 1)[-1].strip().lower()
    return candidate if _SHA256_RE.match(candidate) else None


def is_ebook_key(garage_key: str | None) -> bool:
    return bool(garage_key) and str(garage_key).strip().lower().endswith(EBOOK_EXTENSIONS)


async def ebook_source_ids(pool: FetchPool, source_ids: list[str]) -> set[str]:
    """The ``source_ids`` whose intake envelope's ``garage_key`` ends in
    ``.epub``/``.mobi``/``.azw3`` (case-insensitive). Ids without a sha256,
    or with no matching envelope, are not ebooks."""
    keyed = [(sid, sha) for sid in source_ids if (sha := source_sha256(sid)) is not None]
    if not keyed:
        return set()
    rows = await pool.fetch(
        SOURCE_GARAGE_KEYS_SQL, [sid for sid, _ in keyed], [sha for _, sha in keyed]
    )
    return {str(row["source_id"]) for row in rows if is_ebook_key(row["garage_key"])}


__all__ = [
    "EBOOK_EXTENSIONS",
    "SHA_ENVELOPE_MATCH",
    "SOURCE_GARAGE_KEYS_SQL",
    "FetchPool",
    "ebook_source_ids",
    "is_ebook_key",
    "source_sha256",
]
