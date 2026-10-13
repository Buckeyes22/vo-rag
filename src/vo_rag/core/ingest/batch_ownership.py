"""Envelope ownership for out-of-band ingest batches (VOR-1406).

``ops/rc_ingest_batch.py`` inserts ``file_drop`` envelopes so it can drive the
book_rag pipeline into an arbitrary collection via ``VORAG_RAW_COLLECTION``.
Inserted as ``received`` those envelopes match the book_rag dispatcher's claim
predicate exactly, so a batch aimed at a non-book collection hands the
dispatcher work it ingests into ``vo_rag_books_raw_v1`` -- silently mixing
corpora. Observed 2026-08-14: 4153 Reddit envelopes were claimable and the
dispatcher had to be stopped for the whole run.

The fix is ownership, expressed in the status the batch already controls: an
envelope the batch created is ``processing`` from the moment it exists, because
the batch *is* processing it.

That leaves one hole, and it is the reason this module exists rather than a
one-line status change: the dispatcher reclaims stale ``processing`` envelopes
(a deliberate crash-recovery mechanism). A batch longer than that would have
its not-yet-started work stolen mid-run. The dispatcher
solves the same problem for its own in-flight envelope with
``_heartbeat_processing_envelope``; :func:`touch_owned_envelopes` is that idea
applied to a whole batch's backlog.
"""

from __future__ import annotations

from typing import Any, Final, Protocol

#: Keep well under the dispatcher's 15-minute stale window (DEFAULT_STALE_TIMEOUT).
DEFAULT_OWNERSHIP_HEARTBEAT_S: Final = 60.0

_TOUCH_SQL: Final = """
UPDATE intake_envelopes
SET updated_at = NOW()
WHERE id = ANY($1::text[])
  AND status = 'processing'
"""

__all__ = [
    "DEFAULT_OWNERSHIP_HEARTBEAT_S",
    "touch_owned_envelopes",
]


class _Pool(Protocol):
    async def execute(self, query: str, *args: Any) -> Any: ...


async def touch_owned_envelopes(pool: _Pool, envelope_ids: list[str]) -> int:
    """Refresh ``updated_at`` on envelopes this batch still owns.

    Returns the number of ids submitted (0 when there is nothing to do). Only
    rows still in ``processing`` are touched, so an envelope that already
    reached a terminal status is never resurrected.
    """
    ids = [eid for eid in envelope_ids if eid]
    if not ids:
        return 0
    await pool.execute(_TOUCH_SQL, ids)
    return len(ids)
