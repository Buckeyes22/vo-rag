"""Per-(run, video, stage) checkpoint blob for VCE resumability.

Mirrors `book_rag/checkpoint.py` but keyed at (run_id, video_id, stage)
instead of (workflow_run_id, image_id). One JSON object per (run, video,
stage) so a retry of the same run resumes from where the previous
attempt died rather than re-running every stage from the start.

Persists under::

    vo_rag/checkpoints/vce/<run_id>/<video_id>/<stage>.json

The blob's contents are stage-specific (transcribed text, frame sample
list, etc.) — see :func:`VceCheckpoint.record` and :func:`VceCheckpoint.load`.

``finalize()`` deletes every blob under the prefix once the export is
written successfully. The prefix is the resumability contract; renaming
it without coordination breaks in-flight retries.
"""

from __future__ import annotations

import json
from typing import Any

from vo_rag.pipelines.vce.events import VceStage

CHECKPOINT_PREFIX = "vo_rag/checkpoints/vce"


class VceCheckpoint:
    """Resumable checkpoint blob for one VCE run.

    Args:
        garage: A ``GarageClient`` instance.
        bucket: Bucket name (typically ``vorag-vce-derived``).
        run_id: Identifier for the run; namespaces all checkpoints.
    """

    def __init__(self, garage: Any, bucket: str, *, run_id: str) -> None:
        self._garage = garage
        self._bucket = bucket
        self._prefix = f"{CHECKPOINT_PREFIX}/{run_id}"

    def _key(self, video_id: str, stage: VceStage) -> str:
        return f"{self._prefix}/{video_id}/{stage}.json"

    async def load(self, video_id: str, stage: VceStage) -> dict[str, Any] | None:
        """Load the checkpoint blob for one (video, stage), or ``None`` if absent."""
        key = self._key(video_id, stage)
        if (await self._garage.head_object(self._bucket, key)) is None:
            return None
        raw = await self._garage.get_bytes(self._bucket, key)
        result: dict[str, Any] = json.loads(raw.decode("utf-8"))
        return result

    async def record(self, video_id: str, stage: VceStage, payload: dict[str, Any]) -> None:
        """Persist a checkpoint blob for one (video, stage)."""
        key = self._key(video_id, stage)
        body = json.dumps({"video_id": video_id, "stage": stage, **payload}).encode("utf-8")
        await self._garage.put_bytes(self._bucket, key, body, "application/json")

    async def finalize(self) -> None:
        """Delete every checkpoint object under the run prefix.

        Called after the consolidated export is written. Safe to call on
        an empty prefix — produces zero delete calls.
        """
        for obj in await self._garage.list_objects(self._bucket, self._prefix):
            await self._garage.delete_object(self._bucket, obj["key"])


__all__ = ["CHECKPOINT_PREFIX", "VceCheckpoint"]
