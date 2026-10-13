"""Per-image VLM description checkpoint for book_rag resumability.

Objects live under the `vo_rag/checkpoints/book_rag` prefix.
The GarageClient shape matches `vo_rag.core.storage.garage.GarageClient`,
so the same fake works for unit tests and real S3v4 signing works in
production.

Persists one JSON object per image under a run-scoped Garage prefix so
retries of the same `workflow_run_id` resume from where the previous
attempt died rather than re-running every VLM call from image 0.
"""

from __future__ import annotations

import json

from vo_rag.core.storage.garage import GarageClient

CHECKPOINT_PREFIX = "vo_rag/checkpoints/book_rag"


class Checkpoint:
    def __init__(
        self,
        garage: GarageClient,
        bucket: str,
        *,
        workflow_run_id: str,
    ) -> None:
        self._garage = garage
        self._bucket = bucket
        self._prefix = f"{CHECKPOINT_PREFIX}/{workflow_run_id}/images"

    def _key(self, image_id: str) -> str:
        return f"{self._prefix}/{image_id}.json"

    async def load(self) -> dict[str, object]:
        """Return {image_id: description/evidence} for checkpoint objects.

        Empty dict on a fresh run. Used to seed `descriptions` in
        `do_describe_images` so already-done images are skipped on retry.
        """
        descriptions: dict[str, object] = {}
        for obj in await self._garage.list_objects(self._bucket, self._prefix):
            key = obj["key"]
            image_id = key.rsplit("/", 1)[-1].removesuffix(".json")
            raw = await self._garage.get_bytes(self._bucket, key)
            payload = json.loads(raw.decode("utf-8"))
            descriptions[image_id] = payload["description"]
        return descriptions

    async def record_image(self, image_id: str, description: object) -> None:
        """Persist one image's description or structured evidence.

        Byte-identical bytes for the same (image_id, description) — so
        concurrent writers converge. Under VLM sampling (temperature > 0)
        two concurrent retries may record different descriptions for the
        same image; last-writer-wins, which is acceptable because
        downstream chunking and embedding treat descriptions as opaque
        text.
        """
        payload = json.dumps({"image_id": image_id, "description": description}).encode("utf-8")
        await self._garage.put_bytes(self._bucket, self._key(image_id), payload, "application/json")

    async def finalize(self) -> None:
        """Delete every checkpoint object under the prefix.

        Called after the consolidated `vlm_descriptions.json` artifact has
        been written successfully. Safe to call multiple times: an empty
        prefix produces zero delete calls.
        """
        for obj in await self._garage.list_objects(self._bucket, self._prefix):
            await self._garage.delete_object(self._bucket, obj["key"])


__all__ = ["CHECKPOINT_PREFIX", "Checkpoint"]
