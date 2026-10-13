"""Test doubles for the VCE runner.

Mirrors `tests/pipelines/book_rag/`'s fake pattern — a minimal in-memory
:class:`FakeGarage` that exposes the subset of
:class:`vo_rag.core.storage.garage.GarageClient` the runner uses. The
contract is small (put/get/list/delete + object_exists) which keeps the
fake simple.

Why not mock the real GarageClient? The fake lets us assert what was
written (the "persist first" rule is testable only if the test can see
the write happen).
"""

from __future__ import annotations

import json
from typing import Any


class FakeGarage:
    """In-memory :class:`GarageClient` stand-in for VCE tests."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}
        self.calls: list[tuple[str, str, str]] = []  # (op, bucket, key)

    # ── write ──────────────────────────────────────────────────────────────

    async def put_bytes(
        self, bucket: str, key: str, body: bytes, content_type: str = "application/octet-stream"
    ) -> None:
        self.calls.append(("put_bytes", bucket, key))
        self._objects[(bucket, key)] = body

    async def upload_file(self, bucket: str, key: str, path: Any) -> None:
        self.calls.append(("upload_file", bucket, key))
        self._objects[(bucket, key)] = Path(path).read_bytes()

    # ── read ───────────────────────────────────────────────────────────────

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        self.calls.append(("get_bytes", bucket, key))
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise FileNotFoundError(f"no object at {bucket}/{key}") from exc

    async def object_exists(self, bucket: str, key: str) -> bool:
        return (bucket, key) in self._objects

    async def list_objects(self, bucket: str, prefix: str = "") -> list[dict[str, Any]]:
        return [
            {"bucket": bucket, "key": key}
            for (b, key) in sorted(self._objects)
            if b == bucket and key.startswith(prefix)
        ]

    async def delete_object(self, bucket: str, key: str) -> None:
        self.calls.append(("delete_object", bucket, key))
        self._objects.pop((bucket, key), None)

    async def head_object(self, bucket: str, key: str) -> dict[str, Any] | None:
        if (bucket, key) in self._objects:
            return {"bucket": bucket, "key": key, "size": len(self._objects[(bucket, key)])}
        return None

    # ── inspection helpers (test-only) ─────────────────────────────────────

    def has_object(self, bucket: str, key: str) -> bool:
        return (bucket, key) in self._objects

    def get_json(self, bucket: str, key: str) -> dict[str, Any]:
        raw = self._objects[(bucket, key)]
        decoded: dict[str, Any] = json.loads(raw.decode("utf-8"))
        return decoded


from pathlib import Path  # noqa: E402  (used in upload_file)

__all__ = ["FakeGarage"]
