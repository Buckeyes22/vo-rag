"""Unit tests for book_rag Checkpoint class.

Checkpoints are stored under the `vo_rag/checkpoints/book_rag` prefix. The
GarageClient in the fake below matches the contract of
`vo_rag.core.storage.garage.GarageClient` (the same S3-compatible
async surface used elsewhere in the codebase).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from typing import Any

import pytest

from vo_rag.core.storage.garage import GarageClient
from vo_rag.pipelines.book_rag.checkpoint import CHECKPOINT_PREFIX, Checkpoint


class _FakeGarage:
    """In-memory stand-in for `vo_rag.core.storage.garage.GarageClient` —
    only the four methods `Checkpoint` uses are implemented."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}
        self.calls: list[tuple[str, str, str]] = []

    async def list_objects(self, bucket: str, prefix: str = "") -> list[dict[str, Any]]:
        self.calls.append(("list_objects", bucket, prefix))
        return [
            {
                "key": key,
                "size": len(data),
                "last_modified": dt.datetime(2026, 4, 21),
            }
            for (b, key), data in self.store.items()
            if b == bucket and key.startswith(prefix)
        ]

    async def put_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        self.calls.append(("put_bytes", bucket, key))
        self.store[(bucket, key)] = data

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        self.calls.append(("get_bytes", bucket, key))
        return self.store[(bucket, key)]

    async def delete_object(self, bucket: str, key: str) -> None:
        self.calls.append(("delete_object", bucket, key))
        self.store.pop((bucket, key), None)


def _as_garage(fake: _FakeGarage) -> GarageClient:
    """Cast the in-memory fake to `GarageClient` for the type checker. The
    structural type is satisfied — only the four methods `Checkpoint` uses
    are exercised."""
    return fake  # type: ignore[return-value]


def test_checkpoint_prefix_is_vo_rag_namespaced() -> None:
    assert CHECKPOINT_PREFIX == "vo_rag/checkpoints/book_rag"


@pytest.mark.asyncio
async def test_checkpoint_load_returns_empty_on_fresh_prefix() -> None:
    garage = _FakeGarage()
    checkpoint = Checkpoint(_as_garage(garage), "artifacts", workflow_run_id="run-fresh")
    assert await checkpoint.load() == {}


@pytest.mark.asyncio
async def test_checkpoint_record_and_load_roundtrip() -> None:
    garage = _FakeGarage()
    checkpoint = Checkpoint(_as_garage(garage), "artifacts", workflow_run_id="run-rt")

    await checkpoint.record_image("img-a", "desc-a")
    await checkpoint.record_image("img-b", "desc-b")
    await checkpoint.record_image("img-c", "desc-c")

    loaded = await checkpoint.load()
    assert loaded == {"img-a": "desc-a", "img-b": "desc-b", "img-c": "desc-c"}

    expected_prefix = f"{CHECKPOINT_PREFIX}/run-rt/images"
    assert (
        "artifacts",
        f"{expected_prefix}/img-a.json",
    ) in [(bucket, key) for (bucket, key) in garage.store]
    payload = json.loads(garage.store[("artifacts", f"{expected_prefix}/img-a.json")])
    assert payload == {"image_id": "img-a", "description": "desc-a"}


@pytest.mark.asyncio
async def test_checkpoint_finalize_deletes_all_prefix_objects() -> None:
    garage = _FakeGarage()
    checkpoint = Checkpoint(_as_garage(garage), "artifacts", workflow_run_id="run-fin")

    await checkpoint.record_image("img-1", "desc-1")
    await checkpoint.record_image("img-2", "desc-2")
    await checkpoint.record_image("img-3", "desc-3")

    await checkpoint.finalize()

    assert await checkpoint.load() == {}
    delete_calls = [c for c in garage.calls if c[0] == "delete_object"]
    assert len(delete_calls) == 3


@pytest.mark.asyncio
async def test_checkpoint_concurrent_record_is_idempotent() -> None:
    """Last-writer-wins under temperature > 0 is acceptable — the same payload
    written twice from concurrent tasks must remain byte-identical."""
    garage = _FakeGarage()
    checkpoint = Checkpoint(_as_garage(garage), "artifacts", workflow_run_id="run-race")

    await asyncio.gather(
        checkpoint.record_image("img-x", "same-desc"),
        checkpoint.record_image("img-x", "same-desc"),
    )

    loaded = await checkpoint.load()
    assert loaded == {"img-x": "same-desc"}
    key = f"{CHECKPOINT_PREFIX}/run-race/images/img-x.json"
    payload_a = json.dumps({"image_id": "img-x", "description": "same-desc"}).encode("utf-8")
    assert garage.store[("artifacts", key)] == payload_a


@pytest.mark.asyncio
async def test_checkpoint_finalize_safe_to_call_twice() -> None:
    garage = _FakeGarage()
    checkpoint = Checkpoint(_as_garage(garage), "artifacts", workflow_run_id="run-twice")

    await checkpoint.record_image("img-1", "desc-1")
    await checkpoint.finalize()
    await checkpoint.finalize()

    assert await checkpoint.load() == {}
