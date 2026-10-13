"""Stored-byte hashing contracts for the Garage client."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock

import pytest

from vo_rag.core.storage.garage import (
    GarageClient,
    GarageObjectMetadata,
    GarageObjectTooLarge,
)


class _RawResponse:
    """aiohttp ClientResponse stand-in: ``read()`` takes no size argument."""

    async def __aexit__(self, *args: object) -> None:
        return None

    async def read(self) -> bytes:
        raise AssertionError("raw response read() slurps the whole object")


class _Body:
    """aiobotocore StreamingBody stand-in: ``__aenter__`` yields the wrapped raw
    response (as aiobotocore 2.x does), and only the wrapper's ``read(amt)``
    streams bounded chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.raw = _RawResponse()

    async def __aenter__(self) -> _RawResponse:
        return self.raw

    async def __aexit__(self, *args: object) -> None:
        return None

    async def read(self, amt: int | None = None) -> bytes:
        del amt
        return self.chunks.pop(0) if self.chunks else b""


class _S3:
    def __init__(self, chunks: list[bytes]) -> None:
        self.body = _Body(chunks)
        self.calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> _S3:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get_object(self, **kwargs: Any) -> dict[str, _Body]:
        self.calls.append(kwargs)
        return {"Body": self.body}

    async def copy_object(self, **kwargs: Any) -> None:
        self.calls.append({"operation": "copy", **kwargs})

    async def delete_object(self, **kwargs: Any) -> None:
        self.calls.append({"operation": "delete", **kwargs})


def _client(metadata: GarageObjectMetadata, chunks: list[bytes]) -> tuple[GarageClient, _S3]:
    client = object.__new__(GarageClient)
    client.object_metadata = AsyncMock(return_value=metadata)  # type: ignore[method-assign]
    s3 = _S3(chunks)
    client._client = lambda: s3  # type: ignore[method-assign]
    return client, s3


@pytest.mark.asyncio
async def test_hashes_the_object_body_in_bounded_chunks() -> None:
    client, s3 = _client(
        GarageObjectMetadata(size=6, content_type="text/plain", etag="etag"),
        [b"abc", b"def"],
    )

    verified = await client.verify_object_sha256("bucket", "key", max_bytes=10, chunk_size=3)

    assert verified is not None
    assert verified.sha256 == hashlib.sha256(b"abcdef").hexdigest()
    assert verified.size == 6
    assert s3.calls == [{"Bucket": "bucket", "Key": "key", "IfMatch": '"etag"'}]


@pytest.mark.asyncio
async def test_head_size_rejects_before_object_download() -> None:
    client, s3 = _client(
        GarageObjectMetadata(size=11, content_type="text/plain", etag=None),
        [b"not read"],
    )

    with pytest.raises(GarageObjectTooLarge):
        await client.verify_object_sha256("bucket", "key", max_bytes=10)
    assert s3.calls == []


@pytest.mark.asyncio
async def test_stream_bound_rejects_when_body_exceeds_head_claim() -> None:
    client, _s3 = _client(
        GarageObjectMetadata(size=5, content_type="text/plain", etag=None),
        [b"123456"],
    )

    with pytest.raises(GarageObjectTooLarge):
        await client.verify_object_sha256("bucket", "key", max_bytes=5)


@pytest.mark.asyncio
async def test_changed_or_interrupted_body_never_produces_an_identity() -> None:
    client, _s3 = _client(
        GarageObjectMetadata(size=6, content_type="text/plain", etag=None),
        [b"abc"],
    )

    with pytest.raises(OSError, match="size changed"):
        await client.verify_object_sha256("bucket", "key", max_bytes=10)


@pytest.mark.asyncio
async def test_promotion_is_conditional_and_retains_staged_key_until_db_commit() -> None:
    client, s3 = _client(
        GarageObjectMetadata(size=6, content_type="application/pdf", etag="etag"),
        [],
    )

    key = await client.promote_verified_object(
        "bucket",
        "intake/one/Book.PDF",
        sha256="a" * 64,
        etag="etag",
    )

    assert key == f"verified/sha256/{'a' * 64}.pdf"
    assert s3.calls == [
        {
            "operation": "copy",
            "Bucket": "bucket",
            "Key": key,
            "CopySource": {"Bucket": "bucket", "Key": "intake/one/Book.PDF"},
            "MetadataDirective": "COPY",
            "CopySourceIfMatch": '"etag"',
        }
    ]
