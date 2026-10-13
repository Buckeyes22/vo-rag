"""Authoritative Garage object verification contracts."""

from __future__ import annotations

from typing import Any

import pytest

from vo_rag.core.ingest.object_verification import (
    ObjectVerificationError,
    delete_rejected_staged_object,
    object_is_verified,
    verify_file_drop_object,
)
from vo_rag.core.storage.garage import GarageObjectTooLarge, VerifiedGarageObject

_DIGEST = "a" * 64


class _Garage:
    def __init__(self, result: VerifiedGarageObject | None = None) -> None:
        self.result = result
        self.error: Exception | None = None
        self.deleted: list[tuple[str, str]] = []
        self.promoted: list[tuple[str, str, str, str | None]] = []

    async def verify_object_sha256(
        self,
        bucket: str,
        key: str,
        *,
        max_bytes: int,
    ) -> VerifiedGarageObject | None:
        assert bucket == "artifacts"
        assert key == "intake/book.pdf"
        assert max_bytes == 100
        if self.error is not None:
            raise self.error
        return self.result

    async def delete_object(self, bucket: str, key: str) -> None:
        self.deleted.append((bucket, key))

    async def promote_verified_object(
        self,
        bucket: str,
        staged_key: str,
        *,
        sha256: str,
        etag: str | None,
    ) -> str:
        self.promoted.append((bucket, staged_key, sha256, etag))
        return f"verified/sha256/{sha256}.pdf"


def _verified(**changes: Any) -> VerifiedGarageObject:
    values: dict[str, Any] = {
        "size": 10,
        "content_type": "application/pdf",
        "etag": "etag",
        "sha256": _DIGEST,
    }
    values.update(changes)
    return VerifiedGarageObject(**values)


@pytest.mark.asyncio
async def test_stored_digest_is_authoritative_and_assertion_is_canonicalized() -> None:
    result = await verify_file_drop_object(
        {
            "garage_key": "intake/book.pdf",
            "file_hash": f"  {_DIGEST.upper()}  ",
            "content_type": "application/pdf; charset=binary",
            "size_bytes": 10,
        },
        garage=_Garage(_verified()),
        bucket="artifacts",
        max_bytes=100,
    )

    assert result["file_hash"] == _DIGEST
    assert result["garage_key"] == f"verified/sha256/{_DIGEST}.pdf"
    assert result["object_verification"] == {
        "status": "verified",
        "sha256": _DIGEST,
        "size_bytes": 10,
        "content_type": "application/pdf",
        "etag": "etag",
        "staged_key": "intake/book.pdf",
        "authoritative_key": f"verified/sha256/{_DIGEST}.pdf",
    }
    assert object_is_verified(result) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "verified", "code"),
    [
        ({"file_hash": "abc"}, _verified(), "invalid_file_hash"),
        ({"file_hash": "b" * 64}, _verified(), "file_hash_mismatch"),
        ({"size_bytes": 9}, _verified(), "size_mismatch"),
        ({"content_type": "text/plain"}, _verified(), "content_type_mismatch"),
        ({}, _verified(content_type="image/png"), "content_type_not_allowed"),
        ({}, None, "object_missing"),
    ],
)
async def test_verification_rejects_untrusted_claims(
    raw: dict[str, Any],
    verified: VerifiedGarageObject | None,
    code: str,
) -> None:
    with pytest.raises(ObjectVerificationError, match=code):
        await verify_file_drop_object(
            {"garage_key": "intake/book.pdf", **raw},
            garage=_Garage(verified),
            bucket="artifacts",
            max_bytes=100,
        )


@pytest.mark.asyncio
async def test_actual_oversize_is_rejected_even_without_declared_size() -> None:
    garage = _Garage()
    garage.error = GarageObjectTooLarge("actual object is too large")
    with pytest.raises(ObjectVerificationError, match="object_too_large"):
        await verify_file_drop_object(
            {"garage_key": "intake/book.pdf"},
            garage=garage,
            bucket="artifacts",
            max_bytes=100,
        )


@pytest.mark.asyncio
async def test_binary_octet_stream_is_normalized_to_standard_media_type() -> None:
    result = await verify_file_drop_object(
        {"garage_key": "intake/book.pdf", "content_type": "application/octet-stream"},
        garage=_Garage(_verified(content_type="binary/octet-stream")),
        bucket="artifacts",
        max_bytes=100,
    )
    assert result["content_type"] == "application/octet-stream"


@pytest.mark.asyncio
async def test_cleanup_is_bounded_to_intake_prefix() -> None:
    garage = _Garage()
    assert await delete_rejected_staged_object(garage, "artifacts", "sources/book.pdf") is False
    assert garage.deleted == []
    assert await delete_rejected_staged_object(garage, "artifacts", "intake/book.pdf") is True
    assert garage.deleted == [("artifacts", "intake/book.pdf")]


def test_verification_marker_must_match_authoritative_digest_and_size() -> None:
    raw: dict[str, Any] = {
        "file_hash": _DIGEST,
        "object_verification": {
            "status": "verified",
            "sha256": "b" * 64,
            "size_bytes": 10,
        },
    }
    assert object_is_verified(raw) is False
    verification = raw["object_verification"]
    assert isinstance(verification, dict)
    verification["sha256"] = _DIGEST
    verification["size_bytes"] = True
    assert object_is_verified(raw) is False
