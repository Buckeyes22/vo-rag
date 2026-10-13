"""Authoritative Garage byte verification for file-drop intake."""

from __future__ import annotations

import hmac
import re
from collections.abc import Mapping
from typing import Any

from vo_rag.core.storage.garage import GarageObjectTooLarge, VerifiedGarageObject

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Formats currently accepted by the book parsing router.  Octet-stream remains
# valid for older/external uploaders; byte hashing, not MIME metadata, owns identity.
ALLOWED_BOOK_CONTENT_TYPES = frozenset(
    {
        "application/epub+zip",
        "application/octet-stream",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/x-mobipocket-ebook",
        "text/markdown",
        "text/plain",
    }
)


class ObjectVerificationError(ValueError):
    """A staged object cannot become an authoritative intake source."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def normalize_sha256_assertion(value: object) -> str | None:
    """Return a canonical optional SHA-256 assertion or reject malformed input."""
    if value is None or str(value).strip() == "":
        return None
    assertion = str(value).strip().lower()
    if SHA256_RE.fullmatch(assertion) is None:
        raise ObjectVerificationError("invalid_file_hash", "expected 64 lowercase hex digits")
    return assertion


async def verify_file_drop_object(
    raw_content: Mapping[str, Any],
    *,
    garage: Any,
    bucket: str,
    max_bytes: int,
    allowed_content_types: frozenset[str] = ALLOWED_BOOK_CONTENT_TYPES,
) -> dict[str, Any]:
    """Return normalized raw content whose identity comes from Garage bytes.

    A caller-provided ``file_hash`` is only an assertion.  The returned
    ``file_hash`` is always the digest calculated by ``GarageClient``.
    """
    key_value = raw_content.get("garage_key")
    if not isinstance(key_value, str) or not key_value.strip():
        raise ObjectVerificationError("missing_garage_key", "file drop has no object key")
    key = key_value.strip()
    assertion = normalize_sha256_assertion(raw_content.get("file_hash"))

    try:
        verified: VerifiedGarageObject | None = await garage.verify_object_sha256(
            bucket,
            key,
            max_bytes=max_bytes,
        )
    except GarageObjectTooLarge as exc:
        raise ObjectVerificationError("object_too_large", str(exc)) from exc
    if verified is None:
        raise ObjectVerificationError("object_missing", f"Garage object not found: {key}")

    actual_type = _normalize_content_type(verified.content_type)
    declared_type = _normalize_content_type(raw_content.get("content_type"))
    if actual_type is None or actual_type not in allowed_content_types:
        raise ObjectVerificationError(
            "content_type_not_allowed",
            f"actual content type is {actual_type or 'missing'}",
        )
    if declared_type is not None and declared_type != actual_type:
        raise ObjectVerificationError(
            "content_type_mismatch",
            f"declared={declared_type} actual={actual_type}",
        )

    declared_size = raw_content.get("size_bytes")
    if declared_size is not None:
        if isinstance(declared_size, bool) or not isinstance(declared_size, int):
            raise ObjectVerificationError("invalid_size", "size_bytes must be an integer")
        if declared_size != verified.size:
            raise ObjectVerificationError(
                "size_mismatch",
                f"declared={declared_size} actual={verified.size}",
            )

    if assertion is not None and not hmac.compare_digest(assertion, verified.sha256):
        raise ObjectVerificationError(
            "file_hash_mismatch",
            "client SHA-256 assertion does not match stored bytes",
        )

    authoritative_key = await garage.promote_verified_object(
        bucket,
        key,
        sha256=verified.sha256,
        etag=verified.etag,
    )

    normalized = dict(raw_content)
    normalized.update(
        {
            "garage_key": authoritative_key,
            "file_hash": verified.sha256,
            "content_type": actual_type,
            "size_bytes": verified.size,
            "object_verification": {
                "status": "verified",
                "sha256": verified.sha256,
                "size_bytes": verified.size,
                "content_type": actual_type,
                "etag": verified.etag,
                "staged_key": key,
                "authoritative_key": authoritative_key,
            },
        }
    )
    return normalized


async def delete_rejected_staged_object(garage: Any, bucket: str, key: object) -> bool:
    """Delete only a rejected upload-url object, never an arbitrary Garage key."""
    if not isinstance(key, str) or not key.startswith("intake/"):
        return False
    await garage.delete_object(bucket, key)
    return True


def object_is_verified(raw_content: object) -> bool:
    if not isinstance(raw_content, Mapping):
        return False
    verification = raw_content.get("object_verification")
    if not isinstance(verification, Mapping) or verification.get("status") != "verified":
        return False
    digest = raw_content.get("file_hash")
    verified_digest = verification.get("sha256")
    size = verification.get("size_bytes")
    return (
        isinstance(digest, str)
        and SHA256_RE.fullmatch(digest) is not None
        and isinstance(verified_digest, str)
        and hmac.compare_digest(digest, verified_digest)
        and isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
    )


def _normalize_content_type(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).split(";", 1)[0].strip().lower()
    if normalized == "binary/octet-stream":
        normalized = "application/octet-stream"
    return normalized or None


__all__ = [
    "ALLOWED_BOOK_CONTENT_TYPES",
    "ObjectVerificationError",
    "delete_rejected_staged_object",
    "normalize_sha256_assertion",
    "object_is_verified",
    "verify_file_drop_object",
]
