"""Deterministic UUID + hex helpers for idempotent writes.

`vo_rag.core.storage.qdrant` depends on `deterministic_uuid` to build the
F-024 point ID. The ID is built from the source ID so retries overwrite;
removal filters on the payload ``source_id``. These two functions have no
other dependencies, so keeping them in `core/idempotency` keeps the module a
leaf that core/storage can import.

Every retryable write-path (Qdrant points, Hatchet step keys, etc.) uses
these so retries produce identical IDs rather than duplicates. The digest
formula is fixed: SHA-256 over `"|".join(parts)`, first 16 bytes -> UUID for
`deterministic_uuid`; the same SHA-256 hex prefix for `deterministic_hex`.
Persisted point IDs depend on it, so changing it orphans stored points.
"""

from __future__ import annotations

import hashlib
import uuid


def deterministic_uuid(*parts: str) -> str:
    """Return a stable UUID string from ordered content parts.

    parts are joined by '|' and SHA-256'd; the first 16 bytes become a UUID.
    Stable across processes, Python versions, and host OS.

    Example:
        deterministic_uuid(source_id, str(chunk_index), stage, version_hash)
    """
    if not parts:
        raise ValueError("deterministic_uuid requires at least one part")
    key = "|".join(parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()[:16]
    return str(uuid.UUID(bytes=digest))


def deterministic_hex(*parts: str, length: int = 32) -> str:
    """Shorter hex digest form when a UUID is not required.

    length must be 1..64; default 32 (128 bits).
    """
    if not 1 <= length <= 64:
        raise ValueError("length must be 1..64")
    if not parts:
        raise ValueError("deterministic_hex requires at least one part")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:length]
