from __future__ import annotations

import hashlib
import uuid


def _chunk_id(source_id: str, content: str, stage: str) -> str:
    """Unversioned content-addressed handle for a chunk.

    This is an internal handle (domain-tag keying, sterile-id derivation), NOT
    the stored Qdrant point id; that remains the index-based
    ``deterministic_point_id(source_id, chunk_index, stage)``.
    """
    key = f"{source_id}|{hashlib.sha256(content.encode()).hexdigest()}|{stage}".encode()
    return str(uuid.UUID(bytes=hashlib.sha256(key).digest()[:16]))
