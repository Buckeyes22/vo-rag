"""Opaque keyset cursor helpers."""

from __future__ import annotations

import base64
import json
from datetime import datetime


def encode_cursor(created_at: datetime, id_: str) -> str:
    """Encode created_at/id tuple to URL-safe opaque cursor."""
    raw = json.dumps([created_at.isoformat(), id_]).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    """Decode URL-safe opaque cursor to created_at/id tuple."""
    raw = base64.urlsafe_b64decode(cursor + "===")
    created_at_str, row_id = json.loads(raw)
    return datetime.fromisoformat(created_at_str), row_id
