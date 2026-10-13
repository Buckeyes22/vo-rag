"""Shared helpers for the API surface.

- ``_to_query_hit``: map a landed ``SearchResult`` to the Pydantic
  ``QueryHit`` without mutating ids/payloads (F-031 / core-id invariant).
- ``secure_filename``: a tiny, dependency-free filename sanitizer.
- ``mask_secret``: a JSON-safe redactor for status / health responses so
  operators can paste a response into a chat without leaking API keys,
  Garage secrets, or webhook tokens.
"""

from __future__ import annotations

import re
from typing import Any

from vo_rag.core.retrieval.search import SearchResult

# A deliberately small allowlist — the route layer should not accept
# anything more permissive. Path separators, control chars, and
# Windows-reserved characters are all stripped.
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_DOT_SEGMENT_RE = re.compile(r"^(?:\.\.?)(?:/|$)")

# Field names whose values must be redacted in status / health output.
# Substring match: ``api_key`` also catches ``anthropic_api_key``,
# ``query_api_key``, ``batch_api_key`` etc. — that is the intent.
_SECRET_FIELD_FRAGMENTS: tuple[str, ...] = (
    "api_key",
    "secret_key",
    "access_key",
    "anthropic_api_key",
    "kimi_api_key",
    "webhook_secret",
    "csrf_secret",
    "password",
    "token",
)

_MASKED = "***REDACTED***"


def _to_query_hit(result: SearchResult) -> dict[str, Any]:
    """Convert a core ``SearchResult`` into a plain dict for ``QueryHit``.

    The contract: ``id`` is the Qdrant point id verbatim, ``score`` is
    the merged RRF score, ``collection`` is the collection name, and
    ``payload`` is the unparsed Qdrant payload — no field remapping.
    """
    return {
        "id": str(result.id),
        "score": float(result.score),
        "collection": str(result.collection),
        "payload": dict(result.payload),
    }


def secure_filename(name: str) -> str:
    """Return a filesystem-safe filename.

    Strips path separators, leading dots (path traversal), and
    collapses any run of disallowed characters into a single ``_``.
    An empty result falls back to ``"file"`` so the caller never
    has to handle the empty string.
    """
    if not isinstance(name, str):
        raise TypeError("filename must be a string")

    # Strip any path components — both POSIX and Windows separators.
    basename = name.replace("\\", "/").split("/")[-1]

    # Reject traversal up-front.
    if _DOT_SEGMENT_RE.match(basename):
        basename = basename.lstrip(".")

    cleaned = _SAFE_FILENAME_RE.sub("_", basename).strip("._-")
    return cleaned or "file"


def mask_secret(value: Any) -> Any:
    """Return a redacted copy of ``value`` safe to paste into a chat.

    Strings whose field name (or substring of a key path) matches a
    known secret fragment are replaced with ``***REDACTED***``. Nested
    dicts and lists are walked recursively. Non-secret scalars pass
    through unchanged.
    """
    if isinstance(value, dict):
        return {str(key): mask_secret(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_secret(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_secret(item) for item in value)
    if isinstance(value, str):
        return value
    return value


def _is_secret_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    return any(fragment in lowered for fragment in _SECRET_FIELD_FRAGMENTS)


def mask_secrets(payload: Any) -> Any:
    """Walk ``payload`` and replace values for secret-named fields.

    This is the convenience wrapper route handlers use before
    returning status / health JSON. Unlike :func:`mask_secret` (which
    is a passthrough for raw strings), this one looks at the KEY
    name and redacts the value when the key looks like a secret
    holder.
    """
    if isinstance(payload, dict):
        out: dict[Any, Any] = {}
        for key, value in payload.items():
            if _is_secret_key(key):
                out[key] = _MASKED if value not in (None, "") else ""
            else:
                out[key] = mask_secrets(value)
        return out
    if isinstance(payload, list):
        return [mask_secrets(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(mask_secrets(item) for item in payload)
    return payload


__all__ = [
    "_to_query_hit",
    "mask_secret",
    "mask_secrets",
    "secure_filename",
]
