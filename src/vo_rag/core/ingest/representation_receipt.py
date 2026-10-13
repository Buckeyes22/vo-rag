"""Verify writer payloads and all named vectors against acknowledged Qdrant read-back."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from qdrant_client.models import PointStruct, Record, SparseVector

__all__ = ["canonical_sha256", "representation_receipt", "verify_representation"]


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON without emitting source text or vectors to logs."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vector_json(value: Any) -> Any:
    if isinstance(value, SparseVector):
        return value.model_dump(mode="json")
    return value


def representation_receipt(point: PointStruct | Record) -> dict[str, Any]:
    """Return exact observed fingerprints, distinguishing named modalities."""
    vectors = point.vector
    if not isinstance(vectors, dict):
        raise ValueError("named vectors are required for representation receipts")
    payload = point.payload or {}
    return {
        "point_id": str(point.id),
        "payload_sha256": canonical_sha256(payload),
        "vectors": {
            name: canonical_sha256(_vector_json(value)) for name, value in sorted(vectors.items())
        },
        "content_sha256": canonical_sha256(payload.get("content")),
        "embedding_provenance_sha256": canonical_sha256(payload.get("embedding_provenance")),
        "tag_binding_sha256": canonical_sha256(
            {
                key: value
                for key, value in payload.items()
                if key.startswith("tf_") or key == "domain_tags"
            }
        ),
    }


def _cosine(values: list[float]) -> list[float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("dense and multivector rows must be nonempty and finite")
    scale = max(abs(value) for value in values)
    if scale == 0:
        raise ValueError("cosine vectors must have nonzero norm")
    scaled = [value / scale for value in values]
    norm = math.sqrt(sum(value * value for value in scaled))
    return [value / norm for value in scaled]


def _floats(values: object) -> list[float]:
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int | float) for value in values
    ):
        raise ValueError("vector values must be numeric rows")
    return [float(value) for value in values]


def _close(expected: list[float], actual: list[float]) -> bool:
    return len(expected) == len(actual) and all(
        math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-7)
        for a, b in zip(expected, actual, strict=True)
    )


def verify_representation(expected: PointStruct, actual: Record) -> dict[str, Any]:
    """Validate every modality, allowing only Qdrant's declared cosine/float32 transform.

    Stored fingerprints are exact hashes of observed values. Numeric tolerance
    checks the documented storage transform; it is never called hash parity.
    """
    if str(expected.id) != str(actual.id) or expected.payload != actual.payload:
        raise ValueError("point identity or payload differs from the writer output")
    if not isinstance(expected.vector, dict) or not isinstance(actual.vector, dict):
        raise ValueError("required named vectors are missing")
    if set(expected.vector) != set(actual.vector):
        raise ValueError("named modality inventory differs from writer output")
    for name, wanted in expected.vector.items():
        found = actual.vector[name]
        if isinstance(wanted, SparseVector):
            if (
                not isinstance(found, SparseVector)
                or wanted.indices != found.indices
                or not _close(wanted.values, found.values)
            ):
                raise ValueError(f"{name}: sparse representation mismatch")
        elif isinstance(wanted, list) and wanted and isinstance(wanted[0], list):
            if not isinstance(found, list) or len(wanted) != len(found):
                raise ValueError(f"{name}: multivector inventory mismatch")
            for row, stored in zip(wanted, found, strict=True):
                if not isinstance(stored, list) or not _close(
                    _cosine(_floats(row)), _floats(stored)
                ):
                    raise ValueError(f"{name}: multivector representation mismatch")
        elif isinstance(wanted, list) and isinstance(found, list):
            if not _close(_cosine(_floats(wanted)), _floats(found)):
                raise ValueError(f"{name}: dense representation mismatch")
        else:
            raise ValueError(f"{name}: unsupported writer vector type")
    return {
        "writer": representation_receipt(expected),
        "stored": representation_receipt(actual),
        "storage_transform": "cosine normalization and float32 rounding",
        "verification": "payload_exact_modalities_complete_numeric_transform_verified",
    }
