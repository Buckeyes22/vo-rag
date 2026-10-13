"""Unit tests for `vo_rag.core.storage.points.VoPoint` (VOR-2).

The dense-vector length is the production schema contract (BGE-M3 → 1024d
Cosine) and is enforced at the model boundary. Extra fields are forbidden
to keep the Qdrant write path strictly typed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vo_rag.core.storage.points import VoPoint


def _dense_1024() -> list[float]:
    return [0.0] * 1024


def _valid_kwargs(**overrides):
    base = {
        "source_id": "src-1",
        "chunk_index": 0,
        "stage": "raw",
        "version_hash": None,
        "dense": _dense_1024(),
        "sparse": {1: 0.5, 2: 0.25},
        "bm25": {3: 1.0},
        "colbert": [[0.0] * 1024],
        "payload": {"project_id": "alpha"},
    }
    base.update(overrides)
    return base


def test_vopoint_accepts_minimal_valid_payload() -> None:
    p = VoPoint(**_valid_kwargs())
    assert p.source_id == "src-1"
    assert p.chunk_index == 0
    assert p.stage == "raw"
    assert p.version_hash is None
    assert len(p.dense) == 1024
    assert p.sparse == {1: 0.5, 2: 0.25}
    assert p.bm25 == {3: 1.0}
    assert p.colbert == [[0.0] * 1024]
    assert p.payload == {"project_id": "alpha"}


def test_vopoint_accepts_explicit_version_hash() -> None:
    p = VoPoint(**_valid_kwargs(version_hash="research_v1"))
    assert p.version_hash == "research_v1"


def test_vopoint_rejects_short_dense_vector() -> None:
    with pytest.raises(ValidationError):
        VoPoint(**_valid_kwargs(dense=[0.0] * 1023))


def test_vopoint_rejects_long_dense_vector() -> None:
    with pytest.raises(ValidationError):
        VoPoint(**_valid_kwargs(dense=[0.0] * 1025))


def test_vopoint_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        VoPoint(**_valid_kwargs(extra_field="nope"))


def test_vopoint_requires_required_fields() -> None:
    with pytest.raises(ValidationError):
        VoPoint()  # type: ignore[call-arg]
