"""Numeric boundary tests for representation read-back verification."""

from __future__ import annotations

import math

import pytest
from qdrant_client.models import PointStruct, Record, SparseVector

from vo_rag.core.ingest.representation_receipt import verify_representation


def _pair(dense_expected: list[float], dense_stored: list[float]) -> tuple[PointStruct, Record]:
    expected = PointStruct(
        id=1,
        vector={
            "dense": dense_expected,
            "sparse": SparseVector(indices=[3], values=[0.123456789]),
        },
        payload={"content": "source"},
    )
    actual = Record(
        id=1,
        vector={
            "dense": dense_stored,
            "sparse": SparseVector(indices=[3], values=[0.123456791]),
        },
        payload={"content": "source"},
    )
    return expected, actual


def test_sparse_float32_rounding_is_allowed_with_exact_modality_inventory() -> None:
    expected, actual = _pair([3.0, 4.0], [0.6, 0.8])

    result = verify_representation(expected, actual)

    assert result["verification"] == "payload_exact_modalities_complete_numeric_transform_verified"
    assert result["writer"]["vectors"]["sparse"] != result["stored"]["vectors"]["sparse"]


def test_empty_dense_modality_is_not_a_verified_representation() -> None:
    expected, actual = _pair([], [])

    with pytest.raises(ValueError):
        verify_representation(expected, actual)


def test_zero_norm_dense_modality_is_not_a_cosine_storage_transform() -> None:
    expected, actual = _pair([0.0, 0.0], [0.0, 0.0])

    with pytest.raises(ValueError):
        verify_representation(expected, actual)


def test_empty_colbert_row_is_not_a_verified_multivector() -> None:
    empty_row: list[float] = []
    expected = PointStruct(
        id=1,
        vector={"dense": [1.0, 0.0], "colbert": [empty_row]},
        payload={"content": "source"},
    )
    actual = Record(
        id=1,
        vector={"dense": [1.0, 0.0], "colbert": [empty_row]},
        payload={"content": "source"},
    )

    with pytest.raises(ValueError):
        verify_representation(expected, actual)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_dense_values_are_rejected(invalid: float) -> None:
    expected, actual = _pair([invalid, 1.0], [invalid, 1.0])

    with pytest.raises(ValueError):
        verify_representation(expected, actual)


def test_finite_norm_overflow_cannot_verify_corrupt_zero_vector() -> None:
    finite_vector = [1e308, 1e308]
    expected, actual = _pair(finite_vector, [0.0, 0.0])

    assert all(math.isfinite(value) for value in finite_vector)
    with pytest.raises(ValueError):
        verify_representation(expected, actual)


def test_revision_identity_rejects_duplicate_logical_base_before_admission() -> None:
    from vo_rag.core.storage.source_revision_writer import revision_points

    points = [
        PointStruct(
            id=index,
            vector={"dense": [1.0, 0.0]},
            payload={"source_id": "source", "publication_base_id": "same-logical-node"},
        )
        for index in (1, 2)
    ]
    with pytest.raises(ValueError, match="duplicate writer logical base"):
        revision_points(
            points, corpus="books", collection="vo_rag_books_raw_v1", source_id="source", revision=1
        )
