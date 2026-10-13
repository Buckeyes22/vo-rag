"""Verify representation inventory and Qdrant storage transforms without model requests."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    Record,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from vo_rag.core.ingest.representation_receipt import verify_representation


def test_real_qdrant_dense_normalization_and_sparse_readback() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "test",
        vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)},
        sparse_vectors_config={"bm25": SparseVectorParams()},
    )
    point = PointStruct(
        id=1,
        vector={"dense": [3.0, 4.0], "bm25": SparseVector(indices=[2, 7], values=[1.0, 2.0])},
        payload={"source_id": "s", "content": "example", "source_revision": 1},
    )
    client.upsert("test", points=[point], wait=True)
    stored = client.retrieve("test", ids=[1], with_payload=True, with_vectors=True)[0]
    receipt = verify_representation(point, stored)
    assert receipt["verification"] == "payload_exact_modalities_complete_numeric_transform_verified"
    assert receipt["writer"]["vectors"]["dense"] != receipt["stored"]["vectors"]["dense"]
    client.close()


@pytest.mark.parametrize("corruption", ["payload", "missing_sparse", "dense", "sparse"])
def test_representation_corruption_is_not_success(corruption: str) -> None:
    point = PointStruct(
        id=1,
        vector={"dense": [1.0, 0.0], "sparse": SparseVector(indices=[1], values=[1.0])},
        payload={"content": "source"},
    )
    stored = Record(
        id=1,
        vector={"dense": [1.0, 0.0], "sparse": SparseVector(indices=[1], values=[1.0])},
        payload={"content": "source"},
    )
    assert isinstance(stored.vector, dict)
    if corruption == "payload":
        stored.payload = {"content": "different"}
    elif corruption == "missing_sparse":
        stored.vector.pop("sparse")
    elif corruption == "dense":
        stored.vector["dense"] = [0.0, 1.0]
    else:
        stored.vector["sparse"] = SparseVector(indices=[2], values=[1.0])
    with pytest.raises(ValueError):
        verify_representation(point, stored)
