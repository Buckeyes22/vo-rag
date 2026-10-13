"""Contract tests for vo_rag.core.embedding.schemas.

Verifies:
- EmbedRequest rejects unknown fields (extra='forbid')
- EmbedRequest enforces 1..256 texts, default return flags, max_length=8192
- EmbedRequest / EmbedResponse / SparseVec round-trip
- EmbedResponse defaults dim=1024
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vo_rag.core.embedding.schemas import EmbedRequest, EmbedResponse, SparseVec

# ---------------------------------------------------------------------------
# EmbedRequest
# ---------------------------------------------------------------------------


class TestEmbedRequest:
    def test_defaults(self) -> None:
        req = EmbedRequest(texts=["hello"])
        assert req.texts == ["hello"]
        assert req.return_dense is True
        assert req.return_sparse is True
        assert req.return_colbert is False
        assert req.max_length == 8192

    def test_rejects_empty_texts(self) -> None:
        with pytest.raises(ValidationError):
            EmbedRequest(texts=[])

    def test_rejects_too_many_texts(self) -> None:
        with pytest.raises(ValidationError):
            EmbedRequest(texts=["x"] * 257)

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            EmbedRequest(texts=["x"], not_a_field="boom")  # type: ignore[call-arg]

    def test_round_trip(self) -> None:
        req = EmbedRequest(
            texts=["a", "b"],
            return_dense=True,
            return_sparse=False,
            return_colbert=True,
            max_length=1024,
        )
        dumped = req.model_dump()
        restored = EmbedRequest.model_validate(dumped)
        assert restored == req


# ---------------------------------------------------------------------------
# SparseVec
# ---------------------------------------------------------------------------


class TestSparseVec:
    def test_round_trip(self) -> None:
        sv = SparseVec(indices=[1, 5, 9], values=[0.2, 0.4, 0.6])
        assert sv.indices == [1, 5, 9]
        assert sv.values == [0.2, 0.4, 0.6]


# ---------------------------------------------------------------------------
# EmbedResponse
# ---------------------------------------------------------------------------


class TestEmbedResponse:
    def test_default_dim_is_1024(self) -> None:
        resp = EmbedResponse(
            dense=[[0.0]],
            sparse=[SparseVec(indices=[1], values=[0.5])],
            model="bge-m3",
            latency_ms=12.3,
        )
        assert resp.dim == 1024
        assert resp.model == "bge-m3"
        assert resp.latency_ms == 12.3

    def test_optional_vector_fields(self) -> None:
        # All vector fields are Optional — the server may return only a subset.
        resp = EmbedResponse(
            dense=None,
            sparse=None,
            colbert=None,
            model="bge-m3",
            latency_ms=1.0,
        )
        assert resp.dense is None
        assert resp.sparse is None
        assert resp.colbert is None


@pytest.mark.parametrize("max_length", [0, -1, 8193])
def test_embed_request_rejects_lengths_outside_loaded_model_contract(max_length: int) -> None:
    with pytest.raises(ValidationError):
        EmbedRequest(texts=["usable passage"], max_length=max_length)
