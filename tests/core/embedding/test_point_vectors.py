"""Contract tests for vo_rag.core.embedding.point_vectors.

Verifies:
- sparse_payload_to_int_map normalises both {indices, values} and {token: w} shapes
- downsample_colbert_vectors is deterministic, preserves endpoints, caps at 192
  by default (VORAG_QDRANT_MAX_COLBERT_VECTORS), and respects override
- fallback_colbert_vectors uses the dense vector when BGE-M3 returns nothing
  (VORAG_COLBERT_EMPTY_FALLBACK=dense), hard-fails otherwise
"""

from __future__ import annotations

import pytest

from vo_rag.core.embedding.point_vectors import (
    downsample_colbert_vectors,
    fallback_colbert_vectors,
    include_colbert_at_ingest,
    max_colbert_vectors_per_point,
    sparse_payload_to_int_map,
)

# ---------------------------------------------------------------------------
# include_colbert_at_ingest
# ---------------------------------------------------------------------------


class TestIncludeColbertAtIngest:
    def test_default_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VORAG_BOOK_RAG_INCLUDE_COLBERT", raising=False)
        assert include_colbert_at_ingest() is False

    def test_enabled_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VORAG_BOOK_RAG_INCLUDE_COLBERT", "1")
        assert include_colbert_at_ingest() is True


# ---------------------------------------------------------------------------
# sparse_payload_to_int_map
# ---------------------------------------------------------------------------


class TestSparsePayloadToIntMap:
    def test_indices_values_shape(self) -> None:
        assert sparse_payload_to_int_map({"indices": [1, 5], "values": [0.2, 0.9]}) == {
            1: 0.2,
            5: 0.9,
        }

    def test_plain_mapping_and_none(self) -> None:
        assert sparse_payload_to_int_map({"3": 0.5}) == {3: 0.5}
        assert sparse_payload_to_int_map(None) == {}


# ---------------------------------------------------------------------------
# downsample_colbert_vectors (Qdrant 1 MiB/point cap invariant)
# ---------------------------------------------------------------------------


class TestDownsampleColbert:
    def test_under_limit_passthrough(self) -> None:
        v = [[1.0], [2.0], [3.0]]
        assert downsample_colbert_vectors(v, max_vectors=10) == v

    def test_caps_count_and_preserves_endpoints(self) -> None:
        v = [[float(i)] for i in range(100)]
        out = downsample_colbert_vectors(v, max_vectors=8)
        assert len(out) == 8
        assert out[0] == [0.0]
        assert out[-1] == [99.0]

    def test_empty_and_zero_limit(self) -> None:
        assert downsample_colbert_vectors([]) == []
        assert downsample_colbert_vectors([[1.0]], max_vectors=0) == []

    def test_default_limit_is_positive(self) -> None:
        assert max_colbert_vectors_per_point() >= 1

    def test_default_limit_is_192(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # VOR-2 invariant: Qdrant 1 MiB/point cap → default 192 ColBERT vectors
        # (BGE-M3 returns 0-length arrays on whitespace-only chunks; Qdrant's
        # multivector dim check rejects them).
        monkeypatch.delenv("VORAG_QDRANT_MAX_COLBERT_VECTORS", raising=False)
        assert max_colbert_vectors_per_point() == 192

    def test_default_downsamples_to_192(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No env override → 192 cap; 500 input vectors → exactly 192 output.
        monkeypatch.delenv("VORAG_QDRANT_MAX_COLBERT_VECTORS", raising=False)
        v = [[float(i)] for i in range(500)]
        out = downsample_colbert_vectors(v)
        assert len(out) == 192
        assert out[0] == [0.0]
        assert out[-1] == [499.0]

    def test_deterministic_stride(self) -> None:
        # Determinism: same input → same output, every time.
        v = [[float(i), float(i) + 0.1] for i in range(50)]
        a = downsample_colbert_vectors(v, max_vectors=10)
        b = downsample_colbert_vectors(v, max_vectors=10)
        assert a == b

    def test_env_override_changes_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VORAG_QDRANT_MAX_COLBERT_VECTORS", "5")
        v = [[float(i)] for i in range(20)]
        out = downsample_colbert_vectors(v)
        assert len(out) == 5
        assert out[0] == [0.0]
        assert out[-1] == [19.0]


# ---------------------------------------------------------------------------
# fallback_colbert_vectors (NEVER silent-drop)
# ---------------------------------------------------------------------------


class TestFallbackColbert:
    def test_uses_dense(self) -> None:
        assert fallback_colbert_vectors([0.1, 0.2]) == [[0.1, 0.2]]

    def test_empty_dense_raises(self) -> None:
        with pytest.raises(RuntimeError):
            fallback_colbert_vectors([])

    def test_disabled_fallback_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VORAG_COLBERT_EMPTY_FALLBACK", "off")
        with pytest.raises(RuntimeError):
            fallback_colbert_vectors([0.1])

    def test_default_mode_is_dense(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # VOR-2 invariant: VORAG_COLBERT_EMPTY_FALLBACK defaults to "dense"
        # (NOT silent-drop) so a quality regression cannot slip through.
        monkeypatch.delenv("VORAG_COLBERT_EMPTY_FALLBACK", raising=False)
        dense = [0.3, 0.4, 0.5]
        assert fallback_colbert_vectors(dense) == [dense]
