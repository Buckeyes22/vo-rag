"""Contract tests for ``vo_rag.workers.embedding.model.BgeM3Worker``.

Verifies:

- ``FlagEmbedding.BGEM3FlagModel`` is imported LAZILY inside ``__init__``
  so unit tests can monkeypatch it without installing the GPU stack.
- ``embed()`` forwards ``return_dense``, ``return_sparse``,
  ``return_colbert_vecs``, and ``max_length`` correctly.
- ``embed()`` caps ``batch_size`` at ``min(32, len(texts))``.
- Sparse lexical weights are normalised to integer token ids + float
  weights, ready for Qdrant sparse vectors.
- Dense and ColBERT outputs are surfaced as nested lists of floats so
  JSON serialisation works.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vo_rag.workers.embedding.model import BgeM3Worker

# ---------------------------------------------------------------------------
# Lazy FlagEmbedding import
# ---------------------------------------------------------------------------


def test_flag_embedding_imported_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing a BgeM3Worker must not import FlagEmbedding at the
    module level — we should be able to monkeypatch the module attribute
    and let the constructor resolve the patched symbol on demand."""
    import vo_rag.workers.embedding.model as model_mod

    # Confirm the import is not pinned at module top-level.
    assert "FlagEmbedding" not in model_mod.__dict__
    assert "BGEM3FlagModel" not in model_mod.__dict__


def test_init_does_not_require_real_flag_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monkeypatching the FlagEmbedding module attribute is enough to
    construct a worker without the real GPU dependency installed."""
    calls: list[dict[str, Any]] = []

    class _FakeFlagModel:
        def __init__(self, model_path: str, *, use_fp16: bool, devices: list[str]) -> None:
            calls.append({"model_path": model_path, "use_fp16": use_fp16, "devices": devices})

    fake_mod = SimpleNamespace(BGEM3FlagModel=_FakeFlagModel)
    monkeypatch.setitem(__import__("sys").modules, "FlagEmbedding", fake_mod)

    worker = BgeM3Worker("/fake/bge-m3", use_fp16=False, device="cpu")

    assert worker.device == "cpu"
    assert isinstance(worker.model, _FakeFlagModel)
    assert calls == [{"model_path": "/fake/bge-m3", "use_fp16": False, "devices": ["cpu"]}]


# ---------------------------------------------------------------------------
# embed() forwarding + sparse normalization
# ---------------------------------------------------------------------------


class _FakeBgeModel:
    """Drop-in stand-in for BGEM3FlagModel — records call args, returns
    deterministic dense/sparse/colbert structures."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        max_length: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert_vecs: bool,
    ) -> dict[str, Any]:
        self.last_kwargs = {
            "texts": texts,
            "batch_size": batch_size,
            "max_length": max_length,
            "return_dense": return_dense,
            "return_sparse": return_sparse,
            "return_colbert_vecs": return_colbert_vecs,
        }
        n = len(texts)
        out: dict[str, Any] = {}
        if return_dense:
            # 1024-dim vectors so the response.dim=1024 contract holds
            out["dense_vecs"] = np.arange(n * 1024, dtype=np.float32).reshape(n, 1024)
        if return_sparse:
            # BGE-M3 sparse output: list of dicts whose keys are token ids
            # (already int in the real model, but our test uses both int and
            # string keys to confirm the worker normalises to int).
            out["lexical_weights"] = [
                {1: 0.5, 7: 0.25, 42: 0.1},  # already-int keys
                {"13": 0.3, "99": 0.4},  # str-keyed, must be coerced to int
            ][:n]
        if return_colbert_vecs:
            out["colbert_vecs"] = [np.full((3, 4), 0.5, dtype=np.float32) for _ in range(n)]
        return out


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> BgeM3Worker:
    fake = _FakeBgeModel()
    fake_mod = SimpleNamespace(BGEM3FlagModel=lambda *a, **k: fake)
    monkeypatch.setitem(__import__("sys").modules, "FlagEmbedding", fake_mod)
    return BgeM3Worker("/fake/bge-m3", use_fp16=False, device="cpu")


def test_embed_batches_at_min_32_len(worker: BgeM3Worker) -> None:
    worker.embed(
        ["a", "b", "c"],
        return_dense=True,
        return_sparse=True,
        return_colbert=True,
        max_length=256,
    )
    fake = worker.model
    assert fake.last_kwargs["batch_size"] == min(32, 3)


def test_embed_batches_at_32_when_inputs_exceed(worker: BgeM3Worker) -> None:
    texts = [f"text-{i}" for i in range(64)]
    worker.embed(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert=True,
        max_length=256,
    )
    fake = worker.model
    assert fake.last_kwargs["batch_size"] == 32


def test_embed_forwards_return_flags(worker: BgeM3Worker) -> None:
    worker.embed(
        ["hello"],
        return_dense=True,
        return_sparse=False,
        return_colbert=False,
        max_length=1024,
    )
    fake = worker.model
    assert fake.last_kwargs["return_dense"] is True
    assert fake.last_kwargs["return_sparse"] is False
    assert fake.last_kwargs["return_colbert_vecs"] is False
    assert fake.last_kwargs["max_length"] == 1024


def test_embed_sparse_normalises_lexical_weights(worker: BgeM3Worker) -> None:
    out = worker.embed(
        ["a", "b"],
        return_dense=False,
        return_sparse=True,
        return_colbert=False,
        max_length=512,
    )
    assert "sparse" in out
    sparse = out["sparse"]
    assert len(sparse) == 2
    # First row: int keys → indices are ints, values are floats.
    assert sparse[0]["indices"] == [1, 7, 42]
    assert [type(i) for i in sparse[0]["indices"]] == [int, int, int]
    assert sparse[0]["values"] == [0.5, 0.25, 0.1]
    # Second row: str keys → must be coerced to int.
    assert sparse[1]["indices"] == [13, 99]
    assert [type(i) for i in sparse[1]["indices"]] == [int, int]
    assert sparse[1]["values"] == [0.3, 0.4]


def test_embed_dense_and_colbert_are_json_serialisable(worker: BgeM3Worker) -> None:
    out = worker.embed(
        ["a", "b"],
        return_dense=True,
        return_sparse=False,
        return_colbert=True,
        max_length=512,
    )
    # Both should be plain lists of floats (not numpy arrays), so that
    # FastAPI can json.dumps the response without an encoder.
    dense = out["dense"]
    colbert = out["colbert"]
    assert len(dense) == 2
    assert all(isinstance(row, list) for row in dense)
    assert all(isinstance(x, float) for row in dense for x in row)
    assert len(dense[0]) == 1024  # dim contract

    assert len(colbert) == 2
    assert all(isinstance(token_vec, list) for text_vec in colbert for token_vec in text_vec)
    assert all(
        isinstance(x, float) for text_vec in colbert for token_vec in text_vec for x in token_vec
    )


def test_embed_omits_unrequested_fields(worker: BgeM3Worker) -> None:
    out = worker.embed(
        ["a"],
        return_dense=False,
        return_sparse=False,
        return_colbert=False,
        max_length=512,
    )
    assert "dense" not in out
    assert "sparse" not in out
    assert "colbert" not in out


def test_loaded_tokenizer_reports_actual_input_length_and_truncation(worker: BgeM3Worker) -> None:
    class Tokenizer:
        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
            assert kwargs == {"add_special_tokens": True, "truncation": False, "padding": False}
            return {"input_ids": [[1, 2, 3], [1, 2, 3, 4, 5]]}

    worker.model.tokenizer = Tokenizer()
    out = worker.embed(
        ["short", "long"],
        return_dense=True,
        return_sparse=False,
        return_colbert=False,
        max_length=4,
    )
    assert out["input_token_counts"] == [3, 5]
    assert out["input_truncated"] == [False, True]
    assert out["runtime_identity"]["weights_attestation"] == "unknown"


def test_optional_token_evidence_failure_does_not_fail_embedding(worker: BgeM3Worker) -> None:
    class Tokenizer:
        def __call__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("tokenizer diagnostics unavailable")

    worker.model.tokenizer = Tokenizer()
    out = worker.embed(
        ["usable"], return_dense=True, return_sparse=False, return_colbert=False, max_length=4
    )
    assert len(out["dense"]) == 1
    assert "input_token_counts" not in out and "input_truncated" not in out


def test_observed_config_revisions_do_not_attest_loaded_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeBgeModel()
    fake.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="a" * 40))  # type: ignore[attr-defined]
    fake.tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": "b" * 40})  # type: ignore[attr-defined]
    monkeypatch.setitem(
        __import__("sys").modules,
        "FlagEmbedding",
        SimpleNamespace(BGEM3FlagModel=lambda *a, **k: fake),
    )
    worker = BgeM3Worker("BAAI/bge-m3", device="cpu")
    assert worker.runtime_identity["model_config_revision"] == "a" * 40
    assert worker.runtime_identity["tokenizer_config_revision"] == "b" * 40
    assert worker.runtime_identity["identity_status"] == "observed_config"
    assert worker.runtime_identity["weights_attestation"] == "unknown"
