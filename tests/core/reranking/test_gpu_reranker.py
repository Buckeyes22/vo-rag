"""Tests for GpuReranker (VOR-8).

The relevance/top_k tests are CUDA-gated (run on a box:
``BGE_M3_DEVICE=cuda:0 uv run --no-sync pytest <this> -q``). The revision-pin
lock-in test is torch-only (no GPU) — it stubs ``from_pretrained`` to assert the
B615 integrity controls reach both call sites.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import vo_rag.core.reranking.gpu as gpumod  # noqa: E402
from vo_rag.core.reranking.gpu import GpuReranker  # noqa: E402
from vo_rag.core.retrieval.search import SearchResult  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="GpuReranker needs CUDA")


@cuda
def test_gpu_reranker_orders_by_relevance() -> None:
    rr = GpuReranker(device="cuda:0", batch_size=8)
    cands = [
        SearchResult(
            id="far", score=0.0, collection="c", payload={"content": "the cat sat on a mat"}
        ),
        SearchResult(
            id="near",
            score=0.0,
            collection="c",
            payload={"content": "finite element analysis of a racing chassis under load"},
        ),
    ]
    out = rr.rerank("how is FEA used in chassis design?", cands, top_k=2)
    assert [c.id for c in out] == ["near", "far"]
    assert out[0].score is not None and out[0].score > out[1].score


@cuda
def test_gpu_reranker_respects_top_k() -> None:
    rr = GpuReranker(device="cuda:0", batch_size=8)
    cands = [
        SearchResult(
            id=str(i), score=0.0, collection="c", payload={"content": f"doc {i} about suspension"}
        )
        for i in range(5)
    ]
    out = rr.rerank("suspension geometry", cands, top_k=3)
    assert len(out) == 3


def test_from_pretrained_pins_revision_and_disables_remote_code(monkeypatch) -> None:
    """B615: revision must reach BOTH from_pretrained calls; trust_remote_code=False."""
    calls: dict[str, dict[str, object]] = {}

    class _Stub:
        def to(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
            return self

        def eval(self):  # noqa: ANN201
            return self

    class _Tok:
        @staticmethod
        def from_pretrained(model_id, **kw):  # noqa: ANN001, ANN003, ANN205
            calls["tok"] = kw
            return _Stub()

    class _Model:
        @staticmethod
        def from_pretrained(model_id, **kw):  # noqa: ANN001, ANN003, ANN205
            calls["model"] = kw
            return _Stub()

    monkeypatch.setattr(gpumod, "AutoTokenizer", _Tok)
    monkeypatch.setattr(gpumod, "AutoModelForSequenceClassification", _Model)

    GpuReranker(device="cpu")

    assert calls["tok"].get("revision")
    assert calls["model"].get("revision")
    assert calls["tok"].get("trust_remote_code") is False
    assert calls["model"].get("trust_remote_code") is False
