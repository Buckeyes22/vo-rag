"""RerankerSettings backend-resolution tests (VOR-8)."""

from __future__ import annotations

from vo_rag.core.config.settings import RerankerSettings


def test_remote_backend_when_url_set() -> None:
    s = RerankerSettings(url="http://localhost:8501/rerank")
    assert s.backend == "client"
    assert s.timeout_s == 30
    assert s.model_id == "Alibaba-NLP/gte-reranker-modernbert-base"


def test_default_backend_is_onnx_cpu() -> None:
    s = RerankerSettings()  # no url, model_path defaulted
    assert s.backend == "onnx_cpu"


def test_explicit_gpu_backend_preserved() -> None:
    s = RerankerSettings(backend="gpu")
    assert s.backend == "gpu"


def test_existing_fields_preserved() -> None:
    s = RerankerSettings()
    assert s.batch_size == 32
    assert s.depth_high == 100
