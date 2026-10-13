"""Select the reranker implementation from settings (VOR-8).

- ``client`` → remote HTTP (``RerankerClient``) — what the API process uses when
  pointed at the localhost service.
- ``gpu`` → in-process torch fp16 (``GpuReranker``) — used by the inference
  worker itself, which hosts the model on CUDA.
- ``onnx_cpu`` → in-process ``GteReranker`` (the no-GPU fallback).
- ``cohere`` → Cohere Rerank v3.5 HTTP client.

Heavy backends are imported lazily so the API process running the ``client``
backend never imports torch / onnxruntime.
"""

from __future__ import annotations

from typing import Any

from vo_rag.core.config.settings import RerankerSettings


def build_reranker(settings: RerankerSettings) -> Any:
    """Construct the reranker selected by ``settings.backend``."""
    if settings.backend == "client":
        from vo_rag.core.reranking.client import RerankerClient

        return RerankerClient(settings.url, timeout_s=float(settings.timeout_s))
    if settings.backend == "gpu":
        from vo_rag.core.reranking.gpu import GpuReranker

        return GpuReranker(settings.model_id, batch_size=settings.batch_size)
    if settings.backend == "cohere":
        from vo_rag.core.reranking.cohere import CohereReranker

        return CohereReranker(timeout_s=float(settings.timeout_s))
    from vo_rag.core.reranking.gte import GteReranker

    return GteReranker(settings.model_path)


__all__ = ["build_reranker"]
