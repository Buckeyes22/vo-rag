"""reranking: cross-encoder rerankers.

- ``GteReranker`` — CPU/ONNX int8 (the no-GPU fallback).
- ``GpuReranker`` — torch fp16 on CUDA (the inference worker).
- ``RerankerClient`` — remote HTTP client (the API client backend).

``GteReranker`` is exported lazily via ``__getattr__`` so that importing a
sibling module (``client``/``factory``/``render``) does NOT pull ``gte`` —
and therefore not ``onnxruntime`` — into the client-backend process. The
client backend stays torch/onnx-free (VOR-8 review finding).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vo_rag.core.reranking.gte import GteReranker

__all__ = ["GteReranker"]


def __getattr__(name: str) -> Any:
    if name == "GteReranker":
        from vo_rag.core.reranking.gte import GteReranker

        return GteReranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
