"""GPU integration tests for the BGE-M3 worker.

These tests are SKIPPED by default — they require a CUDA-capable host
with the BAAI/bge-m3 model downloadable. Enable them with::

    VORAG_RUN_GPU_EMBEDDING=1 \\
        uv run pytest tests/workers/embedding/test_gpu_integration.py -v -m integration

The skip is a hard skip (not a xfail) so the default CI stays green on
CPU-only runners.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


_RUN_GPU = os.environ.get("VORAG_RUN_GPU_EMBEDDING", "").strip() == "1"
_SKIP_REASON = (
    "VORAG_RUN_GPU_EMBEDDING not set to 1 — GPU integration tests are "
    "opt-in. Set the env var on a CUDA host to enable."
)


@pytest.mark.skipif(not _RUN_GPU, reason=_SKIP_REASON)
def test_real_bge_m3_encode_round_trip() -> None:
    """End-to-end: real BGE-M3 worker encodes a small batch and returns
    dense+sparse+colbert with dim=1024 and a non-zero latency.

    Requires: GPU, FlagEmbedding installed, BGE_M3_MODEL_PATH resolvable.
    """
    import time

    from vo_rag.workers.embedding.model import BgeM3Worker

    worker = BgeM3Worker(os.environ.get("BGE_M3_MODEL_PATH", "BAAI/bge-m3"))
    t0 = time.perf_counter()
    out = worker.embed(
        ["alpha", "beta"],
        return_dense=True,
        return_sparse=True,
        return_colbert=True,
        max_length=512,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    assert "dense" in out and len(out["dense"][0]) == 1024
    assert "sparse" in out and len(out["sparse"]) == 2
    assert "colbert" in out and len(out["colbert"]) == 2
    assert latency_ms > 0


@pytest.mark.skipif(not _RUN_GPU, reason=_SKIP_REASON)
def test_real_bge_m3_vram_reported_when_loaded() -> None:
    """Once the model is loaded, /health returns status=ready with a device
    string and a vram_used_mb that is either None (no CUDA) or a float >= 0."""
    import asyncio

    import httpx

    from vo_rag.workers.embedding import server as worker_server

    transport = httpx.ASGITransport(app=worker_server.app)

    async def _check() -> dict[str, object]:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        return dict(response.json())

    payload = asyncio.run(_check())
    assert payload["status"] in {"loading", "ready"}
    if payload["status"] == "ready":
        assert "device" in payload
        vram = payload.get("vram_used_mb")
        assert vram is None or (isinstance(vram, (int, float)) and vram >= 0)
