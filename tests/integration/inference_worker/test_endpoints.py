"""Live localhost smoke tests — opt-in only.

These checks hit a *running* localhost unified inference worker over
HTTP and confirm the public surface (``/ready``, ``/health``, ``/ping``,
``/embed``, ``/rerank``) responds with the shapes the rest of vo-rag
relies on. They are SKIPPED unless ``VORAG_INFERENCE_URL`` is set in the
environment, so the normal CI run (no live worker) collects 0 cases.

Run with::

    VORAG_INFERENCE_URL=http://localhost:8501 \
        uv run pytest tests/integration/inference_worker/test_endpoints.py -v

asyncio is intentionally NOT used — the sync ``httpx.Client`` is plenty
for a handful of probe requests against a ready worker.
"""

from __future__ import annotations

import os

import httpx
import pytest

BASE = os.environ.get("VORAG_INFERENCE_URL")  # e.g. http://localhost:8501
pytestmark = pytest.mark.skipif(not BASE, reason="set VORAG_INFERENCE_URL to run live smoke")


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    """One shared sync client for the module — keep-alive, 120s timeout."""
    return httpx.Client(base_url=BASE or "", timeout=120.0)


def test_ready_returns_200(client: httpx.Client) -> None:
    """GET /ready returns 200 once embed + reranker are both loaded."""
    resp = client.get("/ready")
    assert resp.status_code == 200, resp.text


def test_health_reports_current_gpu(client: httpx.Client) -> None:
    """GET /health identifies the Quadro RTX 5000 and its CUDA capacity."""
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload.get("status") == "ready", payload
    vram = payload.get("vram_used_mb")
    assert isinstance(vram, (int, float)), payload
    assert vram > 0, payload
    total_vram = payload.get("vram_total_mb")
    assert isinstance(total_vram, (int, float)), payload
    assert total_vram >= 15_000, payload
    assert payload.get("gpu_name") == "Quadro RTX 5000", payload
    assert payload.get("cuda_compute_capability") == "7.5", payload
    assert payload.get("limits", {}).get("max_concurrency") == 2, payload
    models = payload.get("models", {})
    assert models.get("embedding", {}).get("resident") is True, payload
    assert models.get("embedding", {}).get("device") == "cuda:0", payload
    assert models.get("reranking", {}).get("resident") is True, payload
    assert models.get("reranking", {}).get("device") == "cuda:0", payload


def test_embed_returns_vector_shapes(client: httpx.Client) -> None:
    """POST /embed returns dense[1][1024], sparse[1], colbert[1]."""
    body = {
        "texts": ["hello world"],
        "return_dense": True,
        "return_sparse": True,
        "return_colbert": True,
    }
    resp = client.post("/embed", json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    dense = payload["dense"]
    assert isinstance(dense, list) and len(dense) == 1
    assert isinstance(dense[0], list) and len(dense[0]) == 1024

    sparse = payload["sparse"]
    assert isinstance(sparse, list) and len(sparse) == 1

    colbert = payload["colbert"]
    assert isinstance(colbert, list) and len(colbert) == 1


def test_embed_rejects_oversized_batch(client: httpx.Client) -> None:
    """POST /embed with 33 texts breaches EMBED_MAX_BATCH (default 32) -> 413."""
    body = {"texts": [f"text-{i}" for i in range(33)]}
    resp = client.post("/embed", json=body)
    assert resp.status_code == 413, (resp.status_code, resp.text)


def test_rerank_reorders(client: httpx.Client) -> None:
    """POST /rerank surfaces the chassis-FEA candidate first."""
    body = {
        "query": "finite element analysis of a chassis",
        "top_k": 2,
        "candidates": [
            {"id": "far", "text": "the cat sat on a mat", "payload": {}},
            {"id": "near", "text": "FEA stress analysis of a racing chassis", "payload": {}},
        ],
    }
    resp = client.post("/rerank", json=body)
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert results[0]["id"] == "near", results


def test_rerank_rejects_oversized_depth(client: httpx.Client) -> None:
    """POST /rerank with 101 candidates breaches RERANK_MAX_DEPTH (default 100) -> 413."""
    body = {
        "query": "q",
        "top_k": 5,
        "candidates": [{"id": f"c{i}", "text": f"text {i}", "payload": {}} for i in range(101)],
    }
    resp = client.post("/rerank", json=body)
    assert resp.status_code == 413, (resp.status_code, resp.text)


def test_ping_ok(client: httpx.Client) -> None:
    """GET /ping returns 200 on a ready server (models loaded)."""
    resp = client.get("/ping")
    assert resp.status_code == 200, resp.text


def test_concurrent_embed_and_rerank(client: httpx.Client) -> None:
    """Both /embed and /rerank are live in the same worker process.

    Sequential calls are fine — the point is to confirm neither endpoint
    is gated or short-circuited by the other's state.
    """
    embed_resp = client.post(
        "/embed",
        json={
            "texts": ["hello world"],
            "return_dense": True,
            "return_sparse": True,
            "return_colbert": True,
        },
    )
    rerank_resp = client.post(
        "/rerank",
        json={
            "query": "q",
            "top_k": 1,
            "candidates": [{"id": "a", "text": "alpha", "payload": {}}],
        },
    )
    assert embed_resp.status_code == 200, embed_resp.text
    assert rerank_resp.status_code == 200, rerank_resp.text
