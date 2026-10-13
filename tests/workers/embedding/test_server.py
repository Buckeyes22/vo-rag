"""Contract tests for ``vo_rag.workers.embedding.server``.

Covers:

- ``GET /ping`` 204 before model load / 200 after.
- ``GET /health`` ``{"status": "loading", "vram_used_mb": None}`` before,
  ``{"status": "ready", "device": ...}`` after.
- ``POST /embed`` 503 when the worker hasn't finished loading the model.
- ``POST /embed`` validates the wire-format ``EmbedRequest`` (max 256 texts,
  ``extra="forbid"``, ``max_length=8192``).
- With a fake model injected, ``/embed`` returns dense+sparse+colbert,
  ``model="BAAI/bge-m3"``, ``dim=1024``, ``latency_ms>0``.
- ``/health`` does not query CUDA memory for a CPU-bound model.

All tests run against the ASGI app via ``httpx.ASGITransport`` — no
network sockets, no GPU, no FlagEmbedding import.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from vo_rag.core.embedding.schemas import EmbedRequest, EmbedResponse
from vo_rag.workers.embedding import server

# ---------------------------------------------------------------------------
# /ping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ping_returns_204_while_model_is_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_model", None)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ping")

    assert response.status_code == 204
    assert response.content == b""


@pytest.mark.asyncio
async def test_ping_returns_200_when_model_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_model", SimpleNamespace(device="cpu"))

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ping")

    assert response.status_code == 200
    assert response.content == b""


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_returns_loading_payload_when_model_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_model", None)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "loading", "vram_used_mb": None}


@pytest.mark.asyncio
async def test_health_returns_ready_payload_with_device_when_model_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_model", SimpleNamespace(device="cuda:0"))

    class _Torch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def mem_get_info(device: str) -> tuple[int, int]:
                # 1024 MB total, 512 MB free → 512 MB used
                return 512 * 1024 * 1024, 1024 * 1024 * 1024

    # Fake torch only around the request: where real torch is installed,
    # pytest-randomly reseeds it in the teardown hook before fixture
    # finalizers, and a fake still in sys.modules there leaks everywhere.
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", _Torch)
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["device"] == "cuda:0"
    # 1024-512 = 512 MB used, /1024/1024 in server leaves it as 512.0
    assert payload["vram_used_mb"] == 512.0


@pytest.mark.asyncio
async def test_health_does_not_query_cuda_memory_for_cpu_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_model", SimpleNamespace(device="cpu"))

    class _Torch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def mem_get_info(device: str) -> tuple[int, int]:
                raise AssertionError(f"should not query CUDA memory for {device}")

    # Fake torch only around the request: where real torch is installed,
    # pytest-randomly reseeds it in the teardown hook before fixture
    # finalizers, and a fake still in sys.modules there leaks everywhere.
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", _Torch)
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    payload = response.json()
    assert payload == {"status": "ready", "vram_used_mb": None, "device": "cpu"}


# ---------------------------------------------------------------------------
# /embed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_returns_503_when_model_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_model", None)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/embed",
            json={"texts": ["hello"]},
        )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_embed_rejects_extra_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_model", SimpleNamespace(device="cpu", embed=lambda *_a, **_k: {}))

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/embed",
            json={"texts": ["hello"], "not_a_field": "boom"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_embed_rejects_over_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_model", SimpleNamespace(device="cpu", embed=lambda *_a, **_k: {}))

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/embed",
            json={"texts": ["x"] * 257},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_embed_uses_default_max_length_8192(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wire default is 8192 — confirm the model wrapper receives it."""
    captured: dict[str, Any] = {}

    def fake_embed(
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
        max_length: int,
    ) -> dict[str, Any]:
        captured["texts"] = texts
        captured["return_dense"] = return_dense
        captured["return_sparse"] = return_sparse
        captured["return_colbert"] = return_colbert
        captured["max_length"] = max_length
        return {}

    monkeypatch.setattr(server, "_model", SimpleNamespace(device="cpu", embed=fake_embed))

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/embed",
            json={"texts": ["hello"]},
        )

    assert response.status_code == 200, response.text
    assert captured["max_length"] == 8192
    assert captured["return_dense"] is True
    assert captured["return_sparse"] is True
    assert captured["return_colbert"] is False


@pytest.mark.asyncio
async def test_embed_returns_full_payload_with_fake_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a fake model that returns dense+sparse+colbert, the endpoint
    surfaces the model's tag, dim=1024, and a positive latency."""

    class FakeModel:
        device = "cpu"

        def embed(
            self,
            texts: list[str],
            *,
            return_dense: bool,
            return_sparse: bool,
            return_colbert: bool,
            max_length: int,
        ) -> dict[str, Any]:
            return {
                "dense": [[0.1] * 1024 for _ in texts],
                "sparse": [{"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]} for _ in texts],
                "colbert": [[[0.5] * 4 for _ in range(2)] for _ in texts],
            }

    monkeypatch.setattr(server, "_model", FakeModel())

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/embed",
            json={"texts": ["alpha", "beta"]},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == "BAAI/bge-m3"
    assert payload["dim"] == 1024
    assert payload["latency_ms"] > 0
    assert payload["dense"] is not None
    assert payload["sparse"] is not None
    assert payload["colbert"] is not None
    assert len(payload["dense"]) == 2
    assert len(payload["sparse"]) == 2
    assert len(payload["colbert"]) == 2
    # round-trip the response through the schema to confirm wire-format
    EmbedResponse.model_validate(payload)


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_prometheus_payload() -> None:
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    # Either the default prometheus_client exposition or a real Instrumentator
    # output — both produce text/plain with # HELP lines.
    assert "# HELP" in body or "# TYPE" in body or "python_gc" in body


# ---------------------------------------------------------------------------
# Logging env reading (avoid full settings at module import)
# ---------------------------------------------------------------------------


def test_server_module_does_not_import_full_settings_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker should not import the heavy settings module at import time —
    it only reads ``VORAG_LOG_LEVEL`` / ``VORAG_JSON_LOGS`` from env. We
    re-import the module to force a fresh import and assert no settings call.
    """
    import importlib

    # Sentinel that the settings module would set on import; raise if hit.
    def _explode(*_a: Any, **_k: Any) -> None:
        raise AssertionError("server module imported the heavy settings module")

    settings_mod = type(sys)("vo_rag.core.config.settings")
    settings_mod.get_settings = _explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vo_rag.core.config.settings", settings_mod)
    # Also block via the package's __init__ if it tries to reach settings.
    config_pkg = sys.modules.get("vo_rag.core.config")
    if config_pkg is not None:
        monkeypatch.setattr(config_pkg, "get_settings", _explode, raising=False)

    reloaded = importlib.reload(server)
    assert reloaded is server


def test_load_worker_env_reads_logging_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VORAG_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("VORAG_JSON_LOGS", "0")
    monkeypatch.delenv("BGE_M3_DEVICE", raising=False)
    monkeypatch.delenv("BGE_M3_MODEL_PATH", raising=False)

    cfg = server._load_worker_env()
    assert cfg["log_level"] == "DEBUG"
    assert cfg["json_logs"] is False
    assert cfg["device"] == "cuda:0"
    assert cfg["model_path"] == "BAAI/bge-m3"


def test_load_worker_env_respects_bge_m3_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BGE_M3_DEVICE", "cpu")
    monkeypatch.setenv("BGE_M3_MODEL_PATH", "/srv/models/bge-m3")

    cfg = server._load_worker_env()
    assert cfg["device"] == "cpu"
    assert cfg["model_path"] == "/srv/models/bge-m3"


# ---------------------------------------------------------------------------
# Contract: server uses the canonical EmbedRequest/EmbedResponse schemas
# ---------------------------------------------------------------------------


def test_server_uses_vo_rag_schemas_not_duplicated() -> None:
    """The server must reuse ``vo_rag.core.embedding.schemas`` rather than
    defining its own copy — schemas are part of the worker contract."""
    from vo_rag.workers.embedding import server as srv

    assert srv.EmbedRequest is EmbedRequest
    assert srv.EmbedResponse is EmbedResponse


def test_app_title_uses_vo_rag_brand() -> None:
    assert "vo-rag" in server.app.title
    assert "BGE-M3" in server.app.title
    assert server.app.title == "vo-rag BGE-M3 Embedding Worker"
