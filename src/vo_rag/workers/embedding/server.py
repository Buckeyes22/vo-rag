"""BGE-M3 embedding HTTP worker.

Design notes:

- Title: ``vo-rag BGE-M3 Embedding Worker``.
- ``EmbedRequest`` / ``EmbedResponse`` come from
  ``vo_rag.core.embedding.schemas`` — the worker shares the wire contract
  with the API server. The schemas are NOT duplicated here.
- Logging configuration reads ``VORAG_LOG_LEVEL`` / ``VORAG_JSON_LOGS``
  from env at startup rather than importing the main app's full settings
  module. Workers are independently deployable and must not pull in the
  main app's config stack at import time.
- BGE-M3 env vars default to ``cuda:0`` and ``BAAI/bge-m3``.
- ``/metrics`` uses ``prometheus-client`` directly (not
  ``prometheus-fastapi-instrumentator``) — the latter is not a vo-rag
  dependency, and pulling it in would break CPU-only test runs.

Routes:

- ``GET  /ping``    204 before model load, 200 after.
- ``GET  /health``  ``{"status": "loading", "vram_used_mb": None}`` while
  the model is loading; ``{"status": "ready", "vram_used_mb": ...,
  "device": ...}`` once it is loaded.
- ``POST /embed``   503 while model is loading, otherwise delegates to
  ``BgeM3Worker.embed`` and wraps the result in the canonical
  ``EmbedResponse`` (dense, sparse, colbert, ``model="BAAI/bge-m3"``,
  ``dim=1024``, ``latency_ms``).
- ``GET  /metrics`` Prometheus exposition (default collectors).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client import REGISTRY as _PROM_DEFAULT_REGISTRY

# Re-export the canonical wire-format schemas so callers can type-annotate
# against the worker's contract. Importing them here keeps a single source
# of truth — the server does not redefine them.
from vo_rag.core.embedding.schemas import EmbedRequest, EmbedResponse

from .model import BgeM3Worker

app = FastAPI(title="vo-rag BGE-M3 Embedding Worker")

# Per-handler Prometheus histograms come from ``prometheus-fastapi-instrumentator``
# when it is installed, but the worker does NOT hard-depend on it — that
# keeps CPU-only test runs importable. The default ``prometheus_client``
# collectors (process, python_gc, platform) are always available on
# /metrics; per-handler instrumentation is a future add-on that hooks in
# without changing the route contract.

_model: BgeM3Worker | None = None

# Re-exported for tests + downstream type-checkers — see test_server.py.
__all__ = [
    "EmbedRequest",
    "EmbedResponse",
    "app",
    "_model",
    "_load_worker_env",
    "_configure_logging",
]


def _load_worker_env() -> dict[str, Any]:
    """Read all worker env vars in one place.

    Called from the lifespan startup hook so the module import itself
    stays free of any ``vo_rag.core.config`` plumbing.
    """
    log_level = os.environ.get("VORAG_LOG_LEVEL", "INFO")
    json_logs_raw = os.environ.get("VORAG_JSON_LOGS", "1").strip().lower()
    json_logs = json_logs_raw not in {"0", "false", "no", "off", ""}
    return {
        "log_level": log_level,
        "json_logs": json_logs,
        "device": os.environ.get("BGE_M3_DEVICE", "cuda:0"),
        "model_path": os.environ.get("BGE_M3_MODEL_PATH", "BAAI/bge-m3"),
    }


def _configure_logging(*, json_logs: bool, level: str) -> None:
    """Lightweight structlog setup for the worker.

    Self-contained (no PII redactor dependency) — the worker does not
    emit user payloads, so redaction is unnecessary at this layer.
    """
    normalized_level = level.upper()
    filtering_level = logging._nameToLevel.get(normalized_level, logging.INFO)  # noqa: SLF001

    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.make_filtering_bound_logger(filtering_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    renderer: structlog.types.Processor
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    formatter = structlog.stdlib.ProcessorFormatter(foreign_pre_chain=shared, processor=renderer)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(normalized_level)

    for noisy in ("uvicorn.access", "asyncio", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel("WARNING")


@app.on_event("startup")
async def _load() -> None:
    global _model
    cfg = _load_worker_env()
    _configure_logging(json_logs=cfg["json_logs"], level=cfg["log_level"])
    # FlagEmbedding is GPU-only — import lazily so the worker module stays
    # importable on CPU-only test runs.
    _model = BgeM3Worker(cfg["model_path"], use_fp16=True, device=cfg["device"])


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    out = _model.embed(
        req.texts,
        return_dense=req.return_dense,
        return_sparse=req.return_sparse,
        return_colbert=req.return_colbert,
        max_length=req.max_length,
    )
    out["model"] = "BAAI/bge-m3"
    out["latency_ms"] = (time.perf_counter() - t0) * 1000
    return EmbedResponse(**out)


@app.get("/ping")
async def ping() -> Response:
    if _model is None:
        return Response(status_code=204)
    return Response(status_code=200)


@app.get("/health")
async def health() -> dict[str, float | str | None]:
    if _model is None:
        return {"status": "loading", "vram_used_mb": None}

    # torch is imported lazily — the worker is decoupled from the
    # heavy pytorch install (it's installed by the BGE-M3 image, but
    # CPU test runners / lint envs shouldn't need it). When torch is
    # missing we just omit the VRAM figure; everything else still works.
    vram: float | None = None
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available() and str(_model.device).startswith("cuda"):
        free, total = torch.cuda.mem_get_info(_model.device)
        vram = (total - free) / 1024 / 1024
    return {"status": "ready", "vram_used_mb": vram, "device": _model.device}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus exposition endpoint.

    Returns the default ``prometheus_client`` registry output. When
    ``prometheus-fastapi-instrumentator`` is importable, it has already
    added per-handler histograms to the same registry at module load
    time — the output here is a superset of either path.
    """
    registry: CollectorRegistry = _PROM_DEFAULT_REGISTRY
    payload = generate_latest(registry)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
