"""Unified BGE-M3 embedding + GPU reranking HTTP worker.

This worker extends the standalone embedding worker with a colocated GPU
reranker while preserving the embedding worker's public route contracts:

- ``POST /embed`` keeps the canonical ``EmbedRequest`` / ``EmbedResponse``
  wire format and response shaping.
- ``GET /ping`` is a lightweight liveness probe.
- ``GET /health`` exposes the embedding model load status and CUDA memory
  summary in the same shape as the embedding worker.
- ``GET /metrics`` returns the default ``prometheus_client`` registry output.

The heavy model classes are imported inside the startup hook so test runners
can import this module without CUDA, torch, transformers, or FlagEmbedding.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

import httpx
import structlog
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client import REGISTRY as _PROM_DEFAULT_REGISTRY
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vo_rag.core.embedding.schemas import EmbedRequest, EmbedResponse
from vo_rag.core.observability.metrics import (
    inference_active_requests,
    inference_overload_total,
    inference_request_duration_seconds,
    inference_requests_total,
    vlm_gateway_active_requests,
    vlm_gateway_admission_timeouts_total,
    vlm_gateway_duration_seconds,
    vlm_gateway_queue_seconds,
    vlm_gateway_requests_total,
    vlm_gateway_up,
)
from vo_rag.core.reranking.schemas import RerankCandidate, RerankRequest, RerankResponse
from vo_rag.core.retrieval.search import SearchResult

app = FastAPI(title="vo-rag Unified Inference Worker")

# Runtime-loaded singletons (lazy, GPU-only); typed Any so the heavy model
# imports stay inside startup and tests can inject fakes.
_embed_model: Any = None
_reranker: Any = None
_embedding_resident_device: str | None = None
_reranker_resident_device: str | None = None
_vlm_http_client: httpx.AsyncClient | None = None
_vlm_health_task: asyncio.Task[None] | None = None


class _VlmContentPart(BaseModel):
    """One bounded text or inline-image content part accepted by the gateway."""

    model_config = ConfigDict(extra="forbid")

    type: str
    text: str | None = None
    image_url: dict[str, str] | None = None

    @model_validator(mode="after")
    def _validate_part(self) -> _VlmContentPart:
        if self.type == "text":
            if not self.text or len(self.text) > VLM_MAX_TEXT_CHARS:
                raise ValueError("VLM text parts must contain at most 20000 characters")
            if self.image_url is not None:
                raise ValueError("text content cannot include image_url")
            return self
        if self.type != "image_url" or self.text is not None or self.image_url is None:
            raise ValueError("VLM content parts must be text or image_url")
        url = self.image_url.get("url", "")
        _validate_inline_image(url)
        return self


class _VlmMessage(BaseModel):
    """One gateway chat message with bounded multimodal content."""

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str | list[_VlmContentPart]

    @model_validator(mode="after")
    def _validate_message(self) -> _VlmMessage:
        if self.role not in {"system", "user"}:
            raise ValueError("resident VLM accepts only system and user messages")
        if isinstance(self.content, str) and (
            not self.content or len(self.content) > VLM_MAX_TEXT_CHARS
        ):
            raise ValueError("VLM message text must contain at most 20000 characters")
        return self


class _VlmChatRequest(BaseModel):
    """Narrow OpenAI chat request forwarded to the localhost VLM."""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[_VlmMessage] = Field(min_length=1, max_length=4)
    max_tokens: int = Field(ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    stream: bool = False
    json_schema: dict[str, Any]

    @model_validator(mode="after")
    def _validate_request(self) -> _VlmChatRequest:
        if self.model != VLM_MODEL:
            raise ValueError("unrecognized resident VLM model")
        if self.stream:
            raise ValueError("resident VLM streaming is disabled")
        image_count = sum(
            1
            for message in self.messages
            if isinstance(message.content, list)
            for part in message.content
            if part.type == "image_url"
        )
        if image_count != 1:
            raise ValueError("resident VLM requests require exactly one inline image")
        text_chars = sum(
            len(message.content)
            if isinstance(message.content, str)
            else sum(len(part.text or "") for part in message.content)
            for message in self.messages
        )
        if text_chars > VLM_MAX_TEXT_CHARS:
            raise ValueError("resident VLM request text exceeds the aggregate limit")
        if len(json.dumps(self.json_schema, separators=(",", ":"))) > VLM_MAX_SCHEMA_CHARS:
            raise ValueError("resident VLM JSON schema exceeds the encoded size limit")
        return self


def _llama_cpp_compatible_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove the visual description bound that llama.cpp cannot compile."""
    compatible = copy.deepcopy(schema)
    description_schema = compatible.get("properties", {}).get("description", {})
    if isinstance(description_schema, dict):
        # The 6,000-character application bound becomes ``char{1,6000}``, which
        # llama.cpp rejects as exceeding its safe GBNF repetition limit. The
        # caller still validates the decoded response against the full schema.
        description_schema.pop("maxLength", None)
    return compatible


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _positive_env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _nonnegative_env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


RERANK_MAX_DEPTH = _positive_env_int("VORAG_RERANK_MAX_DEPTH", 100)
EMBED_MAX_BATCH = _positive_env_int("VORAG_EMBED_MAX_BATCH", 32)
# Tune worker concurrency to available GPU memory and workload sizes.
INFERENCE_MAX_CONCURRENCY = _positive_env_int("VORAG_INFERENCE_MAX_CONCURRENCY", 2)
_inference_semaphore = asyncio.BoundedSemaphore(INFERENCE_MAX_CONCURRENCY)
VLM_MODEL = os.environ.get("VORAG_VLM_MODEL", "Qwen3VL-8B-Instruct-Q8_0.gguf")
VLM_UPSTREAM_URL = os.environ.get("VORAG_VLM_UPSTREAM_URL", "http://127.0.0.1:8084/v1")
VLM_QUIET_PERIOD_S = _nonnegative_env_float("VORAG_VLM_QUIET_PERIOD_S", 2.0)
VLM_ADMISSION_WAIT_S = _positive_env_float("VORAG_VLM_ADMISSION_WAIT_S", 120.0)
VLM_UPSTREAM_TIMEOUT_S = _positive_env_float("VORAG_VLM_UPSTREAM_TIMEOUT_S", 300.0)
VLM_MAX_IMAGE_BYTES = _positive_env_int("VORAG_VLM_MAX_IMAGE_BYTES", 16 * 1024 * 1024)
VLM_MAX_TEXT_CHARS = _positive_env_int("VORAG_VLM_MAX_TEXT_CHARS", 20_000)
VLM_MAX_SCHEMA_CHARS = _positive_env_int("VORAG_VLM_MAX_SCHEMA_CHARS", 65_536)
VLM_HEALTH_INTERVAL_S = _positive_env_float("VORAG_VLM_HEALTH_INTERVAL_S", 15.0)
_vlm_semaphore = asyncio.BoundedSemaphore(1)
_retrieval_condition = asyncio.Condition()
_retrieval_active = 0
_retrieval_last_finished = time.monotonic()

__all__ = [
    "EMBED_MAX_BATCH",
    "INFERENCE_MAX_CONCURRENCY",
    "RERANK_MAX_DEPTH",
    "EmbedRequest",
    "EmbedResponse",
    "RerankCandidate",
    "RerankRequest",
    "RerankResponse",
    "app",
    "_configure_logging",
    "_embed_model",
    "_load_worker_env",
    "_reranker",
    "_vlm_http_client",
    "_vlm_health_task",
]


def _validate_inline_image(url: str) -> None:
    """Reject remote/oversized images before the gateway reaches llama.cpp."""
    allowed_prefixes = (
        "data:image/png;base64,",
        "data:image/jpeg;base64,",
        "data:image/webp;base64,",
    )
    prefix = next((candidate for candidate in allowed_prefixes if url.startswith(candidate)), None)
    if prefix is None:
        raise ValueError("resident VLM accepts only inline PNG, JPEG, or WebP images")
    encoded = url[len(prefix) :]
    if len(encoded) > ((VLM_MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError("resident VLM image exceeds the decoded byte limit")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("resident VLM image is not valid base64") from exc
    if not decoded or len(decoded) > VLM_MAX_IMAGE_BYTES:
        raise ValueError("resident VLM image exceeds the decoded byte limit")


@asynccontextmanager
async def _admit(operation: str) -> AsyncIterator[None]:
    """Acquire a bounded model slot or reject immediately as overloaded."""
    if _inference_semaphore.locked():
        inference_overload_total.labels(operation=operation).inc()
        inference_requests_total.labels(operation=operation, outcome="overloaded").inc()
        raise HTTPException(429, "Inference worker is at its concurrency limit")
    await _inference_semaphore.acquire()
    global _retrieval_active, _retrieval_last_finished
    async with _retrieval_condition:
        _retrieval_active += 1
        _retrieval_condition.notify_all()
    inference_active_requests.labels(operation=operation).inc()
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        inference_requests_total.labels(operation=operation, outcome="failure").inc()
        raise
    else:
        inference_requests_total.labels(operation=operation, outcome="success").inc()
    finally:
        inference_request_duration_seconds.labels(operation=operation).observe(
            time.perf_counter() - t0
        )
        inference_active_requests.labels(operation=operation).dec()
        async with _retrieval_condition:
            _retrieval_active -= 1
            _retrieval_last_finished = time.monotonic()
            _retrieval_condition.notify_all()
        _inference_semaphore.release()


async def _wait_for_retrieval_quiet(deadline: float) -> None:
    """Wait until retrieval is inactive and has remained quiet for the policy window."""
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            raise TimeoutError("VLM admission wait expired")
        async with _retrieval_condition:
            quiet_remaining = VLM_QUIET_PERIOD_S - (now - _retrieval_last_finished)
            if _retrieval_active == 0 and quiet_remaining <= 0:
                return
            quiet_wait = max(0.01, quiet_remaining) if _retrieval_active == 0 else remaining
            wait_for = min(remaining, quiet_wait)
            try:
                await asyncio.wait_for(_retrieval_condition.wait(), timeout=wait_for)
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("VLM admission wait expired") from None


@asynccontextmanager
async def _admit_vlm() -> AsyncIterator[None]:
    """Serialize VLM work and admit it only after a bounded retrieval-idle wait."""
    queued_at = time.perf_counter()
    deadline = time.monotonic() + VLM_ADMISSION_WAIT_S
    acquired = False
    try:
        await asyncio.wait_for(_vlm_semaphore.acquire(), timeout=VLM_ADMISSION_WAIT_S)
        acquired = True
        await _wait_for_retrieval_quiet(deadline)
    except TimeoutError as exc:
        if acquired:
            _vlm_semaphore.release()
        vlm_gateway_admission_timeouts_total.inc()
        vlm_gateway_requests_total.labels(outcome="admission_timeout").inc()
        raise HTTPException(
            429,
            "VLM admission wait expired",
            headers={"Retry-After": "2"},
        ) from exc
    except BaseException:
        if acquired:
            _vlm_semaphore.release()
        raise
    vlm_gateway_queue_seconds.observe(time.perf_counter() - queued_at)
    vlm_gateway_active_requests.inc()
    try:
        yield
    finally:
        vlm_gateway_active_requests.dec()
        _vlm_semaphore.release()


def _load_worker_env() -> dict[str, Any]:
    """Read all worker env vars in one place.

    Called from the startup hook so module import stays free of any
    ``vo_rag.core.config`` plumbing.
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
    """Lightweight structlog setup for the worker."""
    normalized_level = level.upper()
    filtering_level = logging.getLevelNamesMapping().get(normalized_level, logging.INFO)

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


def _require_single_model_device(model: Any, *, label: str, expected: str) -> str:
    """Verify that every parameter for one model is on the expected device."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise RuntimeError(f"{label} does not expose model parameters")
    devices = {str(parameter.device) for parameter in parameters()}
    if not devices:
        raise RuntimeError(f"{label} has no parameters to verify")
    if devices != {expected}:
        rendered = ", ".join(sorted(devices))
        raise RuntimeError(f"{label} parameters are on [{rendered}], expected {expected}")
    return expected


def _warm_retrieval_models(*, expected_device: str) -> None:
    """Run bounded inference and prove both retrieval models are CUDA-resident."""
    global _embedding_resident_device, _reranker_resident_device

    _embedding_resident_device = None
    _reranker_resident_device = None
    if _embed_model is None or _reranker is None:
        raise RuntimeError("retrieval models must be constructed before warm-up")
    if not expected_device.startswith("cuda"):
        raise RuntimeError(f"retrieval warm-up requires a CUDA device, got {expected_device}")

    warm_text = "vo-rag retrieval model CUDA residency warm-up"
    embedded = _embed_model.embed(
        [warm_text],
        return_dense=True,
        return_sparse=True,
        return_colbert=True,
        max_length=32,
    )
    required_embedding_outputs = {"dense", "sparse", "colbert"}
    if not required_embedding_outputs.issubset(embedded):
        raise RuntimeError("embedding warm-up did not return every retrieval vector")

    warm_candidate = SearchResult(
        id="startup-warmup",
        score=0.0,
        collection="startup",
        payload={"content": warm_text},
    )
    reranked = _reranker.rerank(warm_text, [warm_candidate], top_k=1)
    if len(reranked) != 1 or reranked[0].id != warm_candidate.id:
        raise RuntimeError("reranker warm-up returned an invalid result")

    embedding_module = getattr(_embed_model.model, "model", None)
    _embedding_resident_device = _require_single_model_device(
        embedding_module,
        label="BGE-M3 embedding model",
        expected=expected_device,
    )
    _reranker_resident_device = _require_single_model_device(
        _reranker.model,
        label="GTE reranking model",
        expected=expected_device,
    )
    logging.getLogger(__name__).info(
        "retrieval models warmed and CUDA-resident: embedding=%s reranking=%s",
        _embedding_resident_device,
        _reranker_resident_device,
    )


def _retrieval_models_ready() -> bool:
    """Return whether both constructed models passed CUDA residency warm-up."""
    return (
        _embed_model is not None
        and _reranker is not None
        and _embedding_resident_device is not None
        and _reranker_resident_device is not None
    )


@app.on_event("startup")
async def _load() -> None:
    global _embed_model, _reranker, _vlm_health_task, _vlm_http_client

    cfg = _load_worker_env()
    _configure_logging(json_logs=cfg["json_logs"], level=cfg["log_level"])

    from vo_rag.core.reranking.gpu import GpuReranker
    from vo_rag.workers.embedding.model import BgeM3Worker

    _embed_model = BgeM3Worker(cfg["model_path"], use_fp16=True, device=cfg["device"])
    _reranker = GpuReranker(device=os.environ.get("BGE_M3_DEVICE", "cuda:0"))
    _warm_retrieval_models(expected_device=cfg["device"])
    _vlm_http_client = httpx.AsyncClient(timeout=VLM_UPSTREAM_TIMEOUT_S)
    _vlm_health_task = asyncio.create_task(_monitor_vlm_health())


@app.on_event("shutdown")
async def _shutdown() -> None:
    """Close the localhost VLM transport without owning llama.cpp's lifecycle."""
    global _vlm_health_task, _vlm_http_client
    if _vlm_health_task is not None:
        _vlm_health_task.cancel()
        with suppress(asyncio.CancelledError):
            await _vlm_health_task
        _vlm_health_task = None
    if _vlm_http_client is not None:
        await _vlm_http_client.aclose()
        _vlm_http_client = None


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    if _embed_model is None:
        raise HTTPException(503, "Model not loaded")
    if len(req.texts) > EMBED_MAX_BATCH:
        raise HTTPException(413, "Embedding batch too large")

    t0 = time.perf_counter()
    async with _admit("embed"):
        out = await asyncio.to_thread(
            _embed_model.embed,
            req.texts,
            return_dense=req.return_dense,
            return_sparse=req.return_sparse,
            return_colbert=req.return_colbert,
            max_length=req.max_length,
        )
    out["model"] = "BAAI/bge-m3"
    out["latency_ms"] = (time.perf_counter() - t0) * 1000
    return EmbedResponse(**out)


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest) -> RerankResponse:
    if _reranker is None:
        raise HTTPException(503, "Reranker not loaded")
    if len(req.candidates) > RERANK_MAX_DEPTH:
        raise HTTPException(413, "Rerank depth too large")

    search_results = [
        SearchResult(id=candidate.id, score=0.0, collection="", payload={"content": candidate.text})
        for candidate in req.candidates
    ]
    async with _admit("rerank"):
        reranked = await asyncio.to_thread(_reranker.rerank, req.query, search_results, req.top_k)
    return RerankResponse(
        results=[
            RerankCandidate(id=result.id, text="", payload={}, score=result.score)
            for result in reranked
        ]
    )


@app.post("/vlm/v1/chat/completions")
async def vlm_chat_completion(req: _VlmChatRequest) -> Response:
    """Forward one bounded visual request after retrieval-idle admission."""
    if _vlm_http_client is None:
        vlm_gateway_requests_total.labels(outcome="unavailable").inc()
        raise HTTPException(503, "Resident VLM transport is not ready")
    async with _admit_vlm():
        started = time.perf_counter()
        upstream_payload = req.model_dump(mode="json", exclude_none=True)
        upstream_payload["json_schema"] = _llama_cpp_compatible_schema(req.json_schema)
        try:
            response = await _vlm_http_client.post(
                f"{VLM_UPSTREAM_URL.rstrip('/')}/chat/completions",
                json=upstream_payload,
            )
        except httpx.HTTPError as exc:
            vlm_gateway_up.set(0)
            vlm_gateway_requests_total.labels(outcome="upstream_failure").inc()
            raise HTTPException(503, "Resident VLM upstream request failed") from exc
        finally:
            vlm_gateway_duration_seconds.observe(time.perf_counter() - started)
    outcome = "success" if response.status_code < 400 else "upstream_error"
    vlm_gateway_up.set(1)
    vlm_gateway_requests_total.labels(outcome=outcome).inc()
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


async def _probe_vlm_health() -> bool:
    """Probe the fixed localhost upstream and update its Prometheus gauge."""
    if _vlm_http_client is None:
        vlm_gateway_up.set(0)
        return False
    health_url = f"{VLM_UPSTREAM_URL.removesuffix('/v1').rstrip('/')}/health"
    try:
        response = await _vlm_http_client.get(health_url, timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError:
        vlm_gateway_up.set(0)
        return False
    vlm_gateway_up.set(1)
    return True


async def _monitor_vlm_health() -> None:
    """Continuously refresh VLM readiness even while the specialist is idle."""
    while True:
        await _probe_vlm_health()
        await asyncio.sleep(VLM_HEALTH_INTERVAL_S)


@app.get("/vlm/health", response_model=None)
async def vlm_health() -> dict[str, Any] | Response:
    """Report localhost llama.cpp readiness without affecting retrieval health."""
    if not await _probe_vlm_health():
        return Response(
            content=json.dumps({"status": "unavailable", "model": VLM_MODEL}),
            status_code=503,
            media_type="application/json",
        )
    return {"status": "ready", "model": VLM_MODEL, "upstream": "localhost"}


@app.get("/ping")
async def ping() -> Response:
    if not _retrieval_models_ready():
        return Response(status_code=204)
    return Response(status_code=200)


@app.get("/health")
async def health() -> dict[str, Any]:
    limits = {
        "embed_max_batch": EMBED_MAX_BATCH,
        "rerank_max_depth": RERANK_MAX_DEPTH,
        "max_concurrency": INFERENCE_MAX_CONCURRENCY,
    }
    model_residency = {
        "embedding": {
            "model": "BAAI/bge-m3",
            "device": _embedding_resident_device,
            "resident": _embedding_resident_device is not None,
        },
        "reranking": {
            "model": "Alibaba-NLP/gte-reranker-modernbert-base",
            "device": _reranker_resident_device,
            "resident": _reranker_resident_device is not None,
        },
    }
    if _embed_model is None:
        return {
            "status": "loading",
            "vram_used_mb": None,
            "models": model_residency,
            "limits": limits,
        }

    vram: float | None = None
    vram_total: float | None = None
    gpu_name: str | None = None
    compute_capability: str | None = None
    try:
        import torch  # type: ignore[import-not-found]  # runtime-only dep
    except ImportError:
        torch = None
    if (
        torch is not None
        and torch.cuda.is_available()
        and str(_embed_model.device).startswith("cuda")
    ):
        free, total = torch.cuda.mem_get_info(_embed_model.device)
        vram = (total - free) / 1024 / 1024
        vram_total = total / 1024 / 1024
        gpu_name = str(torch.cuda.get_device_name(_embed_model.device))
        major, minor = torch.cuda.get_device_capability(_embed_model.device)
        compute_capability = f"{major}.{minor}"
    return {
        "status": "ready" if _retrieval_models_ready() else "loading",
        "vram_used_mb": vram,
        "vram_total_mb": vram_total,
        "gpu_name": gpu_name,
        "cuda_compute_capability": compute_capability,
        "device": _embed_model.device,
        "models": model_residency,
        "limits": limits,
    }


@app.get("/ready")
async def ready() -> Response:
    if not _retrieval_models_ready():
        return Response(status_code=503)
    return Response(status_code=200)


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus exposition endpoint."""
    registry: CollectorRegistry = _PROM_DEFAULT_REGISTRY
    payload = generate_latest(registry)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
