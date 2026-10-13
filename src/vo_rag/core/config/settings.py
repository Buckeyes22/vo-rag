"""vo-rag configuration loader.

Loads config/settings.yaml with ${VAR:default} interpolation, wraps in a
typed Pydantic Settings tree, exposes get_settings() as a cached singleton.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _interpolate(obj: Any) -> Any:
    """Recursively substitute ${VAR} and ${VAR:default} in strings, lists, dicts.

    - ${VAR} with VAR unset -> empty string (matches shell default-empty behavior)
    - ${VAR:default} with VAR unset -> default (literal, no further interpolation)
    - ${VAR:} with VAR unset -> empty string
    """
    if isinstance(obj, str):

        def _sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            return "" if default is None else default

        return _ENV_PATTERN.sub(_sub, obj)
    if isinstance(obj, dict):
        return {k: _interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# Sub-models. Each uses `extra="forbid"` so unknown YAML keys fail loudly.
# --------------------------------------------------------------------------- #
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApiSettings(_Strict):
    # Not read: the bind address is the --host passed to uvicorn.
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str
    api_keys_json: str = ""
    cors_origins: list[str] = Field(default_factory=list)
    request_timeout_s: int = 60


class PortalSettings(_Strict):
    public_url: str = "https://localhost:3080"
    fallback_internal_url: str = "http://127.0.0.1:3080"
    access_team_domain: str = ""
    access_aud: str = ""
    allowed_emails: list[str] = Field(default_factory=list)
    csrf_secret: str = ""
    upload_staging_path: str = "./var/portal-uploads"
    upload_chunk_bytes: int = 8 * 1024 * 1024
    upload_max_bytes: int = 2 * 1024 * 1024 * 1024
    upload_stale_hours: int = 24

    @field_validator("allowed_emails", mode="before")
    @classmethod
    def _parse_allowed_emails(cls, value: object) -> object:
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                import json

                try:
                    parsed = json.loads(raw)
                except ValueError:
                    parsed = None
                if isinstance(parsed, list):
                    return parsed
                raw = raw.strip("[]")
            return [item.strip() for item in raw.split(",") if item.strip()]
        return value


class DatabaseSettings(_Strict):
    url: str
    migrations_dir: str = "src/vo_rag/core/db/migrations"
    pool_min: int = 2
    pool_max: int = 10


class QdrantSettings(_Strict):
    url: str
    prefer_grpc: bool = True
    api_key: str = ""


class FiguresSettings(_Strict):
    """Figure-image origin store (NVMe filesystem)."""

    root: str = "/data/vo-rag/figures"


class GarageBuckets(_Strict):
    artifacts: str = "vorag-artifacts"
    models: str = "vorag-models"
    backups: str = "vorag-backups"


class GarageSettings(_Strict):
    endpoint: str
    access_key: str = ""
    secret_key: str = ""
    region: str = "garage"
    buckets: GarageBuckets = Field(default_factory=GarageBuckets)


class EmbeddingSettings(_Strict):
    batch_url: str
    query_url: str
    query_serverless_url: str = ""
    batch_serverless_url: str = ""
    query_api_key: str = ""
    batch_api_key: str = ""
    timeout_s: int = 60


class ChunkingSettings(_Strict):
    strategy: str = "hierarchical"  # hierarchical | semantic
    semantic_breakpoint_percentile: float = 95.0
    semantic_buffer_size: int = 1

    @model_validator(mode="after")
    def _validate_strategy(self) -> ChunkingSettings:
        allowed = {"hierarchical", "semantic"}
        if self.strategy not in allowed:
            raise ValueError("chunking.strategy must be one of: hierarchical, semantic")
        return self


class VlmSettings(_Strict):
    """Resident visual-specialist settings.

    ``url`` is the service root; callers append ``/v1`` for the OpenAI-compatible
    surface and ``/health`` for readiness. Keeping one root avoids the historical
    split between the unused topology setting and the operator-only runtime vars.
    """

    enabled: bool = False
    url: str = "http://localhost:8501/vlm"
    model: str = "Qwen3VL-8B-Instruct-Q8_0.gguf"
    describe_concurrency: int = Field(default=1, ge=1, le=1)
    max_tokens: int = Field(default=1024, ge=64, le=1024)
    per_image_deadline_s: float = Field(default=300.0, gt=0.0, le=900.0)
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    primary_max_pixels: int = Field(default=1_048_576, ge=65_536, le=1_572_864)
    retry_max_pixels: int = Field(default=1_572_864, ge=65_536, le=1_572_864)

    @model_validator(mode="after")
    def _validate_retry_image_budget(self) -> VlmSettings:
        if self.retry_max_pixels < self.primary_max_pixels:
            raise ValueError("vlm.retry_max_pixels must be >= vlm.primary_max_pixels")
        return self


class RerankerSettings(_Strict):
    model_path: str = "./models/gte-reranker-onnx-int8"
    intra_threads: int = 16
    batch_size: int = 32
    depth_low: int = 20
    depth_high: int = 100
    auto_merge_enabled: bool = False
    auto_merge_threshold: float = 0.5
    # Remote-service fields (VOR-8). ``url`` set ⇒ the API calls a remote
    # /rerank (RerankerClient); empty ⇒ in-process. ``backend=auto`` resolves to
    # ``client`` when a url is set, else ``onnx_cpu``. ``gpu`` is used by the
    # inference worker itself (it hosts GpuReranker on CUDA).
    url: str = ""
    backend: str = "auto"  # auto | client | gpu | onnx_cpu
    model_id: str = "Alibaba-NLP/gte-reranker-modernbert-base"
    timeout_s: int = 30

    @model_validator(mode="after")
    def _resolve_backend(self) -> RerankerSettings:
        if self.backend == "auto":
            object.__setattr__(self, "backend", "client" if self.url else "onnx_cpu")
        return self


class TriageSettings(_Strict):
    local_model_url: str
    confidence_threshold: float = 0.7
    anthropic_api_key: str = ""
    fallback_model: str = "claude-haiku-4-5-20251001"


class ObservabilitySettings(_Strict):
    langfuse_host: str
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    log_level: str = "INFO"
    json_logs: bool = True


class OtelSettings(_Strict):
    enabled: bool = False
    endpoint: str = "http://127.0.0.1:4318"
    service_name: str = "vo-rag"
    service_namespace: str = "vo-rag"
    deployment_environment: str = "dev"
    role: str = "rag"
    host_name: str = ""


class ComplianceSettings(_Strict):
    mode: str = "monitoring"  # monitoring | enforcement
    similarity_flag: float = 0.80
    similarity_reject: float = 0.90
    min_sources: int = 3
    max_single_source_contribution: float = 0.40
    min_page_distance: int = 5
    max_tokens_per_source: int = 2000
    max_verbatim_tokens: int = 100
    max_total_verbatim_pct: float = 0.15


class AnswerSettings(_Strict):
    enabled: bool = False
    provider: str = "anthropic"
    anthropic_api_key: str = ""
    quality_model: str = "claude-sonnet-4-6"
    fast_model: str = "claude-haiku-4-5-20251001"
    audit_model: str = "claude-opus-4-7"
    kimi_model: str = "kimi-k2.6"
    kimi_api_key: str = ""
    kimi_base_url: str = "https://api.moonshot.ai/v1"
    allow_hosted: bool = False
    allow_kimi: bool = False
    allow_local: bool = False
    allow_raw_hosted_egress: bool = False
    # On-prem llama.cpp daily-driver for the LOCAL answer profile (no external
    # egress, so gated on allow_local not allow_hosted). base_url is the server
    # root (the LlamaCppProvider appends /v1/chat/completions), e.g.
    # http://127.0.0.1:8080; model is the served /v1/models id.
    local_base_url: str = ""
    local_model: str = ""
    max_context_chunks: int = 8
    max_output_tokens: int = 1200
    timeout_s: int = 90


class TailscaleSettings(_Strict):
    oauth_client_id: str = ""
    oauth_client_secret: str = ""
    tailnet: str = ""


class R2Settings(_Strict):
    endpoint: str = ""
    access_key: str = ""
    secret_key: str = ""
    bucket: str = "vorag-cloud-staging"


class CloudSettings(_Strict):
    monthly_budget_usd: float = 50.0
    per_job_max_usd: float = 10.0
    tailscale: TailscaleSettings = Field(default_factory=TailscaleSettings)
    r2: R2Settings = Field(default_factory=R2Settings)
    rates: dict[str, dict[str, float]] = Field(default_factory=dict)


class ServerlessSettings(_Strict):
    """RunPod Serverless endpoints.

    Four production-shaped endpoints (text, vlm, dsi, plus the legacy
    ``exp`` slot kept for backward-compat). Each is provisioned with
    ``runpodctl serverless create`` against the official worker-vllm hub
    (cm8h09d9n000008jvh2rqdsmb).

    The ``dsi`` (Document Structure Index) role's endpoint id
    (``SERVERLESS_DSI_ENDPOINT_ID``) is optional, like the legacy ``exp``
    alias; the vlm and text endpoint ids are required.
    """

    api_key: str  # RUNPOD_API_KEY
    vlm_endpoint_id: str
    vlm_model_name: str = "qwen3-vl-32b"
    text_endpoint_id: str
    text_model_name: str = "qwen3-6-27b"
    dsi_endpoint_id: str | None = None
    dsi_model_name: str = "qwen3-6-35b-a3b"
    exp_endpoint_id: str | None = None
    exp_model_name: str = "qwen3-6-35b-a3b"
    base_url_template: str = "https://api.runpod.ai/v2/{endpoint_id}/openai/v1"
    base_url_override: str = ""
    request_timeout_s: int = 600

    def base_url(self, role: str, *, ignore_override: bool = False) -> str:
        if self.base_url_override and not ignore_override:
            return self.base_url_override
        endpoint_id = {
            "vlm": self.vlm_endpoint_id,
            "text": self.text_endpoint_id,
            "dsi": self.dsi_endpoint_id,
            "exp": self.exp_endpoint_id,
        }[role]
        if not endpoint_id:
            raise ValueError(f"serverless role {role!r} has no endpoint_id configured")
        return self.base_url_template.format(endpoint_id=endpoint_id)

    def chat_completions_url(self, role: str) -> str:
        return f"{self.base_url(role)}/chat/completions"

    def model_name(self, role: str) -> str:
        return {
            "vlm": self.vlm_model_name,
            "text": self.text_model_name,
            "dsi": self.dsi_model_name,
            "exp": self.exp_model_name,
        }[role]


# --------------------------------------------------------------------------- #
# VCE settings (plain typed config with no other vo_rag dependencies).
# --------------------------------------------------------------------------- #
class VceBlobConfig(_Strict):
    bucket_raw: str = "vorag-vce-raw"
    bucket_derived: str = "vorag-vce-derived"
    retention_days_raw: int = 90
    retention_days_derived: int = 365


class VceIntakeConfig(_Strict):
    allowed_platforms: list[str] = Field(default_factory=lambda: ["youtube", "local"])
    youtube_url_patterns: list[str] = Field(default_factory=lambda: ["youtube.com", "youtu.be"])


class VceTranscriptionConfig(_Strict):
    model: str = "large-v3"
    diarization: bool = True
    min_speakers: int = 1
    max_speakers: int = 4


class VceFrameSamplingConfig(_Strict):
    max_frames_per_video: int = 1200
    scene_threshold: float = 0.3
    audio_cue_regex: list[str] = Field(
        default_factory=lambda: [
            "look at",
            "see this",
            "compare",
            "right here",
            "notice",
            "brake trace",
            "throttle",
            "apex",
            "delta",
            "line",
        ]
    )
    uniform_fallback_fps: float = 0.25
    cue_window_pre_s: float = 3.0
    cue_window_post_s: float = 8.0
    cue_window_fps: float = 1.0
    extract_concurrency: int = 4


class VceVisualExtractConfig(_Strict):
    prompt_name: str = "vce.visual_extract"
    frames_per_call: int = 4
    max_calls_per_video: int = 100
    workload: str = "vlm-describe"
    candidate_window_pre_s: float = 4.0
    candidate_window_post_s: float = 12.0
    candidate_min_gap_s: float = 4.0


class VceOcrConfig(_Strict):
    enabled: bool = True
    fps: float = 1.0
    engine: str = "glm_ocr_sglang"
    glm_model_server_url: str = "http://localhost:8180"
    glm_model: str = "glm-ocr"
    glm_max_image_px: int = 1280
    glm_timeout_s: float = 180.0
    frame_source: str = "overlay"
    max_frames_per_video: int = 1200
    min_frame_gap_s: float = 0.0


class VceSegmentationConfig(_Strict):
    min_segment_seconds: int = 15
    max_segment_seconds: int = 90
    rerank_workload: str = "vlm-sterilize"


class VceStructuredExtractConfig(_Strict):
    extract_workload: str = "vlm-sterilize"
    max_tokens: int = 4096
    max_records_per_segment: int = 4


class VceValidationConfig(_Strict):
    judge_workload: str = "vlm-sterilize"
    judge_confidence_threshold: float = 0.7


class VceEnrichmentConfig(_Strict):
    embedding_collection_alias: str = "vorag_vce_v1"
    signature_collection_alias: str = "vorag_vce_signature_v1"
    signature_version: int = 1


class VceRetrievalConfig(_Strict):
    default_k: int = 10
    rrf_constant: int = 60


class VceSettings(_Strict):
    blob: VceBlobConfig = Field(default_factory=VceBlobConfig)
    intake: VceIntakeConfig = Field(default_factory=VceIntakeConfig)
    transcription: VceTranscriptionConfig = Field(default_factory=VceTranscriptionConfig)
    frame_sampling: VceFrameSamplingConfig = Field(default_factory=VceFrameSamplingConfig)
    visual_extract: VceVisualExtractConfig = Field(default_factory=VceVisualExtractConfig)
    ocr: VceOcrConfig = Field(default_factory=VceOcrConfig)
    segmentation: VceSegmentationConfig = Field(default_factory=VceSegmentationConfig)
    structured_extract: VceStructuredExtractConfig = Field(
        default_factory=VceStructuredExtractConfig
    )
    validation: VceValidationConfig = Field(default_factory=VceValidationConfig)
    enrichment: VceEnrichmentConfig = Field(default_factory=VceEnrichmentConfig)
    retrieval: VceRetrievalConfig = Field(default_factory=VceRetrievalConfig)
    cost_ceiling_usd_per_video: float = 10.0


# --------------------------------------------------------------------------- #
# Top-level settings
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Top-level settings tree. Unknown keys at this level reject."""

    model_config = SettingsConfigDict(extra="forbid")

    api: ApiSettings
    portal: PortalSettings = Field(default_factory=PortalSettings)
    database: DatabaseSettings
    qdrant: QdrantSettings
    garage: GarageSettings
    figures: FiguresSettings = Field(default_factory=FiguresSettings)
    embedding: EmbeddingSettings
    chunking: ChunkingSettings = ChunkingSettings()
    vlm: VlmSettings
    serverless: ServerlessSettings
    reranker: RerankerSettings
    triage: TriageSettings
    observability: ObservabilitySettings
    otel: OtelSettings = Field(default_factory=OtelSettings)
    compliance: ComplianceSettings
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    cloud: CloudSettings = Field(default_factory=CloudSettings)
    vce: VceSettings = Field(default_factory=VceSettings)


# --------------------------------------------------------------------------- #
# Public loader + cache
# --------------------------------------------------------------------------- #
def _settings_path() -> Path:
    """Location of settings.yaml; overridable via VORAG_SETTINGS_PATH."""
    return Path(os.environ.get("VORAG_SETTINGS_PATH", "config/settings.yaml"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance. Subsequent calls return the same object.

    Raises:
        FileNotFoundError: settings file missing
        pydantic.ValidationError: missing required keys / unknown keys / bad types
    """
    path = _settings_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    interpolated = _interpolate(raw)
    return Settings(**interpolated)


def reset_settings_cache() -> None:
    """Clear the lru_cache. Useful in tests or after hot-reloading settings.yaml."""
    get_settings.cache_clear()


def _handle_sighup(_: int, __: object) -> None:
    """SIGHUP hook to reload settings without process restart."""
    get_settings.cache_clear()


if hasattr(signal, "SIGHUP"):
    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGHUP, _handle_sighup)


__all__ = [
    "Settings",
    "ApiSettings",
    "PortalSettings",
    "DatabaseSettings",
    "QdrantSettings",
    "FiguresSettings",
    "GarageSettings",
    "GarageBuckets",
    "EmbeddingSettings",
    "ChunkingSettings",
    "VlmSettings",
    "RerankerSettings",
    "TriageSettings",
    "ObservabilitySettings",
    "OtelSettings",
    "AnswerSettings",
    "ComplianceSettings",
    "TailscaleSettings",
    "R2Settings",
    "CloudSettings",
    "ServerlessSettings",
    "VceSettings",
    "VceBlobConfig",
    "VceIntakeConfig",
    "VceTranscriptionConfig",
    "VceFrameSamplingConfig",
    "VceVisualExtractConfig",
    "VceOcrConfig",
    "VceSegmentationConfig",
    "VceStructuredExtractConfig",
    "VceValidationConfig",
    "VceEnrichmentConfig",
    "VceRetrievalConfig",
    "get_settings",
    "reset_settings_cache",
]
