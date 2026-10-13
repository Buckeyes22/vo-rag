# syntax=docker/dockerfile:1.7
# BGE-M3 embedding worker.
#
# The vo-rag worker reads BGE_M3_* and VORAG_LOG_* env vars directly and has
# no dependence on config/settings.yaml.
#
# The image is self-contained: it bakes BAAI/bge-m3 into /app/model-cache
# so cold starts are deterministic and the local RunPod batch endpoint
# never has to redownload. Same image is reused for local dev with
# workers/embedding/Dockerfile.local (which only changes the EXPOSE / PORT
# wiring — see that file for the override pattern).
ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.12

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

ARG PYTHON_VERSION
LABEL org.opencontainers.image.title="vo-rag-bge-m3" \
      org.opencontainers.image.description="Local BGE-M3 embedding worker for vo-rag"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/model-cache \
    TRANSFORMERS_CACHE=/app/model-cache \
    BGE_M3_MODEL_PATH=/app/model-cache \
    BGE_M3_DEVICE=cuda:0 \
    VORAG_LOG_LEVEL=INFO \
    VORAG_JSON_LOGS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip \
        git curl ca-certificates \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is the vo-rag repo root — the same image is built from
# the project root as `docker build -f workers/embedding/Dockerfile .`.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the vo-rag package (no extras) so vo_rag.workers.embedding is on
# PYTHONPATH, then layer the BGE-M3 GPU stack on top. torch is pinned to
# the cu124 wheel — matches CUDA 12.4.1 in the FROM line.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 \
    && pip install --no-cache-dir --no-deps -e . \
    && pip install --no-cache-dir \
        "FlagEmbedding>=1.3,<2" \
        "transformers>=4.44,<5" \
        "huggingface-hub>=0.24,<1" \
        "fastapi>=0.115,<1" \
        "uvicorn[standard]>=0.30,<1" \
        "pydantic>=2.7,<3" \
        "pydantic-settings>=2.5" \
        "pyyaml>=6.0" \
        "httpx>=0.27,<1" \
        "structlog>=25.5.0" \
        "prometheus-client>=0.25,<1"

# Bake the model into the image so cold starts don't redownload.
# This adds ~2.3 GB to the image but makes the worker usable offline.
RUN python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="/app/model-cache",
    local_dir_use_symlinks=False,
)
PY

COPY workers/embedding/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Local default port. RunPod batch deployments override $PORT (and
# therefore the EXPOSE) at runtime via the worker-vllm template.
EXPOSE 8090
ENTRYPOINT ["/entrypoint.sh"]
