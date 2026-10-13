#!/bin/bash
# BGE-M3 worker entrypoint — exec uvicorn against the canonical vo-rag
# BGE-M3 module path. Honours $PORT (default 8090) so the same image runs
# locally (8090) and on RunPod batch (80, 8080, etc.) without rebuilding.
set -euo pipefail

exec uvicorn vo_rag.workers.embedding.server:app \
    --host 0.0.0.0 \
    --port "${PORT:-8090}" \
    --workers 1
