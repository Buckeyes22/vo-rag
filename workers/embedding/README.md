# BGE-M3 embedding worker

Local BGE-M3 dense + sparse + ColBERT embedding service. Implements the
canonical vo-rag wire format defined in
[`vo_rag.core.embedding.schemas`](../../src/vo_rag/core/embedding/schemas.py).

## Endpoints

| Method | Path       | Description                                                   |
|--------|------------|---------------------------------------------------------------|
| GET    | `/ping`    | 204 while the model is loading, 200 once loaded.              |
| GET    | `/health`  | `{"status": "loading", "vram_used_mb": null}` or `{"status": "ready", "vram_used_mb": …, "device": "cuda:0"}`. |
| POST   | `/embed`   | Body: `EmbedRequest`. Returns: `EmbedResponse` (dense 1024-dim, sparse Qdrant-friendly `{indices, values}`, optional ColBERT). |
| GET    | `/metrics` | Prometheus exposition (`prometheus_client` default registry). |

## Local run (CPU/GPU dev)

The fastest path — image build, then run:

```bash
docker build -f workers/embedding/Dockerfile -t vo-rag-bge-m3:dev .
docker run --rm --gpus all -p 8090:8090 vo-rag-bge-m3:dev
curl -s http://localhost:8090/ping
# 200 once the model is ready (model is baked into the image)
curl -s http://localhost:8090/health
# {"status": "ready", "vram_used_mb": 1024.0, "device": "cuda:0"}
curl -s -X POST http://localhost:8090/embed \
    -H 'content-type: application/json' \
    -d '{"texts": ["hello", "world"], "return_dense": true, "return_sparse": true, "return_colbert": true}'
```

For a slimmer dev image that defers the model download to a host mount:

```bash
docker build -f workers/embedding/Dockerfile.local -t vo-rag-bge-m3:local .
docker run --rm --gpus all -p 8090:8090 \
    -v /host/cache/bge-m3:/models/bge-m3:ro \
    -e BGE_M3_MODEL_PATH=/models/bge-m3 \
    vo-rag-bge-m3:local
```

## Environment variables

| Var                  | Default            | Notes                                            |
|----------------------|--------------------|--------------------------------------------------|
| `BGE_M3_DEVICE`      | `cuda:0`           | `cpu` is supported for tests; `cuda:N` for GPU.  |
| `BGE_M3_MODEL_PATH`  | `BAAI/bge-m3`      | HF repo id, or a baked local path.               |
| `VORAG_LOG_LEVEL`    | `INFO`             | `DEBUG`/`INFO`/`WARNING`/`ERROR`.                |
| `VORAG_JSON_LOGS`    | `1`                | `0` → pretty console; `1` → JSON.                |
| `PORT`               | `8090`             | Honoured by the entrypoint; sets uvicorn port.   |

## RunPod batch

The canonical `Dockerfile` is reused — push it to a registry, then point
the worker-vllm `text-gen` (or new) template at the image with
`PORT=80` (RunPod batch default). The entrypoint reads `$PORT` and binds
uvicorn to it without an image rebuild.

## GPU integration tests (opt-in)

The default CI stays on CPU. To exercise the worker on a real CUDA host:

```bash
VORAG_RUN_GPU_EMBEDDING=1 \
    uv run pytest tests/workers/embedding/test_gpu_integration.py -v -m integration
```

The integration test is `SKIP`ped by default — see the test file for the
exact skip semantics.

## Source layout

| File                                                | Role                                         |
|-----------------------------------------------------|----------------------------------------------|
| `../../../src/vo_rag/workers/embedding/model.py`    | `BgeM3Worker` — lazy FlagEmbedding wrapper.  |
| `../../../src/vo_rag/workers/embedding/server.py`   | FastAPI app + `/ping` `/health` `/embed` `/metrics`. |
| `../../src/vo_rag/core/embedding/schemas.py`     | Wire-format `EmbedRequest` / `EmbedResponse`.|
| `entrypoint.sh`                                     | `exec uvicorn` against the vo-rag module.    |
| `Dockerfile`                                        | Self-contained: bakes BAAI/bge-m3.           |
| `Dockerfile.local`                                  | Dev variant: model mounted, not baked.       |
