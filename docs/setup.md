# Setup and configuration

[README](../README.md) · [Operations](operations.md) · [Testing](testing.md)

## 1. Developer environment

Use Python 3.12 and uv. The dependency lock pins the resolved packages; the optional `llamaindex` extra is required for adapter tests.

```sh
uv sync --locked --extra dev --extra llamaindex
cp .env.example .env
set -a; . ./.env; set +a
uv run vo-rag version
uv run vo-rag --help
uv run vo-rag roster
make test
```

GPU packages and native parser executables need platform-specific setup.

## 2. Configuration contract

[Settings](../src/vo_rag/core/config/settings.py) loads `config/settings.yaml`, or the file named by `VORAG_SETTINGS_PATH`, interpolating `${VARIABLE:default}`. Models reject unknown configuration keys. `.env` is not loaded automatically: export it explicitly as above. The CLI has a separate optional `VORAG_CLI_CONFIG` file and command-line URL override.

| Area | Principal variables | Public default / requirement |
| --- | --- | --- |
| HTTP | `VORAG_API_KEY`, `VORAG_API_KEYS_JSON`, `VORAG_API_HOST` | Supply a fresh private key. The bind address is the `--host` you pass to uvicorn; `VORAG_API_HOST` fills `api.host`, which nothing reads |
| Metadata | `VORAG_POSTGRES_URL` | Your own initialized Postgres database |
| Retrieval | `QDRANT_URL`, `QDRANT_API_KEY` | Your own Qdrant service and collections |
| Artifact bytes | `GARAGE_ENDPOINT`, `GARAGE_ACCESS_KEY`, `GARAGE_SECRET_KEY` | Your own S3-compatible storage |
| Inference | `BGE_M3_QUERY_URL` (API queries and ingest), `BGE_M3_BATCH_URL` (the `vo-rag build` commands), `RERANKER_URL` | Local example endpoints, all on port 8501: the unified inference worker described below serves embedding, reranking and the vision gateway |
| Answer generation | `ANSWER_ENABLED`, provider allow flags and provider keys | Disabled by default |
| Visual descriptions | `VLM_ENABLED` | Disabled by default |
| Telemetry | `VORAG_OTEL_ENABLED`, `VORAG_OTEL_ENDPOINT` | Disabled; configure your own collector |

[Configuration](configuration.md) explains how the YAML file loads and lists every key and environment variable. Templates contain no valid provider credentials or endpoint identifiers. `.env.example` points `VORAG_POSTGRES_URL` at the local test port without a user or password, and turns the source-publication switches off; supply your own database credentials, and read [ingestion](ingestion.md#source-publication-revisions) before turning publication on.

## 3. Service-backed API

Set up storage and inference before treating the API as ready. Review [migration definitions](../src/vo_rag/core/db/migrations) and [migration preflight](../scripts/check_migrations.py) for your selected database. A preflight check reports migration state; it does not initialize the database or authorize applying schema changes. Creating the tables and collections is a one-time step described in [data model](data-model.md#bootstrapping-a-new-deployment).

The repository includes an inference process. [`vo_rag.workers.inference.server:app`](../src/vo_rag/workers/inference/server.py) is the unified GPU worker that every default URL points at: it serves `/embed` (BGE-M3 dense, sparse and optional ColBERT vectors), `/rerank` (a GPU cross-encoder) and a gateway for figure descriptions. It needs a CUDA GPU and the `runtime` extra (section 5), and it has no container recipe here. A standalone embedding worker with a Dockerfile is in [`workers/embedding`](../workers/embedding).

```sh
# After explicitly configuring authentication, storage and inference:
uv run --extra runtime uvicorn vo_rag.workers.inference.server:app --host 127.0.0.1 --port 8501
uv run uvicorn vo_rag.api.main:app --host 127.0.0.1 --port 8000
uv run python -m vo_rag.pipelines.book_rag.dispatcher   # ingestion does not progress without it
```

The API only records an intake envelope; the dispatcher does the work. Start order, the other processes (MCP server, cost exporter, OCR worker) and their ports are in [operations](operations.md#starting-the-processes).

Use `/health` for process liveness, then `/ready/query` and `/ready/intake` for capability readiness. These probes, `/ready`, `/metrics`, `/docs` and `/openapi.json` are intentionally exempt from API-key authentication. Restrict access through your network or reverse-proxy policy when deploying beyond loopback. A successful liveness response does not mean ingestion or retrieval can run.

## 4. Optional portal

Use the pinned `pnpm@10.33.0` workspace package manager with a compatible Node runtime (validation used Node 22).

```sh
pnpm install --frozen-lockfile
pnpm --filter @vo-rag/portal typecheck
pnpm --filter @vo-rag/portal test
pnpm --filter @vo-rag/portal dev
```

Vite's API proxy defaults to localhost:8000. `VORAG_API_DEV_PROXY_TARGET` and `VORAG_PORTAL_DEV_API_KEY` configure server-side proxy access. Keep credentials out of browser bundles. The portal needs an API for live data; its static fleet panel is a generic example.

## 5. Optional runtime

`uv sync --locked --extra runtime` adds substantial parser, OCR and embedding dependencies. Model weights, GPU drivers, service startup and native parser tools are separate prerequisites. Review their licensing and resource requirements before downloading or running them.
