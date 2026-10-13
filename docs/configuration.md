# Configuration reference

[Setup](setup.md) · [Operations](operations.md) · [Security](security.md)

vo-rag is configured by one YAML file, [`config/settings.yaml`](../config/settings.yaml), plus environment variables that the file and the code read. This page explains how the file is loaded, what each section controls, and which declared keys nothing reads. The two reference tables further down are generated from the code.

## How settings load

[`get_settings()`](../src/vo_rag/core/config/settings.py) does four things, once per process, and caches the result:

1. It reads the file named by `VORAG_SETTINGS_PATH`, or `config/settings.yaml` relative to the working directory when the variable is unset. A missing file raises `FileNotFoundError`; there is no search path.
2. It parses the YAML with `yaml.safe_load`.
3. It substitutes `${VAR}` and `${VAR:default}` in every string, recursively. If `VAR` is set in the environment the value is used, even when it is empty; if it is unset the default is used; `${VAR}` with no default and `VAR` unset becomes the empty string, not an error. The default is literal text and is not interpolated again. Nothing is substituted outside the file, and `.env` is never loaded automatically (export it into the shell first, as in [setup](setup.md#1-developer-environment)).
4. It validates the result against the `Settings` model. Every model in the tree sets `extra="forbid"`, so a misspelled or removed key fails loudly with a validation error instead of being ignored. Types are coerced, so the string `"false"` from `${VLM_ENABLED:false}` becomes a boolean.

Validation also rejects *missing* keys, and it requires eleven of the sections themselves: `api`, `database`, `qdrant`, `garage`, `embedding`, `vlm`, `serverless`, `reranker`, `triage`, `observability` and `compliance`. A file that leaves them out fails with one "field required" error per missing section or key. The other seven sections (`portal`, `figures`, `chunking`, `otel`, `answer`, `cloud`, `vce`) default entirely and may be omitted. Inside the required sections the required leaf keys are `api.api_key`, `database.url`, `qdrant.url`, `garage.endpoint`, `embedding.batch_url`, `embedding.query_url`, `serverless.api_key`, `serverless.vlm_endpoint_id`, `serverless.text_endpoint_id`, `triage.local_model_url` and `observability.langfuse_host`; `vlm`, `reranker` and `compliance` have no required leaves but must still be present, as an empty mapping (`{}`), because a bare `vlm:` is read as null and fails. Values may be empty strings; keys may not be absent. This is the smallest file that validates (checked by loading it through `VORAG_SETTINGS_PATH`); every default then comes from the model:

```yaml
api:
  api_key: "${VORAG_API_KEY:}"
database:
  url: "${VORAG_POSTGRES_URL:postgresql://localhost:5434/vorag}"
qdrant:
  url: "${QDRANT_URL:http://127.0.0.1:6333}"
garage:
  endpoint: "${GARAGE_ENDPOINT:http://localhost:3900}"
embedding:
  batch_url: "${BGE_M3_BATCH_URL:http://localhost:8501/embed}"
  query_url: "${BGE_M3_QUERY_URL:http://localhost:8501/embed}"
vlm: {}
serverless:
  api_key: "${RUNPOD_API_KEY}"
  vlm_endpoint_id: "${SERVERLESS_VLM_ENDPOINT_ID}"
  text_endpoint_id: "${SERVERLESS_TEXT_ENDPOINT_ID}"
reranker: {}
triage:
  local_model_url: "${LOCAL_LLM_URL:http://localhost:8081}"
observability:
  langfuse_host: "${LANGFUSE_HOST:http://localhost:3101}"
compliance: {}
```

`serverless.api_key` and the two endpoint ids use `${VAR}` with no default, which resolves to an empty string when the variable is unset, so the file loads without RunPod configured. Starting from a copy of `config/settings.yaml` is the simpler way to get the same result with the shipped values.

A SIGHUP clears the cached settings object, so the next `get_settings()` call reads the file again. Code that already holds a `Settings` instance keeps it, so restart a process to change its configuration. [`reset_settings_cache()`](../src/vo_rag/core/config/settings.py) does the same thing for tests.

Four things are configured outside this file:

- **The CLI** reads `~/.config/vo-rag/config.yaml` (or `VORAG_CLI_CONFIG`, or `--config`) for `api_url` and `api_key`, and lets `VORAG_API_URL`, `VORAG_API_KEY` and `--url` override them. The commands that talk to the stores directly (`vo-rag corpus ...`, `vo-rag build ...`, `vo-rag ingest run`) use `settings.yaml` instead.
- **The dispatcher, the evaluation gate and many tunables** are read straight from the environment by the code that uses them. They appear in the environment table below with the file that first reads them; the pages that explain them are [ingestion](ingestion.md#the-dispatcher) (`VORAG_BOOK_RAG_*`), [evaluation](evaluation.md) (`VO_RAG_EVAL_*`, `VO_RAG_ALLOW_*`), [reliability](reliability.md) (`VORAG_EMBEDDING_CACHE_*`, `VORAG_SOURCE_PUBLICATION_*`) and [OCR](ocr.md) (`VORAG_OCR_*`).
- **The MCP server** needs `VORAG_MCP_API_KEY` (or `api.api_key`) for its HTTP transport; see [MCP](mcp.md).
- **The workers** read their own environment (`BGE_M3_*`, `VORAG_INFERENCE_*`, `VORAG_VLM_*`, `PORT`); see [operations](operations.md#starting-the-processes).

## Sections

The model has 18 top-level sections. This table says what each is for; the key-by-key table is at the bottom of the page.

| Section | What it controls | Notes |
| --- | --- | --- |
| `api` | The API key (`api_key`), an optional JSON map of scoped keys (`api_keys_json`), and settings for the HTTP server | An empty key makes every protected route answer 401 (fail closed); see [security](security.md#authentication). The bind address and port are the `uvicorn` flags you pass, not these keys |
| `portal` | Cloudflare Access (`access_team_domain`, `access_aud`, `allowed_emails`), the CSRF secret, and the upload size limit (`upload_max_bytes`, 2 GiB) | The portal is described in [portal](portal.md) |
| `database` | The Postgres URL, the migrations directory and the pool size | The URL comes from `VORAG_POSTGRES_URL` |
| `qdrant` | The Qdrant URL, API key and whether to prefer gRPC | `QDRANT_URL`, `QDRANT_API_KEY` |
| `garage` | The S3-compatible endpoint, credentials, region and bucket names | The code uses the `artifacts` bucket (`vorag-artifacts`) |
| `figures` | The filesystem root of the figure store | `VORAG_FIGURES_ROOT` |
| `embedding` | The batch and query embedding URLs and their timeout | Both default to the inference worker on port 8501 |
| `chunking` | The chunking strategy (`hierarchical` or `semantic`) and the semantic splitter's parameters | `vo-rag ingest run --chunking-strategy` overrides it for one run |
| `vlm` | Figure description: whether it is on, the endpoint, the model name and the per-image limits | Off by default (`VLM_ENABLED`); the ranges are validated, for example `describe_concurrency` must be exactly 1 |
| `serverless` | The RunPod serverless endpoints for the `text`, `vlm`, `dsi` and `exp` roles, their model names and the request timeout | `base_url(role)` builds the endpoint URL and raises if the role has no endpoint id |
| `reranker` | The rerank backend (`auto`, `client`, `gpu`, `onnx_cpu`), its URL and timeout, the local model path and batch size, and the parent auto-merge switch | `auto_merge_enabled` is false and the HTTP query route never wires the merge, see [retrieval](retrieval-and-answers.md#parent-auto-merge-is-not-available-over-http) |
| `triage` | Only `anthropic_api_key` is read: it is the fallback key for the answer synthesizer | The remaining keys are required by the model but unused |
| `observability` | Langfuse connection settings and logging defaults | Not read; log level and format come from `VORAG_LOG_LEVEL` and `VORAG_JSON_LOGS` on the workers |
| `otel` | OpenTelemetry export: enabled flag, endpoint, service and resource names | Off by default; see [observability](observability.md#tracing-with-opentelemetry) |
| `compliance` | Thresholds for a similarity and verbatim-overlap policy | Nothing reads this section; the answer gate is supplied by the caller, see [rights and compliance](rights-and-compliance.md#the-answer-compliance-gate) |
| `answer` | Whether answers are on, the provider, the model per profile, the three egress allow flags and the context and output limits | Off by default; the allow flags are explained in [retrieval and answers](retrieval-and-answers.md) |
| `cloud` | The monthly and per-job budgets, Tailscale OAuth, R2 staging credentials and per-provider GPU rates | See [operations](operations.md#cloud-cost-governance) |
| `vce` | The video extraction pipeline's storage, transcription, sampling and validation settings | Only `vce.blob.bucket_derived` is read; see [integrations](integrations.md#video-extraction-vce) |

## Declared but not read

These keys are validated and documented in the table below, but a search of the code finds no reader for them. Setting them has no effect today. The list is accurate for this tree; check it before depending on any of them.

- `api.host`, `api.port`, `api.request_timeout_s` and `api.cors_origins`. The API sets no CORS headers, so apply cross-origin policy at your proxy.
- `portal.public_url`, `portal.fallback_internal_url`, `portal.upload_staging_path`, `portal.upload_chunk_bytes` and `portal.upload_stale_hours`.
- `garage.buckets.models` and `garage.buckets.backups`.
- `embedding.query_serverless_url`, `embedding.batch_serverless_url`, `embedding.query_api_key` and `embedding.batch_api_key`.
- `reranker.intra_threads`, `reranker.depth_low` and `reranker.depth_high`. The CPU reranker takes its thread count from a constructor default, and the candidate depth is set by the retrieval code.
- `triage.local_model_url`, `triage.confidence_threshold` and `triage.fallback_model`.
- `observability.langfuse_*`, `observability.log_level` and `observability.json_logs`.
- Every key under `compliance`.
- `answer.audit_model`.
- `vce.cost_ceiling_usd_per_video` and every `vce` key except `vce.blob.bucket_derived`.

The environment variables that exist only to fill these keys (for example `LANGFUSE_HOST`) are equally inert.

Both tables below are generated by `scripts/gen_reference_docs.py`. Do not edit between the markers; change the code and rerun the generator.

## Environment variables

<!-- BEGIN GENERATED: env-vars -->
| Variable | Default | First read |
| --- | --- | --- |
| `ANSWER_ALLOW_HOSTED` | `'false'` | `config/settings.yaml` |
| `ANSWER_ALLOW_KIMI` | `'false'` | `config/settings.yaml` |
| `ANSWER_ALLOW_LOCAL` | `'false'` | `config/settings.yaml` |
| `ANSWER_ALLOW_RAW_HOSTED_EGRESS` | `'false'` | `config/settings.yaml` |
| `ANSWER_ENABLED` | `'false'` | `config/settings.yaml` |
| `ANSWER_LOCAL_BASE_URL` | `'http://127.0.0.1:8080'` | `config/settings.yaml` |
| `ANSWER_LOCAL_MODEL` | `'unsloth_Qwen3.6-35B-A3B-GGUF_Qwen3.6-35B-A3B-Q8_0.gguf'` | `config/settings.yaml` |
| `ANTHROPIC_API_KEY` | `''` | `config/settings.yaml` |
| `BGE_M3_BATCH_URL` | `'http://localhost:8501/embed'` | `config/settings.yaml` |
| `BGE_M3_DEVICE` | `'cuda:0'` | `src/vo_rag/workers/embedding/server.py` |
| `BGE_M3_MODEL_PATH` | `'BAAI/bge-m3'` | `src/vo_rag/workers/embedding/server.py` |
| `BGE_M3_MODEL_VERSION` | unset | `src/vo_rag/core/embedding/cache.py` |
| `BGE_M3_QUERY_URL` | `'http://localhost:8501/embed'` | `config/settings.yaml` |
| `CI_COMMIT_SHA` | unset | `src/vo_rag/core/observability/cost.py` |
| `CLOUDFLARE_ACCESS_AUD` | `''` | `config/settings.yaml` |
| `CLOUDFLARE_ACCESS_TEAM_DOMAIN` | `''` | `config/settings.yaml` |
| `CLOUD_COST_EXPORTER_PORT` | `'9109'` | `services/cloud_cost_exporter/app.py` |
| `COHERE_API_KEY` | `''` | `src/vo_rag/core/reranking/cohere.py` |
| `DATABASE_URL` | unset | `scripts/eval/dedup_duplicate_books.py` |
| `EMBED_MAX_BATCH_TEXTS` | (computed) | `src/vo_rag/core/embedding/client.py` |
| `GARAGE_ACCESS_KEY` | `''` | `config/settings.yaml` |
| `GARAGE_ENDPOINT` | `'http://localhost:3900'` | `config/settings.yaml` |
| `GARAGE_REGION` | `'garage'` | `config/settings.yaml` |
| `GARAGE_SECRET_KEY` | `''` | `config/settings.yaml` |
| `GIT_SHA` | unset | `src/vo_rag/core/observability/cost.py` |
| `GTE_RERANKER_PATH` | `'models/gte-reranker-onnx-int8'` | `config/settings.yaml` |
| `KIMI_API_KEY` | `''` | `src/vo_rag/core/answer/synthesizer.py` |
| `KIMI_BASE_URL` | `'https://api.moonshot.ai/v1'` | `config/settings.yaml` |
| `KIMI_TIMEOUT_S` | `'90'` | `src/vo_rag/core/llm/client.py` |
| `LANGFUSE_HOST` | `'http://localhost:3101'` | `config/settings.yaml` |
| `LANGFUSE_PUBLIC_KEY` | `''` | `config/settings.yaml` |
| `LANGFUSE_SECRET_KEY` | `''` | `config/settings.yaml` |
| `LOCAL_LLM_TIMEOUT_S` | `'60'` | `src/vo_rag/core/llm/client.py` |
| `LOCAL_LLM_URL` | `'http://localhost:8081'` | `config/settings.yaml` |
| `MOONSHOT_API_KEY` | `''` | `config/settings.yaml` |
| `NDCG_DIR` | `'/data/vo-rag/eval_out/ndcg'` | `scripts/eval/remote_smoke_ndcg.py` |
| `PATH` | `''` | `scripts/tf_classify_lane.py` |
| `QDRANT_API_KEY` | `''` | `config/settings.yaml` |
| `QDRANT_URL` | `'http://127.0.0.1:6400'` | `config/settings.yaml` |
| `R2_ACCESS_KEY` | `''` | `config/settings.yaml` |
| `R2_BUCKET_STAGING` | `'vorag-cloud-staging'` | `config/settings.yaml` |
| `R2_ENDPOINT` | `''` | `config/settings.yaml` |
| `R2_SECRET_KEY` | `''` | `config/settings.yaml` |
| `RERANKER_BACKEND` | `'client'` | `config/settings.yaml` |
| `RERANKER_URL` | `'http://localhost:8501/rerank'` | `config/settings.yaml` |
| `RERANK_DEPTH` | `'100'` | `scripts/eval/remote_smoke_ndcg.py` |
| `RUNPOD_API_KEY` | `''` | `config/settings.yaml` |
| `RUNPOD_REST_API_URL` | `'https://rest.runpod.io/v1'` | `src/vo_rag/core/cloud/runpod_client.py` |
| `SERVERLESS_BASE_URL_OVERRIDE` | `''` | `config/settings.yaml` |
| `SERVERLESS_DSI_ENDPOINT_ID` | `''` | `config/settings.yaml` |
| `SERVERLESS_DSI_MODEL_NAME` | `'qwen3-6-35b-a3b'` | `config/settings.yaml` |
| `SERVERLESS_EMBED_BATCH_URL` | `''` | `config/settings.yaml` |
| `SERVERLESS_EMBED_QUERY_URL` | `''` | `config/settings.yaml` |
| `SERVERLESS_EXP_ENDPOINT_ID` | `''` | `config/settings.yaml` |
| `SERVERLESS_TEXT_ENDPOINT_ID` | unset | `config/settings.yaml` |
| `SERVERLESS_TEXT_MODEL_NAME` | `'qwen3-6-27b'` | `config/settings.yaml` |
| `SERVERLESS_VLM_ENDPOINT_ID` | unset | `config/settings.yaml` |
| `SERVERLESS_VLM_MODEL_NAME` | `'qwen3-vl-32b'` | `config/settings.yaml` |
| `TAILSCALE_OAUTH_CLIENT_ID` | `''` | `config/settings.yaml` |
| `TAILSCALE_OAUTH_CLIENT_SECRET` | `''` | `config/settings.yaml` |
| `TAILSCALE_TAILNET` | `''` | `config/settings.yaml` |
| `VIRTUAL_ENV` | `''` | `scripts/tf_classify_lane.py` |
| `VLM_CONFIDENCE_THRESHOLD` | `'0.65'` | `config/settings.yaml` |
| `VLM_DESCRIBE_CONCURRENCY` | `'1'` | `config/settings.yaml` |
| `VLM_ENABLED` | `'false'` | `config/settings.yaml` |
| `VLM_MAX_TOKENS` | `'1024'` | `config/settings.yaml` |
| `VLM_MODEL` | `'Qwen3VL-8B-Instruct-Q8_0.gguf'` | `config/settings.yaml` |
| `VLM_PER_IMAGE_DEADLINE_S` | `'300'` | `config/settings.yaml` |
| `VLM_PRIMARY_MAX_PIXELS` | `'1048576'` | `config/settings.yaml` |
| `VLM_RETRY_MAX_PIXELS` | `'1572864'` | `config/settings.yaml` |
| `VLM_URL` | `'http://localhost:8501/vlm'` | `config/settings.yaml` |
| `VORAG_API_HOST` | `'127.0.0.1'` | `config/settings.yaml` |
| `VORAG_API_KEY` | `''` | `config/settings.yaml` |
| `VORAG_API_KEYS_JSON` | `''` | `config/settings.yaml` |
| `VORAG_API_URL` | unset | `src/vo_rag/cli/main.py` |
| `VORAG_BOOK_RAG_DISABLE_CORRUPTION_GATE` | `'0'` | `src/vo_rag/pipelines/book_rag/steps/parse.py` |
| `VORAG_BOOK_RAG_DISPATCHER_CLAIM_NOT_BEFORE` | unset | `src/vo_rag/pipelines/book_rag/dispatcher.py` |
| `VORAG_BOOK_RAG_DISPATCHER_HEARTBEAT_INTERVAL_S` | (computed) | `src/vo_rag/pipelines/book_rag/dispatcher.py` |
| `VORAG_BOOK_RAG_DISPATCHER_HEARTBEAT_PATH` | `''` | `src/vo_rag/pipelines/book_rag/dispatcher_health.py` |
| `VORAG_BOOK_RAG_DISPATCHER_HEARTBEAT_STALE_S` | `''` | `src/vo_rag/pipelines/book_rag/dispatcher_health.py` |
| `VORAG_BOOK_RAG_DISPATCHER_ONLY_EXTENSIONS` | unset | `src/vo_rag/pipelines/book_rag/dispatcher.py` |
| `VORAG_BOOK_RAG_DISPATCHER_POLL_INTERVAL_S` | (computed) | `src/vo_rag/pipelines/book_rag/dispatcher.py` |
| `VORAG_BOOK_RAG_DISPATCHER_SHUTDOWN_GRACE_S` | (computed) | `src/vo_rag/pipelines/book_rag/dispatcher.py` |
| `VORAG_BOOK_RAG_DISPATCHER_STALE_TIMEOUT_S` | (computed) | `src/vo_rag/pipelines/book_rag/dispatcher.py` |
| `VORAG_BOOK_RAG_FAIL_ON_PARSE_RED_FLAGS` | unset | `src/vo_rag/pipelines/book_rag/steps/parse.py` |
| `VORAG_BOOK_RAG_FALLBACK_CHAPTER_PAGES` | `'50'` | `src/vo_rag/core/chunking/heading_tracker.py` |
| `VORAG_BOOK_RAG_FALLBACK_MIN_TITLEONLY_CHUNKS` | `'10'` | `src/vo_rag/core/chunking/heading_tracker.py` |
| `VORAG_BOOK_RAG_FALLBACK_SECTION_PAGES` | `'10'` | `src/vo_rag/core/chunking/heading_tracker.py` |
| `VORAG_BOOK_RAG_FORCE_REINGEST` | `''` | `src/vo_rag/pipelines/book_rag/workflow.py` |
| `VORAG_BOOK_RAG_GIANT_FALLBACK_AFTER_S` | (computed) | `src/vo_rag/pipelines/book_rag/dispatcher.py` |
| `VORAG_BOOK_RAG_INCLUDE_COLBERT` | `'0'` | `src/vo_rag/core/embedding/point_vectors.py` |
| `VORAG_BOOK_RAG_LANE_CLASS` | unset | `src/vo_rag/pipelines/book_rag/dispatcher.py` |
| `VORAG_BUILD_COLLECTION_SUFFIX` | `''` | `src/vo_rag/cli/build.py` |
| `VORAG_CHUNKING_STRATEGY` | unset | `src/vo_rag/cli/ingest.py` |
| `VORAG_CHUNK_TOKENIZER_MODEL` | `'BAAI/bge-m3'` | `src/vo_rag/core/chunking/token_counter.py` |
| `VORAG_CHUNK_TOKEN_COUNTER` | `'naive'` | `src/vo_rag/core/chunking/hierarchical.py` |
| `VORAG_CHUNK_TOKEN_COUNTER_FAIL_SOFT` | unset | `src/vo_rag/core/chunking/hierarchical.py` |
| `VORAG_CLI_CONFIG` | `'~/.config/vo-rag/config.yaml'` | `src/vo_rag/cli/main.py` |
| `VORAG_COLBERT_EMPTY_FALLBACK` | `'dense'` | `src/vo_rag/core/embedding/point_vectors.py` |
| `VORAG_DATABASE_URL` | unset | `scripts/eval/dedup_duplicate_books.py` |
| `VORAG_DOCLING_DEVICE` | `''` | `src/vo_rag/core/parsing/parsers/docling.py` |
| `VORAG_DOCLING_PDF_BACKEND` | `''` | `src/vo_rag/core/parsing/parsers/docling.py` |
| `VORAG_EMBEDDING_CACHE_DIR` | `'/var/lib/vo_rag/cache/embeddings'` | `src/vo_rag/core/embedding/cache.py` |
| `VORAG_EMBEDDING_CACHE_ENABLED` | `'1'` | `src/vo_rag/core/embedding/__init__.py` |
| `VORAG_EMBEDDING_CACHE_SIZE_GB` | `'20'` | `src/vo_rag/core/embedding/cache.py` |
| `VORAG_EMBED_MAX_BATCH` | `32` | `src/vo_rag/workers/inference/server.py` |
| `VORAG_EMBED_URL` | required | `scripts/eval/remote_smoke_ndcg.py` |
| `VORAG_FALLBACK_TEXT_CHUNK_MAX` | `'500'` | `src/vo_rag/pipelines/book_rag/steps/chunk_embed.py` |
| `VORAG_FALLBACK_TEXT_CHUNK_TARGET` | `'350'` | `src/vo_rag/pipelines/book_rag/steps/chunk_embed.py` |
| `VORAG_FIGURES_ROOT` | `'./var/figures'` | `config/settings.yaml` |
| `VORAG_GRAPH_27B_FALLBACK` | `'0'` | `src/vo_rag/cli/build.py` |
| `VORAG_GRAPH_DIRECTION_MIN_COVERAGE` | (computed) | `src/vo_rag/cli/build.py` |
| `VORAG_GRAPH_LOCAL_REORIENT` | `'1'` | `src/vo_rag/cli/build.py` |
| `VORAG_GTE_REVISION` | (computed) | `src/vo_rag/core/reranking/gte.py` |
| `VORAG_GUIDED_DECODING_MODE` | `'json_schema'` | `src/vo_rag/core/llm/guided.py` |
| `VORAG_INFERENCE_MAX_CONCURRENCY` | `2` | `src/vo_rag/workers/inference/server.py` |
| `VORAG_INGEST_GENERATIONS` | `'0'` | `src/vo_rag/ops/publication_probe.py` |
| `VORAG_JSON_LOGS` | `'1'` | `src/vo_rag/workers/embedding/server.py` |
| `VORAG_LLAMAINDEX_OTEL_EXPORT` | `''` | `src/vo_rag/integrations/llamaindex/observability.py` |
| `VORAG_LOG_LEVEL` | `'INFO'` | `src/vo_rag/workers/embedding/server.py` |
| `VORAG_MAX_CHUNKS_PER_PAGE` | (computed) | `src/vo_rag/pipelines/book_rag/validator.py` |
| `VORAG_MCP_API_KEY` | `''` | `src/vo_rag/mcp/server.py` |
| `VORAG_MIN_TEXT_CHUNK_WORDS` | `'20'` | `src/vo_rag/core/chunking/section_packer.py` |
| `VORAG_MOBI_CONVERTER` | `'ebook-convert'` | `src/vo_rag/core/parsing/parsers/epub.py` |
| `VORAG_MOBI_CONVERT_TIMEOUT_S` | `'900'` | `src/vo_rag/core/parsing/parsers/epub.py` |
| `VORAG_OCR_FARM_STATE` | unset | `src/vo_rag/cli/corpus_ocr.py` |
| `VORAG_OCR_GPU_MEM_FRACTION` | `''` | `src/vo_rag/ocr_farm/worker/ocr_worker.py` |
| `VORAG_OPERATIONAL_SAMPLE_INTERVAL_S` | `''` | `src/vo_rag/api/operational_sampler.py` |
| `VORAG_OTEL_DEPLOYMENT_ENVIRONMENT` | `'dev'` | `config/settings.yaml` |
| `VORAG_OTEL_ENABLED` | `'false'` | `config/settings.yaml` |
| `VORAG_OTEL_ENDPOINT` | `'http://127.0.0.1:4318'` | `config/settings.yaml` |
| `VORAG_OTEL_HOST_NAME` | `''` | `config/settings.yaml` |
| `VORAG_OTEL_ROLE` | `'rag'` | `config/settings.yaml` |
| `VORAG_OTEL_SERVICE_NAME` | `'vo-rag'` | `config/settings.yaml` |
| `VORAG_OTEL_SERVICE_NAMESPACE` | `'vo-rag'` | `config/settings.yaml` |
| `VORAG_PORTAL_ALLOWED_EMAILS` | `'[]'` | `config/settings.yaml` |
| `VORAG_PORTAL_CSRF_SECRET` | `''` | `config/settings.yaml` |
| `VORAG_PORTAL_FALLBACK_INTERNAL_URL` | `'http://127.0.0.1:3080'` | `config/settings.yaml` |
| `VORAG_PORTAL_PUBLIC_URL` | `'http://localhost:3080'` | `config/settings.yaml` |
| `VORAG_PORTAL_UPLOAD_CHUNK_BYTES` | `'8388608'` | `config/settings.yaml` |
| `VORAG_PORTAL_UPLOAD_MAX_BYTES` | `'2147483648'` | `config/settings.yaml` |
| `VORAG_PORTAL_UPLOAD_STAGING_PATH` | `'./var/portal-uploads'` | `config/settings.yaml` |
| `VORAG_PORTAL_UPLOAD_STALE_HOURS` | `'24'` | `config/settings.yaml` |
| `VORAG_POSTGRES_URL` | `'postgresql://localhost:5434/vorag'` | `config/settings.yaml` |
| `VORAG_QDRANT_MAX_COLBERT_VECTORS` | `'192'` | `src/vo_rag/core/embedding/point_vectors.py` |
| `VORAG_RAW_COLLECTION` | unset | `src/vo_rag/pipelines/book_rag/runner.py` |
| `VORAG_REDACTION_LEVEL` | `'standard'` | `src/vo_rag/core/observability/redaction.py` |
| `VORAG_RERANKER_REVISION` | (computed) | `src/vo_rag/core/reranking/gpu.py` |
| `VORAG_RERANK_MAX_DEPTH` | `100` | `src/vo_rag/workers/inference/server.py` |
| `VORAG_RERANK_URL` | required | `scripts/eval/remote_smoke_ndcg.py` |
| `VORAG_RUNAWAY_MIN_MEAN_TOKENS` | (computed) | `src/vo_rag/pipelines/book_rag/validator.py` |
| `VORAG_SETTINGS_PATH` | `'config/settings.yaml'` | `src/vo_rag/core/config/settings.py` |
| `VORAG_SOURCE_PUBLICATION_ENABLED` | `'true'` | `src/vo_rag/core/storage/client_factory.py` |
| `VORAG_SOURCE_PUBLICATION_WRITES_ENABLED` | `'false'` | `src/vo_rag/core/storage/client_factory.py` |
| `VORAG_TF_TAGGER_URL` | `''` | `scripts/tf_classify_lane.py` |
| `VORAG_VCE_BASELINE_ROWS_PATH` | `'./examples/vce-baseline.json'` | `src/vo_rag/pipelines/vce/baseline.py` |
| `VORAG_VCE_SPEACHES_URL` | unset | `src/vo_rag/pipelines/vce/transcription.py` |
| `VORAG_VCE_STUB_FIXTURE_DIR` | `'tests/pipelines/vce/fixtures/transcripts'` | `src/vo_rag/cli/vce_impl.py` |
| `VORAG_VCE_TARGET_COLLECTION` | `'vorag_vce_v1'` | `src/vo_rag/pipelines/vce/workflow.py` |
| `VORAG_VCE_TRANSCRIPTION_API_KEY` | unset | `src/vo_rag/pipelines/vce/transcription.py` |
| `VORAG_VCE_TRANSCRIPTION_ENDPOINT_URL` | unset | `src/vo_rag/pipelines/vce/transcription.py` |
| `VORAG_VCE_TRANSCRIPTION_LANGUAGE` | unset | `src/vo_rag/pipelines/vce/transcription.py` |
| `VORAG_VCE_TRANSCRIPTION_MODEL` | `'tiny'` | `src/vo_rag/pipelines/vce/transcription.py` |
| `VORAG_VLM_ADMISSION_WAIT_S` | `120.0` | `src/vo_rag/workers/inference/server.py` |
| `VORAG_VLM_HEALTH_INTERVAL_S` | `15.0` | `src/vo_rag/workers/inference/server.py` |
| `VORAG_VLM_MAX_IMAGE_BYTES` | (computed) | `src/vo_rag/workers/inference/server.py` |
| `VORAG_VLM_MAX_SCHEMA_CHARS` | `65536` | `src/vo_rag/workers/inference/server.py` |
| `VORAG_VLM_MAX_TEXT_CHARS` | `20000` | `src/vo_rag/workers/inference/server.py` |
| `VORAG_VLM_MODEL` | `'Qwen3VL-8B-Instruct-Q8_0.gguf'` | `src/vo_rag/workers/inference/server.py` |
| `VORAG_VLM_QUIET_PERIOD_S` | `2.0` | `src/vo_rag/workers/inference/server.py` |
| `VORAG_VLM_UPSTREAM_TIMEOUT_S` | `300.0` | `src/vo_rag/workers/inference/server.py` |
| `VORAG_VLM_UPSTREAM_URL` | `'http://127.0.0.1:8084/v1'` | `src/vo_rag/workers/inference/server.py` |
| `VO_RAG_ALLOW_CORPUS_SNAPSHOT_MISMATCH` | unset | `src/vo_rag/core/eval/gate.py` |
| `VO_RAG_ALLOW_LEGACY_BASELINE` | unset | `src/vo_rag/core/eval/gate.py` |
| `VO_RAG_BOOK_RAG_DEGENERATE_MIN_CHUNKS` | (computed) | `src/vo_rag/pipelines/book_rag/validator.py` |
| `VO_RAG_BOOK_RAG_DEGENERATE_MIN_CHUNKS_PER_PAGE` | (computed) | `src/vo_rag/pipelines/book_rag/validator.py` |
| `VO_RAG_BOOK_RAG_DEGENERATE_MIN_PAGES` | (computed) | `src/vo_rag/pipelines/book_rag/validator.py` |
| `VO_RAG_BOOK_RAG_DEGENERATE_MIN_TOKENS_PER_PAGE` | (computed) | `src/vo_rag/pipelines/book_rag/validator.py` |
| `VO_RAG_BOOK_RAG_METADATA_GUARD` | `'fail'` | `src/vo_rag/pipelines/book_rag/validator.py` |
| `VO_RAG_BOOK_RAG_METADATA_REQUIRED` | unset | `src/vo_rag/pipelines/book_rag/validator.py` |
| `VO_RAG_BOOK_RAG_MIN_BOOK_PAGES` | (computed) | `src/vo_rag/pipelines/book_rag/validator.py` |
| `VO_RAG_EVAL_COLLECTION` | unset | `src/vo_rag/core/eval/registered.py` |
| `VO_RAG_EVAL_MODE` | `'replay_smoke'` | `scripts/run_eval_gate.py` |
| `VO_RAG_EVAL_RELEVANCE_MODE` | `'chunk_id'` | `src/vo_rag/core/eval/registered.py` |
| `VO_RAG_GARAGE_BUCKET` | unset | `src/vo_rag/pipelines/book_rag/visual_sidecar.py` |
| `VO_RAG_LIVE_CORPUS_SNAPSHOT` | unset | `src/vo_rag/core/eval/gate.py` |
| `VO_RAG_MIN_VLM_DESCRIBE_YIELD` | `0.9` | `src/vo_rag/pipelines/book_rag/visual_sidecar.py` |
| `VO_RAG_VLM_DESCRIBE_CONCURRENCY` | unset | `src/vo_rag/pipelines/book_rag/visual_sidecar.py` |
| `VO_RAG_VLM_MAX_TOKENS` | (computed) | `src/vo_rag/pipelines/book_rag/visual_sidecar.py` |
| `VO_RAG_VLM_PER_IMAGE_DEADLINE_S` | (computed) | `src/vo_rag/pipelines/book_rag/visual_sidecar.py` |
<!-- END GENERATED: env-vars -->

## Configuration keys

Defaults shown are the model defaults; `config/settings.yaml` can override them.

<!-- BEGIN GENERATED: config-keys -->
| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `api.host` | `str` | `'127.0.0.1'` | Not read: the bind address is the --host passed to uvicorn. |
| `api.port` | `int` | `8000` | - |
| `api.api_key` | `str` | required | - |
| `api.api_keys_json` | `str` | `''` | - |
| `api.cors_origins` | `list[str]` | `[]` | - |
| `api.request_timeout_s` | `int` | `60` | - |
| `portal.public_url` | `str` | `'https://localhost:3080'` | - |
| `portal.fallback_internal_url` | `str` | `'http://127.0.0.1:3080'` | - |
| `portal.access_team_domain` | `str` | `''` | - |
| `portal.access_aud` | `str` | `''` | - |
| `portal.allowed_emails` | `list[str]` | `[]` | - |
| `portal.csrf_secret` | `str` | `''` | - |
| `portal.upload_staging_path` | `str` | `'./var/portal-uploads'` | - |
| `portal.upload_chunk_bytes` | `int` | `8388608` | - |
| `portal.upload_max_bytes` | `int` | `2147483648` | - |
| `portal.upload_stale_hours` | `int` | `24` | - |
| `database.url` | `str` | required | - |
| `database.migrations_dir` | `str` | `'src/vo_rag/core/db/migrations'` | - |
| `database.pool_min` | `int` | `2` | - |
| `database.pool_max` | `int` | `10` | - |
| `qdrant.url` | `str` | required | - |
| `qdrant.prefer_grpc` | `bool` | `True` | - |
| `qdrant.api_key` | `str` | `''` | - |
| `garage.endpoint` | `str` | required | - |
| `garage.access_key` | `str` | `''` | - |
| `garage.secret_key` | `str` | `''` | - |
| `garage.region` | `str` | `'garage'` | - |
| `garage.buckets.artifacts` | `str` | `'vorag-artifacts'` | - |
| `garage.buckets.models` | `str` | `'vorag-models'` | - |
| `garage.buckets.backups` | `str` | `'vorag-backups'` | - |
| `figures.root` | `str` | `'/data/vo-rag/figures'` | - |
| `embedding.batch_url` | `str` | required | - |
| `embedding.query_url` | `str` | required | - |
| `embedding.query_serverless_url` | `str` | `''` | - |
| `embedding.batch_serverless_url` | `str` | `''` | - |
| `embedding.query_api_key` | `str` | `''` | - |
| `embedding.batch_api_key` | `str` | `''` | - |
| `embedding.timeout_s` | `int` | `60` | - |
| `chunking.strategy` | `str` | `'hierarchical'` | hierarchical \| semantic |
| `chunking.semantic_breakpoint_percentile` | `float` | `95.0` | - |
| `chunking.semantic_buffer_size` | `int` | `1` | - |
| `vlm.enabled` | `bool` | `False` | - |
| `vlm.url` | `str` | `'http://localhost:8501/vlm'` | - |
| `vlm.model` | `str` | `'Qwen3VL-8B-Instruct-Q8_0.gguf'` | - |
| `vlm.describe_concurrency` | `int` | `1` | - |
| `vlm.max_tokens` | `int` | `1024` | - |
| `vlm.per_image_deadline_s` | `float` | `300.0` | - |
| `vlm.confidence_threshold` | `float` | `0.65` | - |
| `vlm.primary_max_pixels` | `int` | `1048576` | - |
| `vlm.retry_max_pixels` | `int` | `1572864` | - |
| `serverless.api_key` | `str` | required | RUNPOD_API_KEY |
| `serverless.vlm_endpoint_id` | `str` | required | - |
| `serverless.vlm_model_name` | `str` | `'qwen3-vl-32b'` | - |
| `serverless.text_endpoint_id` | `str` | required | - |
| `serverless.text_model_name` | `str` | `'qwen3-6-27b'` | - |
| `serverless.dsi_endpoint_id` | `str \| None` | `None` | - |
| `serverless.dsi_model_name` | `str` | `'qwen3-6-35b-a3b'` | - |
| `serverless.exp_endpoint_id` | `str \| None` | `None` | - |
| `serverless.exp_model_name` | `str` | `'qwen3-6-35b-a3b'` | - |
| `serverless.base_url_template` | `str` | `'https://api.runpod.ai/v2/{endpoint_id}/openai/v1'` | - |
| `serverless.base_url_override` | `str` | `''` | - |
| `serverless.request_timeout_s` | `int` | `600` | - |
| `reranker.model_path` | `str` | `'./models/gte-reranker-onnx-int8'` | - |
| `reranker.intra_threads` | `int` | `16` | - |
| `reranker.batch_size` | `int` | `32` | - |
| `reranker.depth_low` | `int` | `20` | - |
| `reranker.depth_high` | `int` | `100` | - |
| `reranker.auto_merge_enabled` | `bool` | `False` | - |
| `reranker.auto_merge_threshold` | `float` | `0.5` | - |
| `reranker.url` | `str` | `''` | Remote-service fields. ``url`` set ⇒ the API calls a remote /rerank (RerankerClient); empty ⇒ in-process. ``backend=auto`` resolves to ``client`` when a url is set, else ``onnx_cpu``. ``gpu`` is used by the inference worker itself (it hosts GpuReranker on CUDA). |
| `reranker.backend` | `str` | `'auto'` | auto \| client \| gpu \| onnx_cpu |
| `reranker.model_id` | `str` | `'Alibaba-NLP/gte-reranker-modernbert-base'` | - |
| `reranker.timeout_s` | `int` | `30` | - |
| `triage.local_model_url` | `str` | required | - |
| `triage.confidence_threshold` | `float` | `0.7` | - |
| `triage.anthropic_api_key` | `str` | `''` | - |
| `triage.fallback_model` | `str` | `'claude-haiku-4-5-20251001'` | - |
| `observability.langfuse_host` | `str` | required | - |
| `observability.langfuse_public_key` | `str` | `''` | - |
| `observability.langfuse_secret_key` | `str` | `''` | - |
| `observability.log_level` | `str` | `'INFO'` | - |
| `observability.json_logs` | `bool` | `True` | - |
| `otel.enabled` | `bool` | `False` | - |
| `otel.endpoint` | `str` | `'http://127.0.0.1:4318'` | - |
| `otel.service_name` | `str` | `'vo-rag'` | - |
| `otel.service_namespace` | `str` | `'vo-rag'` | - |
| `otel.deployment_environment` | `str` | `'dev'` | - |
| `otel.role` | `str` | `'rag'` | - |
| `otel.host_name` | `str` | `''` | - |
| `compliance.mode` | `str` | `'monitoring'` | monitoring \| enforcement |
| `compliance.similarity_flag` | `float` | `0.8` | - |
| `compliance.similarity_reject` | `float` | `0.9` | - |
| `compliance.min_sources` | `int` | `3` | - |
| `compliance.max_single_source_contribution` | `float` | `0.4` | - |
| `compliance.min_page_distance` | `int` | `5` | - |
| `compliance.max_tokens_per_source` | `int` | `2000` | - |
| `compliance.max_verbatim_tokens` | `int` | `100` | - |
| `compliance.max_total_verbatim_pct` | `float` | `0.15` | - |
| `answer.enabled` | `bool` | `False` | - |
| `answer.provider` | `str` | `'anthropic'` | - |
| `answer.anthropic_api_key` | `str` | `''` | - |
| `answer.quality_model` | `str` | `'claude-sonnet-4-6'` | - |
| `answer.fast_model` | `str` | `'claude-haiku-4-5-20251001'` | - |
| `answer.audit_model` | `str` | `'claude-opus-4-7'` | - |
| `answer.kimi_model` | `str` | `'kimi-k2.6'` | - |
| `answer.kimi_api_key` | `str` | `''` | - |
| `answer.kimi_base_url` | `str` | `'https://api.moonshot.ai/v1'` | - |
| `answer.allow_hosted` | `bool` | `False` | - |
| `answer.allow_kimi` | `bool` | `False` | - |
| `answer.allow_local` | `bool` | `False` | - |
| `answer.allow_raw_hosted_egress` | `bool` | `False` | - |
| `answer.local_base_url` | `str` | `''` | On-prem llama.cpp daily-driver for the LOCAL answer profile (no external egress, so gated on allow_local not allow_hosted). base_url is the server root (the LlamaCppProvider appends /v1/chat/completions), e.g. http://127.0.0.1:8080; model is the served /v1/models id. |
| `answer.local_model` | `str` | `''` | - |
| `answer.max_context_chunks` | `int` | `8` | - |
| `answer.max_output_tokens` | `int` | `1200` | - |
| `answer.timeout_s` | `int` | `90` | - |
| `cloud.monthly_budget_usd` | `float` | `50.0` | - |
| `cloud.per_job_max_usd` | `float` | `10.0` | - |
| `cloud.tailscale.oauth_client_id` | `str` | `''` | - |
| `cloud.tailscale.oauth_client_secret` | `str` | `''` | - |
| `cloud.tailscale.tailnet` | `str` | `''` | - |
| `cloud.r2.endpoint` | `str` | `''` | - |
| `cloud.r2.access_key` | `str` | `''` | - |
| `cloud.r2.secret_key` | `str` | `''` | - |
| `cloud.r2.bucket` | `str` | `'vorag-cloud-staging'` | - |
| `cloud.rates` | `dict[str, dict[str, float]]` | `{}` | - |
| `vce.blob.bucket_raw` | `str` | `'vorag-vce-raw'` | - |
| `vce.blob.bucket_derived` | `str` | `'vorag-vce-derived'` | - |
| `vce.blob.retention_days_raw` | `int` | `90` | - |
| `vce.blob.retention_days_derived` | `int` | `365` | - |
| `vce.intake.allowed_platforms` | `list[str]` | `['youtube', 'local']` | - |
| `vce.intake.youtube_url_patterns` | `list[str]` | `['youtube.com', 'youtu.be']` | - |
| `vce.transcription.model` | `str` | `'large-v3'` | - |
| `vce.transcription.diarization` | `bool` | `True` | - |
| `vce.transcription.min_speakers` | `int` | `1` | - |
| `vce.transcription.max_speakers` | `int` | `4` | - |
| `vce.frame_sampling.max_frames_per_video` | `int` | `1200` | - |
| `vce.frame_sampling.scene_threshold` | `float` | `0.3` | - |
| `vce.frame_sampling.audio_cue_regex` | `list[str]` | `['look at', 'see this', 'compare', 'right here', 'notice', 'brake trace', 'throttle', 'apex', 'delta', 'line']` | - |
| `vce.frame_sampling.uniform_fallback_fps` | `float` | `0.25` | - |
| `vce.frame_sampling.cue_window_pre_s` | `float` | `3.0` | - |
| `vce.frame_sampling.cue_window_post_s` | `float` | `8.0` | - |
| `vce.frame_sampling.cue_window_fps` | `float` | `1.0` | - |
| `vce.frame_sampling.extract_concurrency` | `int` | `4` | - |
| `vce.visual_extract.prompt_name` | `str` | `'vce.visual_extract'` | - |
| `vce.visual_extract.frames_per_call` | `int` | `4` | - |
| `vce.visual_extract.max_calls_per_video` | `int` | `100` | - |
| `vce.visual_extract.workload` | `str` | `'vlm-describe'` | - |
| `vce.visual_extract.candidate_window_pre_s` | `float` | `4.0` | - |
| `vce.visual_extract.candidate_window_post_s` | `float` | `12.0` | - |
| `vce.visual_extract.candidate_min_gap_s` | `float` | `4.0` | - |
| `vce.ocr.enabled` | `bool` | `True` | - |
| `vce.ocr.fps` | `float` | `1.0` | - |
| `vce.ocr.engine` | `str` | `'glm_ocr_sglang'` | - |
| `vce.ocr.glm_model_server_url` | `str` | `'http://localhost:8180'` | - |
| `vce.ocr.glm_model` | `str` | `'glm-ocr'` | - |
| `vce.ocr.glm_max_image_px` | `int` | `1280` | - |
| `vce.ocr.glm_timeout_s` | `float` | `180.0` | - |
| `vce.ocr.frame_source` | `str` | `'overlay'` | - |
| `vce.ocr.max_frames_per_video` | `int` | `1200` | - |
| `vce.ocr.min_frame_gap_s` | `float` | `0.0` | - |
| `vce.segmentation.min_segment_seconds` | `int` | `15` | - |
| `vce.segmentation.max_segment_seconds` | `int` | `90` | - |
| `vce.segmentation.rerank_workload` | `str` | `'vlm-sterilize'` | - |
| `vce.structured_extract.extract_workload` | `str` | `'vlm-sterilize'` | - |
| `vce.structured_extract.max_tokens` | `int` | `4096` | - |
| `vce.structured_extract.max_records_per_segment` | `int` | `4` | - |
| `vce.validation.judge_workload` | `str` | `'vlm-sterilize'` | - |
| `vce.validation.judge_confidence_threshold` | `float` | `0.7` | - |
| `vce.enrichment.embedding_collection_alias` | `str` | `'vorag_vce_v1'` | - |
| `vce.enrichment.signature_collection_alias` | `str` | `'vorag_vce_signature_v1'` | - |
| `vce.enrichment.signature_version` | `int` | `1` | - |
| `vce.retrieval.default_k` | `int` | `10` | - |
| `vce.retrieval.rrf_constant` | `int` | `60` | - |
| `vce.cost_ceiling_usd_per_video` | `float` | `10.0` | - |
<!-- END GENERATED: config-keys -->
