# RunPod vLLM worker surface

This directory ships the **contract** and **local harness** for the three
RunPod Serverless vLLM workloads vo-rag runs:

| Workload | Serverless role | Default model | OpenAI shape |
|---|---|---|---|
| `text-gen` | `text` | `qwen3-6-27b` | text-only chat completions |
| `vlm` | `vlm` | `qwen3-vl-32b` | multimodal (text + `image_url`) chat completions |
| `dsi` | `dsi` | `qwen3-6-35b-a3b` | text-only JSON-mode chat completions |

The Python package under `src/vo_rag/workers/runpod_vllm/` is the
runtime side: Pydantic wire shapes, the workload→role mapping, and a
small FastAPI harness used by tests and local dev. It is **not** the
deployment surface — production endpoints are provisioned from the JSON
templates in `templates/` using `runpodctl serverless create` (see the
*Deploy phase* section below).

## What's in the package

```
src/vo_rag/workers/runpod_vllm/
├── __init__.py     # re-exports RunpodWorkload, ServerlessRole, role_for_workload, ...
├── schemas.py      # pure Pydantic: ChatContentPart, ChatMessage, ServerlessWorkloadRequest/Result
├── contracts.py    # role_for_workload / client_for_workload / build_chat_payload
└── harness.py      # local FastAPI app — POST /openai/v1/chat/completions + GET /health
```

The package never imports `httpx` or any RunPod SDK directly. The
**only** transport seam is `vo_rag.core.cloud.serverless.ServerlessClient`,
which already owns the `(1, 3, 9)` retry loop and the 600s timeout for
`api.runpod.ai` traffic. A static test in
`tests/workers/runpod_vllm/test_contracts.py::test_workers_runpod_vllm_has_no_raw_httpx_or_runpod_sdk`
asserts this no-bypass invariant.

## Local harness

The harness is a tiny OpenAI-compatible ASGI app used by the test suite
and by humans poking at the workload surface from a CLI.

```bash
# Run the harness in-process and curl it
python -m vo_rag.workers.runpod_vllm.harness --port 8188 &
curl -s http://127.0.0.1:8188/health | jq
# {"mode":"local-harness","gpu":false}

curl -s http://127.0.0.1:8188/openai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
```

`fail_first_n=N` makes the harness return HTTP 502 for the first N
requests, then succeed. Tests use this to exercise `ServerlessClient`'s
retry path without real sleeps (they pass `retry_delays=(0, 0, 0)`).

The harness is **not** production. It is a double; it does not run vLLM,
does not call `api.runpod.ai`, and must never be exposed on a public
network.

## Deploy phase — NOT in this repo

The following are **deploy-phase** operations and are intentionally not
committed here. They live in the deploy pipeline and operator runbooks:

- `runpodctl serverless create` / `runpodctl serverless update` for each
  of the three workloads.
- The actual RunPod **endpoint IDs** for staging and production
  (`SERVERLESS_TEXT_ENDPOINT_ID`, `SERVERLESS_VLM_ENDPOINT_ID`,
  `SERVERLESS_DSI_ENDPOINT_ID`) — set in `.env` / Vault, not in git.
- **FlashBoot** (RunPod's cold-start mitigation) toggle, scaling curve
  overrides, and per-environment max-concurrency overrides.
- **Grafana** dashboard wiring and Prometheus scrape config that point at
  the new endpoints.

Templates under `templates/` describe the *shape* of each endpoint
(workload, role, model name, base URL, scaling envelope, health check
command, env vars). The deploy pipeline reads these and emits
`runpodctl` invocations.

### Why `worker-vllm`, not plain `vllm/vllm-openai`

The official `runpod/worker-vllm` hub is what wraps vLLM with RunPod's
`/run` and `/status` queue-handler endpoints. `vllm/vllm-openai` is a
much older base image that only serves the OpenAI HTTP shape — it has
no job-queue handler, no cold-start telemetry, and no RunPod-native
scaling. Every template in `templates/` therefore pins
`runpod/worker-vllm` (or any hub name that resolves to the same
contract) and explicitly notes the choice in `image_notes`.
