# workers

Five worker surfaces are declared as `active` and covered by contract tests.

The worker roster is the single source of truth — declared once in
`pyproject.toml` under `[tool.vo_rag].workers` and loaded at runtime
by `vo_rag.workers.registry.load_worker_registry()`. Every
non-`misc` workload is validated against the static
workload→location map in `vo_rag.core.cloud.router` at load time,
so a registry/placement drift is a load-time error rather than a
runtime one.

## Roster

| Worker | Workload | Location | Entrypoint | GPU | Test level | Status | Deploy artifact(s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `embedding` | `embedding` | `local` | `vo_rag.workers.embedding.server:app` | yes | `unit+gpu-integration` | `active` | `workers/embedding/Dockerfile`, `workers/embedding/Dockerfile.local` |
| `runpod_text_gen` | `text-gen` | `runpod_serverless` | `worker-vllm hub` | yes | `contract+live` | `active` | `workers/runpod_vllm/templates/text-gen.json` |
| `runpod_vlm` | `vlm` | `runpod_serverless` | `worker-vllm hub` | yes | `contract+live` | `active` | `workers/runpod_vllm/templates/vlm.json` |
| `runpod_dsi` | `dsi` | `runpod_serverless` | `worker-vllm hub` | yes | `contract+live` | `active` | `workers/runpod_vllm/templates/dsi.json` |
| `cloud_cost_exporter` | `misc` | `local` | `services.cloud_cost_exporter.app:app` | no | `unit+db-integration` | `active` | `services/cloud_cost_exporter/Dockerfile` |

**Planned / deprecated entries:** none. The roster is intentionally
narrow — five workers, one per workload plus the accounting
exporter. New workloads go through `vo_rag.core.cloud.router`
(add the workload + location) and then the registry
(add the `WorkerSpec`). Drift between the two is rejected at
registry-load time.

## Sub-tree index

- `workers/embedding/` — local BGE-M3 dense + sparse + ColBERT
  server. Source package: `vo_rag.workers.embedding`. HTTP contract:
  `vo_rag.core.embedding.schemas.EmbedRequest` / `EmbedResponse`.
  Clients live in `vo_rag.core.embedding.client`. See
  `workers/embedding/README.md` for endpoints, env vars, and
  local-run commands. The two Dockerfiles are referenced by the
  registry row above; `Dockerfile` bakes the model, `Dockerfile.local`
  defers it to a host mount.
- `workers/runpod_vllm/` — RunPod Serverless vLLM contract layer
  and local harness. Source package: `vo_rag.workers.runpod_vllm`
  with three sub-modules: `schemas` (pure Pydantic wire shapes),
  `contracts` (workload→role mapping + `build_chat_payload`),
  `harness` (a small FastAPI app used by tests + local dev).
  Templates under `workers/runpod_vllm/templates/{text-gen,vlm,dsi}.json`
  describe the *shape* of each serverless endpoint; the
  `runpodctl serverless create|update` invocations that turn
  those templates into live endpoints are deploy-phase and
  intentionally not committed. See `workers/runpod_vllm/README.md`
  for the workload table, harness usage, the "no `httpx` / no
  RunPod SDK in this package" invariant, and the deploy-phase
  scope note.

## Reuse

The HTTP API, CLI and MCP server call the same `vo_rag.core.cloud.*` seams
that the registry-declared workers expose. See the contract tests in
`tests/integration/test_compute_contracts.py`.
