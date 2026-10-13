# Engineering decisions and limitations

[Architecture](architecture.md) · [Testing](testing.md) · [Operations](operations.md)

## Decisions visible in the code

| Decision | Benefit | Cost / obligation |
| --- | --- | --- |
| Separate Postgres, Qdrant and object storage | Distinct state, evidence and artifact responsibilities | Cross-store reconciliation and recovery are explicit engineering work |
| Injected surface services and workflow clients | Hermetic tests without GPU, network or live databases | Fakes cannot prove deployed behavior or external compatibility |
| Thin HTTP routes over retrieval helpers | Core retrieval policy is reusable across callers | Each surface still needs its own auth/delivery tests |
| Plain async book orchestration, with no external workflow engine | Typed steps, ordered by what each step needs from the one before, that run against fakes and fail visibly | Workflow ownership, dispatch and durable recovery need operational setup; see [below](#a-plain-async-dag-instead-of-a-workflow-engine) |
| Explicit optional-dependency boundaries | Core development avoids the heaviest runtime packages | Parser/GPU/provider features require extra setup and validation |
| Liveness separate from capability readiness | Avoids accepting a running but unusable process | Monitoring must evaluate the right endpoint and dependency |
| Rights/policy modes and scoped evidence access | Restricts which evidence crosses a given boundary | Policy labels cannot establish ownership or legal permission |
| Duplicate detection groups by title, author and exact year, and never joins editions | Two editions of one book cannot be planned for deletion against each other; an undated copy is kept and surfaced | A copy with no year in a multi-edition title needs a person to resolve it; see [below](#duplicate-detection-never-bridges-editions) |
| A fixed workload-to-location table instead of dynamic routing | Placement is auditable and the worker roster is checked against it when it loads | Moving a workload is a code change, and there is no automatic burst or failover; see [below](#a-static-workload-router) |
| A kill switch that revokes network access and compute but does not rotate storage keys | It stays inside what the platforms still offer, and cloud workers hold only expiring credentials anyway | Rotating long-lived keys after an incident is a manual dashboard task; see [below](#a-kill-switch-without-key-rotation) |
| Unknown page counts never cause a skip | A failed page-count read cannot silently drop a real book | The short-document guard that uses this rule is not called by the pipeline yet |

### A plain async DAG instead of a workflow engine

[`workflow.py`](../src/vo_rag/pipelines/book_rag/workflow.py) sequences the ingest as one async class over a typed context, and each step lives in its own small module (`steps/parse.py`, `steps/chunk_embed.py`, `steps/sterilize.py`). It is deliberately not a Hatchet or similar engine graph. Every service a step uses (object store, Qdrant, embedding client, LLM client, retriever, parser router) is injected, so the whole sequence runs in a test against fakes with no GPU and no live service. Splitting the steps into focused modules keeps each one small enough to read and test alone. A consequence worth stating: the durable state that matters (the envelope row and its heartbeat, the generation lease, the publication revision) lives in Postgres and Qdrant, so there is no second state store inside an engine to reconcile with them. The failure modes that matter are specific to this domain (an ambiguous commit across two stores, a book reclaimed from a dead worker), and the recovery code for them is written in this repository either way.

The step order is chosen for correctness, not for convenience: the source identity is derived first, so the duplicate check can end a rerun before any paid work; parsing comes before embedding because it fails cheaply; sterile points are written after raw chunks because they derive from them one for one; the artifact row is written after the points so the validator can check that it exists; the structural validator runs before the model-based smoke gate because it is cheap; and generation activation is last, so everything it publishes has been checked. The table with the reason for each position is in [ingestion](ingestion.md#the-workflow). The cost is operational: a process has to run the dispatcher, and an interrupted run recovers through the reclaim and lease rules, not through an engine's retry.

### Duplicate detection never bridges editions

`vo-rag corpus duplicates` plans which copies of a title to delete. Normalizing a title deliberately strips its year, so a plan on the normalized title alone would group the 2013 and 2019 editions of a book and mark one for deletion. Connecting books transitively would fail in a subtler way: an undated copy would be connected to both dated editions and so join them. The rule is therefore exact equality with no bridging. When a title-and-author bucket holds at most one distinct year, everything in it is one group. When it holds two or more, each exact year is its own group and the undated copies become a separate keep-only group (`ambiguous_year`), so the delete list can never contain something ambiguous. The behavior, an example and the keep rule are in [operations](operations.md#corpus-maintenance-loops).

### A static workload router

[`core/cloud/router.py`](../src/vo_rag/core/cloud/router.py) is a five-entry table: `text-gen`, `vlm` and `dsi` run on RunPod serverless; `embedding` and `query-handling` run locally. A fixed table is used in place of a run-time choice by burst capacity and data sovereignty. A table can be read in one glance, never sends work somewhere nobody intended because of a capacity guess, and turns drift between the table and the declared workers into an error when the roster loads, not in production. The price is that there is no automatic burst or failover. Three things hold it in place: the worker registry in `pyproject.toml` fails to load if a declared worker disagrees with the table, the admin status route reports the placement, and the agent helper refuses a `homelab_only` envelope a serverless workload. The cost is rigidity; see [integrations](integrations.md#static-workload-router).

### A kill switch without key rotation

The emergency teardown ([`core/cloud/kill_switch.py`](../src/vo_rag/core/cloud/kill_switch.py)) denies the cloud-worker network tag, deletes the tagged devices and terminates the RunPod pods whose names start with `vorag-`. There is no key-rotation step: Cloudflare offers no REST endpoint for rotating permanent R2 access keys. The repository bounds exposure in time instead: cloud workers receive presigned URLs that expire, never the long-lived keys, so there is little for a rotation to protect. The remaining obligation is yours: after an incident, rotate permanent keys by hand in the provider's dashboard. See [operations](operations.md#cloud-cost-governance).

## Where the rest of the rationale lives

The design reasoning sits next to the behavior it explains, so each page can be read alone. This index says where to find it.

| Question | Answer in |
| --- | --- |
| Why a dedicated heartbeat thread, and why lanes for giant books | [ingestion](ingestion.md#heartbeat-and-stale-reclaim), [reliability](reliability.md#heartbeat-thread) |
| Why ambiguous commits are reconciled, and what a lease fences | [reliability](reliability.md#leases-generations-and-activation-reconciliation) |
| Why validation has eight checks and the floors it uses | [ingestion](ingestion.md#the-eight-check-validator) |
| Why zip archives are measured before they are read, and why pandoc is a subprocess | [parsing and chunking](parsing-and-chunking.md#epub-and-mobi), [why pandoc](parsing-and-chunking.md#why-pandoc-is-run-as-a-subprocess) |
| Why OCR-variant running heads need a run of pages | [parsing and chunking](parsing-and-chunking.md#how-elements-are-routed) |
| How the OCR cache label steers the parser, and how the farm limits cost | [parsing and chunking](parsing-and-chunking.md#the-router), [OCR](ocr.md#cost-controls) |
| Why fusion has two dedup passes and only RRF ships | [retrieval and answers](retrieval-and-answers.md) |
| Why the answer call fixes `allowed_domains` as a literal | [retrieval and answers](retrieval-and-answers.md#why-allowed_domains-is-a-literal) |
| Why collection names are a persistence contract | [data model](data-model.md) |
| Why the embedding client retries and caps request size | [reliability](reliability.md#retries-and-backoff) |
| Why status bodies are built from named fields, and why `/health` stayed minimal | [security](security.md#secrets-and-redaction), [probes and exemptions](security.md#probes-and-exemptions) |
| Which attacks the URL gate answers | [security](security.md#url-safety) |
| Why quarantine is the default trust level, and why IDs are deterministic | [security](security.md#trust-levels-and-quarantine-by-default-f-031), [data model](data-model.md) |
| Why contextual retrieval is not built and why the layers have STOP/GO gates | [advanced layers](advanced-layers.md) |
| Why evaluation baselines are bound to their inputs | [evaluation](evaluation.md#baselines-are-bound-to-their-inputs) |
| Why the reranker is a pre-quantized ONNX model and there is no adaptive reranker | [retrieval and answers](retrieval-and-answers.md) |

## Implemented, experimental and planned

**Implemented and contract-tested:** book/document workflow, API/CLI/MCP surfaces, retrieval and evidence handling, parser routing, authentication, URL safety, redaction, readiness, telemetry seams and React portal components.

**Implemented with external-service or experimental boundaries:** GPU parsing/embedding, visual descriptions, advanced summary/graph/DSI paths, generation-scoped ingest, source-publication writes, hosted synthesis, cloud worker providers, RunPod OCR farm (requires the separately installed pitwall GPU-broker CLI), VCE export/transcription. Tests cover many component contracts; their complete deployed workflows need their own service setup.

**Planned in the registry:** code-RAG, research-agent, idea intake, article extraction, structured scraping, URL snapshotting, repository analysis, tool evaluation and VCE are not advertised as active producers by the project registry. Some related adapters or VCE code exist; that does not upgrade their registry status or establish end-to-end support. `vo-rag roster` reports the code's current declaration.

## Limitations

What is not measured or not tested:

- No live production deployment has been exercised; scale, latency and retrieval quality are not measured.
- There is no public benchmark. `make eval-demo` checks evaluation wiring on synthetic data only.
- Service-backed features (GPU parsing and embedding, hosted synthesis, cloud workers, telemetry export) need their own services and configuration, and have no automated lane here.
- Fresh installation across every platform, distributed recovery and live provider behavior have not been tested. There is no tested production backup and restore procedure.
- Dependencies, models and documents you add keep their own licenses.

What exists in code but is not wired, or does not work with the defaults:

- **Sterilization has no runtime switch.** `BookRagConfig.sterilize_enabled` defaults to false and the production runner never sets it; there is no environment variable or CLI flag. Without sterile evidence the `publication` and `partner` modes have nothing to search until you build the pipeline yourself ([rights and compliance](rights-and-compliance.md)).
- **Public answers fail closed.** In `publication` and `partner` mode `POST /api/v1/answer` returns 403 until a compliance gate is supplied, and nothing in the shipped code supplies one.
- **The layer STOP/GO functions are library code.** No `vo-rag build` command runs them, and DSI routing is not on the query path ([advanced layers](advanced-layers.md)).
- **`vo-rag build graph` refuses to run under the default settings**, because source publication readers are on by default; set `VORAG_SOURCE_PUBLICATION_ENABLED=false` for that command.
- **Parent auto-merge is unwired on HTTP.** The query route passes no parent fetch function, so the merge never happens there even with the setting on.
- **Twelve metrics have no emitter.** They appear on `/metrics` and stay at zero ([observability](observability.md#defined-with-no-emitter-in-this-tree)).
- **The kill switch has no caller.** Nothing triggers `CloudKillSwitch.activate`, and nothing writes the `cloud_kill_log` table that the exporter counts.
- **`vo-rag corpus purge --yes` refuses while publication readers are enabled**, which is the default; the dry run works ([operations](operations.md#corpus-maintenance-loops)).
- **Only file-drop envelopes are ingested.** URL, text and idea envelopes are stored and wait for pipelines that are not implemented, so most portal ingest tabs create envelopes that stay `received`.
- **Source repair has producers for two targets only** (the books-raw and Reddit collections); typed and derived collections are refused.
- **The short-document guard `non_book_skip_by_pages` is defined and tested but called by nothing**, and the video pipeline's frame sampling and visual extraction are placeholders and its `idempotency` check is unimplemented ([integrations](integrations.md#video-extraction-vce)).
- **Declared configuration is partly inert.** Most `vce` keys and about three dozen others in `settings.yaml` are validated but read by nothing ([configuration](configuration.md#declared-but-not-read)), including `api.cors_origins` and the `compliance` thresholds.
- **The API sets no CORS headers, applies no rate limit and installs no exception handlers**, so a failed dependency surfaces as a bare 500 ([security](security.md), [reliability](reliability.md#partial-failure-table)). Put those controls at your proxy.
- **The portal queries in `internal` mode only**, and its fleet panel and console links are static examples ([portal](portal.md)).
- **One pipeline is active.** `book_rag` is the only producer the roster marks active.
