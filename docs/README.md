# Documentation

Start with the [root README](../README.md), then [capabilities](capabilities.md) for the full feature list and [architecture](architecture.md) for how the pieces fit.

## Understand the system

| Page | Covers |
| --- | --- |
| [Capabilities](capabilities.md) | Every capability, its entry point and the page that explains it |
| [Architecture](architecture.md) | Processes, package map, dependency rules, the LLM seam, the settings loader |
| [Data model](data-model.md) | Postgres tables, Qdrant collections, object-store keys, caches, deterministic IDs |
| [Data and API flows](data-and-api.md) | Ingestion, query and answer paths with a request example |
| [Engineering decisions](engineering.md) | Design decisions and their costs, an index of where the rest of the rationale lives, and current limitations |

## Ingest and retrieve

| Page | Covers |
| --- | --- |
| [Ingestion](ingestion.md) | Intake surfaces, dispatcher, workflow steps, generations, validator, smoke gate |
| [Parsing and chunking](parsing-and-chunking.md) | Parser routing, preflight, quality gates, hierarchical chunks |
| [OCR](ocr.md) | Docling OCR, the OCR farm, cache keys and cost controls |
| [Retrieval and answers](retrieval-and-answers.md) | Vector lanes, fusion, reranking, answer profiles and citations |
| [Rights and compliance](rights-and-compliance.md) | Sterile evidence, license classes, query modes, the compliance gate |
| [Advanced layers](advanced-layers.md) | Summary tree, RAPTOR, DSI and entity graph |
| [Integrations](integrations.md) | LlamaIndex adapters, the video extraction pipeline, example domain packs, the static workload router |

## Run and operate

| Page | Covers |
| --- | --- |
| [Setup](setup.md) | Developer environment, configuration contract, the inference worker, service-backed API, portal |
| [Configuration](configuration.md) | How settings load, what each section controls, keys that nothing reads, and the generated variable and key tables |
| [Operations](operations.md) | Process start commands and topology, readiness, corpus maintenance loops, source repair, migrations, cost governance and the kill switch, `brain_sync`, troubleshooting |
| [Reliability](reliability.md) | Idempotency, leases, publication revisions, retries |
| [Observability](observability.md) | Metrics, sampler, tracing, logs, audit tables |
| [Security](security.md) | Trust boundaries, authentication, data handling |
| [Evaluation](evaluation.md) | Evaluation fixtures, gates and the smoke gate |
| [Testing](testing.md) | Test lanes, markers, reproducible counts, the integration variables and the CI script |

## Interfaces

| Page | Covers |
| --- | --- |
| [HTTP API](api.md) | Route reference and two worked examples |
| [CLI](cli.md) | Command reference, how the CLI finds the API, and worked examples |
| [MCP server](mcp.md) | How to start the server, tool reference, transports and worked examples |
| [Portal](portal.md) | Pages, how the portal talks to the API, the sign-in and CSRF flow, running and deploying it |

Redistributed assets: the SCOWL dictionary and copyright notice reside together under `src/vo_rag/pipelines/book_rag/data/`. No model weights or document corpora ship here. Review licenses of separately installed dependencies, model weights and any documents you add before redistribution.
