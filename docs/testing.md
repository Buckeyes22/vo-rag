# Testing and reproducibility

[README](../README.md) · [Engineering limitations](engineering.md) · [Evaluation](evaluation.md) · [Operations](operations.md)

The test strategy has one rule: the default run needs nothing but Python. Every service the code talks to (Postgres, Qdrant, the object store, the embedding and rerank workers, model providers) is injected, and the suite supplies fakes. A small, separate lane runs the same contracts against real, disposable Postgres and Qdrant. This page explains the lanes, the markers, the numbers you can reproduce, and the checks `scripts/ci.sh` runs.

## The lanes

| Lane | Command | Needs | What it proves |
| --- | --- | --- | --- |
| Hermetic backend | `make test` | Python and the locked dev dependencies | API, orchestration, retrieval, state transitions and security contracts against fakes |
| Lint and types | `make lint`, `make typecheck` | Same | `ruff` rules and `mypy --strict` over `src` and `tests` |
| Evaluation wiring | `make eval-demo` | Same | The evaluation gate runs end to end on a synthetic fixture (see [evaluation](evaluation.md)) |
| Integration | `scripts/testinfra-ci.sh`, or `pytest -m integration` against your own services | Docker, or a disposable Postgres and Qdrant | Migrations, Qdrant collection behavior, the dispatcher's claim SQL, publication revisions, corpus maintenance queries |
| Live | `VORAG_RUN_LIVE=1 pytest -m live` | A RunPod key and endpoint | One test of the serverless transport against a real endpoint |
| Portal | `pnpm --filter @vo-rag/portal test` | Node 22 and pnpm | Components, the API client, the stream hook and the nginx template, with the network mocked |

```sh
uv sync --locked --extra dev --extra llamaindex
make test
make lint
make eval-demo
```

`make test` runs `pytest -m "not integration and not slow and not live"`. Skips stay visible in the summary (`-ra` is in the pytest options), so a missing optional dependency shows up as a skip with its reason, never as a silent pass. A passing hermetic suite is not proof that your database, GPU or provider services work.

## What the numbers are

The hermetic selection holds about 5,000 tests in about 420 test files, from a total of about 5,200 collected. You can reproduce the counts instead of trusting this page:

```sh
uv run --extra dev --extra llamaindex pytest --collect-only -q -m "not integration and not slow and not live" | tail -1
uv run --extra dev --extra llamaindex pytest --collect-only -q | tail -1
```

The same numbers feed the checker that keeps the docs honest: `scripts/gen_reference_docs.py` writes them to [`docs/_measured.json`](_measured.json), and `scripts/check_docs.py` fails if a count in the prose disagrees. This is why the prose says "about" for counts that change with every commit.

| Marker | Meaning | In `make test` | Tests today |
| --- | --- | --- | --- |
| `integration` | Needs live Postgres or Qdrant (the testinfra stack) | No | about 120 |
| `slow` | Long-running | No | none |
| `live` | Calls a real external service | No | 1 |
| `security` | Abuse and non-disclosure guards: auth, SSRF, injection | Yes | about 160 |
| `property` | Hypothesis property tests | Yes | none |
| `release` | Release-gate checks, run only with `-m release` | Yes, but none exist | none |
| `perf` | `pytest-benchmark` micro-benchmarks | Yes, but none exist | none |

The markers are declared in `pyproject.toml` and `--strict-markers` is on, so a typo in a marker is an error. Four of the seven are declared without any test using them yet. Tests run in random order (`pytest-randomly`); the seed is printed in the header, and `--randomly-seed=<seed>` replays an order.

## Where the tests are

The tree under [`tests`](../tests) mirrors the source tree, so the tests for `core/parsing` are in `tests/core/parsing`. The weight follows the risk:

| Area | Tests (about) | Covers |
| --- | --- | --- |
| `core/parsing`, `core/chunking` | 1,000 | Parser routing, preflight, the quality gates, EPUB repairs, heading and contents filters, the hierarchical chunker |
| `pipelines/book_rag` | 490 | The dispatcher's claim and heartbeat logic, the workflow, validator, smoke gate, visual sidecar |
| `integration/llamaindex` | 480 | The adapter tier and the ingest workflow port (needs the `llamaindex` extra; the subtree drops itself without it) |
| `core/corpus` | 290 | Health, purge, duplicates, canary gates, rechunk preview, tag store |
| `core/retrieval`, `core/storage` | 390 | Fusion, query modes, Qdrant behavior, publication revisions |
| `core/graph`, `core/summary`, `core/raptor`, `core/dsi` | 430 | The advanced layers |
| `core/ingest`, `core/llm`, `core/observability`, `core/embedding` | 360 | Intake normalization, the LLM seam and retries, metrics, the embedding client |
| `ocr_farm`, `workers`, `cli`, `api`, `mcp`, `security`, `scripts` | the rest | The OCR farm, workers, each command surface, URL safety, and the doc tooling itself |

Test data is synthetic. Nothing under `tests` contains a copyrighted book or a real person's text; fixtures such as the two-document evaluation corpus in [`tests/fixtures/eval/book_rag`](../tests/fixtures/eval/book_rag) are invented. Fakes for the service seams live in [`tests/fakes`](../tests/fakes). A top-level `conftest.py` redirects the CLI's config lookup to a temporary directory, so a test run can never read your real `~/.config/vo-rag/config.yaml`.

## Evaluation boundaries

The two-document public fixture under `tests/fixtures/eval/book_rag` is synthetic. Chunk and passage relevance files exercise the evaluator, not generalization. Its zero baseline floors mean it does not gate quality.

For retrieval-quality evaluation, create an independent corpus you have rights to use, queries and judgments, bind the baseline to the actual corpus and qrels hashes, and use the real retriever. Replay scores must not become the evidence for a retrieval-quality claim.

## The integration lane

[`docker-compose.testinfra.yml`](../docker-compose.testinfra.yml) defines two disposable services bound to loopback: Postgres 16 and Qdrant 1.17, both on `tmpfs` so nothing survives `down`. Their credentials (`vorag` / `vorag` for Postgres, none for Qdrant) are public and must never be reused anywhere else. Integration tests read their targets from the environment and skip, with a reason, when it is unset:

| Variable | Used for |
| --- | --- |
| `VORAG_TEST_DATABASE_URL` | The disposable Postgres, for example `postgresql://vorag:vorag@127.0.0.1:5444/vorag_test` |
| `VORAG_TEST_QDRANT_URL` | The disposable Qdrant, for example `http://127.0.0.1:6334` |
| `VORAG_TEST_QDRANT_API_KEY` | A key, only if your Qdrant requires one |
| `VORAG_REQUIRE_TESTINFRA` | Set to `1` to turn a missing URL into a failure instead of a skip |
| `VORAG_TEST_POSTGRES_PORT`, `VORAG_TEST_QDRANT_PORT` | Host ports the compose file publishes (defaults 5444 and 6334) |

```sh
docker compose -f docker-compose.testinfra.yml up -d --wait
VORAG_TEST_DATABASE_URL=postgresql://vorag:vorag@127.0.0.1:5444/vorag_test \
VORAG_TEST_QDRANT_URL=http://127.0.0.1:6334 \
uv run --no-sync pytest -m integration -rs
docker compose -f docker-compose.testinfra.yml down --volumes
```

[`scripts/testinfra-ci.sh`](../scripts/testinfra-ci.sh) is the unattended version of that sequence. It names the compose project after the run (`vo-rag-testinfra-<run id>-<attempt>`) and asks Docker for free ports, so parallel jobs cannot collide with each other or with a developer's stack; it removes leftover stacks of that name older than two hours (`VORAG_TESTINFRA_STALE_S`) because a cancelled job never runs its cleanup; when it runs inside a container it joins the stack's network and uses service names; it runs a fixed list of integration files with `VORAG_REQUIRE_TESTINFRA=1`; and it tears the stack down and prints server logs if anything failed. It does not run the whole `integration` marker, only the files it lists.

Two names that look like inputs are not. `VORAG_TEST_LANE_CLASS` and `VORAG_TEST_EXT` are scratch variables that two dispatcher unit tests set for themselves while checking the parsing of the real settings `VORAG_BOOK_RAG_LANE_CLASS` and `VORAG_BOOK_RAG_DISPATCHER_ONLY_EXTENSIONS`; you never export them.

Real parsing and embedding, hosted answer quality, GPU behavior, the OCR farm against real hosts, and telemetry export have no automated lane here. They need your own services and your own validation before you operate those paths.

## The pipeline script

[`scripts/ci.sh`](../scripts/ci.sh) is the single entry point for a CI system; run it locally to get the same result. It stops at the first failing stage.

```mermaid
flowchart LR
    B["Locked bootstrap: uv sync --locked --extra dev --extra llamaindex"] --> L["ruff check and format check"]
    L --> T["mypy --strict"]
    T --> P["pytest with coverage, hermetic selection"]
    P --> S["bandit over src"]
    S --> K["detect-secrets over tracked files"]
    K --> E["Eval gate, replay_smoke mode"]
    E --> G(["ALL GREEN"])
```

The eval gate stage uses the replay mode on purpose: it checks the evaluation wiring against the synthetic fixture and says nothing about retrieval quality. Coverage has a floor of 60% (`fail_under` in `pyproject.toml`) and leaves out the SQL query helpers, which only run against a live Postgres and are covered by the integration lane. The integration lane is a separate script because it needs Docker. The portal has its own checks.

## The portal

```sh
corepack enable
pnpm install --frozen-lockfile
pnpm --filter @vo-rag/portal typecheck
pnpm --filter @vo-rag/portal test
```

The portal has type checking and about 75 unit tests across 16 test files (Node 22). The tests use mocks; they do not exercise a corpus-backed deployment. What the portal does is described in [portal](portal.md).
