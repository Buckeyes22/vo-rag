# Command-line reference

The `vo-rag` command is the operator interface to the pipeline. It is installed as a console script that points at `vo_rag.cli.main:cli`, and every command below is a leaf of that Click tree. Run `vo-rag <command> --help` for the same information on a terminal.

The table is generated from the command tree by `scripts/gen_reference_docs.py`. Do not edit between the markers; change the command and rerun the generator.

## Connecting

Two kinds of command exist. `vo-rag query`, `answer`, `status`, `health`, `admin status` and the `ingest file`, `file-drop`, `text`, `url` and `idea` commands call the HTTP API. Everything else works on its own: `preflight`, `pipeline`, `roster`, `version`, `config show`, `vce validate` and `eval-gate` (which replays the bundled evaluation fixture by default) need nothing running, while `ingest run`, `build`, `vce run`, `corpus ...` and `corpus-repair ...` open Postgres, Qdrant or the object store directly from [`config/settings.yaml`](../config/settings.yaml), so they need the same environment as the API (see [configuration](configuration.md)).

The API address and key are resolved in this order: the `--url` option or `VORAG_API_URL`, then `api_url` in `~/.config/vo-rag/config.yaml` (override the file with `--config` or `VORAG_CLI_CONFIG`), then `http://localhost:8000`; the key comes from `VORAG_API_KEY` or `api_key` in that file. `vo-rag config show` prints what was resolved, with the key masked to its last four characters:

```text
$ VORAG_API_KEY=example-key-1234 uv run vo-rag --config /tmp/none.yaml config show
{
  "api_key": "***1234",
  "api_url": "http://localhost:8000",
  "config_path": "/tmp/none.yaml",
  "file": {}
}
```

<!-- BEGIN GENERATED: cli-commands -->
| Command | Summary | Options (type, default) |
| --- | --- | --- |
| `vo-rag admin status` | Show admin status. | - |
| `vo-rag answer` | Generate an answer for the query. | `QUERY_TEXT` (text, required)<br>`--profile` (choice: quality\|fast\|kimi_benchmark\|local, default 'quality')<br>`--context-scope` (choice: sterile\|raw, default 'sterile')<br>`--include-retrieval/--no-include-retrieval` (flag, default True)<br>`--mode` (choice: internal\|publication\|partner, default 'internal')<br>`--collection` (text)<br>`--top-k` (integer range, default 8)<br>`--rerank/--no-rerank` (flag, default True)<br>`--visual-mode` (choice: intent_aware\|baseline\|visual_only\|visual_track)<br>`--format` (choice: json\|table, default 'table') |
| `vo-rag build dsi` | Build DSI record for a single book and persist to Qdrant. | `--source-id` (text, required)<br>`--collection-version` (text, default 'vo_rag_books_raw_v1')<br>`--n-representative-chunks` (integer, default 16)<br>`--max-tokens` (integer, default 3000)<br>`--version-hash` (text) |
| `vo-rag build graph` | Build knowledge graph for a single book and persist to Qdrant. | `--source-id` (text, required)<br>`--max-tokens` (integer, default 4000)<br>`--chunks-per-call` (integer, default 6)<br>`--max-concurrent` (integer, default 24)<br>`--version-hash` (text) |
| `vo-rag build raptor` | Build RAPTOR tree for a single book and persist to Qdrant. | `--source-id` (text, required)<br>`--max-tokens` (integer, default 2500)<br>`--version-hash` (text) |
| `vo-rag build summary` | Build summary tree for a single book and persist to Qdrant. | `--source-id` (text, required)<br>`--max-concurrent` (integer, default 8)<br>`--version-hash` (text) |
| `vo-rag config show` | Show resolved CLI configuration with masked key. | - |
| `vo-rag corpus boilerplate` | Scan for (`--scan`) or remove (`--delete`, needs `--yes` to apply) boilerplate chunks. | `--scan` (flag)<br>`--delete` (flag)<br>`--yes` (flag)<br>`--source-id` (text, repeatable) |
| `vo-rag corpus canary-report` | Read-only gate report for a pre-batch canary sample. | `ENVELOPE_ID...` (text, required)<br>`--total-pages` (integer)<br>`--lanes` (integer, default 1)<br>`--include-queue-wait` (flag)<br>`--json` (flag) |
| `vo-rag corpus duplicates` | Group duplicate-titled books and plan a keep/delete list. | `--report` (file, required) |
| `vo-rag corpus health` | Report degenerate, dominant-section and unpaginated books, test points, boilerplate points, duplicate groups with a planned delete, and empty-figure, low-text-density, image-only-suspect, too-few-chunks and no-chunks books; exit 1 unless clean. | `--source-id` (text, repeatable)<br>`--json` (flag) |
| `vo-rag corpus ocr-farm bootstrap` | Create each worker's venv (CUDA wheels) and verify torch sees the GPU. | `--hosts` (text) |
| `vo-rag corpus ocr-farm plan` | List the books to OCR with total pages and estimated hours (read-only). | - |
| `vo-rag corpus ocr-farm run` | OCR the backlog on the workers and store ocr/<sha256>/docling.json. | `--limit` (integer range)<br>`--no-requeue` (flag)<br>`--hosts` (text) |
| `vo-rag corpus ocr-farm runpod collect` | Pull results, merge, upload (docling.json then meta.json), verify; terminate the pods and remove the ssh aliases only when every book verifies. | `--job-dir` (directory, required)<br>`--overwrite` (flag)<br>`--pitwall` (text, default 'pitwall') |
| `vo-rag corpus ocr-farm runpod plan` | Estimate pages, GPU-hours, cost and pod reservations; refuse if the reservations exceed the remaining budget or a pod's exceeds the per-request cap. | `--backlog` (file, required)<br>`--job-dir` (directory, required)<br>`--pages-per-gpu-hour` (float range, default 6000)<br>`--pods` (integer range, default 4)<br>`--gpu-type` (text, default 'NVIDIA A40')<br>`--cloud` (choice: secure\|community, default 'secure')<br>`--setup-minutes` (integer range, default 15)<br>`--price-per-gpu-hour` (float range)<br>`--setup-concurrency` (integer range, default 8)<br>`--force-replan` (flag)<br>`--pitwall` (text, default 'pitwall') |
| `vo-rag corpus ocr-farm runpod run` | Stage units, create pods (TTL = estimate + 25%), bootstrap, start the pools. | `--job-dir` (directory, required)<br>`--image` (text, default 'runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04')<br>`--ssh-key` (file, default ~/.ssh/id_ed25519)<br>`--max-workers` (integer range, default 8)<br>`--setup-concurrency` (integer range)<br>`--pitwall` (text, default 'pitwall') |
| `vo-rag corpus ocr-farm runpod verify` | Check every staged book's ocr/<sha256>/ cache in Garage (read-only). | `--job-dir` (directory, required) |
| `vo-rag corpus purge` | Remove every trace of each SOURCE_ID from Qdrant and Postgres. | `SOURCE_ID...` (text, required)<br>`--yes` (flag)<br>`--drop-categories` (flag) |
| `vo-rag corpus rechunk-preview` | Re-chunk stored parses with the current chunker and diff against live Qdrant chunks. | `--source` (text, repeatable)<br>`--sources-file` (file)<br>`--reparse` (flag)<br>`--show-drops` (integer range, default 0)<br>`--json` (flag) |
| `vo-rag corpus tf-books inputs` | Write one book-draft job per track-and-field source. | `--out` (file, required) |
| `vo-rag corpus tf-books load` | Load the reviewed sheet: every valid row is stored with reviewed=true. | `SHEET_PATH` (file, required) |
| `vo-rag corpus tf-books load-drafts` | Store every valid model draft as the book's record (reviewed=false); no human step. | `RESULTS_PATH` (file, required) |
| `vo-rag corpus tf-books sheet` | Write the model's book drafts as a review sheet (TSV), one row per draft. | `RESULTS_PATH` (file, required)<br>`--out` (file, required) |
| `vo-rag corpus tf-chunks apply` | Re-stamp Qdrant from stored tags without calling the model (after re-ingest). | `--source` (text, repeatable) |
| `vo-rag corpus tf-chunks export` | Write per-lane job files for untagged or stale chunks of every book with a record. | `--shards` (integer range, default 5)<br>`--out-dir` (directory, required)<br>`--source` (text, repeatable)<br>`--batch` (integer range, default 10) |
| `vo-rag corpus tf-chunks load` | Validate chunk-batch results, store them, and stamp the Qdrant payload. | `RESULTS_PATHS` (file, required) |
| `vo-rag corpus tf-chunks rekey` | Move stored chunk tags from their pre- keys to the current ones. | `--source` (text, repeatable) |
| `vo-rag corpus tf-chunks sample` | Review sheet: `per_book` tagged chunks per source, spread across the book. | `--source` (text, required, repeatable)<br>`--n` (integer range, default 20)<br>`--out` (file, required) |
| `vo-rag corpus tf-chunks status` | Tagged / stale / untagged point counts per T&F book, then totals. | `--failed` (file, repeatable) |
| `vo-rag corpus-repair apply` | Apply one eligible source repair with its declared finite budget. | `--manifest` (file, required)<br>`--state-dir` (directory, required) |
| `vo-rag corpus-repair plan` | Validate immutable evidence locally; performs no service calls. | `--manifest` (file, required) |
| `vo-rag corpus-repair preflight` | Read a fresh complete source census and active revision token. | `--manifest` (file, required)<br>`--state-dir` (directory, required) |
| `vo-rag corpus-repair resume` | Resume the same immutable source job from private checkpoints. | `--manifest` (file, required)<br>`--state-dir` (directory, required) |
| `vo-rag corpus-repair verify` | Verify publication receipts and source-filtered query boundary. | `--manifest` (file, required)<br>`--state-dir` (directory, required) |
| `vo-rag eval-gate` | Run the retrieval eval gate against the active pipeline roster. | - |
| `vo-rag health` | Check API health endpoint (no auth required). | - |
| `vo-rag ingest file` | - | `PATH` (file, required)<br>`--title` (text)<br>`--author` (text)<br>`--tags` (text, default '')<br>`--sovereignty` (text, default 'cloud_ok') |
| `vo-rag ingest file-drop` | - | `TARGETS` (text, required)<br>`--title` (text)<br>`--author` (text)<br>`--tags` (text, default '')<br>`--sovereignty` (text, default 'cloud_ok')<br>`--preflight` (file)<br>`--allow-rejects` (flag) |
| `vo-rag ingest idea` | - | `IDEA` (text, required)<br>`--tags` (text, default '')<br>`--title` (text)<br>`--sovereignty` (text, default 'cloud_ok') |
| `vo-rag ingest run` | Run one stored envelope through the book_rag ingest DAG. | `--envelope-id` (text, required)<br>`--chunking-strategy` (choice: hierarchical\|semantic, env VORAG_CHUNKING_STRATEGY) |
| `vo-rag ingest text` | - | `TEXT` (text, required)<br>`--title` (text)<br>`--tags` (text, default '')<br>`--sovereignty` (text, default 'cloud_ok') |
| `vo-rag ingest url` | - | `URL` (text, required)<br>`--title` (text)<br>`--sovereignty` (text, default 'cloud_ok') |
| `vo-rag pipeline list` | List the pipeline roster (from the single source of truth). | - |
| `vo-rag pipeline status` | Show pipeline roster summary counts. | `--json` (flag) |
| `vo-rag preflight` | Route or reject every book in TARGET (a directory or a manifest CSV) before ingest -- CPU-only, no network, no Qdrant, no Postgres. | `TARGET` (path, required)<br>`--out` (file)<br>`--csv` (file)<br>`--check-cache` (flag)<br>`--jobs` (integer, default 1)<br>`--timeout` (float, default 180.0)<br>`--catalog` (path)<br>`--canary-out` (file)<br>`--canary-n` (integer, default 20) |
| `vo-rag query` | Query indexed knowledge artifacts. | `QUERY_TEXT` (text, required)<br>`--collection` (text)<br>`--top-k` (integer range, default 20)<br>`--rerank/--no-rerank` (flag, default True)<br>`--mode` (choice: internal\|publication\|partner, default 'internal')<br>`--visual-mode` (choice: intent_aware\|baseline\|visual_only\|visual_track)<br>`--format` (choice: json\|table, default 'table') |
| `vo-rag roster` | List the pipeline roster (legacy alias for pipeline list). | - |
| `vo-rag status` | Show system status and optionally watch. | `--watch` (flag) |
| `vo-rag vce run` | Run the VCE stage DAG and write the export. | `--export-dir` (path, required)<br>`--video-ids` (text, repeatable)<br>`--executor` (choice: stub\|local\|remote\|speaches, default 'stub')<br>`--baseline` (path)<br>`--publish-path` (path)<br>`--run-id` (text)<br>`--json` (flag) |
| `vo-rag vce status` | Operator smoke — per-stage state counts from stage_run_log. | `--video-id` (text)<br>`--json` (flag) |
| `vo-rag vce validate` | Run the seven export validator checks against a published export directory. | `EXPORT_DIR` (path, required)<br>`--baseline` (path)<br>`--coverage-threshold` (float, default 0.9)<br>`--json` (flag) |
| `vo-rag version` | Print the vo-rag version. | - |
<!-- END GENERATED: cli-commands -->

## Worked examples

### Check files before you ingest them

`vo-rag preflight` decides, on the CPU and with no network, Postgres or Qdrant, whether each file can be ingested and which route it will take. This is real output from running it in this repository on the one-file example directory:

```text
$ uv run vo-rag preflight examples/
{
  "ok": true,
  "schema_version": "2",
  "command": "preflight",
  "counts": {
    "route": {
      "text": 1
    },
    "verdict": {
      "ok": 1
    }
  },
  "duplicate_groups": {},
  "results": [
    {
      "path": "examples/synthetic-document.md",
      "sha256": "ba0c211fd47dd0dcde8c2e21b32219f4bf3cd3c31f8681b38f6604b7de6aca3e",
      "fingerprint": null,
      "kind": "text",
      "pages": null,
      "route": "text",
      "verdict": "ok",
      "reasons": []
    }
  ]
}
```

For a directory of real books you would add `--out report.json` and pass the report to `vo-rag ingest file-drop --preflight report.json`, which refuses rejected files unless you pass `--allow-rejects`. Every check, route and option is explained in [parsing and chunking](parsing-and-chunking.md#preflight).

### Ask a question and read the result

With an API running and a book ingested, `vo-rag query` posts to `POST /api/v1/query` and prints a table. This is example output (the ids, scores and text are invented, and the table is rendered by the CLI's own renderer):

```text
$ export VORAG_API_KEY=...        # your own key
$ uv run vo-rag query "How does the sample library store passages?" --top-k 2 --no-rerank
                                                     Query results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ id                                   ┃ score  ┃ collection          ┃ preview                                        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 3f1c9e52-7a0b-5c1e-9d4a-2b6e8f0a1c77 │ 0.8213 │ vo_rag_books_raw_v1 │ A passage is stored with the heading path that │
│                                      │        │                     │ leads to it, so a hit can be cited.            │
│ a07d44b1-12c9-5e3b-8f60-91d2c3b4e5f6 │ 0.7502 │ vo_rag_books_raw_v1 │ Each chunk keeps its page number and source id │
│                                      │        │                     │ in the payload.                                │
└──────────────────────────────────────┴────────┴─────────────────────┴────────────────────────────────────────────────┘
```

The preview is the first 100 characters of the hit's text. Add `--format json` to get the full response, which is the `QueryResponse` documented in [retrieval and answers](retrieval-and-answers.md#request-fields): a `results` array whose items carry `id`, `score`, `collection` and the whole `payload` (source, page, heading path). `--mode publication` or `--mode partner` search the sterile text instead of the raw text; see [rights and compliance](rights-and-compliance.md).

### Keep a corpus clean

The `corpus` group is the maintenance loop: `health` finds defective books, `duplicates` plans which copies to remove, `purge` removes a book everywhere, and `canary-report` and `rechunk-preview` check the result. The loop, every finding and example output are in [operations](operations.md#corpus-maintenance-loops).
