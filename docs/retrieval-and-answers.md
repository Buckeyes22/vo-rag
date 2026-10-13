# Retrieval and answers

[README](../README.md) · [Architecture](architecture.md) · [Data model](data-model.md) · [Rights and compliance](rights-and-compliance.md) · [HTTP API](api.md)

This page covers the two read paths. `POST /api/v1/query` turns a question into ranked evidence: it embeds the question once, searches several vector lanes in each collection, fuses the ranked lists, removes duplicates and optionally reranks. `POST /api/v1/answer` runs that same retrieval and then asks a model to write a cited answer from the evidence, with prompt-injection defenses around the model call. Which collections a request may read, and when an answer is allowed to leave the building, are rights questions and live on [Rights and compliance](rights-and-compliance.md).

## What a query does

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API route
    participant X as execute_retrieval
    participant H as HybridRetriever
    participant E as Embedding service
    participant Q as Qdrant (publication client)
    participant R as Reranker

    C->>A: POST /api/v1/query
    A->>A: authenticate, resolve collections, check key scope
    A->>A: write query_log row (best effort)
    A->>X: query, mode, collections, filters, top_k, rerank_fn
    X->>X: capture publication pointers for this query
    X->>H: cross_collection_search
    H->>E: embed the query once (dense, sparse, ColBERT)
    H->>H: BM25 token ids computed locally
    par one search per collection (and per visual track)
        H->>Q: dense + sparse + BM25 prefetch, server-side RRF
        Q-->>H: ranked points, license filter applied
    end
    H->>H: cross-collection RRF (k = 60) and two-pass dedup
    X->>X: exact-evidence dedup, source-balanced candidate pool
    opt rerank requested and a reranker loaded
        X->>R: query + candidate pool
        R-->>X: cross-encoder scores, best first
    end
    X-->>A: at most top_k hits
    A-->>C: QueryResponse
```

The route is [`api/routes/query.py`](../src/vo_rag/api/routes/query.py). It is thin: it resolves collections, enforces the API key's mode and collection scope, writes a `query_log` row and calls [`execute_retrieval`](../src/vo_rag/core/retrieval/execution.py), the single entry point the API, the MCP `vo_rag_query` tool and the LlamaIndex retriever all share. `HybridRetriever` ([`search.py`](../src/vo_rag/core/retrieval/search.py)) owns the vector search; it never reaches past `QdrantStore` into the Qdrant client.

## Vectors: what is stored and what ranks

Every point in a searchable collection carries four named vectors. BGE-M3 produces three of them and a local tokenizer produces the fourth ([data model](data-model.md#qdrant-collections) lists the schema).

| Vector | What it is | Used for ranking |
| --- | --- | --- |
| `dense` | 1024-dimension cosine embedding of the chunk | Yes |
| `sparse` | BGE-M3 learned lexical weights | Yes |
| `bm25` | Hashed-token term weights from a local tokenizer; the token id is a 4-byte BLAKE2b hash, so the writer and the query path produce identical ids without a shared vocabulary file | Yes |
| `colbert` | Per-token multivector, stored with no ANN index | **No** |

ColBERT is optional at ingest. Real token vectors cost roughly ten times the storage and a third of the embedding compute, so they are off unless `VORAG_BOOK_RAG_INCLUDE_COLBERT=1`. The collection schema still requires a `colbert` vector on every point, so by default a point stores its dense vector as a one-row stand-in (`colbert_representation` is `dense_substitute` in the payload; real vectors are `token_vectors`). The query path still asks the embedding service for ColBERT vectors, but [`search_hybrid`](../src/vo_rag/core/storage/qdrant.py) does not use them: it prefetches the dense, sparse and BM25 lanes and fuses those three. The reason is stated at the call site: token vectors and dense stand-ins share a dimension but not a representation, so scoring one against the other would compare different things. Reranking is done by a cross-encoder (below), not by late interaction.

The query is embedded once per request into a `VectorBundle` and the same bundle is reused for every collection and every visual track.

## Collections searched per mode

With no `collections` in the request, `resolve_collections` ([`query_mode.py`](../src/vo_rag/core/retrieval/query_mode.py)) picks the book collection for the mode and adds the mode-agnostic collections:

| Mode | Book collection | Plus (all modes) |
| --- | --- | --- |
| `internal` | `vo_rag_books_raw_v1` | `vo_rag_articles_v1`, `vo_rag_repos_v1`, `vo_rag_research_v1`, `vo_rag_ideas_v1`, `vo_rag_tools_v1`, `vo_rag_urls_v1`, `vo_rag_structured_v1`, `vo_rag_code_chunks_v1`, `vo_rag_reddit_v1` |
| `publication` and `partner` | `vo_rag_books_sterile_v1` | the same nine |

An explicit `collections` list is honored, except that `vo_rag_books_raw_v1` is rewritten to `vo_rag_books_sterile_v1` in the two public modes. The derived-layer collections (summary tree, RAPTOR, DSI, graph) are not in the defaults; see [Advanced layers](advanced-layers.md). The collection name strings are the persistence contract: every read and write targets one of them, and renaming one orphans every point stored under it. Publication and partner modes also add a license filter, and `publication` and `partner` retrieval returns no book evidence until sterilization has been enabled and has filled the sterile collection; both points are on [Rights and compliance](rights-and-compliance.md).

## Fusion

Fusion happens at two levels.

**Inside one collection**, Qdrant fuses the three lanes server-side with its built-in reciprocal-rank fusion (`Fusion.RRF`). Each lane prefetches `prefetch_depth` candidates (100 by default, never fewer than `top_k`). vo-rag does not set Qdrant's rank constant. The summary-tree, RAPTOR and graph-node collections get one extra rule: their BM25 lane only admits points whose `bm25_representation` is `hashed_tokens_v1`, because earlier builders wrote learned sparse ids into that vector and those must not vote. Graph edges are dense-only.

**Across collections**, [`fusion.py`](../src/vo_rag/core/retrieval/fusion.py) merges the per-collection lists. Each list contributes `1 / (k + rank)` to a passage's score with `k = 60` (the classic constant from Cormack et al.) and `rank` counting from 1. A passage that is first in one list and third in another scores 1/61 + 1/63, about 0.0323. Changing `k` changes every merged ranking relative to any earlier baseline, so `k_rrf` is a keyword-only argument and an accidental positional value cannot change it silently.

The merge dedups in two passes, and the order matters:

1. **Per-source pass.** The key is `(source_id, parent_chunk_id or chunk_id)`, falling back to the storage point id for legacy points that carry neither. It collapses a raw and sterile pair that share a parent chunk, and it stops one chunk counting twice inside a single list. Independent lists still each cast a vote.
2. **Cross-variant pass.** The key is `(logical_book_key, content_hash)`. It collapses byte-distinct ingest copies of the same book so one passage surfaces once. The copies keep their best contribution rather than adding up, so duplicate copies cannot manufacture consensus.

When passages collapse, the surviving hit carries a `duplicate_provenance` list in its payload so every original citation identity is still available. A final exact-evidence pass collapses hits with identical text and context while keeping the best-ranked one first.

The module's own comments call this two-pass key sequence the most correct thing in the file and warn against simplifying it. A single key either leaves ingest copies in the result or merges passages that are genuinely different sections of one page, so the pair of keys is deliberate. **Only RRF ships.** A convex combination of normalized scores (min-max, z-score or rank normalization) was considered and dropped because no A/B decision called for it; the module therefore exports no alternative fusion and no fusion factory, and `k` is the only knob.

## Visual-aware retrieval

Figures and tables are far outnumbered by text chunks in every book, so on concept queries they would never reach the top results. `visual_mode` controls how many result slots they get (the default is `intent_aware`).

| `visual_mode` | Behavior |
| --- | --- |
| `intent_aware` | Detects visual intent in the query (phrases such as "show me" or "looks like", or words such as "diagram", "figure" or "chart"). With visual intent, 70% of the slots go to visual chunks; otherwise 30% |
| `baseline` | No chunk-type filtering |
| `visual_only` | Only `figure_composite`, `figure` and `table` chunks |
| `visual_track` | A fixed 60/40 text to visual split |

The two quota modes run a text search and a visual search in parallel, put the visual hits first, and then backfill from whichever track has spare hits, because a quota expresses a preference and is not a reason to discard available results. A caller-supplied `chunk_type` filter is never overwritten.

## Filters and visibility

`filters` is a flat object: a scalar matches exactly, a list matches any value, and `{"gte"|"gt"|"lte"|"lt": n}` is a numeric range, for example `{"page_number": {"gte": 100}}`. A clause the retriever cannot translate (an unknown bound name, a non-numeric bound, an empty object) is rejected with 422 at the API boundary instead of becoming a range that silently matches everything. The filter is combined with the mode's license filter using AND, so a caller's filter can narrow a public-mode query but never widen it.

Two visibility rules apply to every search without being requested. Points in `staging`, `failed`, `quarantined` or `superseded` ingest generations are excluded (a point with no `ingest_state` counts as active). For books and Reddit, a per-query read view pins the active revision of each source so one query sees one consistent version of it; see [Rights and compliance](rights-and-compliance.md#publication-read-view).

## Candidate depth and reranking

`top_k` is the size of the answer, not the size of the search. `execute_retrieval` widens the search so a reranker has something to choose from:

| Quantity | Formula | `top_k` = 10 | `top_k` = 20 |
| --- | --- | --- | --- |
| Rerank pool | `max(top_k, min(200, max(40, 4 * top_k)))` | 40 | 80 |
| Candidates fetched when reranking | `min(400, max(100, 2 * pool))` | 100 | 160 |
| Candidates fetched without reranking | `top_k` | 10 | 20 |

The reason is stated plainly in the code: a cross-encoder cannot recover a passage that was discarded before it ran. The pool is also source-balanced. The first half is simply the best by retrieval rank; the rest is filled round-robin across sources, so one long book cannot occupy every cross-encoder slot while strong passages from other books wait outside. A query over 4000 characters is refused.

Rerankers are interchangeable behind one `rerank(query, candidates, top_k)` call, chosen by `reranker.backend` in settings ([`factory.py`](../src/vo_rag/core/reranking/factory.py)):

| Backend | What runs | Notes |
| --- | --- | --- |
| `client` | HTTP call to a `/rerank` service ([`client.py`](../src/vo_rag/core/reranking/client.py)) | The default in `config/settings.yaml`, pointing at the inference worker on `localhost:8501`. Retries 429, 502, 503, 504 and transport errors with exponential backoff, four attempts in all. The API process stays free of torch and onnxruntime |
| `gpu` | A transformers sequence-classification model in fp16 on CUDA ([`gpu.py`](../src/vo_rag/core/reranking/gpu.py)) | Used by the inference worker itself. Default model `Alibaba-NLP/gte-reranker-modernbert-base` |
| `onnx_cpu` | `GteReranker`, an int8 ONNX cross-encoder on CPU ([`gte.py`](../src/vo_rag/core/reranking/gte.py)) | The no-GPU fallback. Batch 32, max length 512, 16 intra-op threads by default |
| `cohere` | Cohere Rerank v3.5 over HTTPS ([`cohere.py`](../src/vo_rag/core/reranking/cohere.py)) | Sends candidate text to a hosted service; needs `COHERE_API_KEY` |

`GteReranker` loads the quantized model file and its tokenizer when it is constructed, so a missing file fails at boot and not on the first query. It consumes a pre-exported int8 model, which is why `optimum` is not a dependency. The tokenizer revision is pinned through `VORAG_GTE_REVISION` so a future Hub download cannot silently follow a moving branch. There is no adaptive reranker: nothing in the query path calls for one, so each backend is a single class with one `rerank` method and the candidate depth is set by `execute_retrieval`, not by the reranker.

Each candidate is rendered as its section title followed by its text before scoring ([`render.py`](../src/vo_rag/core/reranking/render.py)). After reranking, a hit's `score` is the cross-encoder logit, which can be above 1 or negative. Without reranking it is the cross-collection RRF sum, which is small (a first place in one list is 1/61, about 0.0164). Do not compare scores across the two.

Two different failures behave differently. If the reranker could not be **built** at startup (a missing model file, say), the API still serves queries in fusion order even when the request says `rerank: true`, and the readiness probe reports the reranker as unavailable with the reason. If it was built but fails **at query time**, as with the default `client` backend when the rerank service is down, the call retries and then raises, nothing catches it, and a request with `rerank: true` (the default) returns 500. Send `rerank: false` to bypass a failing service.

## Parent auto-merge is not available over HTTP

[`auto_merge.py`](../src/vo_rag/core/retrieval/auto_merge.py) can replace a group of retrieved child chunks with their parent when enough siblings appear (`reranker.auto_merge_threshold`, default 0.5). It is off by default (`reranker.auto_merge_enabled: false`) and the HTTP query route never wires it: it passes `fetch_parent_fn=None` ([`api/routes/query.py`](../src/vo_rag/api/routes/query.py)), so even with the setting on the route merges nothing. The answer route does not pass the arguments at all. A library caller of `execute_retrieval` can supply a fetch function, but the merge keys on `parent_id` and `parent_child_count` payload fields that nothing in `src/` writes today, so on current data there would be no groups to merge.

## Request fields

`QueryRequest` ([`api/schemas.py`](../src/vo_rag/api/schemas.py)) rejects unknown fields.

| Field | Default | Meaning |
| --- | --- | --- |
| `query` | required | 1 to 4000 characters, not blank |
| `collections` | mode default | Non-empty list of collection names |
| `top_k` | 20 | 1 to 200 |
| `filters` | none | See above |
| `rerank` | `true` | Rerank when a reranker is loaded |
| `mode` | `internal` | `internal`, `publication` or `partner` |
| `visual_mode` | `intent_aware` | One of the four modes above |

An API key can be scoped to specific modes and collections; a request outside the key's scope gets 403 (`mode_not_allowed_for_api_key` or `collection_not_allowed_for_api_key`). The caller name is the `X-VO-RAG-Caller` header when one is sent (normalized to `api`, `cli`, `mcp` or `portal`, with anything else becoming `api`) and otherwise the caller bound to the key or Access identity; it is written to `query_log`. A scoped key whose bound caller differs from the header gets 403 `caller_not_allowed_for_api_key`; see [security](security.md#the-caller-label-is-not-authorization).

Example request (an illustration, not captured output):

```sh
curl -sS http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" -H "X-API-Key: $VORAG_API_KEY" \
  -d '{
        "query": "How does a plate heat exchanger transfer heat?",
        "top_k": 2,
        "mode": "internal",
        "rerank": true,
        "filters": {"license_class": "cleared"}
      }'
```

Example response (shape from `QueryResponse` and `QueryHit`; IDs and text are placeholders, and real payloads carry more fields):

```json
{
  "results": [
    {
      "id": "5f0c2a64-3b1e-5d8e-9a47-0c1d2e3f4a5b",
      "score": 6.81,
      "collection": "vo_rag_books_raw_v1",
      "payload": {
        "source_id": "vo-rag-corpus-sha256:9cd33ad5263959401ed620f416fea121c6deecc80cc1bf7f34a32b9a16973921",
        "chunk_id": "8d1e7c52-4f3a-5b60-a2d9-6e7f80919a2b",
        "chunk_type": "text",
        "page_number": 212,
        "section_path": "Heat exchangers > Plate designs",
        "section_title": "Plate designs",
        "content": "Thin corrugated plates separate the hot and cold streams...",
        "source_book": "Thermal Systems Handbook (A. Author)",
        "license_class": "cleared"
      }
    },
    {
      "id": "a3b4c5d6-e7f8-5a9b-8c0d-1e2f3a4b5c6d",
      "score": 5.12,
      "collection": "vo_rag_books_raw_v1",
      "payload": {
        "source_id": "vo-rag-corpus-sha256:9cd33ad5263959401ed620f416fea121c6deecc80cc1bf7f34a32b9a16973921",
        "chunk_id": "c1d2e3f4-a5b6-5c7d-8e9f-0a1b2c3d4e5f",
        "chunk_type": "figure_composite",
        "page_number": 214,
        "section_path": "Heat exchangers > Plate designs",
        "section_title": "Plate designs",
        "content": "Caption: Flow paths between adjacent plates.",
        "source_book": "Thermal Systems Handbook (A. Author)",
        "license_class": "cleared"
      }
    }
  ]
}
```

In `internal` mode a hit whose license class is not `cleared` or `public_domain` gains a `_vo_rag_license_warning` payload key such as `source_license_class=unlicensed`, so an operator sees what they are looking at. The same request works from the CLI (`vo-rag query "..." --mode internal --top-k 2 --format json`) and the MCP tool `vo_rag_query`.

## What an answer does

```mermaid
sequenceDiagram
    participant C as Client
    participant A as Answer route
    participant X as execute_retrieval
    participant G as Compliance gate
    participant S as AnswerSynthesizer
    participant L as LLM provider
    participant P as Postgres

    C->>A: POST /api/v1/answer
    A->>A: resolve collections (context_scope), check key scope
    A->>A: public mode without a gate: 403 before any retrieval
    A->>A: hosted egress guard for raw context
    A->>X: retrieve evidence
    X-->>A: hits
    opt publication or partner mode
        A->>G: check(mode, hits, caller)
        G-->>A: AnswerGateDecision (deny, error or invalid is 403)
    end
    A->>S: synthesize(query, hits, profile)
    Note over S,L: trust boundary: retrieved text is untrusted
    S->>S: select profile, refuse if not enabled
    S->>S: wrap evidence in an untrusted block, add canary and hard rule
    S->>L: JSON-mode completion, temperature 0
    L-->>S: draft answer
    S->>S: validate output (canary, echo, leak, URLs)
    S-->>A: draft, signals, should_quarantine
    A->>P: answer_log and citation_assignments rows
    A-->>C: AnswerResponse
```

The route is [`api/routes/answer.py`](../src/vo_rag/api/routes/answer.py) and the synthesis is [`core/answer/synthesizer.py`](../src/vo_rag/core/answer/synthesizer.py). The compliance gate, the 403 outcomes and how to supply a gate are documented in full on [Rights and compliance](rights-and-compliance.md#the-answer-compliance-gate); this section covers everything around it.

**Collections and scope.** `AnswerRequest.context_scope` defaults to `sterile`, which remaps the raw book collection to the sterile one in every mode, `internal` included. Raw evidence reaches synthesis only for `internal` mode with `context_scope: "raw"`. That is deliberate: raw, unredacted book text should not reach a model call unless the caller asks for it by name. If hosted answers are enabled and raw egress is not (`answer.allow_raw_hosted_egress`), a request that resolves to a raw collection is refused with 403 `raw_context_hosted_egress_forbidden`.

**Evidence formatting.** The synthesizer sends at most `answer.max_context_chunks` hits (8 by default). Each is rendered as a block labeled `[S1]`, `[S2]` and so on, with its point id, chunk id, source id, title, page and content, joined by `---` separators.

**The model call.** The system prompt tells the model to answer only from the supplied context, to treat that context as untrusted reference text, to put an `[S#]` label on every factual sentence, to paraphrase instead of quoting, and to return an empty answer with low confidence and a list of gaps when the context is insufficient. The call is JSON-mode at temperature 0 with `answer.max_output_tokens` (1200) as the output cap. The model returns an `AnswerDraft`: `answer`, `confidence`, `cited_chunk_ids`, `gaps` and `follow_up_questions`.

### Profiles and hosted gating

| `profile` | Provider and default model | Enabled by |
| --- | --- | --- |
| `quality` (default) | Anthropic, `claude-sonnet-4-6` | `answer.allow_hosted` and an Anthropic key |
| `fast` | Anthropic, `claude-haiku-4-5-20251001` | `answer.allow_hosted` and an Anthropic key |
| `kimi_benchmark` | Kimi (Moonshot API), `kimi-k2.6` | `answer.allow_hosted`, `answer.allow_kimi` and a Kimi key |
| `local` | A llama.cpp server at `answer.local_base_url` | `answer.allow_local`, a base URL and a model name; no external egress, so it is not gated on `allow_hosted` |

All of these flags are off by default (`config/settings.yaml`, `answer:`). A request for a profile that is not enabled gets 503 `answer_generation_not_configured`, so with an unmodified configuration the default `quality` profile is refused. The model ids are configuration, not code: change `quality_model`, `fast_model`, `kimi_model` or `local_model` in settings. A misconfiguration fails loudly instead of falling back to another provider. That is the point of the gating: sending retrieved text to a hosted model is egress, so each hosted route needs an explicit switch, and the benchmark route needs a second one.

### Why `allowed_domains=[]` is a literal

The synthesizer calls `complete_json_protected` with `allowed_domains=[]` written directly at the call site. It is not read from settings and cannot be changed by a request. An answer is supposed to cite internal chunks, never external pages, so any URL in the model's output is treated as a possible exfiltration channel (a prompt-injected instruction could ask the model to encode data into a link). With an empty allowlist the validator flags every URL it finds. Keeping the value immutable means no configuration or request can quietly widen what an answer may link to.

### The injection-defense trio

Retrieved text is attacker-controllable: a book, a web page or an OCR error can contain instructions. Every call over such text goes through [`core/llm/protected.py`](../src/vo_rag/core/llm/protected.py), which applies three layers:

1. **Wrap.** [`untrusted.py`](../src/vo_rag/core/llm/untrusted.py) HTML-escapes the evidence and fences it in `<document trust="untrusted">`. If the evidence contains a literal closing tag or the call's canary, the call is refused before the model sees anything, and the refusal is final (it is never retried).
2. **Canary.** A fresh random token is generated for each call and written into a hard rule appended to the system prompt: the fenced content is data, instructions inside it are never followed, and the canary must never be printed. Because the token is new for every call, a red-team fixture cannot target a fixed value, and any appearance in the output is conclusive evidence that the model leaked its prompt.
3. **Validate.** [`output_validator.py`](../src/vo_rag/core/llm/output_validator.py) inspects the response for `canary_echoed`, `verbatim_untrusted_echo` (a 120-character or longer span copied from the evidence), `system_prompt_leaked` (80 characters or more) and `url_exfil_attempt`. A signal at confidence 0.8 or higher marks the result for quarantine. The verbatim check also flags an answer that reproduces long spans of the source.

A quarantined answer is still returned: the response carries `warnings` including `prompt_injection_suspected` and the signal kinds, and `compliance.should_quarantine` is `true`. The audit row records `prompt_injection_suspected` as its compliance outcome. Withholding is a decision for the caller, who has the signals.

### Citations

An answer cites in two ways. In the text, the model labels sentences `[S1]`, `[S2]` and so on, which refer to the blocks in the evidence. In the response, `citations` is the model's `cited_chunk_ids` list, and `sources` is the sorted unique `source_id` set of the retrieved hits. The route returns the model's cited ids as given and does not cross-check them against the retrieved set, so a consumer that needs strict grounding should compare `citations` with the chunk ids in `retrieval_results`. When the model returns no usable answer, `answer` is `null`, `confidence` is 0 and `citations` is empty.

With `include_retrieval: true` (the default) the response includes the evidence as `retrieval_results`, but with credential-like keys and text-bearing keys (`content`, `text`, `summary`, anything containing `token`, `secret`, `password`, `api_key` or `credential`) removed from each payload. Unlike `/query`, `/answer` does not hand the evidence text back.

The route writes one `answer_log` row (profile, model, collections, context scope, retrieved count, cost, latency, compliance outcome, warnings) and one `citation_assignments` row per cited id at tier `chunk`. Persistence failures are logged and never fail the response.

### Request fields

`AnswerRequest` has the `QueryRequest` fields with different bounds and defaults, plus three of its own.

| Field | Default | Meaning |
| --- | --- | --- |
| `query`, `collections`, `filters`, `rerank`, `mode`, `visual_mode` | as for `/query` | Same meaning |
| `top_k` | 8 | 1 to 50; only the first `answer.max_context_chunks` hits (8 by default) are sent to the model |
| `profile` | `quality` | `quality`, `fast`, `kimi_benchmark` or `local` |
| `context_scope` | `sterile` | `sterile` or `raw` |
| `include_retrieval` | `true` | Return the redacted evidence |

### Errors

| Status and detail | Cause |
| --- | --- |
| 503 `retriever_unavailable`, `answer_synthesizer_unavailable` | A service slot did not start |
| 403 `answer_compliance_gate_not_configured` | Public mode and no gate; returned before retrieval |
| 403 `answer_compliance_gate_denied`, `_error`, `_invalid` | The gate refused, raised or was malformed |
| 403 `raw_context_hosted_egress_forbidden` | Raw collections with hosted answers on and raw egress off |
| 422 | Evidence from the `vo_rag_brain_v1` knowledge index is incomplete or too large to send whole |
| 503 `answer_generation_not_configured` | The requested profile is not enabled or lacks a key |
| 500 `answer_generation_failed` | Any other synthesis failure, including a refused untrusted block |

Example request (an illustration):

```sh
curl -sS http://127.0.0.1:8000/api/v1/answer \
  -H "Content-Type: application/json" -H "X-API-Key: $VORAG_API_KEY" \
  -H "X-VO-RAG-Caller: portal" \
  -d '{
        "query": "How does a plate heat exchanger transfer heat?",
        "mode": "publication",
        "profile": "quality",
        "top_k": 4
      }'
```

Example response with a compliance gate configured (shape from `AnswerResponse`; ids and text are placeholders):

```json
{
  "answer_id": "01M4SD7D19Z7E090ET39M7ZKJX",
  "answer": "Heat moves from the hot stream to the cold stream through thin corrugated plates, and the corrugations create turbulence that raises the transfer rate [S1]. Adjacent plates carry the two streams in alternating channels [S2].",
  "confidence": 0.82,
  "citations": [
    "2b7e9f10-6a3c-5d41-8f52-7a8b9c0d1e2f",
    "e4f5a6b7-c8d9-5e0f-a1b2-c3d4e5f6a7b8"
  ],
  "sources": [
    "vo-rag-corpus-sha256:9cd33ad5263959401ed620f416fea121c6deecc80cc1bf7f34a32b9a16973921"
  ],
  "retrieval_results": [
    {
      "id": "7c8d9e0f-1a2b-5c3d-9e4f-5a6b7c8d9e0f",
      "score": 6.43,
      "collection": "vo_rag_books_sterile_v1",
      "payload": {
        "source_id": "vo-rag-corpus-sha256:9cd33ad5263959401ed620f416fea121c6deecc80cc1bf7f34a32b9a16973921",
        "chunk_id": "2b7e9f10-6a3c-5d41-8f52-7a8b9c0d1e2f",
        "chunk_type": "text",
        "page_number": 212,
        "section_path": "Heat exchangers > Plate designs",
        "license_class": "cleared"
      }
    },
    {
      "id": "0e1f2a3b-4c5d-5e6f-8a7b-8c9d0e1f2a3b",
      "score": 5.97,
      "collection": "vo_rag_books_sterile_v1",
      "payload": {
        "source_id": "vo-rag-corpus-sha256:9cd33ad5263959401ed620f416fea121c6deecc80cc1bf7f34a32b9a16973921",
        "chunk_id": "e4f5a6b7-c8d9-5e0f-a1b2-c3d4e5f6a7b8",
        "chunk_type": "text",
        "page_number": 213,
        "section_path": "Heat exchangers > Plate designs",
        "license_class": "cleared"
      }
    }
  ],
  "compliance": {
    "status": "allowed",
    "should_quarantine": false,
    "reason": null,
    "pipeline": "query_answer",
    "details": {
      "parent_ids": [
        "2b7e9f10-6a3c-5d41-8f52-7a8b9c0d1e2f",
        "e4f5a6b7-c8d9-5e0f-a1b2-c3d4e5f6a7b8"
      ],
      "signals": []
    }
  },
  "warnings": [],
  "model": "claude-sonnet-4-6",
  "mode": "publication"
}
```

`compliance.status` is `"allowed"` only when a gate was consulted and allowed the request. In `internal` mode no gate runs and the status is `"not_configured"`.

## Code map

| Concern | Code |
| --- | --- |
| Query route, answer route | [`api/routes/query.py`](../src/vo_rag/api/routes/query.py), [`api/routes/answer.py`](../src/vo_rag/api/routes/answer.py) |
| Request and response models | [`api/schemas.py`](../src/vo_rag/api/schemas.py) |
| Retrieval entry point | [`core/retrieval/execution.py`](../src/vo_rag/core/retrieval/execution.py) |
| Hybrid search, filters, visual dispatch | [`core/retrieval/search.py`](../src/vo_rag/core/retrieval/search.py), [`visual_intent.py`](../src/vo_rag/core/retrieval/visual_intent.py) |
| Fusion and dedup | [`core/retrieval/fusion.py`](../src/vo_rag/core/retrieval/fusion.py) |
| BM25 tokenizer | [`core/retrieval/bm25_tokenizer.py`](../src/vo_rag/core/retrieval/bm25_tokenizer.py) |
| Per-collection hybrid search | [`core/storage/qdrant.py`](../src/vo_rag/core/storage/qdrant.py) |
| Rerankers | [`core/reranking`](../src/vo_rag/core/reranking) |
| Answer synthesis and context | [`core/answer/synthesizer.py`](../src/vo_rag/core/answer/synthesizer.py), [`core/answer/context.py`](../src/vo_rag/core/answer/context.py) |
| Injection defense | [`core/llm/protected.py`](../src/vo_rag/core/llm/protected.py), [`untrusted.py`](../src/vo_rag/core/llm/untrusted.py), [`output_validator.py`](../src/vo_rag/core/llm/output_validator.py) |

## Related

[Rights and compliance](rights-and-compliance.md) · [Advanced layers](advanced-layers.md) · [Evaluation](evaluation.md) · [Security](security.md) · [Parsing and chunking](parsing-and-chunking.md) for how chunks and their embeddings are produced
