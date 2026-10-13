# MCP server reference

The MCP server exposes the query, intake and status operations to agent clients. It is built by `build_mcp_server()` in [`mcp/server.py`](../src/vo_rag/mcp/server.py) and runs over stdio or streamable HTTP.

The table is generated from the registered tools by `scripts/gen_reference_docs.py`. Do not edit between the markers; change the tool and rerun the generator.

## Starting the server

```sh
# Local agent client: the default transport, no network port, no authentication.
uv run python -m vo_rag.mcp.server --transport stdio

# Remote client: streamable HTTP on 127.0.0.1:8200, served at /mcp.
VORAG_MCP_API_KEY=... uv run python -m vo_rag.mcp.server --transport streamable-http --host 127.0.0.1 --port 8200
```

The server builds its own service container (database pool, Qdrant, embedding and rerank clients) when it starts, from the same [settings](configuration.md) the API uses, and calls retrieval and envelope insertion directly; it does not go through the API. The HTTP transport refuses to start unless `VORAG_MCP_API_KEY` or `api.api_key` is set, and then requires `Authorization: Bearer <key>` on every request. Stdio has no authentication because the client owns the process. Keep the HTTP transport on loopback or behind your own proxy.

A client registers the stdio server with a command and arguments. This is an example in the `mcpServers` shape that several clients use; the exact file and keys depend on your client:

```json
{
  "mcpServers": {
    "vo-rag": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/vo-rag", "python", "-m", "vo_rag.mcp.server", "--transport", "stdio"],
      "env": {"VORAG_SETTINGS_PATH": "/path/to/vo-rag/config/settings.yaml"}
    }
  }
}
```

The server announces itself as `vo-rag`, so clients that namespace tools by server show them as `vo_rag_query`, `vo_rag_ingest`, `vo_rag_ingest_stream` and `vo_rag_status` under that name. The query tool caps `top_k` at 50 and defaults to 5, because its output goes into a model's context window. A tool failure comes back as a result with `isError` set and a redacted message, not a protocol error: stack traces, credentialed URLs, `key=value` secrets and file paths are removed ([security](security.md#secrets-and-redaction)).

<!-- BEGIN GENERATED: mcp-tools -->
| Tool | Description | Input fields |
| --- | --- | --- |
| `vo_rag_query` | Search the vo-rag knowledge layer. Returns ranked hits with source attribution. filters: scalar = exact match, list = any-of, {"gte"\|"gt"\|"lte"\|"lt": n} = numeric range (e.g. {"page_number": {"gte": 10}}). | `query` (required), `collections`, `top_k`, `mode`, `filters`, `rerank`, `visual_mode` |
| `vo_rag_ingest` | Submit content to vo-rag. Returns envelope_id and dispatch status. | `source_type` (required), `content` (required), `metadata`, `sovereignty` |
| `vo_rag_ingest_stream` | Stream ingest progress for an envelope. Reports per-step progress via MCP notifications. | `envelope_id` (required) |
| `vo_rag_status` | Check vo-rag system status. No args: system status. With envelope_id: envelope progress. | `envelope_id` |
<!-- END GENERATED: mcp-tools -->

## Worked examples

Each tool returns two views of one result: `content`, a short text rendering for a model to read, and `structuredContent`, the typed result. Both examples are built from the real result models and renderers; the ids and text are invented.

### Search

A client calls `vo_rag_query` with its arguments:

```json
{"method": "tools/call", "params": {"name": "vo_rag_query", "arguments": {"query": "How does the sample library store passages?", "top_k": 3}}}
```

and receives:

```json
{
  "content": [
    {
      "type": "text",
      "text": "#1 [0.821] 1 Storing passages\n  Source: Handbook (page 3)\n  A passage is stored with the heading path that leads to it, so a hit can be cited."
    }
  ],
  "structuredContent": {
    "ok": true,
    "code": null,
    "error": null,
    "command": "vo_rag_query",
    "schema_version": 1,
    "results": [
      {
        "id": "3f1c9e52-7a0b-5c1e-9d4a-2b6e8f0a1c77",
        "score": 0.8213,
        "collection": "vo_rag_books_raw_v1",
        "title": "1 Storing passages",
        "source": "Handbook",
        "page": 3,
        "image": null,
        "payload": {"source_title": "Handbook", "section_title": "1 Storing passages", "page_number": 3, "content": "A passage is stored with the heading path that leads to it, so a hit can be cited."}
      }
    ],
    "mode": "internal",
    "top_k": 3,
    "reranked": true,
    "warnings": []
  },
  "isError": false
}
```

`mode` defaults to `internal`, which reads raw text, so the same trust rules as the API apply: give an agent the `publication` or `partner` mode if it should not see raw text ([rights and compliance](rights-and-compliance.md)). `filters` takes the same flat object as the API, for example `{"page_number": {"gte": 100}}`.

### Follow an ingest

`vo_rag_ingest` stores an envelope and returns its id and a `dispatch_status` of `submitted`, `streamed` or `not_configured`; the MCP tool never produces a verified file-drop, so the dispatcher does not claim what it stores ([ingestion](ingestion.md#intake-surfaces)). `vo_rag_status` with an `envelope_id` reports the state of one envelope. The text view of that result is:

```text
Envelope: 01M4SBAPR8NMHWTQJKZRQ0JFPY
Source:   file_drop
Status:   processing
Priority: medium
Sovereignty: cloud_ok
```

Without arguments `vo_rag_status` returns the system view instead: one line per backing service (`ok`, `failed: <reason>` or `not configured`) and the capability availability, which reflects configuration and is not a live health check. `vo_rag_ingest_stream` runs the ingest for an envelope inside the MCP server process and reports per-step progress as MCP notifications, so that process needs the same runtime dependencies as the dispatcher.
