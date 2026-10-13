# HTTP API reference

The API is a FastAPI application built by `create_app()` in [`api/main.py`](../src/vo_rag/api/main.py) and served as `vo_rag.api.main:app`. Every route requires an API key except the ones marked exempt in [`api/middleware.py`](../src/vo_rag/api/middleware.py).

The table is generated from the application's route table by `scripts/gen_reference_docs.py`. Do not edit between the markers; change the route and rerun the generator.

Send the key in the `X-API-Key` header. A deployment behind Cloudflare Access can instead accept the Access JWT, and browser mutations then need a CSRF token; the full rules, the probe exemptions and the 401 body are in [security](security.md#authentication). Request and response shapes come from the pydantic models in [`api/schemas.py`](../src/vo_rag/api/schemas.py), and the running app serves the same contract at `/openapi.json` and `/docs`. Narrative walkthroughs of the main flows are in [ingestion](ingestion.md#intake-surfaces) (intake), [retrieval and answers](retrieval-and-answers.md) (query and answer) and [data and API flows](data-and-api.md).

<!-- BEGIN GENERATED: api-routes -->
| Method | Path | Endpoint | Auth | Request model | Response model |
| --- | --- | --- | --- | --- | --- |
| GET | `/api/v1/admin/status` | `admin_status` | API key | - | `dict[str, Any]` |
| POST | `/api/v1/answer` | `answer` | API key | `AnswerRequest` | `AnswerResponse` |
| GET | `/api/v1/artifacts/{artifact_id}/images` | `artifact_images` | API key | - | `dict[str, Any]` |
| GET | `/api/v1/artifacts/{artifact_id}` | `artifact_detail` | API key | - | `dict[str, Any]` |
| GET | `/api/v1/artifacts` | `artifacts_list` | API key | - | `dict[str, Any]` |
| GET | `/api/v1/books/{source_id}/pages/{page}` | `book_page` | API key | - | `BookPageResponse` |
| GET | `/api/v1/books/{source_id}/summaries` | `book_summaries` | API key | - | `BookSummariesResponse` |
| GET | `/api/v1/books/{source_id}` | `book_overview` | API key | - | `BookOverviewResponse` |
| GET | `/api/v1/collections/stats` | `collections_stats` | API key | - | `dict[str, Any]` |
| GET | `/api/v1/envelopes/{envelope_id}` | `envelope_detail` | API key | - | `dict[str, Any]` |
| GET | `/api/v1/envelopes` | `envelopes_list` | API key | - | `dict[str, Any]` |
| GET | `/api/v1/figures/{src}/{filename}` | `get_figure` | API key | - | - |
| POST | `/api/v1/intake/upload-url` | `upload_url` | API key | `UploadUrlRequest` | `UploadUrlResponse` |
| GET | `/api/v1/intake/{envelope_id}/stream` | `stream_intake` | API key | - | - |
| POST | `/api/v1/intake` | `submit_intake` | API key | `IntakeSubmission` | `EnvelopeResponse` |
| GET | `/api/v1/portal/session` | `portal_session` | API key | - | `dict[str, Any]` |
| POST | `/api/v1/query` | `query` | API key | `QueryRequest` | `QueryResponse` |
| GET | `/api/v1/rag/status` | `rag_status` | API key | - | - |
| GET | `/docs/oauth2-redirect` | `swagger_ui_redirect` | API key | - | - |
| GET | `/docs` | `swagger_ui_html` | exempt | - | - |
| GET | `/health` | `health` | exempt | - | `dict[str, str]` |
| GET | `/metrics` | `metrics` | exempt | - | - |
| GET | `/openapi.json` | `openapi` | exempt | - | - |
| GET | `/ready/intake` | `intake_ready` | exempt | - | - |
| GET | `/ready/query` | `ready` | exempt | - | - |
| GET | `/ready` | `ready` | exempt | - | - |
| GET | `/redoc` | `redoc_html` | API key | - | - |
| GET | `/status` | `status_route` | API key | - | `dict[str, Any]` |
<!-- END GENERATED: api-routes -->

## Worked examples

All responses below are examples built from the response models; ids and times are invented.

### Find failed ingests

`GET /api/v1/envelopes` lists intake envelopes newest first, with optional `status` and `limit` (1 to 200, default 50) filters. After a dispatcher run, this is the quickest way to see what failed and why:

```sh
curl -sS "http://127.0.0.1:8000/api/v1/envelopes?status=failed&limit=1" \
  -H "X-API-Key: $VORAG_API_KEY"
```

```json
{
  "items": [
    {
      "id": "01M4SBAPR8NMHWTQJKZRQ0JFPY",
      "source_type": "file_drop",
      "source_data": {"garage_key": "verified/sha256/9cd33ad5.pdf"},
      "raw_content": {"filename": "handbook.pdf", "file_hash": "9cd33ad5", "size_bytes": 48213},
      "metadata": {"title": "Handbook"},
      "triage_result": null,
      "status": "failed",
      "target_pipeline": null,
      "priority": "medium",
      "sovereignty": "cloud_ok",
      "error": "parse failed: file has no extractable text",
      "created_at": "2026-10-12T09:30:00Z",
      "updated_at": "2026-10-12T09:30:00Z"
    }
  ],
  "next_cursor": null
}
```

`next_cursor` is `null` when there is no older page, but the route accepts no cursor parameter, so it returns only the newest `limit` envelopes; filter by `status` to narrow the list. The status values are the `EnvelopeStatus` enum ([ingestion](ingestion.md#envelope-lifecycle)). `GET /api/v1/envelopes/{envelope_id}` returns one item in the same shape, or 404 `envelope_not_found`.

### Read a stored page

`GET /api/v1/books/{source_id}/pages/{page}` returns the raw chunks that sit on one page of an ingested book, which is how the portal shows a book. The path takes the full `source_id`, and `page` starts at 1:

```sh
curl -sS "http://127.0.0.1:8000/api/v1/books/vo-rag-corpus-sha256:9cd33ad5.../pages/3" \
  -H "X-API-Key: $VORAG_API_KEY"
```

```json
{
  "page": 3,
  "max_page": 240,
  "chunks": [
    {
      "chunk_id": "3f1c9e52-7a0b-5c1e-9d4a-2b6e8f0a1c77",
      "section_title": "Handbook > 1 Storing passages",
      "text": "A passage is stored with the heading path that leads to it ...",
      "has_image": false,
      "image_id": null,
      "vlm_description": null,
      "token_count": 212
    }
  ]
}
```

The route reads the raw collection, so treat access to it as access to raw text. `max_page` comes from the book's artifact row and is `null` when none exists. The sibling routes `/api/v1/books/{source_id}` and `/api/v1/books/{source_id}/summaries` return the overview and the advanced-layer summaries ([advanced layers](advanced-layers.md)).
