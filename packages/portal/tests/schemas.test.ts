import {
  CollectionStatsSchema,
  HealthResponseSchema,
  IntakeEnvelopeSchema,
  KnowledgeArtifactSchema,
  PortalSessionSchema,
  QueryRequestSchema,
  QueryResponseSchema,
  RagStatusSchema,
  StatusResponseSchema,
  TriageResultSchema,
  UploadUrlResponseSchema,
} from "@/schemas";

describe("Zod schemas", () => {
  it("TriageResult rejects unknown disposition", () => {
    expect(() =>
      TriageResultSchema.parse({
        input_type: "url",
        disposition: "nope",
        target_pipeline: "url_snapshot",
        priority: "low",
        confidence: 0.5,
      }),
    ).toThrow();
  });

  it("TargetPipeline enum covers vo-rag envelope pipelines", () => {
    for (const pipeline of [
      "book_rag",
      "repo_analysis",
      "research_agent",
      "idea_capture",
      "article_extraction",
      "tool_evaluation",
      "url_snapshot",
      "structured_scrape",
      "vce",
      "code_rag",
    ]) {
      const env = IntakeEnvelopeSchema.parse({
        envelope_id: "x",
        source_type: "url",
        status: "received",
        triage_result: {
          input_type: "url",
          disposition: "process",
          target_pipeline: pipeline,
          priority: "low",
          confidence: 0.9,
        },
      });

      expect(env.triage_result?.target_pipeline).toBe(pipeline);
    }
  });

  it("QueryRequest rejects top_k > 200", () => {
    expect(() => QueryRequestSchema.parse({ query: "x", top_k: 201 })).toThrow();
  });

  it("QueryRequest only accepts the internal mode", () => {
    expect(QueryRequestSchema.parse({ query: "x", mode: "internal" }).mode).toBe("internal");
    expect(() => QueryRequestSchema.parse({ query: "x", mode: "publication" })).toThrow();
  });

  it("QueryRequest rejects empty query", () => {
    expect(() => QueryRequestSchema.parse({ query: "" })).toThrow();
  });

  it("PortalSession defaults to the internal-only mode list", () => {
    const session = PortalSessionSchema.parse({
      user: { email: "admin@example.com" },
      auth_source: "api_key",
      csrf_token: null,
      capabilities: {
        single_admin: true,
        chunked_upload: false,
        safe_publication_mode: false,
        legacy_api_keys: true,
      },
    });

    expect(session.modes).toEqual(["internal"]);
  });

  it("HealthResponse accepts the real liveness body", () => {
    expect(
      HealthResponseSchema.parse({
        status: "ok",
        service: "vo-rag",
      }),
    ).toEqual({ status: "ok", service: "vo-rag" });
  });

  it("StatusResponse accepts the real operator status shape", () => {
    const status = StatusResponseSchema.parse({
      ok: true,
      service: "vo-rag",
      version: "0.1.0",
      envelope_counts: { received: 3 },
      active_workers: null,
      queue_depth: 2,
      qdrant: { ok: true, collections: ["vo_rag_books_raw_v1"], points: { vo_rag_books_raw_v1: 10 } },
      cloud: { ok: false, error: "not_configured" },
      services: {
        db_pool: { ok: true },
        qdrant: { ok: true, collections: 2 },
        reranker: { ok: true },
        embedding_server: { ok: true, status_code: 200, device: "cuda" },
      },
    });

    expect(status.services.db_pool).toMatchObject({ ok: true });
    expect(status.queue_depth).toBe(2);
  });

  it("RagStatus accepts production collection readiness checks", () => {
    const rag = RagStatusSchema.parse({
      ready: true,
      db_pool: { ok: true },
      qdrant: { ok: true, collections: 2 },
      embedding_server: { ok: true, status_code: 200, device: "cuda" },
      raw_collection: { ok: true, collection: "vo_rag_books_raw_v1", points: 5 },
      sterile_collection: { ok: true, collection: "vo_rag_books_sterile_v1", points: 7 },
      raw_collection_has_points: true,
      sterile_collection_has_points: true,
    });

    expect(rag.sterile_collection?.points).toBe(7);
  });

  it("QueryResponse preserves hits as payload-first vo-rag results", () => {
    const result = QueryResponseSchema.parse({
      results: [
        {
          id: "p1",
          score: 0.9,
          collection: "vo_rag_books_sterile_v1",
          payload: {
            content: "x",
            section_title: "Corner Entry",
            page_number: 12,
            image_ref: "book/page.png",
          },
        },
      ],
      elapsed_ms: 12,
    });

    expect(result.results[0]?.score).toBe(0.9);
    expect(result.results[0]?.payload).toMatchObject({ content: "x", page_number: 12 });
    expect("snippet" in (result.results[0] as Record<string, unknown>)).toBe(false);
  });

  it("CollectionStats keeps the backend keyed-by-collection shape", () => {
    const stats = CollectionStatsSchema.parse({
      collections: {
        vo_rag_books_raw_v1: { points: 10 },
        vo_rag_books_sterile_v1: { points: 20 },
      },
    });

    expect(stats.collections.vo_rag_books_raw_v1?.points).toBe(10);
  });

  it("UploadUrlResponse includes bucket and expiry", () => {
    const upload = UploadUrlResponseSchema.parse({
      upload_url: "https://garage/upload",
      garage_key: "intake/01H/book.pdf",
      bucket: "artifacts",
      expires_in_s: 21600,
    });

    expect(upload.bucket).toBe("artifacts");
  });

  it("KnowledgeArtifact accepts the model_dump field set", () => {
    const artifact = KnowledgeArtifactSchema.parse({
      id: "a",
      envelope_id: "e",
      pipeline: "book_rag",
      artifact_type: "summary",
      title: "Going Faster",
      content: { summary: "x" },
      tags: ["book"],
      priority: "medium",
      status: "active",
      license_class: "unknown",
      trust_level: "quarantined",
      qdrant_collection: "vo_rag_books_raw_v1",
      qdrant_point_ids: ["p1"],
      source_provenance: [],
      created_at: "2026-04-13T00:00:00Z",
      updated_at: "2026-04-13T00:00:00Z",
    });

    expect(artifact.qdrant_point_ids).toEqual(["p1"]);
  });
});
