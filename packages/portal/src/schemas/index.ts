import { z } from "zod";

export const EnvelopeStatusSchema = z.enum([
  "received",
  "triaging",
  "triaged",
  "dispatched",
  "processing",
  "completed",
  "failed",
  "discarded",
]);

export const IntakeSourceSchema = z.enum([
  "github_star",
  "file_drop",
  "url",
  "text_prompt",
  "api",
]);

export const TargetPipelineSchema = z.enum([
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
]);

export const LicenseClassSchema = z.enum([
  "cleared",
  "pending",
  "unlicensed",
  "public_domain",
  "unknown",
]);

export const PrioritySchema = z.enum(["high", "medium", "low", "noise"]);

export const DispositionSchema = z.enum([
  "process",
  "bookmark_only",
  "discard",
  "needs_human",
]);

export const SovereigntySchema = z.enum([
  "homelab_only",
  "cloud_ok",
  "cloud_ok_no_persist",
]);

export const QueryModeSchema = z.enum(["internal"]);

export const PortalSessionSchema = z
  .object({
    user: z.object({ email: z.string() }).passthrough(),
    auth_source: z.string(),
    csrf_token: z.string().nullable().optional(),
    capabilities: z
      .object({
        single_admin: z.boolean(),
        chunked_upload: z.boolean(),
        safe_publication_mode: z.boolean(),
        legacy_api_keys: z.boolean().optional(),
      })
      .passthrough(),
    modes: z.array(z.string()).default(["internal"]),
  })
  .passthrough();

export const TriageResultSchema = z
  .object({
    input_type: z.string(),
    disposition: DispositionSchema,
    priority: PrioritySchema,
    tags: z.array(z.string()).optional(),
    target_pipeline: TargetPipelineSchema.nullable().optional(),
    confidence: z.number(),
    reasoning: z.string().optional(),
    used_fallback: z.boolean().optional(),
  })
  .passthrough();

export const IntakeEnvelopeSchema = z
  .object({
    id: z.string().optional(),
    envelope_id: z.string().optional(),
    source_type: IntakeSourceSchema,
    source_data: z.record(z.unknown()).optional(),
    raw_content: z.record(z.unknown()).optional(),
    metadata: z.record(z.unknown()).optional(),
    triage_result: TriageResultSchema.nullable().optional(),
    status: EnvelopeStatusSchema,
    target_pipeline: TargetPipelineSchema.nullable().optional(),
    priority: PrioritySchema.optional(),
    sovereignty: SovereigntySchema.optional(),
    error: z.string().nullable().optional(),
    created_at: z.string().optional(),
    updated_at: z.string().optional(),
    dispatch_status: z.string().optional(),
  })
  .passthrough()
  .transform((value) => ({
    ...value,
    envelope_id: value.envelope_id ?? value.id ?? "",
  }));

export const KnowledgeArtifactSchema = z
  .object({
    id: z.string(),
    envelope_id: z.string().optional(),
    pipeline: TargetPipelineSchema.optional(),
    artifact_type: z.string().optional(),
    title: z.string().optional(),
    content: z.record(z.unknown()).optional(),
    tags: z.array(z.string()).optional(),
    priority: PrioritySchema.optional(),
    status: z.string().optional(),
    license_class: LicenseClassSchema.optional(),
    trust_level: z.string().optional(),
    qdrant_collection: z.string().nullable().optional(),
    qdrant_point_ids: z.array(z.string()).optional(),
    source_provenance: z.array(z.record(z.unknown())).optional(),
    created_at: z.string().optional(),
    updated_at: z.string().optional(),
  })
  .passthrough();

export const QueryRequestSchema = z.object({
  query: z.string().min(1),
  collections: z.array(z.string()).optional(),
  top_k: z.number().int().min(1).max(200).optional(),
  filters: z.record(z.unknown()).optional(),
  rerank: z.boolean().optional(),
  mode: QueryModeSchema.optional(),
});

export const QueryHitSchema = z.object({
  id: z.string(),
  score: z.number(),
  collection: z.string(),
  payload: z.record(z.unknown()),
});

export const QueryResponseSchema = z
  .object({
    results: z.array(QueryHitSchema),
  })
  .passthrough();

export const UploadUrlRequestSchema = z.object({
  filename: z.string().min(1),
  content_type: z.string().min(1),
  size_bytes: z.number().int().min(1),
  sovereignty: SovereigntySchema.optional(),
});

export const UploadUrlResponseSchema = z
  .object({
    upload_url: z.string(),
    garage_key: z.string(),
    bucket: z.string(),
    expires_in_s: z.number(),
  })
  .passthrough();

export const IntakeSubmissionSchema = z.object({
  source_type: z.string(),
  source_data: z.record(z.unknown()).optional(),
  raw_content: z.record(z.unknown()),
  metadata: z.record(z.unknown()).optional(),
  priority: z.string().optional(),
  sovereignty: z.string().optional(),
  target_pipeline: z.string().nullable().optional(),
});

export const CollectionStatsSchema = z
  .object({
    collections: z.record(
      z
        .object({
          points: z.number(),
        })
        .passthrough(),
    ),
  })
  .passthrough();

export const HealthCheckSchema = z.union([
  z.string(),
  z.boolean(),
  z
    .object({
      ok: z.boolean().optional(),
      ready: z.boolean().optional(),
      error: z.string().optional(),
      reason: z.string().optional(),
    })
    .passthrough(),
]);

export const HealthResponseSchema = z
  .object({
    status: z.string(),
    service: z.string(),
  })
  .passthrough();

export const GpuStatusSchema = z.object({
  node: z.string(),
  vram_used_mb: z.number(),
  util_pct: z.number(),
});

export const StatusResponseSchema = z
  .object({
    ok: z.boolean(),
    service: z.string(),
    version: z.string(),
    envelope_counts: z.record(z.number()).default({}),
    active_workers: z.unknown().nullable().optional(),
    queue_depth: z.number().optional(),
    qdrant: z.record(z.unknown()).optional(),
    cloud: z.record(z.unknown()).optional(),
    services: z.record(HealthCheckSchema).default({}),
    gpus: z.array(GpuStatusSchema).default([]),
  })
  .passthrough();

export const CountCheckSchema = z
  .object({
    ok: z.boolean().optional(),
    collection: z.string().optional(),
    points: z.number().optional(),
    count: z.number().optional(),
    error: z.string().optional(),
  })
  .passthrough();

export const RagStatusSchema = z
  .object({
    ready: z.boolean(),
    db_pool: HealthCheckSchema.optional(),
    qdrant: HealthCheckSchema.optional(),
    embedding_server: HealthCheckSchema.optional(),
    raw_collection: CountCheckSchema.optional(),
    sterile_collection: CountCheckSchema.optional(),
    raw_collection_has_points: z.boolean().optional(),
    sterile_collection_has_points: z.boolean().optional(),
  })
  .passthrough();

export type TriageResult = z.infer<typeof TriageResultSchema>;
export type IntakeEnvelope = z.infer<typeof IntakeEnvelopeSchema>;
export type KnowledgeArtifact = z.infer<typeof KnowledgeArtifactSchema>;
export type QueryRequest = z.infer<typeof QueryRequestSchema>;
export type QueryHit = z.infer<typeof QueryHitSchema>;
export type QueryResponse = z.infer<typeof QueryResponseSchema>;
export type UploadUrlRequest = z.infer<typeof UploadUrlRequestSchema>;
export type UploadUrlResponse = z.infer<typeof UploadUrlResponseSchema>;
export type IntakeSubmission = z.infer<typeof IntakeSubmissionSchema>;
export type CollectionStats = z.infer<typeof CollectionStatsSchema>;
export type HealthCheck = z.infer<typeof HealthCheckSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
export type StatusResponse = z.infer<typeof StatusResponseSchema>;
export type RagStatus = z.infer<typeof RagStatusSchema>;
export type QueryMode = z.infer<typeof QueryModeSchema>;
export type PortalSession = z.infer<typeof PortalSessionSchema>;
