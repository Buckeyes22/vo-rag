import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryOptions,
} from "@tanstack/react-query";
import { z } from "zod";
import { useAppStore } from "@/stores/app";
import {
  CollectionStatsSchema,
  HealthResponseSchema,
  IntakeSubmissionSchema,
  IntakeEnvelopeSchema,
  KnowledgeArtifactSchema,
  PortalSessionSchema,
  QueryRequestSchema,
  QueryResponseSchema,
  RagStatusSchema,
  StatusResponseSchema,
  UploadUrlRequestSchema,
  UploadUrlResponseSchema,
  type CollectionStats,
  type HealthResponse,
  type IntakeSubmission,
  type IntakeEnvelope,
  type KnowledgeArtifact,
  type PortalSession,
  type QueryRequest,
  type QueryResponse,
  type RagStatus,
  type StatusResponse,
} from "@/schemas";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Typed JSON fetch + zod validation. Exported so sibling API modules
 *  share the auth/CSRF/error handling. */
export function useApi() {
  const csrfToken = useAppStore((s) => s.csrfToken);

  return async function api<T>(
    path: string,
    init: RequestInit,
    schema: z.ZodType<T, z.ZodTypeDef, unknown>,
  ): Promise<T> {
    const headers: Record<string, string> = {
      "X-VO-RAG-Caller": "portal",
      ...(csrfToken && !["GET", "HEAD", "OPTIONS"].includes(init.method ?? "GET")
        ? { "X-CSRF-Token": csrfToken }
        : {}),
      ...((init.headers as Record<string, string> | undefined) ?? {}),
    };
    if (init.body !== undefined && !("Content-Type" in headers)) {
      headers["Content-Type"] = "application/json";
    }

    const res = await fetch(path, {
      ...init,
      credentials: "same-origin",
      headers,
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new ApiError(res.status, `${res.status} ${res.statusText} ${body}`.trim());
    }

    const json: unknown = await res.json();
    return schema.parse(json);
  };
}

export function usePortalSession() {
  const api = useApi();
  return useQuery({
    queryKey: ["portal-session"],
    queryFn: () => api("/api/v1/portal/session", { method: "GET" }, PortalSessionSchema),
    staleTime: 5 * 60_000,
    refetchInterval: 10 * 60_000,
  });
}

export function useHealth(opts?: Partial<UseQueryOptions<HealthResponse>>) {
  const api = useApi();
  return useQuery({
    queryKey: ["health"],
    queryFn: () => api("/health", { method: "GET" }, HealthResponseSchema),
    refetchInterval: 15_000,
    ...opts,
  });
}

export function useStatus() {
  const api = useApi();
  return useQuery({
    queryKey: ["status"],
    queryFn: () => api("/status", { method: "GET" }, StatusResponseSchema),
    refetchInterval: 15_000,
  });
}

export function useCollectionStats() {
  const api = useApi();
  return useQuery({
    queryKey: ["collections-stats"],
    queryFn: () =>
      api(
        "/api/v1/collections/stats",
        { method: "GET" },
        CollectionStatsSchema,
      ),
  });
}

export function useRagStatus() {
  const api = useApi();
  return useQuery({
    queryKey: ["rag-status"],
    queryFn: () => api("/api/v1/rag/status", { method: "GET" }, RagStatusSchema),
    refetchInterval: 15_000,
  });
}

export function useArtifacts(params: { pipeline?: string; limit?: number } = {}) {
  const api = useApi();

  const ArtifactStatsSchema = z.object({
    text_chunks: z.number(),
    figure_chunks: z.number(),
    pages: z.number().nullable().optional(),
    domain_tags: z.array(z.string()).default([]),
  });

  const ArtifactLayersSchema = z.object({
    summary_tree: z.number().default(0),
    raptor: z.number().default(0),
    dsi: z.number().default(0),
    graph_nodes: z.number().default(0),
  });

  const ArtifactListItemSchema = KnowledgeArtifactSchema.extend({
    content: z
      .object({
        stats: ArtifactStatsSchema.optional(),
        layers: ArtifactLayersSchema.optional(),
      })
      .passthrough()
      .optional(),
  });

  const ArtifactListSchema = z.object({
    items: z.array(ArtifactListItemSchema),
    next_cursor: z.string().nullable().optional(),
  });

  return useInfiniteQuery({
    queryKey: ["artifacts", params],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) => {
      const qs = new URLSearchParams();
      if (params.pipeline) {
        qs.set("pipeline", params.pipeline);
      }
      if (params.limit !== undefined) {
        qs.set("limit", String(params.limit));
      }
      if (pageParam) {
        qs.set("cursor", pageParam);
      }
      return api(`/api/v1/artifacts?${qs}`, { method: "GET" }, ArtifactListSchema);
    },
    getNextPageParam: (last) => last.next_cursor ?? null,
  });
}

export function useArtifact(id: string | undefined) {
  const api = useApi();
  return useQuery({
    queryKey: ["artifact", id],
    queryFn: () =>
      api(
        `/api/v1/artifacts/${encodeURIComponent(id!)}`,
        { method: "GET" },
        KnowledgeArtifactSchema,
      ),
    enabled: Boolean(id),
  });
}

const ArtifactImageSchema = z.object({
  id: z.string(),
  url: z.string(),
  caption: z.string().nullable().optional(),
});

const ArtifactImagesResponseSchema = z.object({
  items: z.array(ArtifactImageSchema),
});

export function useArtifactImages(artifactId: string | undefined) {
  const api = useApi();
  return useQuery({
    queryKey: ["artifact-images", artifactId],
    queryFn: () =>
      api(
        `/api/v1/artifacts/${encodeURIComponent(artifactId!)}/images`,
        { method: "GET" },
        ArtifactImagesResponseSchema,
      ),
    enabled: Boolean(artifactId),
  });
}

export function useEnvelopes(params: { limit?: number; status?: string } = {}) {
  const api = useApi();
  const qs = new URLSearchParams();
  if (params.limit !== undefined) {
    qs.set("limit", String(params.limit));
  }
  if (params.status) {
    qs.set("status", params.status);
  }

  const EnvelopeListSchema = z.object({
    items: z.array(IntakeEnvelopeSchema),
    next_cursor: z.string().nullable().optional(),
  });

  return useQuery({
    queryKey: ["envelopes", params],
    queryFn: () => api(`/api/v1/envelopes?${qs}`, { method: "GET" }, EnvelopeListSchema),
    refetchInterval: 5_000,
  });
}

export function useEnvelope(id: string | undefined) {
  const api = useApi();
  return useQuery({
    queryKey: ["envelope", id],
    queryFn: () =>
      api(
        `/api/v1/envelopes/${encodeURIComponent(id!)}`,
        { method: "GET" },
        IntakeEnvelopeSchema,
      ),
    enabled: Boolean(id),
    refetchInterval: 1_500,
  });
}

export function useUploadUrl() {
  const api = useApi();
  return useMutation({
    mutationFn: (meta: { filename: string; content_type: string; size_bytes: number }) =>
      api(
        "/api/v1/intake/upload-url",
        {
          method: "POST",
          body: JSON.stringify(UploadUrlRequestSchema.parse(meta)),
        },
        UploadUrlResponseSchema,
      ),
  });
}

export function useIntakeSubmit() {
  const api = useApi();
  const qc = useQueryClient();

  return useMutation({
    mutationFn: (body: IntakeSubmission) =>
      api(
        "/api/v1/intake",
        { method: "POST", body: JSON.stringify(IntakeSubmissionSchema.parse(body)) },
        IntakeEnvelopeSchema,
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["envelopes"] }),
  });
}

export function useQueryRun() {
  const api = useApi();
  return useMutation({
    mutationFn: (req: QueryRequest) =>
      api(
        "/api/v1/query",
        {
          method: "POST",
          body: JSON.stringify(QueryRequestSchema.parse(req)),
        },
        QueryResponseSchema,
      ),
  });
}

const BookOverviewStatsSchema = z.object({
  text_chunks: z.number(),
  figure_chunks: z.number(),
  pages: z.number().nullable(),
  domain_tags: z.array(z.string()),
});

const BookOverviewLayersSchema = z.object({
  summary_tree: z.number(),
  raptor: z.number(),
  dsi: z.number(),
  graph_nodes: z.number(),
});

const BookOverviewSchema = z.object({
  source_id: z.string(),
  title: z.string().nullable(),
  author: z.string().nullable(),
  license_class: z.string().nullable(),
  artifact_id: z.string().nullable(),
  stats: BookOverviewStatsSchema.nullable(),
  layers: BookOverviewLayersSchema,
});

export function useBookOverview(sourceId: string | undefined) {
  const api = useApi();
  return useQuery({
    queryKey: ["book-overview", sourceId],
    queryFn: () =>
      api(
        `/api/v1/books/${encodeURIComponent(sourceId!)}`,
        { method: "GET" },
        BookOverviewSchema,
      ),
    enabled: Boolean(sourceId),
  });
}

const BookPageChunkSchema = z.object({
  chunk_id: z.string(),
  section_title: z.string().nullable(),
  text: z.string(),
  has_image: z.boolean(),
  image_id: z.string().nullable(),
  vlm_description: z.string().nullable(),
  token_count: z.number().nullable(),
});

const BookPageSchema = z.object({
  page: z.number(),
  max_page: z.number().nullable(),
  chunks: z.array(BookPageChunkSchema),
});

export function useBookPage(sourceId: string | undefined, page: number) {
  const api = useApi();
  return useQuery({
    queryKey: ["book-page", sourceId, page],
    queryFn: () =>
      api(
        `/api/v1/books/${encodeURIComponent(sourceId!)}/pages/${page}`,
        { method: "GET" },
        BookPageSchema,
      ),
    enabled: Boolean(sourceId) && page >= 1,
  });
}

const BookDsiSchema = z.object({
  chapter_outline: z.string().nullable(),
  key_concepts: z.array(z.string()),
  key_methods: z.array(z.string()),
  key_equations_or_models: z.array(z.string()),
  prerequisite_topics: z.array(z.string()),
  figure_heavy_topics: z.array(z.string()),
}).nullable();

const BookSummaryTreeEntrySchema = z.object({
  display_path: z.string().nullable(),
  key_points: z.array(z.string()),
});

const BookSummaryTreeSchema = z.object({
  book: z.array(BookSummaryTreeEntrySchema),
  chapter: z.array(BookSummaryTreeEntrySchema),
  section_count: z.number(),
}).nullable();

const BookGraphSchema = z.object({
  node_count: z.number(),
  top_nodes: z.array(z.object({
    label: z.string(),
    node_type: z.string(),
    mentions: z.number(),
  })),
}).nullable();

const BookRaptorSchema = z.object({
  node_count: z.number(),
}).nullable();

const BookSummariesSchema = z.object({
  dsi: BookDsiSchema,
  summary_tree: BookSummaryTreeSchema,
  graph: BookGraphSchema,
  raptor: BookRaptorSchema,
});

export function useBookSummaries(sourceId: string | undefined, opts: { enabled: boolean }) {
  const api = useApi();
  return useQuery({
    queryKey: ["book-summaries", sourceId],
    queryFn: () =>
      api(
        `/api/v1/books/${encodeURIComponent(sourceId!)}/summaries`,
        { method: "GET" },
        BookSummariesSchema,
      ),
    enabled: Boolean(sourceId) && opts.enabled,
  });
}

export type {
  HealthResponse,
  StatusResponse,
  CollectionStats,
  IntakeEnvelope,
  KnowledgeArtifact,
  QueryResponse,
  QueryRequest,
  RagStatus,
  PortalSession,
};
