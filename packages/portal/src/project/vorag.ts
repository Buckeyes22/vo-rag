export type FacetStatus = "wired" | "partial" | "external" | "not_wired";

export type ProjectFacet = {
  id: string;
  label: string;
  route: string;
  status: FacetStatus;
  surface: string;
  endpoints: string[];
  signals: string[];
};

export const PIPELINES = [
  {
    id: "book_rag",
    label: "Book RAG",
    input: "PDF, EPUB, MOBI",
    collection: "vo_rag_books_raw_v1 / vo_rag_books_sterile_v1",
  },
] as const;

export const COLLECTIONS = ["vo_rag_books_raw_v1", "vo_rag_books_sterile_v1"] as const;

export const PROJECT_FACETS: ProjectFacet[] = [
  {
    id: "intake",
    label: "Intake",
    route: "/ingest",
    status: "wired",
    surface: "File, URL, text, idea, API",
    endpoints: ["POST /api/v1/intake", "POST /api/v1/intake/upload-url"],
    signals: ["recent envelopes", "triage result", "needs-human queue"],
  },
  {
    id: "knowledge",
    label: "Knowledge Library",
    route: "/library",
    status: "wired",
    surface: "Artifacts, collections, provenance, image attachments",
    endpoints: ["GET /api/v1/artifacts", "GET /api/v1/artifacts/{id}", "GET /api/v1/collections/stats"],
    signals: ["artifact count", "collection point counts", "license class"],
  },
  {
    id: "query",
    label: "Retrieval",
    route: "/query",
    status: "wired",
    surface: "Hybrid retrieval, modes, filters, rerank control",
    endpoints: ["POST /api/v1/query", "POST /api/v1/answer"],
    signals: ["result scores", "source payloads", "compliance mode"],
  },
  {
    id: "pipelines",
    label: "Pipelines",
    route: "/pipelines",
    status: "wired",
    surface: "Envelope queue and active work",
    endpoints: ["GET /api/v1/envelopes", "GET /api/v1/envelopes/{id}", "GET /status"],
    signals: ["queue depth", "status counts", "raw content"],
  },
  {
    id: "book_rag",
    label: "Book RAG",
    route: "/book-rag",
    status: "partial",
    surface: "GLM-OCR path, raw/sterile collections, artifacts, readiness",
    endpoints: ["GET /api/v1/rag/status", "GET /api/v1/collections/stats"],
    signals: ["raw points", "sterile points", "artifact count", "GLM-OCR health"],
  },
  {
    id: "system",
    label: "System",
    route: "/system",
    status: "wired",
    surface: "Core services, readiness, external consoles",
    endpoints: ["GET /health", "GET /status", "GET /api/v1/rag/status"],
    signals: ["service health", "queue depth", "RAG readiness"],
  },
  {
    id: "ops",
    label: "Ops",
    route: "/ops",
    status: "partial",
    surface: "Fleet, deployment targets, docs, readiness artifacts",
    endpoints: ["localhost portal", "Grafana", "Hatchet", "Langfuse"],
    signals: ["deployment state", "readiness docs", "service links"],
  },
];

export const BOOK_RAG_STAGES = [
  "register_source",
  "parse",
  "visual",
  "chunk_embed",
  "sterilize",
  "register_artifact",
  "validate",
  "retrieval_smoke",
] as const;

export const FLEET_NODES = [
  { name: "API host", role: "vo-rag API + Qdrant (vo-rag-qdrant :6400) + ingest workers + figure store (/data/vo-rag/figures)" },
  { name: "Inference host", role: "BGE-M3 embed + rerank (:8501)" },
  { name: "RunPod serverless", role: "text/DSI/VLM generation endpoints" },
  { name: "File share", role: "book source share and figure backup target" },
] as const;
