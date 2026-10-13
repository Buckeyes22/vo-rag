import { useStatus } from "@/api/client";
import { healthTone, normalizeHealth } from "@/utils/health";

const LABELS: Record<string, string> = {
  db_pool: "DB Pool",
  postgres: "Postgres",
  qdrant: "Qdrant",
  hatchet: "Hatchet",
  embedding: "Embedding",
  embedding_server: "Embedding",
  glm_ocr: "GLM-OCR",
  migrations: "Migrations",
  vlm: "VLM",
  garage: "Garage",
};

export default function ServiceHealthGrid() {
  const { data } = useStatus();
  const entries = Object.entries(data?.services ?? {});

  return (
    <section aria-label="Service health" className="grid grid-cols-2 md:grid-cols-3 gap-2">
      {entries.map(([k, v]) => {
        const status = normalizeHealth(v);
        return (
        <div key={k} className="rounded border border-zinc-800 p-3 flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${healthTone(status)}`} />
          <span className="text-sm text-zinc-200">{LABELS[k] ?? k}</span>
          <span className="ml-auto text-xs text-zinc-500">{status}</span>
        </div>
        );
      })}
    </section>
  );
}
