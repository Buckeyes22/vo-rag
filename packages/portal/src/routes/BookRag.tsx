import MetricCard from "@/components/dashboard/MetricCard";
import LiveTriage from "@/components/ingest/LiveTriage";
import { useCollectionStats, useEnvelopes, useRagStatus } from "@/api/client";
import { BOOK_RAG_STAGES } from "@/project/vorag";
import { healthTone, normalizeHealth } from "@/utils/health";

function countFor(collections: Record<string, { points: number }>, name: string) {
  return collections[name]?.points ?? 0;
}

export default function BookRag() {
  const rag = useRagStatus();
  const collections = useCollectionStats();
  const processing = useEnvelopes({ limit: 5, status: "processing" });
  const collectionMap = collections.data?.collections ?? {};
  const rawPoints = countFor(collectionMap, "vo_rag_books_raw_v1");
  const sterilePoints = countFor(collectionMap, "vo_rag_books_sterile_v1");
  const inFlightEnvelope =
    processing.data?.items.find((envelope) => envelope.target_pipeline === "book_rag") ??
    processing.data?.items[0] ??
    null;
  const checks = [
    ["DB Pool", rag.data?.db_pool],
    ["Qdrant", rag.data?.qdrant],
    ["Embedding", rag.data?.embedding_server],
  ] as const;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold">Book RAG</h2>
          <div className="mt-1 text-sm text-zinc-400">GLM-OCR, VLM composites, raw and sterile retrieval.</div>
        </div>
        <span
          className={`rounded px-3 py-1 text-sm ${
            rag.data?.ready ? "bg-emerald-600 text-white" : "bg-amber-600/80 text-white"
          }`}
        >
          {rag.data?.ready ? "ready" : "pending"}
        </span>
      </header>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Raw Points" value={rawPoints.toLocaleString()} detail="vo_rag_books_raw_v1" />
        <MetricCard label="Sterile Points" value={sterilePoints.toLocaleString()} detail="vo_rag_books_sterile_v1" />
        <MetricCard
          label="Sterile Populated"
          value={rag.data?.sterile_collection_has_points ? "yes" : "no"}
          detail="sterile collection readiness"
        />
        <MetricCard
          label="Raw Populated"
          value={rag.data?.raw_collection_has_points ? "yes" : "no"}
        />
      </div>

      <section aria-label="Book RAG readiness" className="grid grid-cols-1 gap-2 md:grid-cols-3">
        {checks.map(([label, value]) => {
          const status = normalizeHealth(value);
          return (
            <div key={label} className="rounded border border-zinc-800 p-3">
              <div className="flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${healthTone(status)}`} />
                <span className="text-sm font-medium text-zinc-200">{label}</span>
                <span className="ml-auto text-xs text-zinc-500">{status}</span>
              </div>
            </div>
          );
        })}
      </section>

      {inFlightEnvelope && (
        <section aria-label="In-flight book ingest" className="space-y-3">
          <div>
            <h3 className="text-lg font-semibold">In-flight book ingest</h3>
            <div className="mt-1 text-sm text-zinc-500">
              Most recent processing envelope <code>{inFlightEnvelope.envelope_id}</code>
            </div>
          </div>
          <LiveTriage envelopeId={inFlightEnvelope.envelope_id} />
        </section>
      )}

      <section aria-label="Book RAG stages" className="space-y-3">
        <h3 className="text-lg font-semibold">Workflow Stages</h3>
        <div className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-5">
          {BOOK_RAG_STAGES.map((stage, index) => (
            <div key={stage} className="rounded border border-zinc-800 p-3">
              <div className="text-xs text-zinc-500">{String(index + 1).padStart(2, "0")}</div>
              <div className="mt-1 text-sm text-zinc-200">{stage}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
