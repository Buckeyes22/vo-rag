import FacetGrid from "@/components/dashboard/FacetGrid";
import MetricCard from "@/components/dashboard/MetricCard";
import StatusBadge from "@/components/dashboard/StatusBadge";
import { useCollectionStats, useHealth, useRagStatus, useStatus } from "@/api/client";
import { PIPELINES, PROJECT_FACETS } from "@/project/vorag";
import { normalizeHealth } from "@/utils/health";

export default function Overview() {
  const health = useHealth();
  const status = useStatus();
  const collections = useCollectionStats();
  const rag = useRagStatus();

  const collectionRows = Object.entries(collections.data?.collections ?? {});
  const totalPoints = collectionRows.reduce((sum, [, collection]) => sum + collection.points, 0);
  const queueDepth = status.data?.queue_depth ?? 0;
  const serviceStates = Object.values(status.data?.services ?? {}).map((value) =>
    normalizeHealth(value),
  );
  const downServices = serviceStates.filter((state) => state === "down").length;
  const wiredFacets = PROJECT_FACETS.filter((facet) => facet.status === "wired").length;
  const healthStatus = health.data?.status ?? "unknown";

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold">Overview</h2>
          <div className="mt-1 text-sm text-zinc-400">
            Universal intake, retrieval, pipeline, compliance, cloud, and training surface.
          </div>
        </div>
        <StatusBadge status={downServices > 0 ? "partial" : "wired"} />
      </header>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Queue Depth" value={queueDepth} detail="received through processing" />
        <MetricCard label="Collections" value={collectionRows.length} detail="Qdrant collections visible" />
        <MetricCard label="Vector Points" value={totalPoints.toLocaleString()} detail="across known collections" />
        <MetricCard
          label="API Health"
          value={healthStatus}
          detail={health.data?.service ?? "vo-rag"}
        />
        <MetricCard
          label="RAG Ready"
          value={rag.data?.ready ? "yes" : "no"}
          detail={rag.data?.raw_collection_has_points ? "raw collection populated" : "raw readiness pending"}
        />
      </div>

      <section aria-label="Pipeline collection map" className="space-y-3">
        <h3 className="text-lg font-semibold text-zinc-100">Pipeline Map</h3>
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 xl:grid-cols-3">
          {PIPELINES.map((pipeline) => (
            <article key={pipeline.id} className="rounded border border-zinc-800 p-3">
              <div className="font-medium text-zinc-100">{pipeline.label}</div>
              <div className="mt-1 text-sm text-zinc-400">{pipeline.input}</div>
              <div className="mt-2 text-xs text-zinc-500">{pipeline.collection}</div>
            </article>
          ))}
        </div>
      </section>

      <section aria-label="Coverage summary" className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <MetricCard label="Wired Facets" value={`${wiredFacets}/${PROJECT_FACETS.length}`} />
        <MetricCard label="Partial Facets" value={PROJECT_FACETS.filter((f) => f.status === "partial").length} />
        <MetricCard label="External Facets" value={PROJECT_FACETS.filter((f) => f.status === "external").length} />
      </section>

      <FacetGrid facets={PROJECT_FACETS} />
    </div>
  );
}
