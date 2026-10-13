import { Link } from "react-router-dom";
import { useArtifacts } from "@/api/client";

export default function ArtifactList({ pipeline }: { pipeline: string | undefined }) {
  const limit = 50;
  const params = pipeline ? { pipeline, limit } : { limit };
  const { data, isLoading, isError, fetchNextPage, hasNextPage, isFetchingNextPage } =
    useArtifacts(params);
  if (isLoading) {
    return <div className="text-zinc-500 p-3">Loading…</div>;
  }
  if (isError) {
    return (
      <div role="alert" className="text-red-400 p-3">
        Failed to load artifacts.
      </div>
    );
  }
  const items = data?.pages.flatMap((p) => p.items) ?? [];
  if (items.length === 0) {
    return <div className="text-zinc-500 p-3">No artifacts.</div>;
  }
  return (
    <div>
      <ul className="divide-y divide-zinc-800 rounded border border-zinc-800">
        {items.map((a) => {
          const stats = a.content?.stats;
          const statsLine = stats
            ? [
                `${stats.text_chunks} text`,
                ...(stats.figure_chunks > 0 ? [`${stats.figure_chunks} figures`] : []),
                ...(stats.pages !== null && stats.pages !== undefined ? [`${stats.pages} pp`] : []),
                ...(stats.domain_tags.length ? [stats.domain_tags.slice(0, 3).join(", ")] : []),
              ].join(" · ")
            : undefined;
          const layers = a.content?.layers;
          const layerChips = layers
            ? ([
                [layers.dsi, "DSI", "border-emerald-800 text-emerald-400"],
                [layers.summary_tree, "Summaries", "border-sky-800 text-sky-400"],
                [layers.graph_nodes, "Graph", "border-violet-800 text-violet-400"],
                [layers.raptor, "RAPTOR", "border-amber-800 text-amber-400"],
              ] as const).filter(([count]) => count > 0)
            : [];

          return (
            <li key={a.id} className="p-3">
              <Link to={`/library/${encodeURIComponent(a.id)}`} className="text-blue-300 hover:underline">
                {a.title ?? a.id}
              </Link>
              <div className="text-xs text-zinc-500">
                {statsLine ?? (
                  <>
                    {a.pipeline} · {a.created_at}
                    {a.priority ? ` · ${a.priority}` : ""}
                    {a.tags && a.tags.length ? ` · ${a.tags.map((t) => `#${t}`).join(" ")}` : ""}
                  </>
                )}
              </div>
              {layerChips.length > 0 ? (
                <div className="mt-1 flex flex-wrap gap-1">
                  {layerChips.map(([, label, cls]) => (
                    <span key={label} className={`rounded border px-1.5 py-0.5 text-[10px] ${cls}`}>
                      {label}
                    </span>
                  ))}
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
      {hasNextPage ? (
        <button
          type="button"
          onClick={() => void fetchNextPage()}
          disabled={isFetchingNextPage}
          className="mt-3 w-full rounded border border-zinc-700 px-3 py-2 text-sm text-zinc-200 hover:bg-zinc-800 disabled:opacity-50"
        >
          {isFetchingNextPage ? "Loading…" : "Load more"}
        </button>
      ) : null}
    </div>
  );
}
