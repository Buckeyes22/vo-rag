import { useState } from "react";
import { Link } from "react-router-dom";
import type { KnowledgeArtifact } from "@/schemas";
import {
  useBookOverview,
  useBookPage,
  useBookSummaries,
} from "@/api/client";
import ImageGallery from "./ImageGallery";

type Tab = "read" | "summaries" | "details";

export default function BookView({ artifact }: { artifact: KnowledgeArtifact }) {
  const sourceId = artifact.content?.source_id as string | undefined;
  const [tab, setTab] = useState<Tab>("read");
  const [page, setPage] = useState(1);

  const overview = useBookOverview(sourceId);
  const pageData = useBookPage(sourceId, page);

  const layers = overview.data?.layers;
  const hasSummaries =
    layers !== undefined &&
    (layers.summary_tree > 0 ||
      layers.raptor > 0 ||
      layers.dsi > 0 ||
      layers.graph_nodes > 0);
  const summaries = useBookSummaries(sourceId, { enabled: hasSummaries });

  const title = artifact.title ?? overview.data?.title ?? artifact.id;
  const author = overview.data?.author;
  const stats = overview.data?.stats;
  const licenseClass = overview.data?.license_class ?? artifact.license_class;

  const queryParams = new URLSearchParams({
    mode: "internal",
    artifact_id: artifact.id,
  });
  if (artifact.qdrant_collection) {
    queryParams.set("cols", artifact.qdrant_collection);
  }

  if (overview.isLoading) {
    return <div className="text-zinc-500">Loading…</div>;
  }
  if (overview.isError) {
    return (
      <div role="alert" className="text-red-400">
        Failed to load book overview.
      </div>
    );
  }

  const tabs: { key: Tab; label: string; hidden?: boolean }[] = [
    { key: "read", label: "Read" },
    { key: "summaries", label: "Summaries", hidden: !hasSummaries },
    { key: "details", label: "Details" },
  ];

  return (
    <div className="space-y-4">
      <header>
        <h2 className="text-xl font-semibold">{title}</h2>
        {author && <div className="text-sm text-zinc-400">{author}</div>}
        <div className="text-xs text-zinc-500">
          {stats && (
            <>
              {stats.text_chunks} text · {stats.figure_chunks} figures
              {stats.pages != null ? ` · ${stats.pages} pp` : ""}
            </>
          )}
        </div>
        {stats && stats.domain_tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1">
            {stats.domain_tags.map((tag) => (
              <span
                key={tag}
                className="rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-300"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
        {licenseClass && (
          <span className="mt-1 inline-block rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400">
            {licenseClass}
          </span>
        )}
        <Link
          to={`/query?${queryParams.toString()}`}
          className="mt-3 inline-block rounded border border-zinc-700 px-3 py-2 text-sm text-zinc-200 hover:bg-zinc-800"
        >
          Query this output
        </Link>
      </header>

      <nav className="flex gap-1 border-b border-zinc-800" role="tablist">
        {tabs.map(
          (t) =>
            !t.hidden && (
              <button
                key={t.key}
                role="tab"
                aria-selected={tab === t.key}
                onClick={() => setTab(t.key)}
                className={`px-3 py-2 text-sm ${
                  tab === t.key
                    ? "border-b-2 border-zinc-200 text-zinc-100"
                    : "text-zinc-500 hover:text-zinc-300"
                }`}
              >
                {t.label}
              </button>
            ),
        )}
      </nav>

      {tab === "read" && (
        <ReadTab
          page={page}
          setPage={setPage}
          pageData={pageData}
        />
      )}
      {tab === "summaries" && summaries.data && (
        <SummariesTab data={summaries.data} />
      )}
      {tab === "summaries" && summaries.isLoading && (
        <div className="text-zinc-500">Loading…</div>
      )}
      {tab === "summaries" && summaries.isError && (
        <div role="alert" className="text-red-400">
          Failed to load summaries.
        </div>
      )}
      {tab === "details" && <DetailsTab artifact={artifact} />}
    </div>
  );
}

function ReadTab({
  page,
  setPage,
  pageData,
}: {
  page: number;
  setPage: (p: number) => void;
  pageData: ReturnType<typeof useBookPage>;
}) {
  if (pageData.isLoading) {
    return <div className="text-zinc-500">Loading…</div>;
  }
  if (pageData.isError) {
    return (
      <div role="alert" className="text-red-400">
        Failed to load page.
      </div>
    );
  }

  const data = pageData.data;
  if (!data) return null;

  const maxPage = data.max_page;
  const hasContent = data.chunks.length > 0;
  let lastSection: string | null = null;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 text-sm text-zinc-400">
        <button
          disabled={page <= 1}
          onClick={() => setPage(page - 1)}
          className="rounded border border-zinc-700 px-2 py-1 hover:bg-zinc-800 disabled:opacity-40 disabled:hover:bg-transparent"
        >
          Prev
        </button>
        <span>
          Page {page}
          {maxPage != null ? ` of ${maxPage}` : ""}
        </span>
        <button
          disabled={maxPage != null && page >= maxPage}
          onClick={() => setPage(page + 1)}
          className="rounded border border-zinc-700 px-2 py-1 hover:bg-zinc-800 disabled:opacity-40 disabled:hover:bg-transparent"
        >
          Next
        </button>
        {maxPage != null && (
          <label className="flex items-center gap-1 ml-2">
            Go to
            <input
              type="number"
              min={1}
              max={maxPage}
              className="w-16 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm text-zinc-200"
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  const v = parseInt(
                    (e.target as HTMLInputElement).value,
                    10,
                  );
                  if (v >= 1 && v <= (maxPage ?? Infinity)) setPage(v);
                }
              }}
            />
          </label>
        )}
      </div>

      {!hasContent && (
        <p className="text-zinc-500 italic">Blank page</p>
      )}

      <div className="max-w-prose space-y-3">
        {data.chunks.map((chunk) => {
          const showHeading =
            chunk.section_title != null && chunk.section_title !== lastSection;
          if (showHeading) lastSection = chunk.section_title;
          return (
            <div key={chunk.chunk_id}>
              {showHeading && (
                <h4 className="text-sm font-semibold text-zinc-300 mb-1">
                  {chunk.section_title}
                </h4>
              )}
              <p className="text-zinc-300 text-sm whitespace-pre-wrap leading-relaxed">
                {chunk.text}
              </p>
              {chunk.has_image && (
                <figure className="mt-2 rounded border border-zinc-700 p-3">
                  <div className="text-xs font-semibold text-zinc-400 mb-1">
                    Figure
                  </div>
                  <figcaption className="text-xs text-zinc-400">
                    {chunk.vlm_description ?? "Image available after figure re-ingest"}
                  </figcaption>
                </figure>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SummariesTab({
  data,
}: {
  data: NonNullable<ReturnType<typeof useBookSummaries>["data"]>;
}) {
  return (
    <div className="space-y-6">
      {data.dsi && <DsiSection dsi={data.dsi} />}
      {data.summary_tree && <SummaryTreeSection tree={data.summary_tree} />}
      {data.graph && <GraphSection graph={data.graph} />}
      {data.raptor && <RaptorSection raptor={data.raptor} />}
    </div>
  );
}

function ChipList({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1">
      {items.map((item) => (
        <span
          key={item}
          className="rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-300"
        >
          {item}
        </span>
      ))}
    </div>
  );
}

function DsiSection({
  dsi,
}: {
  dsi: NonNullable<NonNullable<ReturnType<typeof useBookSummaries>["data"]>["dsi"]>;
}) {
  return (
    <section className="rounded border border-zinc-800 p-4 space-y-3">
      <h3 className="text-sm font-semibold text-zinc-200">About this book</h3>
      {dsi.chapter_outline && (
        <p className="text-sm text-zinc-300">{dsi.chapter_outline}</p>
      )}
      {dsi.key_concepts.length > 0 && (
        <div>
          <div className="text-xs text-zinc-500 mb-1">Key concepts</div>
          <ChipList items={dsi.key_concepts} />
        </div>
      )}
      {dsi.key_methods.length > 0 && (
        <div>
          <div className="text-xs text-zinc-500 mb-1">Key methods</div>
          <ChipList items={dsi.key_methods} />
        </div>
      )}
      {dsi.key_equations_or_models.length > 0 && (
        <div>
          <div className="text-xs text-zinc-500 mb-1">
            Key equations / models
          </div>
          <ChipList items={dsi.key_equations_or_models} />
        </div>
      )}
      {dsi.prerequisite_topics.length > 0 && (
        <div>
          <div className="text-xs text-zinc-500 mb-1">Prerequisites</div>
          <ChipList items={dsi.prerequisite_topics} />
        </div>
      )}
      {dsi.figure_heavy_topics.length > 0 && (
        <div>
          <div className="text-xs text-zinc-500 mb-1">
            Figure-heavy topics
          </div>
          <ChipList items={dsi.figure_heavy_topics} />
        </div>
      )}
    </section>
  );
}

function SummaryTreeSection({
  tree,
}: {
  tree: NonNullable<NonNullable<ReturnType<typeof useBookSummaries>["data"]>["summary_tree"]>;
}) {
  return (
    <section className="rounded border border-zinc-800 p-4 space-y-3">
      <h3 className="text-sm font-semibold text-zinc-200">
        Summary tree
      </h3>
      {tree.book.length > 0 && (
        <div>
          <div className="text-xs text-zinc-500 mb-1">Book-level</div>
          <ul className="list-disc list-inside text-sm text-zinc-300 space-y-1">
            {tree.book.flatMap((entry, i) =>
              entry.key_points.map((pt, j) => (
                <li key={`${i}-${j}`}>{pt}</li>
              )),
            )}
          </ul>
        </div>
      )}
      {tree.chapter.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-zinc-500 mb-1">Chapters</div>
          {tree.chapter.map((ch, i) => (
            <details key={i} className="rounded border border-zinc-800">
              <summary className="cursor-pointer px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-800">
                {ch.display_path ?? `Chapter ${i + 1}`}
              </summary>
              <div className="px-3 pb-2">
                <ul className="list-disc list-inside text-sm text-zinc-400 space-y-1">
                  {ch.key_points.map((pt, j) => (
                    <li key={j}>{pt}</li>
                  ))}
                </ul>
              </div>
            </details>
          ))}
        </div>
      )}
      <p className="text-xs text-zinc-500">
        +{tree.section_count} section summaries indexed
      </p>
    </section>
  );
}

function GraphSection({
  graph,
}: {
  graph: NonNullable<NonNullable<ReturnType<typeof useBookSummaries>["data"]>["graph"]>;
}) {
  return (
    <section className="rounded border border-zinc-800 p-4 space-y-3">
      <h3 className="text-sm font-semibold text-zinc-200">Knowledge graph</h3>
      {graph.top_nodes.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {graph.top_nodes.map((node) => (
            <span
              key={node.label}
              className="rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-300"
            >
              {node.label} ({node.mentions})
            </span>
          ))}
        </div>
      )}
      <div className="text-xs text-zinc-500">
        {graph.node_count} total nodes
      </div>
    </section>
  );
}

function RaptorSection({
  raptor,
}: {
  raptor: NonNullable<NonNullable<ReturnType<typeof useBookSummaries>["data"]>["raptor"]>;
}) {
  return (
    <section className="rounded border border-zinc-800 p-4">
      <p className="text-sm text-zinc-300">
        {raptor.node_count} hierarchical cluster nodes indexed
      </p>
    </section>
  );
}

function DetailsTab({ artifact }: { artifact: KnowledgeArtifact }) {
  return (
    <div className="space-y-4">
      <ImageGallery artifactId={artifact.id} />
      <pre className="rounded bg-zinc-900 p-3 text-xs overflow-auto text-zinc-300">
        {JSON.stringify(artifact, null, 2)}
      </pre>
    </div>
  );
}
