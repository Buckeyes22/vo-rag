const TABS = [
  { id: "all", label: "All", pipeline: undefined },
  { id: "book_rag", label: "Books", pipeline: "book_rag" },
  { id: "repo_analysis", label: "Repos", pipeline: "repo_analysis" },
  { id: "research_agent", label: "Research", pipeline: "research_agent" },
  { id: "idea_capture", label: "Ideas", pipeline: "idea_capture" },
  { id: "article_extraction", label: "Articles", pipeline: "article_extraction" },
  { id: "tool_evaluation", label: "Tools", pipeline: "tool_evaluation" },
  { id: "url_snapshot", label: "URLs", pipeline: "url_snapshot" },
] as const;

export type CollectionId = (typeof TABS)[number]["id"];

export default function CollectionTabs({
  active,
  onChange,
}: {
  active: CollectionId;
  onChange: (id: CollectionId) => void;
}) {
  return (
    <div role="tablist" className="flex flex-wrap gap-2 border-b border-zinc-800">
      {TABS.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={active === t.id}
          onClick={() => onChange(t.id)}
          className={`px-3 py-2 text-sm ${active === t.id ? "border-b-2 border-blue-500 text-blue-300" : "text-zinc-400"}`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

export { TABS };
