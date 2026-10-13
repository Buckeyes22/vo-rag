export default function ArtifactFilters({
  search,
  onSearch,
}: {
  search: string;
  onSearch: (v: string) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        value={search}
        onChange={(e) => onSearch(e.target.value)}
        placeholder="Search within list..."
        aria-label="Search artifacts"
        className="rounded bg-zinc-900 border border-zinc-700 px-3 py-2 text-sm w-full max-w-sm"
      />
    </div>
  );
}
