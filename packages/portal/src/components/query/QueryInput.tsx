export default function QueryInput({
  value,
  onChange,
  topK,
  onTopK,
  rerank,
  onRerank,
  onSubmit,
}: {
  value: string;
  onChange: (v: string) => void;
  topK: number;
  onTopK: (v: number) => void;
  rerank: boolean;
  onRerank: (v: boolean) => void;
  onSubmit: () => void;
}) {
  return (
    <div className="space-y-2">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            onSubmit();
          }
        }}
        rows={3}
        placeholder="Ask anything… (Ctrl/Cmd+Enter to submit)"
        aria-label="Query"
        className="w-full rounded border border-zinc-700 bg-zinc-900 px-3 py-2"
      />
      <div className="flex items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          top_k
          <input
            type="range"
            min={1}
            max={200}
            value={topK}
            onChange={(e) => onTopK(Number(e.target.value))}
            aria-label="top_k"
          />
          <span>{topK}</span>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={rerank}
            onChange={(e) => onRerank(e.target.checked)}
            aria-label="Rerank"
          />
          rerank
        </label>
        <button onClick={onSubmit} className="rounded bg-blue-600 px-4 py-2 text-white">
          Search
        </button>
      </div>
    </div>
  );
}
