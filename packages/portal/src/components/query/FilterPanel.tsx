import { useState } from "react";

export default function FilterPanel({
  filters,
  onChange,
}: {
  filters: Record<string, string>;
  onChange: (f: Record<string, string>) => void;
}) {
  const [key, setKey] = useState("");
  const [val, setVal] = useState("");

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2 text-xs">
        {Object.entries(filters).map(([k, v]) => (
          <span key={k} className="rounded bg-zinc-800 px-2 py-1">
            {k}={v}
            <button
              onClick={() => {
                const next = { ...filters };
                delete next[k];
                onChange(next);
              }}
              className="ml-2 text-zinc-400 hover:text-red-400"
              aria-label={`Remove ${k}`}
            >
              ×
            </button>
          </span>
        ))}
      </div>
      <div className="flex gap-2">
        <input
          value={key}
          onChange={(e) => setKey(e.target.value)}
          placeholder="key"
          className="w-32 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs"
          aria-label="Filter key"
        />
        <input
          value={val}
          onChange={(e) => setVal(e.target.value)}
          placeholder="value"
          className="w-40 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs"
          aria-label="Filter value"
        />
        <button
          onClick={() => {
            if (!key) {
              return;
            }
            onChange({ ...filters, [key]: val });
            setKey("");
            setVal("");
          }}
          className="rounded bg-zinc-800 px-2 py-1 text-xs text-zinc-200"
        >
          Add
        </button>
      </div>
    </div>
  );
}
