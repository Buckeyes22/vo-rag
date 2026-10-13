import { COLLECTIONS } from "@/project/vorag";

export default function CollectionPicker({
  selected,
  onChange,
}: {
  selected: string[];
  onChange: (v: string[]) => void;
}) {
  return (
    <div aria-label="Collections" className="flex flex-wrap gap-2">
      {COLLECTIONS.map((c) => {
        const on = selected.includes(c);
        return (
          <button
            key={c}
            onClick={() => onChange(on ? selected.filter((x) => x !== c) : [...selected, c])}
            aria-pressed={on}
            className={`rounded border px-2 py-1 text-xs ${on ? "border-blue-500 bg-blue-600/20 text-blue-200" : "border-zinc-700 text-zinc-400"}`}
          >
            {c}
          </button>
        );
      })}
    </div>
  );
}
