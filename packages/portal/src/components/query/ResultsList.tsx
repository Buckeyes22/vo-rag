import type { QueryHit, QueryMode } from "@/schemas";
import ResultCard from "./ResultCard";

export default function ResultsList({ hits, mode }: { hits: QueryHit[]; mode: QueryMode }) {
  if (hits.length === 0) {
    return <div className="text-zinc-500">No results.</div>;
  }

  return (
    <div className="space-y-3">
      {hits.map((h, i) => (
        <ResultCard key={h.id ?? `${h.collection}-${i}`} hit={h} rank={i + 1} mode={mode} />
      ))}
    </div>
  );
}
