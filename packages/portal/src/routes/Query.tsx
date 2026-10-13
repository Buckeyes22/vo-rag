import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import ModeSelector from "@/components/query/ModeSelector";
import CollectionPicker from "@/components/query/CollectionPicker";
import FilterPanel from "@/components/query/FilterPanel";
import QueryInput from "@/components/query/QueryInput";
import ResultsList from "@/components/query/ResultsList";
import { useQueryRun } from "@/api/client";
import { useAppStore } from "@/stores/app";
import type { QueryHit } from "@/schemas";

export default function Query() {
  const [params, setParams] = useSearchParams();
  const setMode = useAppStore((s) => s.setQueryMode);
  const mode = useAppStore((s) => s.queryMode);

  const [q, setQ] = useState(params.get("q") ?? "");
  const [cols, setCols] = useState<string[]>(() => {
    const raw = params.get("cols");
    return raw ? raw.split(",").filter(Boolean) : [];
  });
  const [filters, setFilters] = useState<Record<string, string>>(() => {
    const entries: Record<string, string> = {};
    for (const key of ["artifact_id", "source_id", "license_class", "pipeline"]) {
      const value = params.get(key);
      if (value) {
        entries[key] = value;
      }
    }
    return entries;
  });
  const [topK, setTopK] = useState(20);
  const [rerank, setRerank] = useState(true);
  const [hits, setHits] = useState<QueryHit[]>([]);
  const run = useQueryRun();

  useEffect(() => {
    const urlMode = params.get("mode");
    if (urlMode === "internal") {
      setMode(urlMode);
    }
  }, [params, setMode]);

  const filtersValid = useMemo(() => filters, [filters]);

  async function onSubmit() {
    setParams({
      q,
      mode,
      ...(cols.length ? { cols: cols.join(",") } : {}),
    });
    const resp = await run.mutateAsync({
      query: q,
      ...(cols.length ? { collections: cols } : {}),
      mode,
      top_k: Math.min(Math.max(topK, 1), 200),
      rerank,
      ...(Object.keys(filtersValid).length ? { filters: filtersValid } : {}),
    });
    setHits(resp.results);
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold">Query</h2>
        <ModeSelector />
      </div>
      <CollectionPicker selected={cols} onChange={setCols} />
      <FilterPanel filters={filters} onChange={setFilters} />
      <QueryInput
        value={q}
        onChange={setQ}
        topK={topK}
        onTopK={setTopK}
        rerank={rerank}
        onRerank={setRerank}
        onSubmit={onSubmit}
      />
      {run.isError && <div role="alert" className="text-red-400">{String(run.error)}</div>}
      <ResultsList hits={hits} mode={mode} />
    </div>
  );
}
