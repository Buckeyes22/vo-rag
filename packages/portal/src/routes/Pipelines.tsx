import { useState } from "react";
import ActiveRuns from "@/components/pipelines/ActiveRuns";
import EnvelopeList from "@/components/pipelines/EnvelopeList";
import EnvelopeDetail from "@/components/pipelines/EnvelopeDetail";

export default function Pipelines() {
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Pipelines</h2>
      <ActiveRuns />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <EnvelopeList onSelect={setSelected} />
        {selected && <EnvelopeDetail id={selected} />}
      </div>
    </div>
  );
}
