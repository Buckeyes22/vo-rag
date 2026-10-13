import { useState } from "react";
import IngestForm from "@/components/ingest/IngestForm";
import LiveTriage from "@/components/ingest/LiveTriage";
import RecentSubmissions from "@/components/ingest/RecentSubmissions";

export default function Ingest() {
  const [lastId, setLastId] = useState<string | null>(null);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Ingest</h2>
      <span className="sr-only">Ingest</span>

      <IngestForm onSubmitted={setLastId} />
      {lastId && <LiveTriage envelopeId={lastId} />}
      <RecentSubmissions />
    </div>
  );
}
