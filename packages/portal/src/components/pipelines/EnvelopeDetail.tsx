import { useEnvelope } from "@/api/client";

export default function EnvelopeDetail({ id }: { id: string }) {
  const { data: env } = useEnvelope(id);

  if (!env) {
    return <div className="text-zinc-500">Loading envelope…</div>;
  }

  return (
    <section aria-label="Envelope detail" className="space-y-2 rounded border border-zinc-800 p-3">
      <header className="text-sm">
        <div>
          <code>{env.envelope_id}</code>
        </div>
        <div>
          Status: <span className="text-blue-300">{env.status}</span>
          {env.error ? ` — ${env.error}` : ""}
        </div>
      </header>
      {env.triage_result && (
        <div className="text-xs text-zinc-400">
          triage: {env.triage_result.target_pipeline ?? "unassigned"} / {env.triage_result.disposition} /
          priority={env.triage_result.priority} / conf={env.triage_result.confidence.toFixed(2)}
        </div>
      )}
      <div>
        <h4 className="mb-1 text-xs font-semibold text-zinc-400">Raw Content</h4>
        <pre className="max-h-64 overflow-auto rounded bg-zinc-950 p-2 text-[11px] text-zinc-300">
          {JSON.stringify(env.raw_content ?? {}, null, 2)}
        </pre>
      </div>
    </section>
  );
}
