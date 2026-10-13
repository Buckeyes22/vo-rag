import { useEnvelopes } from "@/api/client";

export default function RecentSubmissions() {
  const { data } = useEnvelopes({ limit: 10 });
  const items = data?.items ?? [];

  return (
    <section aria-label="Recent submissions">
      <h3 className="mb-2 text-lg font-semibold text-zinc-200">Recent submissions</h3>
      <ul className="divide-y divide-zinc-800 rounded border border-zinc-800">
        {items.length === 0 && <li className="p-3 text-zinc-500">No submissions yet.</li>}
        {items.map((envelope) => (
          <li key={envelope.envelope_id} className="flex items-center justify-between p-3">
            <div className="text-sm">
              <span className="text-zinc-400">{envelope.source_type}</span> {envelope.envelope_id}
            </div>
            <span className="rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-300">
              {envelope.status}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
