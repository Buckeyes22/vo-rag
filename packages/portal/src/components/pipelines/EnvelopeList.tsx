import { useState } from "react";
import { useEnvelopes } from "@/api/client";

const STATUSES = ["", "received", "triaging", "processing", "completed", "failed", "discarded"];

export default function EnvelopeList({
  onSelect,
}: {
  onSelect: (id: string) => void;
}) {
  const [status, setStatus] = useState("");
  const { data } = useEnvelopes({ limit: 25, ...(status ? { status } : {}) });

  return (
    <section aria-label="Envelopes">
      <div className="mb-2 flex items-center gap-2">
        <label className="text-sm text-zinc-400">Status</label>
        <select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          aria-label="Status filter"
          className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm"
        >
          {STATUSES.map((s) => (
            <option key={s} value={s}>
              {s || "all"}
            </option>
          ))}
        </select>
      </div>
      <ul className="divide-y divide-zinc-800 rounded border border-zinc-800">
        {(data?.items ?? []).map((e) => (
          <li key={e.envelope_id}>
            <button
              onClick={() => onSelect(e.envelope_id)}
              className="flex w-full items-center justify-between p-3 text-left hover:bg-zinc-900"
            >
              <span className="text-sm text-zinc-300">
                {e.source_type} · {e.envelope_id}
              </span>
              <span className="rounded bg-zinc-800 px-2 py-0.5 text-xs">{e.status}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
