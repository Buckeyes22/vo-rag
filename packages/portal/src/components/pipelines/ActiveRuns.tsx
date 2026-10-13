import { useStatus } from "@/api/client";

export default function ActiveRuns() {
  const { data } = useStatus();
  const runs = data?.queue_depth ?? 0;

  return (
    <section aria-label="Active pipeline runs" className="rounded border border-zinc-800 p-3">
      <h3 className="font-semibold">Active queue</h3>
      <div className="mt-1 text-3xl text-blue-300">{runs}</div>
      <p className="mt-1 text-xs text-zinc-500">active, queued, or processing envelopes</p>
    </section>
  );
}
