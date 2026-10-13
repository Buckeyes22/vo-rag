import { useStatus } from "@/api/client";

export default function GpuStrip() {
  const { data } = useStatus();
  const gpus = data?.gpus ?? [];

  if (gpus.length === 0) {
    return <div className="text-zinc-500 text-sm">No GPU telemetry.</div>;
  }

  return (
    <section aria-label="GPU strip" className="flex flex-wrap gap-2">
      {gpus.map((g, i) => (
        <div key={i} className="rounded border border-zinc-800 p-3 text-xs">
          <div className="text-zinc-300">{g.node}</div>
          <div className="text-zinc-500">{g.vram_used_mb} MB</div>
          <div className="text-zinc-500">{g.util_pct}% util</div>
        </div>
      ))}
    </section>
  );
}
