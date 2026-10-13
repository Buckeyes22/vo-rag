import type { FacetStatus } from "@/project/vorag";

const LABELS: Record<FacetStatus, string> = {
  wired: "wired",
  partial: "partial",
  external: "external",
  not_wired: "not wired",
};

const CLASSES: Record<FacetStatus, string> = {
  wired: "border-emerald-500/40 bg-emerald-500/10 text-emerald-200",
  partial: "border-amber-500/40 bg-amber-500/10 text-amber-200",
  external: "border-cyan-500/40 bg-cyan-500/10 text-cyan-200",
  not_wired: "border-zinc-600 bg-zinc-800 text-zinc-300",
};

export default function StatusBadge({ status }: { status: FacetStatus }) {
  return (
    <span className={`inline-flex rounded border px-2 py-0.5 text-[11px] uppercase tracking-wide ${CLASSES[status]}`}>
      {LABELS[status]}
    </span>
  );
}
