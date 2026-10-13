import type { KnowledgeArtifact } from "@/schemas";

export default function ProvenanceSidebar({ artifact }: { artifact: KnowledgeArtifact }) {
  return (
    <aside aria-label="Provenance" className="rounded border border-zinc-800 p-3 text-sm space-y-1">
      <div className="font-semibold text-zinc-200">Provenance</div>
      <div>
        License: <span className="text-zinc-300">{artifact.license_class ?? "unknown"}</span>
      </div>
      {/* Full provenance chain rendered from artifact.provenance when that field
          exists on the API response (3.3.1). Kept minimal here. */}
    </aside>
  );
}
