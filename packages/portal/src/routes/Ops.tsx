import ExternalLinks from "@/components/system/ExternalLinks";
import { FLEET_NODES } from "@/project/vorag";

export default function Ops() {
  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-2xl font-semibold">Ops</h2>
        <div className="mt-1 text-sm text-zinc-400">Fleet roles and consoles.</div>
      </header>

      <ExternalLinks />

      <section aria-label="Fleet nodes" className="space-y-3">
        <h3 className="text-lg font-semibold">Fleet</h3>
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {FLEET_NODES.map((node) => (
            <article key={node.name} className="rounded border border-zinc-800 p-3">
              <div className="font-medium text-zinc-100">{node.name}</div>
              <div className="mt-1 text-sm text-zinc-400">{node.role}</div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
