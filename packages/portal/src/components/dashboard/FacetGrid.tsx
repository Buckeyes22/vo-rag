import { Link } from "react-router-dom";
import StatusBadge from "@/components/dashboard/StatusBadge";
import type { ProjectFacet } from "@/project/vorag";

export default function FacetGrid({ facets }: { facets: ProjectFacet[] }) {
  return (
    <section aria-label="Project facet coverage" className="space-y-3">
      <h3 className="text-lg font-semibold text-zinc-100">Facet Coverage</h3>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        {facets.map((facet) => (
          <article key={facet.id} className="rounded border border-zinc-800 p-3">
            <header className="flex items-start justify-between gap-3">
              <div>
                <Link to={facet.route} className="font-medium text-blue-300 hover:underline">
                  {facet.label}
                </Link>
                <div className="mt-1 text-sm text-zinc-400">{facet.surface}</div>
              </div>
              <StatusBadge status={facet.status} />
            </header>
            <div className="mt-3 grid grid-cols-1 gap-3 text-xs md:grid-cols-2">
              <div>
                <div className="mb-1 text-zinc-500">Endpoints</div>
                <ul className="space-y-1 text-zinc-300">
                  {facet.endpoints.map((endpoint) => (
                    <li key={endpoint}>
                      <code>{endpoint}</code>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <div className="mb-1 text-zinc-500">Signals</div>
                <ul className="space-y-1 text-zinc-300">
                  {facet.signals.map((signal) => (
                    <li key={signal}>{signal}</li>
                  ))}
                </ul>
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
