import { NavLink } from "react-router-dom";

type NavItem = { to: string; label: string };
type NavGroup = { label: string; links: NavItem[] };

const GROUPS: NavGroup[] = [
  {
    label: "Operate",
    links: [
      { to: "/overview", label: "Overview" },
      { to: "/ingest", label: "Ingest" },
      { to: "/library", label: "Library" },
      { to: "/query", label: "Query" },
      { to: "/pipelines", label: "Pipelines" },
    ],
  },
  {
    label: "Core",
    links: [
      { to: "/book-rag", label: "Book RAG" },
      { to: "/system", label: "System" },
    ],
  },
  {
    label: "Platform",
    links: [
      { to: "/ops", label: "Ops" },
      { to: "/settings", label: "Settings" },
    ],
  },
];

export default function Sidebar({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  return (
    <aside
      aria-label="Primary"
      className={`fixed inset-y-0 left-0 z-40 w-64 overflow-y-auto border-r border-zinc-800 bg-zinc-900 p-4 transform transition-transform md:static md:translate-x-0 ${
        open ? "translate-x-0" : "-translate-x-full"
      }`}
    >
      <div className="mb-6 text-lg font-bold text-blue-400">vo-rag</div>
      <nav className="space-y-5">
        {GROUPS.map((group) => (
          <section key={group.label} aria-label={group.label}>
            <div className="mb-2 px-3 text-[11px] uppercase tracking-wide text-zinc-500">
              {group.label}
            </div>
            <div className="flex flex-col gap-1">
              {group.links.map((link) => (
                <NavLink
                  key={link.to}
                  to={link.to}
                  onClick={onClose}
                  className={({ isActive }) =>
                    `rounded px-3 py-2 text-sm ${
                      isActive
                        ? "bg-blue-600/20 text-blue-300"
                        : "text-zinc-300 hover:bg-zinc-800"
                    }`
                  }
                >
                  {link.label}
                </NavLink>
              ))}
            </div>
          </section>
        ))}
      </nav>
    </aside>
  );
}
