const LINKS = [
  { label: "Hatchet", href: "http://localhost:8889" },
  { label: "Langfuse", href: "http://localhost:3101" },
  { label: "Grafana", href: "http://localhost:3000" },
];

export default function ExternalLinks() {
  return (
    <section aria-label="External links" className="flex gap-2">
      {LINKS.map((l) => (
        <a
          key={l.label}
          href={l.href}
          target="_blank"
          rel="noopener noreferrer"
          className="rounded bg-zinc-800 px-3 py-1 text-sm text-zinc-200 hover:bg-zinc-700"
        >
          {l.label} ↗
        </a>
      ))}
    </section>
  );
}
