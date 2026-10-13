import SystemHealthPill from "./SystemHealthPill";

export default function Header({ onToggleSidebar }: { onToggleSidebar: () => void }) {
  return (
    <header className="sticky top-0 z-30 flex items-center justify-between border-b border-zinc-800 bg-zinc-950 px-4 py-3">
      <button
        onClick={onToggleSidebar}
        className="md:hidden rounded border border-zinc-700 px-2 py-1 text-zinc-300"
        aria-label="Toggle sidebar"
      >
        ≡
      </button>
      <h1 className="text-lg font-semibold text-zinc-100">vo-rag Portal</h1>
      <SystemHealthPill />
    </header>
  );
}
