import { usePortalSession } from "@/api/client";
import { useAppStore } from "@/stores/app";

export default function Settings() {
  const queryMode = useAppStore((s) => s.queryMode);
  const setQueryMode = useAppStore((s) => s.setQueryMode);
  const safePublicationMode = useAppStore((s) => s.safePublicationMode);
  const setSafePublicationMode = useAppStore((s) => s.setSafePublicationMode);
  const session = usePortalSession();
  const email = session.data?.user.email ?? "Unknown";
  const authSource = session.data?.auth_source ?? "unknown";

  return (
    <section className="max-w-xl p-6">
      <h2 className="mb-4 text-2xl font-semibold">Settings</h2>

      <div className="rounded border border-zinc-800 bg-zinc-900 p-4">
        <div className="text-xs uppercase text-zinc-500">Account</div>
        <div className="mt-2 text-zinc-100">{email}</div>
        <div className="mt-1 text-sm text-zinc-400">{authSource}</div>
        <a
          href="/cdn-cgi/access/logout"
          className="mt-3 inline-block rounded border border-zinc-700 px-3 py-2 text-sm text-zinc-200 hover:bg-zinc-800"
        >
          Logout
        </a>
      </div>

      <label className="mb-2 mt-6 block text-zinc-300" htmlFor="default-query-mode">
        Default query mode
      </label>
      <select
        id="default-query-mode"
        value={queryMode}
        onChange={(event) => setQueryMode(event.target.value as typeof queryMode)}
        className="rounded border border-zinc-700 bg-zinc-900 px-3 py-2 text-zinc-100"
        aria-label="Default query mode"
      >
        <option value="internal">internal</option>
      </select>

      <label className="mt-6 flex items-center gap-3 text-sm text-zinc-300">
        <input
          type="checkbox"
          checked={safePublicationMode}
          onChange={(event) => setSafePublicationMode(event.target.checked)}
          aria-label="Safe publication mode"
        />
        Safe publication mode
      </label>
    </section>
  );
}
