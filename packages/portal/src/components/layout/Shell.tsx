import { useEffect, useState } from "react";
import { Outlet } from "react-router-dom";
import { usePortalSession } from "@/api/client";
import { useAppStore } from "@/stores/app";
import Header from "./Header";
import Sidebar from "./Sidebar";

export default function Shell() {
  const [open, setOpen] = useState(false);
  const setSession = useAppStore((s) => s.setSession);
  const session = usePortalSession();

  useEffect(() => {
    if (!session.data) {
      return;
    }
    setSession({
      csrfToken: session.data.csrf_token ?? null,
      email: session.data.user.email,
      authSource: session.data.auth_source,
    });
  }, [session.data, setSession]);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 md:flex">
      <Sidebar open={open} onClose={() => setOpen(false)} />
      <div className="flex-1 md:ml-0">
        <Header onToggleSidebar={() => setOpen((o) => !o)} />
        <main className="p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
