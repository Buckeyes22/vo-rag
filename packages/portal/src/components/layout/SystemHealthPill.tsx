import { useNavigate } from "react-router-dom";
import { useHealth } from "@/api/client";

export default function SystemHealthPill() {
  const nav = useNavigate();
  const { data, isLoading, isError } = useHealth();

  let color = "bg-zinc-600";
  let label = "unknown";
  if (isLoading) {
    color = "bg-zinc-600";
    label = "loading";
  } else if (isError) {
    color = "bg-red-600";
    label = "down";
  } else if (data) {
    const healthy = data.status === "ok";
    color = healthy ? "bg-green-600" : "bg-red-600";
    label = healthy ? "healthy" : "degraded";
  }

  return (
    <button
      onClick={() => nav("/system")}
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs text-white ${color}`}
      aria-label={`System health: ${label}`}
    >
      <span className="h-2 w-2 rounded-full bg-white/80" />
      {label}
    </button>
  );
}
