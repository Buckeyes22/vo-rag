export type NormalizedHealth = "ok" | "degraded" | "down" | "unknown";

export function normalizeHealth(value: unknown): NormalizedHealth {
  if (value === true) {
    return "ok";
  }
  if (value === false || value == null) {
    return "down";
  }
  if (typeof value === "string") {
    const lowered = value.toLowerCase();
    if (lowered === "ok" || lowered === "healthy" || lowered === "ready") {
      return "ok";
    }
    if (lowered === "degraded" || lowered === "partial") {
      return "degraded";
    }
    if (lowered === "unknown" || lowered === "loading") {
      return "unknown";
    }
    return "down";
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (record.ok === true || record.ready === true) {
      return "ok";
    }
    if (record.ok === false || record.ready === false || record.error) {
      return "down";
    }
    return "unknown";
  }
  return "unknown";
}

export function healthTone(status: NormalizedHealth): string {
  if (status === "ok") {
    return "bg-emerald-500";
  }
  if (status === "degraded") {
    return "bg-amber-500";
  }
  if (status === "down") {
    return "bg-rose-500";
  }
  return "bg-zinc-500";
}
