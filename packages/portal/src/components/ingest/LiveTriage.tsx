import { useEffect, useState } from "react";
import { useEnvelope } from "@/api/client";
import { useIngestStream, type IngestStreamEvent } from "@/api/stream";

const TERMINAL = new Set(["completed", "failed", "discarded"]);
const NON_STEP_EVENTS = new Set(["image_described", "done"]);

function stringField(data: Record<string, unknown>, key: string): string | null {
  const value = data[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function numberField(data: Record<string, unknown>, key: string): number | null {
  const value = data[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function booleanField(data: Record<string, unknown>, key: string): boolean | null {
  const value = data[key];
  return typeof value === "boolean" ? value : null;
}

function detailText(event: IngestStreamEvent): string | null {
  const detail = event.data.detail;
  if (typeof detail === "string" && detail.length > 0) {
    return detail;
  }
  if (detail === undefined || detail === null) {
    return null;
  }
  if (typeof detail === "number" || typeof detail === "boolean") {
    return String(detail);
  }

  try {
    return JSON.stringify(detail);
  } catch {
    return "detail unavailable";
  }
}

function stepName(event: IngestStreamEvent): string {
  return stringField(event.data, "step") ?? event.event;
}

function latestImageEvent(events: IngestStreamEvent[]): IngestStreamEvent | null {
  const imageEvents = events.filter((event) => event.event === "image_described");
  return imageEvents.at(-1) ?? null;
}

function QualityWarnings({ value }: { value: unknown }) {
  if (!Array.isArray(value)) return null;
  const warnings = value.filter(
    (entry): entry is Record<string, unknown> =>
      entry !== null && typeof entry === "object" && !Array.isArray(entry),
  );
  if (warnings.length === 0) return null;
  return (
    <div role="status" className="rounded border border-amber-900/60 p-3 text-amber-300">
      <p>Ingestion quality warnings</p>
      <ul className="list-disc pl-5">
        {warnings.map((warning, index) => (
          <li key={index}>
            {stringField(warning, "stage") ?? "quality"}: {stringField(warning, "code") ?? "unknown"}
            {stringField(warning, "details") && ` — ${String(warning.details).slice(0, 500)}`}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function LiveTriage({ envelopeId }: { envelopeId: string }) {
  const [pollUntil] = useState(() => Date.now() + 20_000);
  const [stopped, setStopped] = useState(false);
  const stream = useIngestStream(envelopeId);
  const fallbackEnabled = stream.status === "error" && !stopped;
  const { data } = useEnvelope(fallbackEnabled ? envelopeId : undefined);
  const stepEvents = stream.events.filter((event) => !NON_STEP_EVENTS.has(event.event));
  const imageEvent = latestImageEvent(stream.events);
  const done = stream.result;

  useEffect(() => {
    if (stream.status !== "error" || !data) {
      return;
    }

    if (data.triage_result || TERMINAL.has(data.status) || Date.now() > pollUntil) {
      setStopped(true);
    }
  }, [data, pollUntil, stream.status]);

  if (stream.status !== "error") {
    const imageIndex = imageEvent ? numberField(imageEvent.data, "image_index") : null;
    const imageTotal = imageEvent ? numberField(imageEvent.data, "total") : null;
    const doneStatus = done ? stringField(done, "status") ?? "done" : null;
    const chunkCount = done ? numberField(done, "chunk_count") : null;
    const idempotentNoop = done ? booleanField(done, "idempotent_noop") : null;

    return (
      <div className="space-y-3 rounded border border-zinc-800 p-3 text-sm">
        <div className="flex flex-wrap items-center gap-2">
          <span>
            Submitted - envelope <code>{envelopeId}</code>
          </span>
          <span
            className={`rounded px-2 py-0.5 text-xs ${
              stream.status === "done"
                ? "bg-emerald-600/20 text-emerald-300"
                : "bg-blue-600/20 text-blue-300"
            }`}
          >
            {stream.status}
          </span>
        </div>

        <div className="space-y-2" aria-label="Ingest stream timeline">
          {stepEvents.length === 0 && (
            <div className="text-zinc-500">Waiting for stream events...</div>
          )}
          {stepEvents.map((event, index) => {
            const pct = numberField(event.data, "pct");
            const detail = detailText(event);
            return (
              <div
                key={`${event.event}-${index}`}
                className="grid grid-cols-[0.75rem_minmax(0,1fr)_auto] items-start gap-2"
              >
                <span className="mt-1 h-2 w-2 rounded-full bg-blue-400" />
                <div className="min-w-0">
                  <div className="font-medium text-zinc-200">{stepName(event)}</div>
                  {detail && <div className="break-words text-zinc-400">{detail}</div>}
                </div>
                {pct !== null && <span className="text-xs text-zinc-500">{Math.round(pct)}%</span>}
              </div>
            );
          })}
        </div>

        {imageIndex !== null && imageTotal !== null && (
          <div className="rounded border border-zinc-800 bg-zinc-950 px-3 py-2 text-zinc-300">
            described {imageIndex}/{imageTotal} figures
          </div>
        )}

        {done && <QualityWarnings value={done.quality_warnings} />}
        {done && (
          <div className="rounded border border-emerald-900/60 bg-emerald-950/20 p-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium text-emerald-300">{doneStatus}</span>
              {chunkCount !== null && <span className="text-zinc-300">{chunkCount} chunks</span>}
              {idempotentNoop !== null && (
                <span className="rounded bg-zinc-900 px-2 py-0.5 text-xs text-zinc-400">
                  {idempotentNoop ? "idempotent noop" : "new ingest"}
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="rounded border border-amber-900/60 p-3 text-sm text-amber-300">
        Live stream unavailable - polling envelope status...
      </div>
    );
  }

  const t = data.triage_result;
  return (
    <div className="space-y-1 rounded border border-zinc-800 p-3 text-sm">
      <div>
        Submitted - envelope <code>{data.envelope_id}</code>
      </div>
      <div>
        Status: <span className="text-blue-300">{data.status}</span>
      </div>
      <QualityWarnings value={data.metadata?.quality_warnings} />
      {t && (
        <>
          <div>
            Triaged as <span className="text-blue-300">{t.target_pipeline ?? "unassigned"}</span> (priority=
            {t.priority}, confidence={t.confidence.toFixed(2)})
          </div>
          <div className="text-zinc-400">disposition: {t.disposition}</div>
        </>
      )}
      {stopped && !t && (
        <div className="text-zinc-400">(triage not observed within 20s window - use Pipelines view)</div>
      )}
    </div>
  );
}
