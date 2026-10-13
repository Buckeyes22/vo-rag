import { useCallback, useEffect, useRef, useState } from "react";

export const INGEST_STREAM_EVENT_NAMES = [
  "register_source",
  "parse",
  "visual",
  "chunk_embed",
  "sterilize",
  "register_artifact",
  "validate",
  "retrieval_smoke",
  "image_described",
  "done",
] as const;

export type IngestStreamEventName = (typeof INGEST_STREAM_EVENT_NAMES)[number];

export type IngestStreamEvent = {
  event: string;
  data: Record<string, unknown>;
};

export type IngestDonePayload = {
  source_id?: string;
  status?: string;
  qdrant_point_ids?: string[];
  raw_point_ids?: string[];
  sterile_point_ids?: string[];
  smoke_decision?: unknown;
  chunk_count?: number;
  idempotent_noop?: boolean;
  [key: string]: unknown;
};

export type IngestStreamStatus = "idle" | "streaming" | "done" | "error";

export type IngestStreamOptions = {
  onData?: (ev: IngestStreamEvent) => void;
};

export type IngestStreamState = {
  events: IngestStreamEvent[];
  status: IngestStreamStatus;
  error: string | null;
  result: IngestDonePayload | null;
  close: () => void;
};

function parseEventData(data: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(data) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return { value: parsed };
  } catch {
    return { raw: data };
  }
}

/**
 * Subscribe to one ingest run's SSE stream. This mirrors the official
 * LlamaIndex `useHandler(handlerId)` shape with `envelopeId` as the handler id:
 * `events` is this portal's retained callback log, `onData` maps to
 * `subscribeToEvents({ onData })`, `close` maps to `unsubscribe()`, and
 * `result` mirrors the terminal `state.result`.
 *
 * The GET `/api/v1/intake/{envelopeId}/stream` starts the ingest run
 * server-side; call this once for a submitted envelope, not as attach-only
 * polling.
 */
export function useIngestStream(
  envelopeId: string | undefined,
  options: IngestStreamOptions = {},
): IngestStreamState {
  const [events, setEvents] = useState<IngestStreamEvent[]>([]);
  const [status, setStatus] = useState<IngestStreamStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<IngestDonePayload | null>(null);
  const sourceRef = useRef<EventSource | null>(null);
  const onDataRef = useRef<IngestStreamOptions["onData"]>(options.onData);

  useEffect(() => {
    onDataRef.current = options.onData;
  }, [options.onData]);

  const close = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
  }, []);

  useEffect(() => {
    if (!envelopeId) {
      close();
      setEvents([]);
      setError(null);
      setResult(null);
      setStatus("idle");
      return;
    }

    setEvents([]);
    setError(null);
    setResult(null);
    setStatus("streaming");

    if (typeof EventSource === "undefined") {
      setStatus("error");
      setError("eventsource_unavailable");
      return;
    }

    const source = new EventSource(
      `/api/v1/intake/${encodeURIComponent(envelopeId)}/stream`,
    );
    sourceRef.current = source;

    const push = (ev: MessageEvent) => {
      const event = { event: ev.type, data: parseEventData(String(ev.data)) };
      setEvents((prev) => [...prev, event]);
      onDataRef.current?.(event);

      if (ev.type === "done") {
        setResult(event.data);
        setStatus("done");
        source.close();
      }
    };

    for (const name of INGEST_STREAM_EVENT_NAMES) {
      source.addEventListener(name, push);
    }
    source.onmessage = push;
    source.onerror = () => {
      if (source.readyState === EventSource.CLOSED) {
        setStatus((current) => (current === "done" ? current : "error"));
        setError("stream_closed");
      }
    };

    return () => {
      source.close();
      if (sourceRef.current === source) {
        sourceRef.current = null;
      }
    };
  }, [close, envelopeId]);

  return { events, status, error, result, close };
}
