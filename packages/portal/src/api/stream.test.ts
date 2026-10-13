import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { useIngestStream, type IngestStreamEvent } from "./stream";

class FakeEventSource {
  static readonly CLOSED = 2;
  static instances: FakeEventSource[] = [];

  listeners = new Map<string, Array<(ev: MessageEvent) => void>>();
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = 0;

  constructor(public readonly url: string) {
    FakeEventSource.instances.push(this);
  }

  addEventListener(name: string, cb: (ev: MessageEvent) => void) {
    this.listeners.set(name, [...(this.listeners.get(name) ?? []), cb]);
  }

  emit(name: string, data: unknown) {
    const ev = { type: name, data: JSON.stringify(data) } as MessageEvent;
    for (const cb of this.listeners.get(name) ?? []) {
      cb(ev);
    }
  }

  close() {
    this.readyState = FakeEventSource.CLOSED;
  }
}

function currentSource(): FakeEventSource {
  const source = FakeEventSource.instances[0];
  if (!source) {
    throw new Error("expected EventSource to be created");
  }
  return source;
}

beforeEach(() => {
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource as unknown as typeof EventSource);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

test("collects named ingest events and terminates on done", async () => {
  const { result } = renderHook(() => useIngestStream("01JENV"));
  const source = currentSource();

  expect(source.url).toBe("/api/v1/intake/01JENV/stream");

  act(() => {
    source.emit("parse", {
      envelope_id: "01JENV",
      step: "parse",
      detail: "parsed document",
      pct: 20,
    });
    source.emit("image_described", {
      envelope_id: "01JENV",
      image_index: 1,
      total: 4,
    });
    source.emit("done", {
      source_id: "src_1",
      status: "completed",
      chunk_count: 42,
      idempotent_noop: false,
    });
  });

  await waitFor(() => expect(result.current.status).toBe("done"));
  expect(result.current.events.map((event) => event.event)).toEqual([
    "parse",
    "image_described",
    "done",
  ]);
  expect(source.readyState).toBe(FakeEventSource.CLOSED);
});

test("exposes the parsed done payload as result", async () => {
  const { result } = renderHook(() => useIngestStream("01JENV"));
  const source = currentSource();

  act(() => {
    source.emit("done", {
      source_id: "src_1",
      status: "completed",
      qdrant_point_ids: ["q1"],
      raw_point_ids: ["r1"],
      sterile_point_ids: ["s1"],
      smoke_decision: { ok: true },
      chunk_count: 12,
      idempotent_noop: true,
    });
  });

  await waitFor(() =>
    expect(result.current.result).toMatchObject({
      source_id: "src_1",
      status: "completed",
      chunk_count: 12,
      idempotent_noop: true,
    }),
  );
});

test("calls onData for each streamed event", async () => {
  const onData = vi.fn<(ev: IngestStreamEvent) => void>();
  renderHook(() => useIngestStream("01JENV", { onData }));
  const source = currentSource();

  act(() => {
    source.emit("parse", {
      envelope_id: "01JENV",
      step: "parse",
      detail: "parsed document",
      pct: 20,
    });
    source.emit("image_described", {
      envelope_id: "01JENV",
      image_index: 2,
      total: 4,
    });
    source.emit("done", {
      source_id: "src_1",
      status: "completed",
      chunk_count: 42,
      idempotent_noop: false,
    });
  });

  await waitFor(() => expect(onData).toHaveBeenCalledTimes(3));
  expect(onData.mock.calls.map(([event]) => event.event)).toEqual([
    "parse",
    "image_described",
    "done",
  ]);
});
