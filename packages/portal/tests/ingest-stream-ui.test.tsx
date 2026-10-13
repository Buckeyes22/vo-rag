import type { ReactNode } from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import LiveTriage from "@/components/ingest/LiveTriage";
import BookRag from "@/routes/BookRag";

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

function wrap(children: ReactNode) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });

  return render(<QueryClientProvider client={queryClient}>{children}</QueryClientProvider>);
}

describe("live ingest stream UI", () => {
  beforeEach(() => {
    FakeEventSource.instances = [];
    vi.stubGlobal("EventSource", FakeEventSource as unknown as typeof EventSource);
    vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders stream steps, figure progress, and terminal done details", async () => {
    wrap(<LiveTriage envelopeId="01JENV" />);
    const source = currentSource();

    act(() => {
      source.emit("parse", {
        envelope_id: "01JENV",
        step: "parse",
        detail: "parsed document",
        pct: 25,
      });
      source.emit("image_described", {
        envelope_id: "01JENV",
        image_index: 2,
        total: 5,
      });
      source.emit("done", {
        source_id: "src_1",
        status: "completed",
        chunk_count: 42,
        idempotent_noop: false,
      });
    });

    expect(await screen.findByText("parse")).toBeInTheDocument();
    expect(screen.getByText("parsed document")).toBeInTheDocument();
    expect(screen.getByText("25%")).toBeInTheDocument();
    expect(screen.getByText("described 2/5 figures")).toBeInTheDocument();
    expect(screen.getByText("completed")).toBeInTheDocument();
    expect(screen.getByText("42 chunks")).toBeInTheDocument();
    expect(screen.getByText("new ingest")).toBeInTheDocument();
  });

  it("attaches BookRag to the most recent processing envelope", async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockImplementation(async (url) => {
      const href = String(url);
      if (href.includes("/api/v1/rag/status")) {
        return new Response(
          JSON.stringify({
            ready: true,
            db_pool: "ok",
            qdrant: "ok",
            embedding_server: "ok",
            sterile_collection_has_points: true,
            raw_collection_has_points: true,
          }),
          { status: 200 },
        );
      }
      if (href.includes("/api/v1/collections/stats")) {
        return new Response(
          JSON.stringify({
            collections: {
              vo_rag_books_raw_v1: { points: 10 },
              vo_rag_books_sterile_v1: { points: 8 },
            },
          }),
          { status: 200 },
        );
      }
      if (href.includes("/api/v1/envelopes")) {
        return new Response(
          JSON.stringify({
            items: [
              {
                envelope_id: "01BOOK",
                source_type: "file_drop",
                status: "processing",
                target_pipeline: "book_rag",
              },
            ],
          }),
          { status: 200 },
        );
      }
      return new Response("{}", { status: 200 });
    });

    wrap(<BookRag />);

    await screen.findByText("In-flight book ingest");
    await waitFor(() => expect(FakeEventSource.instances[0]?.url).toBe("/api/v1/intake/01BOOK/stream"));
  });
});
