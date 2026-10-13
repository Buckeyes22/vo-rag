import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import System from "@/routes/System";
import { vi, beforeEach, afterEach } from "vitest";
import { useAppStore } from "@/stores/app";

function wrap() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/system"]}>
        <Routes>
          <Route path="/system" element={<System />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("System view", () => {
  beforeEach(() => {
    vi.spyOn(globalThis, "fetch");
    useAppStore.setState({ queryMode: "internal", csrfToken: "csrf" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  function mockAll() {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockImplementation(async (url) => {
      const s = String(url);
      if (s.endsWith("/health") || s.includes("/health")) {
        return new Response(
          JSON.stringify({
            status: "ok",
            service: "vo-rag",
          }),
          { status: 200 },
        );
      }
      if (s.endsWith("/status") || s.includes("/status")) {
        return new Response(
          JSON.stringify({
            ok: false,
            service: "vo-rag",
            version: "0.1.0",
            envelope_counts: {},
            active_workers: null,
            queue_depth: 0,
            qdrant: { ok: false, error: "degraded", collections: [] },
            cloud: { ok: false, error: "not_configured" },
            services: {
              db_pool: { ok: true },
              qdrant: { ok: false, error: "degraded" },
              reranker: { ok: true },
              embedding_server: { ok: true, status_code: 200, device: "cuda" },
            },
          }),
          { status: 200 },
        );
      }
      return new Response("{}", { status: 200 });
    });
  }

  it("service health renders with status colors", async () => {
    mockAll();
    wrap();
    await waitFor(() => expect(screen.getByText(/DB Pool/i)).toBeInTheDocument());
    expect(screen.getByText(/Qdrant/i)).toBeInTheDocument();
  });

  it("GPU strip is optional when status omits telemetry", async () => {
    mockAll();
    wrap();
    await waitFor(() => expect(screen.getByText(/No GPU telemetry/i)).toBeInTheDocument());
  });

  it("external links have rel=noopener and target=_blank", async () => {
    mockAll();
    wrap();
    const links = screen.getAllByRole("link");
    for (const l of links) {
      expect(l.getAttribute("target")).toBe("_blank");
      expect(l.getAttribute("rel")).toMatch(/noopener/);
    }
  });
});
