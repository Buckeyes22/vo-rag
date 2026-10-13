import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, it, vi } from "vitest";
import Pipelines from "@/routes/Pipelines";
import { useAppStore } from "@/stores/app";

function wrap() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });

  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/pipelines"]}>
        <Routes>
          <Route path="/pipelines" element={<Pipelines />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Pipelines view", () => {
  beforeEach(() => {
    vi.spyOn(globalThis, "fetch");
    useAppStore.setState({ queryMode: "internal", csrfToken: "csrf" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders sections", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockImplementation(async (url) => {
      if (String(url).includes("/status")) {
        return new Response(
          JSON.stringify({
            ok: true,
            service: "vo-rag",
            version: "0.1.0",
            envelope_counts: {},
            active_workers: null,
            queue_depth: 3,
            qdrant: { ok: true, collections: [], points: {} },
            cloud: { ok: false, error: "not_configured" },
            services: {
              db_pool: { ok: true },
              qdrant: { ok: true },
              reranker: { ok: true },
              embedding_server: { ok: true },
            },
          }),
          { status: 200 },
        );
      }
      if (String(url).includes("/envelopes")) {
        return new Response(JSON.stringify({ items: [] }), { status: 200 });
      }
      return new Response("{}", { status: 200 });
    });

    wrap();

    await waitFor(() => expect(screen.getByText("3")).toBeInTheDocument());
    expect(screen.getByRole("region", { name: /active pipeline runs/i })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /envelopes/i })).toBeInTheDocument();
  });

  it("status filter changes GET param", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValue(new Response(JSON.stringify({ items: [] }), { status: 200 }));

    wrap();
    fireEvent.change(screen.getByLabelText("Status filter"), {
      target: { value: "failed" },
    });

    await waitFor(() => {
      const called = fetchSpy.mock.calls.some(
        ([u]) => String(u).includes("/api/v1/envelopes") && String(u).includes("status=failed"),
      );
      expect(called).toBe(true);
    });
  });

});
