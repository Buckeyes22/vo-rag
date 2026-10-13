import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Query from "@/routes/Query";
import { useAppStore } from "@/stores/app";

function wrap(entry = "/query") {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[entry]}>
        <Routes>
          <Route path="/query" element={<Query />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Query view", () => {
  beforeEach(() => {
    vi.spyOn(globalThis, "fetch");
    useAppStore.setState({ queryMode: "internal", csrfToken: "csrf" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("mode selector exposes the internal-only backend mode", () => {
    wrap();
    expect(screen.getByText("internal")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "publication" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "partner" })).not.toBeInTheDocument();
    expect(useAppStore.getState().queryMode).toBe("internal");
  });

  it("URL state ignores legacy non-internal modes", async () => {
    wrap("/query?q=test&mode=partner");
    await waitFor(() => {
      expect(useAppStore.getState().queryMode).toBe("internal");
    });
  });

  it("top_k slider clamped 1..200 before send", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValue(new Response(JSON.stringify({ results: [] }), { status: 200 }));
    wrap();
    fireEvent.change(screen.getByLabelText("Query"), { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: /^Search$/i }));
    await waitFor(() => {
      const call = fetchSpy.mock.calls.find(([u]) => String(u).includes("/api/v1/query"));
      expect(call).toBeDefined();
      const body = JSON.parse((call![1] as RequestInit).body as string);
      expect(body.top_k).toBeGreaterThanOrEqual(1);
      expect(body.top_k).toBeLessThanOrEqual(200);
    });
  });

  it("filter panel round-trips key=value into request body", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValue(new Response(JSON.stringify({ results: [] }), { status: 200 }));
    wrap();
    fireEvent.change(screen.getByLabelText("Query"), { target: { value: "hello" } });
    fireEvent.change(screen.getByLabelText("Filter key"), { target: { value: "domain_tags" } });
    fireEvent.change(screen.getByLabelText("Filter value"), { target: { value: "braking" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    fireEvent.click(screen.getByRole("button", { name: /^Search$/i }));
    await waitFor(() => {
      const call = fetchSpy.mock.calls.find(([u]) => String(u).includes("/api/v1/query"));
      const body = JSON.parse((call![1] as RequestInit).body as string);
      expect(body.filters.domain_tags).toBe("braking");
    });
  });

  it("renders results from vo-rag payload fields", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValue(
      new Response(
        JSON.stringify({
          results: [
            {
              id: "hit1",
              score: 0.87,
              collection: "vo_rag_books_sterile_v1",
              payload: {
                content: "trail braking text",
                section_title: "Brake Release",
                page_number: 42,
                vlm_description: "Annotated corner-entry diagram",
                image_ref: "book/page.png",
              },
            },
          ],
        }),
        { status: 200 },
      ),
    );
    wrap();
    fireEvent.change(screen.getByLabelText("Query"), { target: { value: "trail braking" } });
    fireEvent.click(screen.getByRole("button", { name: /^Search$/i }));
    await waitFor(() => {
      expect(screen.getByText(/trail braking text/)).toBeInTheDocument();
    });
    expect(screen.getByText(/Brake Release/)).toBeInTheDocument();
    expect(screen.getByText(/p\. 42/)).toBeInTheDocument();
    const img = screen.getByRole("img", { name: "Annotated corner-entry diagram" });
    expect(img).toHaveAttribute("src", "/api/v1/figures/book/page.png");
    expect(screen.getByText(/score 0\.870/)).toBeInTheDocument();
  });

  it("POSTs only the internal query mode", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValue(
      new Response(
        JSON.stringify({
          results: [],
        }),
        { status: 200 },
      ),
    );
    wrap();
    fireEvent.change(screen.getByLabelText("Query"), { target: { value: "x" } });
    fireEvent.click(screen.getByRole("button", { name: /^Search$/i }));
    await waitFor(() => {
      const call = fetchSpy.mock.calls.find(([u]) => String(u).includes("/api/v1/query"));
      expect(call).toBeDefined();
      const body = JSON.parse((call![1] as RequestInit).body as string);
      expect(body.mode).toBe("internal");
    });
  });

  it("renders vlm_description as text fallback when image_ref is absent", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValue(
      new Response(
        JSON.stringify({
          results: [
            {
              id: "hit-noimg",
              score: 0.5,
              collection: "vo_rag_books_sterile_v1",
              payload: {
                content: "some text",
                vlm_description: "diagram without image",
              },
            },
          ],
        }),
        { status: 200 },
      ),
    );
    wrap();
    fireEvent.change(screen.getByLabelText("Query"), { target: { value: "diagram" } });
    fireEvent.click(screen.getByRole("button", { name: /^Search$/i }));
    await waitFor(() => {
      expect(screen.getByText(/diagram without image/)).toBeInTheDocument();
    });
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});
