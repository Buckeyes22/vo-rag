import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { type ReactNode } from "react";
import SystemHealthPill from "@/components/layout/SystemHealthPill";
import { vi, beforeEach, afterEach } from "vitest";

function wrap(children: ReactNode) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });

  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("SystemHealthPill", () => {
  beforeEach(() => {
    vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("green when all services ok", async () => {
    vi.mocked(globalThis.fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          status: "ok",
          service: "vo-rag",
        }),
        { status: 200 },
      ),
    );

    wrap(<SystemHealthPill />);

    await waitFor(() =>
      expect(screen.getByRole("button")).toHaveAttribute("aria-label", expect.stringMatching(/healthy/i)),
    );
    expect(screen.getByRole("button").className).toMatch(/bg-green-600/);
  });

  it("red when liveness status is not ok", async () => {
    vi.mocked(globalThis.fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          status: "down",
          service: "vo-rag",
        }),
        { status: 200 },
      ),
    );

    wrap(<SystemHealthPill />);

    await waitFor(() =>
      expect(screen.getByRole("button")).toHaveAttribute(
        "aria-label",
        expect.stringMatching(/degraded/i),
      ),
    );
    expect(screen.getByRole("button").className).toMatch(/bg-red-600/);
  });

  it("red when health fetch errors", async () => {
    vi.mocked(globalThis.fetch).mockResolvedValue(new Response("bad", { status: 500 }));

    wrap(<SystemHealthPill />);

    await waitFor(() =>
      expect(screen.getByRole("button")).toHaveAttribute("aria-label", expect.stringMatching(/down/i)),
    );
  });
});
