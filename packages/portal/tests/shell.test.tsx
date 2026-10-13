import { render, screen, fireEvent, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "@/App";
import { vi, beforeEach, afterEach } from "vitest";

function renderApp(initialRoute = "/") {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[initialRoute]}>
        <App />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Shell + routing", () => {
  beforeEach(() => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          status: "ok",
          service: "vo-rag",
        }),
        { status: 200 },
      ),
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders comprehensive sidebar links", () => {
    renderApp();
    const primary = within(screen.getByLabelText("Primary"));

    for (const label of [
      "Overview",
      "Ingest",
      "Library",
      "Query",
      "Pipelines",
      "Book RAG",
      "System",
      "Ops",
      "Settings",
    ]) {
      expect(primary.getByRole("link", { name: label })).toBeInTheDocument();
    }
  });

  it("index route redirects to /overview", () => {
    renderApp("/");
    expect(screen.getByRole("heading", { name: /overview/i })).toBeInTheDocument();
  });

  it("clicking a sidebar link navigates", () => {
    renderApp("/ingest");
    fireEvent.click(screen.getByRole("link", { name: "Library" }));
    expect(screen.getByRole("heading", { name: /library/i })).toBeInTheDocument();
  });

  it("Settings route renders the Settings page", () => {
    renderApp("/settings");
    expect(screen.getByRole("heading", { name: /settings/i })).toBeInTheDocument();
    expect(screen.queryByLabelText(/API Key/i)).not.toBeInTheDocument();
    expect(screen.getByLabelText(/Default query mode/i)).toBeInTheDocument();
  });

  it("does not offer an Acquisition section", async () => {
    renderApp();
    expect(screen.queryByRole("link", { name: /acquisition/i })).toBeNull();
  });
});
