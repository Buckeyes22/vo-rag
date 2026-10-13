import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { vi, beforeEach, afterEach } from "vitest";
import Library from "@/routes/Library";
import LibraryDetail from "@/routes/LibraryDetail";
import { useAppStore } from "@/stores/app";

function wrap(initialEntry = "/library") {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route path="/library" element={<Library />} />
          <Route path="/library/:id" element={<LibraryDetail />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function artifact(overrides: Record<string, unknown> = {}) {
  return {
    id: "01ART",
    envelope_id: "01ENV",
    pipeline: "book_rag",
    artifact_type: "summary",
    title: "Going Faster",
    content: { summary: "x" },
    tags: ["book"],
    priority: "medium",
    status: "active",
    trust_level: "quarantined",
    qdrant_collection: "vo_rag_books_raw_v1",
    qdrant_point_ids: [],
    source_provenance: [],
    created_at: "2026-04-13T00:00:00Z",
    updated_at: "2026-04-13T00:00:00Z",
    license_class: "unknown",
    ...overrides,
  };
}

describe("Library", () => {
  beforeEach(() => {
    vi.spyOn(globalThis, "fetch");
    useAppStore.setState({ queryMode: "internal", csrfToken: "csrf" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders 8 collection tabs", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValue(new Response(JSON.stringify({ items: [] }), { status: 200 }));
    wrap();

    for (const label of ["All", "Books", "Repos", "Research", "Ideas", "Articles", "Tools", "URLs"]) {
      expect(screen.getByRole("tab", { name: label })).toBeInTheDocument();
    }
  });

  it("clicking Books tab issues GET with pipeline=book_rag", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValue(new Response(JSON.stringify({ items: [] }), { status: 200 }));
    wrap();

    fireEvent.click(screen.getByRole("tab", { name: "Books" }));

    await waitFor(() => {
      const called = fetchSpy.mock.calls.some(([u]) =>
        String(u).includes("/api/v1/artifacts") && String(u).includes("pipeline=book_rag"),
      );
      expect(called).toBe(true);
    });
  });

  it("empty list renders 'No artifacts.'", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValue(new Response(JSON.stringify({ items: [] }), { status: 200 }));
    wrap();
    await waitFor(() => expect(screen.getByText("No artifacts.")).toBeInTheDocument());
  });

  it("renders advanced-layer chips when content.layers is present", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValue(
      new Response(
        JSON.stringify({
          items: [
            artifact({
              content: {
                source_id: "vo-rag-corpus-sha256:abc",
                layers: { summary_tree: 60, raptor: 9, dsi: 1, graph_nodes: 253 },
              },
            }),
            artifact({ id: "01NOLAYERS", title: "Plain Book", content: { summary: "x" } }),
          ],
        }),
        { status: 200 },
      ),
    );
    wrap();
    await waitFor(() => expect(screen.getByText("DSI")).toBeInTheDocument());
    expect(screen.getByText("Summaries")).toBeInTheDocument();
    expect(screen.getByText("Graph")).toBeInTheDocument();
    expect(screen.getByText("RAPTOR")).toBeInTheDocument();
    expect(screen.getByText("Plain Book")).toBeInTheDocument();
  });

  it("Load more fetches the next cursor page and appends items", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          items: [artifact({ id: "01PAGE1", title: "First Page Book" })],
          next_cursor: "cur123",
        }),
        { status: 200 },
      ),
    );
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          items: [artifact({ id: "01PAGE2", title: "Second Page Book" })],
          next_cursor: null,
        }),
        { status: 200 },
      ),
    );
    wrap();
    await waitFor(() => expect(screen.getByText("First Page Book")).toBeInTheDocument());
    const loadMore = screen.getByRole("button", { name: "Load more" });
    loadMore.click();
    await waitFor(() => expect(screen.getByText("Second Page Book")).toBeInTheDocument());
    expect(screen.getByText("First Page Book")).toBeInTheDocument();
    const secondCall = String(fetchSpy.mock.calls[1]?.[0] ?? "");
    expect(secondCall).toContain("cursor=cur123");
    expect(screen.queryByRole("button", { name: "Load more" })).not.toBeInTheDocument();
  });

  it("renders corpus stats when artifact content includes stats", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValue(
      new Response(
        JSON.stringify({
          items: [
            artifact({
              content: {
                summary: "x",
                stats: {
                  text_chunks: 144,
                  figure_chunks: 3,
                  pages: 276,
                  domain_tags: ["vehicle dynamics", "setup", "racing", "chassis"],
                },
              },
            }),
          ],
        }),
        { status: 200 },
      ),
    );

    wrap();

    await waitFor(() => expect(screen.getByRole("link", { name: "Going Faster" })).toBeInTheDocument());
    expect(
      screen.getByText("144 text · 3 figures · 276 pp · vehicle dynamics, setup, racing"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/chassis/)).not.toBeInTheDocument();
  });

  it("falls back to metadata when artifact stats are absent", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockResolvedValue(
      new Response(
        JSON.stringify({
          items: [
            artifact({
              title: "Legacy note",
              tags: ["legacy", "book"],
              priority: "high",
            }),
          ],
        }),
        { status: 200 },
      ),
    );

    wrap();

    await waitFor(() => expect(screen.getByRole("link", { name: "Legacy note" })).toBeInTheDocument());
    expect(
      screen.getByText("book_rag · 2026-04-13T00:00:00Z · high · #legacy #book"),
    ).toBeInTheDocument();
  });

  it("detail view fetches artifact by id", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockImplementation(async (url) => {
      if (String(url).includes("/api/v1/artifacts/01ART") && !String(url).includes("/images")) {
        return new Response(JSON.stringify(artifact()), { status: 200 });
      }
      if (String(url).includes("/images")) {
        return new Response(JSON.stringify({ items: [] }), { status: 200 });
      }
      return new Response("{}", { status: 200 });
    });

    wrap("/library/01ART");
    await waitFor(() => expect(screen.getByText("Going Faster")).toBeInTheDocument());
    expect(screen.getByText(/book_rag · 2026-04-13T00:00:00Z/)).toBeInTheDocument();
    expect(screen.getAllByText(/unknown/).length).toBeGreaterThan(0);
    expect(screen.getByRole("link", { name: /query this output/i })).toHaveAttribute(
      "href",
      expect.stringContaining("mode=internal"),
    );
  });
});

const bookOverview = (overrides: Record<string, unknown> = {}) => ({
  source_id: "src-123",
  title: "The Art of Racing",
  author: "A.N. Author",
  license_class: "public_domain",
  artifact_id: "01ART",
  stats: {
    text_chunks: 200,
    figure_chunks: 5,
    pages: 120,
    domain_tags: ["racing", "physics"],
  },
  layers: {
    summary_tree: 1,
    raptor: 1,
    dsi: 1,
    graph_nodes: 10,
  },
  ...overrides,
});

const bookPage = (overrides: Record<string, unknown> = {}) => ({
  page: 1,
  max_page: 120,
  chunks: [
    {
      chunk_id: "c1",
      section_title: "Chapter 1",
      text: "It was a dark and stormy night.",
      has_image: false,
      image_id: null,
      vlm_description: null,
      token_count: 8,
    },
    {
      chunk_id: "c2",
      section_title: "Chapter 1",
      text: "The car roared through the bend.",
      has_image: true,
      image_id: "img-1",
      vlm_description: "A race car on a wet track",
      token_count: 7,
    },
  ],
  ...overrides,
});

const bookSummaries = (overrides: Record<string, unknown> = {}) => ({
  dsi: {
    chapter_outline: "A book about racing techniques.",
    key_concepts: ["traction", "apex"],
    key_methods: ["trail braking"],
    key_equations_or_models: ["F=ma"],
    prerequisite_topics: ["basic physics"],
    figure_heavy_topics: ["suspension geometry"],
  },
  summary_tree: {
    book: [{ display_path: null, key_points: ["Racing is fast", "Braking matters"] }],
    chapter: [{ display_path: "Ch 1: Intro", key_points: ["Intro to racing"] }],
    section_count: 42,
  },
  graph: {
    node_count: 150,
    top_nodes: [
      { label: "Traction", node_type: "concept", mentions: 23 },
      { label: "Apex", node_type: "concept", mentions: 15 },
    ],
  },
  raptor: { node_count: 80 },
  ...overrides,
});

describe("BookView", () => {
  beforeEach(() => {
    vi.spyOn(globalThis, "fetch");
    useAppStore.setState({ queryMode: "internal", csrfToken: "csrf" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders BookView header and Read tab chunks for book_rag with source_id", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockImplementation(async (url) => {
      const u = String(url);
      if (u.includes("/api/v1/artifacts/01ART") && !u.includes("/images") && !u.includes("/books")) {
        return new Response(
          JSON.stringify(
            artifact({ content: { summary: "x", source_id: "src-123" } }),
          ),
          { status: 200 },
        );
      }
      if (u.includes("/api/v1/books/src-123/pages/")) {
        return new Response(JSON.stringify(bookPage()), { status: 200 });
      }
      if (u.endsWith("/api/v1/books/src-123") && !u.includes("/pages") && !u.includes("/summaries")) {
        return new Response(JSON.stringify(bookOverview()), { status: 200 });
      }
      if (u.includes("/images")) {
        return new Response(JSON.stringify({ items: [] }), { status: 200 });
      }
      return new Response("{}", { status: 200 });
    });

    wrap("/library/01ART");

    await waitFor(() =>
      expect(screen.getByText("Going Faster")).toBeInTheDocument(),
    );
    expect(screen.getByText("A.N. Author")).toBeInTheDocument();
    expect(screen.getByText("200 text · 5 figures · 120 pp")).toBeInTheDocument();
    expect(screen.getByText("racing")).toBeInTheDocument();
    expect(screen.getByText("physics")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /query this output/i })).toHaveAttribute(
      "href",
      expect.stringContaining("mode=internal"),
    );

    await waitFor(() =>
      expect(screen.getByText("It was a dark and stormy night.")).toBeInTheDocument(),
    );
    expect(screen.getByText("Chapter 1")).toBeInTheDocument();
    expect(screen.getByText("The car roared through the bend.")).toBeInTheDocument();
    expect(screen.getByText("A race car on a wet track")).toBeInTheDocument();
    expect(screen.getByText("Page 1 of 120")).toBeInTheDocument();
  });

  it("page navigation triggers next page fetch", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockImplementation(async (url) => {
      const u = String(url);
      if (u.includes("/api/v1/artifacts/01ART") && !u.includes("/images") && !u.includes("/books")) {
        return new Response(
          JSON.stringify(
            artifact({ content: { summary: "x", source_id: "src-123" } }),
          ),
          { status: 200 },
        );
      }
      if (u.includes("/api/v1/books/src-123/pages/2")) {
        return new Response(
          JSON.stringify(
            bookPage({
              page: 2,
              chunks: [
                {
                  chunk_id: "c3",
                  section_title: null,
                  text: "Page two content.",
                  has_image: false,
                  image_id: null,
                  vlm_description: null,
                  token_count: 3,
                },
              ],
            }),
          ),
          { status: 200 },
        );
      }
      if (u.includes("/api/v1/books/src-123/pages/")) {
        return new Response(JSON.stringify(bookPage()), { status: 200 });
      }
      if (u.endsWith("/api/v1/books/src-123") && !u.includes("/pages") && !u.includes("/summaries")) {
        return new Response(JSON.stringify(bookOverview()), { status: 200 });
      }
      if (u.includes("/images")) {
        return new Response(JSON.stringify({ items: [] }), { status: 200 });
      }
      return new Response("{}", { status: 200 });
    });

    wrap("/library/01ART");

    await waitFor(() =>
      expect(screen.getByText("Page 1 of 120")).toBeInTheDocument(),
    );

    fireEvent.click(screen.getByRole("button", { name: "Next" }));

    await waitFor(() =>
      expect(screen.getByText("Page two content.")).toBeInTheDocument(),
    );
    expect(screen.getByText("Page 2 of 120")).toBeInTheDocument();
  });

  it("Summaries tab renders DSI and summary tree content", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockImplementation(async (url) => {
      const u = String(url);
      if (u.includes("/api/v1/artifacts/01ART") && !u.includes("/images") && !u.includes("/books")) {
        return new Response(
          JSON.stringify(
            artifact({ content: { summary: "x", source_id: "src-123" } }),
          ),
          { status: 200 },
        );
      }
      if (u.includes("/api/v1/books/src-123/pages/")) {
        return new Response(JSON.stringify(bookPage()), { status: 200 });
      }
      if (u.endsWith("/api/v1/books/src-123") && !u.includes("/pages") && !u.includes("/summaries")) {
        return new Response(JSON.stringify(bookOverview()), { status: 200 });
      }
      if (u.includes("/api/v1/books/src-123/summaries")) {
        return new Response(JSON.stringify(bookSummaries()), { status: 200 });
      }
      if (u.includes("/images")) {
        return new Response(JSON.stringify({ items: [] }), { status: 200 });
      }
      return new Response("{}", { status: 200 });
    });

    wrap("/library/01ART");

    await waitFor(() =>
      expect(screen.getByRole("tab", { name: "Summaries" })).toBeInTheDocument(),
    );

    fireEvent.click(screen.getByRole("tab", { name: "Summaries" }));

    await waitFor(() =>
      expect(screen.getByText("A book about racing techniques.")).toBeInTheDocument(),
    );
    expect(screen.getByText("traction")).toBeInTheDocument();
    expect(screen.getByText("apex")).toBeInTheDocument();
    expect(screen.getByText("trail braking")).toBeInTheDocument();
    expect(screen.getByText("Racing is fast")).toBeInTheDocument();
    expect(screen.getByText("Braking matters")).toBeInTheDocument();
    expect(screen.getByText("+42 section summaries indexed")).toBeInTheDocument();
    expect(screen.getByText("Traction (23)")).toBeInTheDocument();
    expect(screen.getByText("150 total nodes")).toBeInTheDocument();
    expect(screen.getByText("80 hierarchical cluster nodes indexed")).toBeInTheDocument();
  });

  it("non-book artifact without source_id still renders JSON fallback", async () => {
    const fetchSpy = vi.mocked(globalThis.fetch);
    fetchSpy.mockImplementation(async (url) => {
      const u = String(url);
      if (u.includes("/api/v1/artifacts/02RES") && !u.includes("/images")) {
        return new Response(
          JSON.stringify(
            artifact({
              id: "02RES",
              pipeline: "repo_analysis",
              title: "My Repo",
              content: { summary: "repo summary" },
            }),
          ),
          { status: 200 },
        );
      }
      return new Response("{}", { status: 200 });
    });

    wrap("/library/02RES");

    await waitFor(() =>
      expect(screen.getByText("My Repo")).toBeInTheDocument(),
    );
    expect(screen.getByRole("link", { name: /query this output/i })).toBeInTheDocument();
    expect(document.querySelector("pre")).toBeInTheDocument();
  });
});
