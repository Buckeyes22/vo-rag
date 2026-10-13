import type { ReactNode } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Ingest from "@/routes/Ingest";
import { useAppStore } from "@/stores/app";

function wrap(children: ReactNode) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Ingest view", () => {
  beforeEach(() => {
    vi.spyOn(globalThis, "fetch");
    useAppStore.setState({ queryMode: "internal", csrfToken: "csrf" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders all intake launchers with Article active by default", () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ items: [] }), { status: 200 }));

    wrap(<Ingest />);

    for (const label of [
      "Article",
      "Repository",
      "Research",
      "Idea",
      "File",
      "Tool/Library",
      "Snapshot",
    ]) {
      expect(screen.getByRole("tab", { name: label })).toBeInTheDocument();
    }
    expect(screen.getByRole("tab", { name: "Article" })).toHaveAttribute("aria-selected", "true");
  });

  it("URL submission POSTs to /api/v1/intake", async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockImplementation(async (url, init) => {
      if (String(url).includes("/api/v1/envelopes")) {
        return new Response(JSON.stringify({ items: [] }), { status: 200 });
      }
      if (String(url).includes("/api/v1/intake") && (init as RequestInit)?.method === "POST") {
        return new Response(
          JSON.stringify({ envelope_id: "01H", source_type: "url", status: "received" }),
          { status: 200 },
        );
      }
      return new Response("{}", { status: 200 });
    });

    wrap(<Ingest />);

    fireEvent.change(screen.getByLabelText("URL"), { target: { value: "https://example.com" } });
    fireEvent.click(screen.getByRole("button", { name: /Submit Article/i }));

    await waitFor(() => {
      const called = fetchMock.mock.calls.some(
        ([url, init]) =>
          String(url).includes("/api/v1/intake") && (init as RequestInit | undefined)?.method === "POST",
      );
      expect(called).toBe(true);
    });
  });

  it("Idea submission includes source_data pipeline_hint", async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockImplementation(async (url, init) => {
      if (String(url).includes("/api/v1/envelopes")) {
        return new Response(JSON.stringify({ items: [] }), { status: 200 });
      }
      if (String(url).includes("/api/v1/intake") && (init as RequestInit)?.method === "POST") {
        return new Response(
          JSON.stringify({ envelope_id: "01H", source_type: "text_prompt", status: "received" }),
          { status: 200 },
        );
      }
      return new Response("{}", { status: 200 });
    });

    wrap(<Ingest />);

    fireEvent.click(screen.getByRole("tab", { name: "Idea" }));
    fireEvent.change(screen.getByLabelText("Idea"), { target: { value: "thought" } });
    fireEvent.click(screen.getByRole("button", { name: /Submit Idea/i }));

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([url, init]) =>
          String(url).includes("/api/v1/intake") && (init as RequestInit | undefined)?.method === "POST",
      );
      expect(call).toBeDefined();
      const body = JSON.parse((call?.[1] as RequestInit).body as string) as {
        source_data: { pipeline_hint: string };
        metadata: { hint_pipeline?: string };
      };
      expect(body.source_data.pipeline_hint).toBe("idea_capture");
      expect(body.metadata.hint_pipeline).toBeUndefined();
    });
  });

  it("Tag input parses comma-separated tags into array", async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    let captured: unknown = null;
    fetchMock.mockImplementation(async (url, init) => {
      if (String(url).includes("/api/v1/intake") && (init as RequestInit)?.method === "POST") {
        captured = JSON.parse((init as RequestInit).body as string);
        return new Response(
          JSON.stringify({ envelope_id: "01H", source_type: "url", status: "received" }),
          { status: 200 },
        );
      }
      return new Response(JSON.stringify({ items: [] }), { status: 200 });
    });

    wrap(<Ingest />);

    fireEvent.change(screen.getByLabelText("URL"), { target: { value: "https://x" } });
    fireEvent.change(screen.getByLabelText("Tags"), { target: { value: "braking, technique, apex" } });
    fireEvent.click(screen.getByRole("button", { name: /Submit Article/i }));

    await waitFor(() => {
      expect((captured as { metadata: { tags: string[] } }).metadata.tags).toEqual([
        "braking",
        "technique",
        "apex",
      ]);
    });
  });

  it("File upload uses presigned PUT then submits file_drop intake", async () => {
    const digest = vi.fn().mockResolvedValue(new Uint8Array([0xab, 0xcd]).buffer);
    vi.stubGlobal("crypto", { subtle: { digest } });

    const fetchMock = vi.mocked(globalThis.fetch);
    const capturedBodies: unknown[] = [];
    fetchMock.mockImplementation(async (url, init) => {
      const method = (init as RequestInit | undefined)?.method ?? "GET";
      const href = String(url);
      if (href.includes("/api/v1/envelopes")) {
        return new Response(JSON.stringify({ items: [] }), { status: 200 });
      }
      if (href.includes("/api/v1/intake/upload-url") && method === "POST") {
        capturedBodies.push(JSON.parse((init as RequestInit).body as string));
        return new Response(
          JSON.stringify({
            upload_url: "https://garage.example/upload",
            garage_key: "intake/01H/book.pdf",
            bucket: "artifacts",
            expires_in_s: 21600,
          }),
          { status: 200 },
        );
      }
      if (href === "https://garage.example/upload" && method === "PUT") {
        expect((init as RequestInit).headers).toMatchObject({ "Content-Type": "application/pdf" });
        expect((init as RequestInit).body).toBeInstanceOf(File);
        return new Response("", { status: 200 });
      }
      if (href.includes("/api/v1/intake") && method === "POST") {
        capturedBodies.push(JSON.parse((init as RequestInit).body as string));
        return new Response(
          JSON.stringify({
            envelope_id: "01H",
            source_type: "file_drop",
            status: "received",
            priority: "medium",
            sovereignty: "cloud_ok",
            created_at: "2026-06-11T00:00:00Z",
            updated_at: "2026-06-11T00:00:00Z",
          }),
          { status: 200 },
        );
      }
      return new Response("{}", { status: 200 });
    });

    wrap(<Ingest />);

    fireEvent.click(screen.getByRole("tab", { name: "File" }));
    const file = new File(["pdf bytes"], "book.pdf", { type: "application/pdf" });
    fireEvent.change(screen.getByLabelText("File"), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: /Upload File/i }));

    await waitFor(() => {
      const intakeBody = capturedBodies.find(
        (body): body is {
          source_type: string;
          source_data: Record<string, unknown>;
          raw_content: Record<string, unknown>;
        } =>
          Boolean(body) &&
          typeof body === "object" &&
          body !== null &&
          "source_type" in body &&
          (body as { source_type?: string }).source_type === "file_drop",
      );
      expect(intakeBody).toBeDefined();
      expect(intakeBody?.source_data.garage_key).toBe("intake/01H/book.pdf");
      expect(intakeBody?.raw_content).toMatchObject({
        garage_key: "intake/01H/book.pdf",
        file_hash: "abcd",
        filename: "book.pdf",
        content_type: "application/pdf",
      });
    });
    expect(digest).toHaveBeenCalledWith("SHA-256", expect.any(ArrayBuffer));
  });
});
