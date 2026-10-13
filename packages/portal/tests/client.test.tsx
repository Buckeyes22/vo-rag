import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import { ApiError, useArtifactImages, useHealth, useQueryRun, useUploadUrl } from "@/api/client";
import { useAppStore } from "@/stores/app";

function wrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe("api client hooks", () => {
  beforeEach(() => {
    vi.spyOn(globalThis, "fetch");
    useAppStore.setState({
      queryMode: "internal",
      csrfToken: "csrf-xyz",
      sessionEmail: "admin@example.com",
      authSource: "cloudflare_access",
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("useHealth sends portal caller without browser API key", async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          status: "ok",
          service: "vo-rag",
        }),
        { status: 200 },
      ),
    );

    const { result } = renderHook(() => useHealth(), { wrapper: wrapper() });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const headers = (fetchMock.mock.calls[0]?.[1] as RequestInit).headers as Record<
      string,
      string
    >;
    expect(headers["X-API-Key"]).toBeUndefined();
    expect(headers["X-VO-RAG-Caller"]).toBe("portal");
  });

  it("useHealth error -> ApiError with status code", async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockResolvedValue(new Response("nope", { status: 401, statusText: "Unauthorized" }));

    const { result } = renderHook(() => useHealth(), { wrapper: wrapper() });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeInstanceOf(ApiError);
    expect((result.current.error as ApiError).status).toBe(401);
  });

  it("useHealth Zod failure rejects invalid liveness values", async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ status: [], service: "vo-rag" }), { status: 200 }),
    );

    const { result } = renderHook(() => useHealth(), { wrapper: wrapper() });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });
  });

  it("useQueryRun POSTs request body", async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ results: [] }), { status: 200 }));

    const { result } = renderHook(() => useQueryRun(), { wrapper: wrapper() });
    await result.current.mutateAsync({ query: "trail braking" });

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const headers = init.headers as Record<string, string>;
    expect(init.method).toBe("POST");
    expect(headers["X-CSRF-Token"]).toBe("csrf-xyz");
    expect(JSON.parse(init.body as string).query).toBe("trail braking");
  });

  it("useUploadUrl parses bucket and expiry from the presigned response", async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          upload_url: "https://garage/upload",
          garage_key: "intake/01H/book.pdf",
          bucket: "artifacts",
          expires_in_s: 21600,
        }),
        { status: 200 },
      ),
    );

    const { result } = renderHook(() => useUploadUrl(), { wrapper: wrapper() });
    const response = await result.current.mutateAsync({
      filename: "book.pdf",
      content_type: "application/pdf",
      size_bytes: 12,
    });

    expect(response.bucket).toBe("artifacts");
    expect(response.expires_in_s).toBe(21600);
  });

  it("useArtifactImages accepts null captions and no expires_at field", async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          items: [{ id: "image-1", url: "/api/v1/figures/book/page.png", caption: null }],
        }),
        { status: 200 },
      ),
    );

    const { result } = renderHook(() => useArtifactImages("artifact-1"), { wrapper: wrapper() });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });
    expect(result.current.data?.items[0]?.caption).toBeNull();
  });
});
