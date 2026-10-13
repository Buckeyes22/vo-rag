import { render, screen } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import LiveTriage from "./LiveTriage";

const state = vi.hoisted(() => ({ fallback: false }));
vi.mock("@/api/stream", () => ({
  useIngestStream: () => ({
    events: [], status: state.fallback ? "error" : "done", error: null,
    result: { status: "completed", chunk_count: 3, quality_warnings: [
      { stage: "parse", code: "low_page_coverage", details: "Extracted 1/12 pages" },
    ] },
  }),
}));
vi.mock("@/api/client", () => ({ useEnvelope: () => ({ data: state.fallback ? {
  envelope_id: "test-envelope", status: "completed", metadata: {
    quality_warnings: [{ stage: "validation", code: "audit_unavailable" }],
  },
} : undefined }) }));
beforeEach(() => { state.fallback = false; });

test("completed usable ingestion displays its quality warning", () => {
  render(<LiveTriage envelopeId="test-envelope" />);
  expect(screen.getByText("completed")).toBeTruthy();
  expect(screen.getByText("Ingestion quality warnings")).toBeTruthy();
  expect(screen.getByText(/low_page_coverage/)).toBeTruthy();
  expect(screen.getByText(/Extracted 1\/12 pages/)).toBeTruthy();
});

test("envelope polling retains quality warnings after the stream fails", () => {
  state.fallback = true;
  render(<LiveTriage envelopeId="test-envelope" />);
  expect(screen.getByText("completed")).toBeTruthy();
  expect(screen.getByText("Ingestion quality warnings")).toBeTruthy();
  expect(screen.getByText(/audit_unavailable/)).toBeTruthy();
});
