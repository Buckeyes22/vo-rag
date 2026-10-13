import type { QueryHit, QueryMode } from "@/schemas";
import FigureImage from "@/components/figures/FigureImage";

function payloadString(payload: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = payload[key];
    if (typeof value === "string" && value.trim()) {
      return value;
    }
  }
  return undefined;
}

function payloadPage(payload: Record<string, unknown>): string | undefined {
  const value = payload.page_number;
  if (typeof value === "number" || typeof value === "string") {
    return String(value);
  }
  return undefined;
}

export default function ResultCard({
  hit,
  rank,
}: {
  hit: QueryHit;
  rank: number;
  mode: QueryMode;
}) {
  const title =
    payloadString(hit.payload, ["section_title", "title", "source_title", "source_book"]) ??
    hit.collection;
  const content =
    payloadString(hit.payload, ["content", "snippet", "text", "sterile_text", "summary"]) ??
    "(no content)";
  const pageNumber = payloadPage(hit.payload);
  const vlmDescription = payloadString(hit.payload, ["vlm_description"]);
  const imageRef = payloadString(hit.payload, ["image_ref"]);

  return (
    <article className="space-y-2 rounded border border-zinc-800 p-3">
      <header className="flex items-center justify-between text-xs text-zinc-500">
        <span>
          #{rank} · {hit.collection}
        </span>
        <span>score {hit.score.toFixed(3)}</span>
      </header>
      <div>
        <div className="font-medium text-zinc-200">{title}</div>
        {pageNumber && <div className="text-xs text-zinc-500">p. {pageNumber}</div>}
      </div>
      <div className="text-sm text-zinc-300">{content}</div>
      {imageRef && (
        <FigureImage imageRef={imageRef} caption={vlmDescription} />
      )}
      {!imageRef && vlmDescription && (
        <div className="rounded border border-zinc-800 bg-zinc-950 p-2 text-xs text-zinc-400">
          {vlmDescription}
        </div>
      )}
    </article>
  );
}
