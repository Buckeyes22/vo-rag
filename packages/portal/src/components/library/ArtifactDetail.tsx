import { Link } from "react-router-dom";
import type { KnowledgeArtifact } from "@/schemas";
import BookView from "./BookView";

export default function ArtifactDetail({ artifact }: { artifact: KnowledgeArtifact }) {
  if (
    artifact.pipeline === "book_rag" &&
    typeof artifact.content?.source_id === "string"
  ) {
    return <BookView artifact={artifact} />;
  }

  const queryParams = new URLSearchParams({
    mode: "internal",
    artifact_id: artifact.id,
  });
  if (artifact.qdrant_collection) {
    queryParams.set("cols", artifact.qdrant_collection);
  }

  return (
    <div className="space-y-4">
      <header>
        <h2 className="text-xl font-semibold">{artifact.title ?? artifact.id}</h2>
        <div className="text-xs text-zinc-500">
          {artifact.pipeline} · {artifact.created_at}
          {artifact.license_class ? ` · ${artifact.license_class}` : ""}
        </div>
        <Link
          to={`/query?${queryParams.toString()}`}
          className="mt-3 inline-block rounded border border-zinc-700 px-3 py-2 text-sm text-zinc-200 hover:bg-zinc-800"
        >
          Query this output
        </Link>
      </header>

      <pre className="rounded bg-zinc-900 p-3 text-xs overflow-auto text-zinc-300">
        {JSON.stringify(artifact, null, 2)}
      </pre>
    </div>
  );
}
