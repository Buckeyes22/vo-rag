import { useParams, Link } from "react-router-dom";
import { useArtifact } from "@/api/client";
import ArtifactDetail from "@/components/library/ArtifactDetail";
import ProvenanceSidebar from "@/components/library/ProvenanceSidebar";

export default function LibraryDetail() {
  const { id } = useParams();
  const { data, isLoading, isError } = useArtifact(id);

  if (isLoading) {
    return <div className="text-zinc-500">Loading…</div>;
  }
  if (isError) {
    return (
      <div role="alert" className="text-red-400">
        Failed to load artifact.
      </div>
    );
  }
  if (!data) {
    return <div className="text-zinc-500">Not found.</div>;
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-[1fr_18rem] gap-6">
      <div>
        <Link to="/library" className="text-xs text-zinc-400 hover:underline">
          ← Library
        </Link>
        <ArtifactDetail artifact={data} />
      </div>
      <ProvenanceSidebar artifact={data} />
    </div>
  );
}
