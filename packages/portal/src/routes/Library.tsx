import { useState } from "react";
import CollectionTabs, { TABS, type CollectionId } from "@/components/library/CollectionTabs";
import ArtifactList from "@/components/library/ArtifactList";
import ArtifactFilters from "@/components/library/ArtifactFilters";

export default function Library() {
  const [tab, setTab] = useState<CollectionId>("all");
  const [search, setSearch] = useState("");
  const pipeline = TABS.find((t) => t.id === tab)?.pipeline;

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold">Library</h2>
      <p className="sr-only">Library</p>
      <CollectionTabs active={tab} onChange={setTab} />
      <ArtifactFilters search={search} onSearch={setSearch} />
      <ArtifactList pipeline={pipeline} />
    </div>
  );
}
