import { useState } from "react";
import FileUpload from "@/components/ingest/FileUpload";
import IdeaInput from "@/components/ingest/IdeaInput";
import TextInput from "@/components/ingest/TextInput";
import UrlInput from "@/components/ingest/UrlInput";

type Tab =
  | "article"
  | "repo"
  | "research"
  | "idea"
  | "file"
  | "tool"
  | "snapshot";

const TABS: ReadonlyArray<{ id: Tab; label: string }> = [
  { id: "article", label: "Article" },
  { id: "repo", label: "Repository" },
  { id: "research", label: "Research" },
  { id: "idea", label: "Idea" },
  { id: "file", label: "File" },
  { id: "tool", label: "Tool/Library" },
  { id: "snapshot", label: "Snapshot" },
];

export default function IngestForm({ onSubmitted }: { onSubmitted: (id: string) => void }) {
  const [tab, setTab] = useState<Tab>("article");

  return (
    <div className="space-y-4">
      <div role="tablist" className="flex gap-2 border-b border-zinc-800">
        {TABS.map((tabDef) => (
          <button
            key={tabDef.id}
            role="tab"
            aria-selected={tab === tabDef.id}
            onClick={() => setTab(tabDef.id)}
            className={`px-3 py-2 text-sm ${
              tab === tabDef.id ? "border-b-2 border-blue-500 text-blue-300" : "text-zinc-400"
            }`}
          >
            {tabDef.label}
          </button>
        ))}
      </div>

      <div>
        {tab === "article" && (
          <UrlInput
            onSubmitted={onSubmitted}
            pipelineHint="article_extraction"
            buttonLabel="Submit Article"
          />
        )}
        {tab === "repo" && (
          <UrlInput
            onSubmitted={onSubmitted}
            pipelineHint="repo_analysis"
            label="Repository URL"
            buttonLabel="Submit Repository"
          />
        )}
        {tab === "research" && (
          <TextInput
            onSubmitted={onSubmitted}
            pipelineHint="research_agent"
            label="Research prompt"
            placeholder="Question or research brief..."
            buttonLabel="Submit Research"
          />
        )}
        {tab === "idea" && <IdeaInput onSubmitted={onSubmitted} />}
        {tab === "file" && <FileUpload onSubmitted={onSubmitted} />}
        {tab === "tool" && (
          <UrlInput
            onSubmitted={onSubmitted}
            pipelineHint="tool_evaluation"
            label="Tool or library URL"
            buttonLabel="Submit Tool"
          />
        )}
        {tab === "snapshot" && (
          <UrlInput
            onSubmitted={onSubmitted}
            pipelineHint="url_snapshot"
            label="Snapshot URL"
            buttonLabel="Submit Snapshot"
          />
        )}
      </div>
    </div>
  );
}
