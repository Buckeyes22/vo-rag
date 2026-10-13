import { useState } from "react";
import { useIntakeSubmit } from "@/api/client";

type Props = {
  onSubmitted: (id: string) => void;
  pipelineHint?: "article_extraction" | "repo_analysis" | "tool_evaluation" | "url_snapshot" | "";
  label?: string;
  buttonLabel?: string;
};

export default function UrlInput({
  onSubmitted,
  pipelineHint = "",
  label = "URL",
  buttonLabel = "Submit URL",
}: Props) {
  const [url, setUrl] = useState("");
  const [title, setTitle] = useState("");
  const [tags, setTags] = useState("");
  const intake = useIntakeSubmit();

  async function onSubmit() {
    const env = await intake.mutateAsync({
      source_type: "url",
      source_data: pipelineHint ? { pipeline_hint: pipelineHint } : {},
      raw_content: { url },
      metadata: {
        ...(title ? { title } : {}),
        ...(tags
          ? { tags: tags.split(",").map((t) => t.trim()).filter(Boolean) }
          : {}),
      },
    });
    onSubmitted(env.envelope_id);
  }

  return (
    <div className="space-y-3">
      <input
        value={url}
        onChange={(event) => setUrl(event.target.value)}
        placeholder="https://..."
        className="w-full rounded border border-zinc-700 bg-zinc-900 px-3 py-2"
        aria-label={label}
      />
      <input
        value={title}
        onChange={(event) => setTitle(event.target.value)}
        placeholder="Title (optional)"
        className="w-full rounded border border-zinc-700 bg-zinc-900 px-3 py-2"
        aria-label="Title"
      />
      <input
        value={tags}
        onChange={(event) => setTags(event.target.value)}
        placeholder="Tags (optional)"
        className="w-full rounded border border-zinc-700 bg-zinc-900 px-3 py-2"
        aria-label="Tags"
      />
      <button
        onClick={onSubmit}
        disabled={!url}
        className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
      >
        {buttonLabel}
      </button>
    </div>
  );
}
