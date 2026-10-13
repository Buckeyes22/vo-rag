import { useState } from "react";
import { useIntakeSubmit } from "@/api/client";

export default function TextInput({
  onSubmitted,
  pipelineHint = "research_agent",
  label = "Text",
  placeholder = "Research question or free text...",
  buttonLabel = "Submit Text",
}: {
  onSubmitted: (id: string) => void;
  pipelineHint?: "research_agent" | "";
  label?: string;
  placeholder?: string;
  buttonLabel?: string;
}) {
  const [text, setText] = useState("");
  const intake = useIntakeSubmit();

  async function onSubmit() {
    const env = await intake.mutateAsync({
      source_type: "text_prompt",
      source_data: pipelineHint ? { pipeline_hint: pipelineHint } : {},
      raw_content: { text },
      metadata: {},
    });
    onSubmitted(env.envelope_id);
  }

  return (
    <div className="space-y-3">
      <textarea
        value={text}
        onChange={(event) => setText(event.target.value)}
        rows={6}
        placeholder={placeholder}
        className="w-full rounded border border-zinc-700 bg-zinc-900 px-3 py-2"
        aria-label={label}
      />
      <button
        onClick={onSubmit}
        disabled={!text.trim()}
        className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
      >
        {buttonLabel}
      </button>
    </div>
  );
}
