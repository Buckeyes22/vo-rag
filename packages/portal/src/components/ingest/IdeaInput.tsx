import { useState } from "react";
import { useIntakeSubmit } from "@/api/client";

export default function IdeaInput({ onSubmitted }: { onSubmitted: (id: string) => void }) {
  const [text, setText] = useState("");
  const [buildable, setBuildable] = useState(false);
  const intake = useIntakeSubmit();

  async function onSubmit() {
    const env = await intake.mutateAsync({
      source_type: "text_prompt",
      source_data: { pipeline_hint: "idea_capture" },
      raw_content: { text },
      metadata: { buildable },
    });
    onSubmitted(env.envelope_id);
  }

  return (
    <div className="space-y-3">
      <textarea
        value={text}
        onChange={(event) => setText(event.target.value)}
        rows={6}
        placeholder="Capture the thought..."
        className="w-full rounded border border-zinc-700 bg-zinc-900 px-3 py-2"
        aria-label="Idea"
      />
      <label className="inline-flex items-center gap-2 text-sm text-zinc-300">
        <input
          type="checkbox"
          checked={buildable}
          onChange={(event) => setBuildable(event.target.checked)}
          aria-label="Buildable"
        />
        Buildable?
      </label>
      <button
        onClick={onSubmit}
        disabled={!text.trim()}
        className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
      >
        Submit Idea
      </button>
    </div>
  );
}
