import { useState } from "react";
import { useIntakeSubmit, useUploadUrl } from "@/api/client";

type UploadState = "idle" | "uploading" | "done" | "error";

function hexFromBuffer(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function sha256Hex(file: File): Promise<string> {
  const bytes =
    typeof file.arrayBuffer === "function"
      ? await file.arrayBuffer()
      : await new Response(file).arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return hexFromBuffer(digest);
}

export default function FileUpload({ onSubmitted }: { onSubmitted: (id: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<UploadState>("idle");
  const [error, setError] = useState("");
  const uploadUrl = useUploadUrl();
  const intake = useIntakeSubmit();

  async function onSubmit() {
    if (!file) {
      return;
    }

    const contentType = file.type || "application/octet-stream";
    setState("uploading");
    setError("");

    try {
      const upload = await uploadUrl.mutateAsync({
        filename: file.name,
        content_type: contentType,
        size_bytes: file.size,
      });
      const put = await fetch(upload.upload_url, {
        method: "PUT",
        headers: { "Content-Type": contentType },
        body: file,
      });
      if (!put.ok) {
        throw new Error(`upload_failed:${put.status}`);
      }

      const fileHash = await sha256Hex(file);
      const env = await intake.mutateAsync({
        source_type: "file_drop",
        source_data: { garage_key: upload.garage_key },
        raw_content: {
          garage_key: upload.garage_key,
          file_hash: fileHash,
          filename: file.name,
          content_type: contentType,
        },
      });
      setState("done");
      onSubmitted(env.envelope_id);
    } catch (err) {
      setState("error");
      setError(err instanceof Error ? err.message : "upload_failed");
    }
  }

  return (
    <div className="space-y-3">
      <input
        type="file"
        aria-label="File"
        onChange={(event) => setFile(event.currentTarget.files?.[0] ?? null)}
        className="w-full rounded border border-zinc-700 bg-zinc-900 px-3 py-2"
      />
      <button
        onClick={onSubmit}
        disabled={!file || state === "uploading"}
        className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
      >
        {state === "uploading" ? "Uploading..." : "Upload File"}
      </button>
      {state === "done" && <div className="text-sm text-emerald-400">File submitted.</div>}
      {state === "error" && (
        <div role="alert" className="text-sm text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}
