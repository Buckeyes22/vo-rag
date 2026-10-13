import { useArtifactImages } from "@/api/client";

export default function ImageGallery({ artifactId }: { artifactId: string }) {
  const { data, isError } = useArtifactImages(artifactId);
  if (isError) {
    return <div className="text-zinc-500 text-sm">No images.</div>;
  }
  const items = data?.items ?? [];
  if (items.length === 0) {
    return null;
  }
  return (
    <section aria-label="Image gallery" className="mt-4">
      <h3 className="text-sm font-semibold text-zinc-300 mb-2">Images</h3>
      <div className="grid grid-cols-3 gap-2">
        {items.map((im) => (
          <figure key={im.id} className="rounded border border-zinc-800 p-1">
            <img src={im.url} alt={im.caption ?? im.id} className="w-full h-auto" />
            {im.caption && <figcaption className="text-xs text-zinc-400 mt-1">{im.caption}</figcaption>}
          </figure>
        ))}
      </div>
    </section>
  );
}
