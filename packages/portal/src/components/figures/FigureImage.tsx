/**
 * Figure image from the vo-rag figure store. `imageRef` is the opaque
 * relative key from hit payloads ({src64hex}/pNNN_imgNNN.ext); the serving
 * route is immutable-cacheable so browsers cache aggressively. Same-origin
 * cookies (Cloudflare Access) authenticate the request; in dev the Vite
 * proxy injects the API key.
 */
export default function FigureImage({
  imageRef,
  caption,
  className,
}: {
  imageRef: string;
  caption?: string | null | undefined;
  className?: string;
}) {
  return (
    <figure className={className}>
      <img
        src={`/api/v1/figures/${imageRef}`}
        alt={caption ?? "extracted figure"}
        loading="lazy"
        className="max-h-64 rounded border border-zinc-800 object-contain"
        onError={(e) => {
          // Missing file (404): degrade to description-only per the spec.
          (e.currentTarget.closest("figure") as HTMLElement).style.display = "none";
        }}
      />
      {caption ? (
        <figcaption className="mt-1 text-xs text-zinc-400">{caption}</figcaption>
      ) : null}
    </figure>
  );
}
