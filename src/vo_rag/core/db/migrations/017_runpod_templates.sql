-- Cache of private RunPod "My Templates" IDs keyed by logical template name
-- and the image SHA they reference.

CREATE TABLE IF NOT EXISTS runpod_templates (
    template_id       TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    image_sha         TEXT NOT NULL,
    image_ref         TEXT NOT NULL,              -- e.g. "registry.example.com/cloud-worker:abc123"
    registry_auth_id  TEXT,                       -- RunPod registry credential id (nullable if public image)
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, image_sha)
);

CREATE INDEX IF NOT EXISTS runpod_templates_image_sha_idx
    ON runpod_templates (image_sha);
