-- VOR-1443: track-and-field contextual tags. Book records (auto-loaded drafts; `reviewed` once the operator checks them)
-- and chunk classifications keyed by content hash so re-ingest re-applies them.
CREATE TABLE IF NOT EXISTS tf_book_tags (
    source_id TEXT PRIMARY KEY,
    tags JSONB NOT NULL,
    reviewed BOOLEAN NOT NULL DEFAULT FALSE,
    tag_version INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tf_chunk_tags (
    source_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    tags JSONB NOT NULL,
    tag_version INTEGER NOT NULL,
    tagger_model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_id, content_sha256)
);
