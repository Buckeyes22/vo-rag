CREATE TABLE IF NOT EXISTS sterile_samples (
    id                  TEXT PRIMARY KEY,
    parent_chunk_id     TEXT NOT NULL,
    sterile_chunk_id    TEXT NOT NULL,
    source_id           TEXT NOT NULL,
    raw_content         TEXT NOT NULL,
    sterile_content     TEXT NOT NULL,
    sterilization_model TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sterile_samples_source ON sterile_samples(source_id);
CREATE INDEX IF NOT EXISTS idx_sterile_samples_time   ON sterile_samples(created_at DESC);
