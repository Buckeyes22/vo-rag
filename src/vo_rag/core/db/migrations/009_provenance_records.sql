CREATE TABLE IF NOT EXISTS provenance_records (
    output_id            TEXT PRIMARY KEY,
    query_text           TEXT,
    query_mode           TEXT NOT NULL CHECK (query_mode IN ('internal','publication','partner')),
    total_sources        INT  NOT NULL,
    collection_searched  TEXT NOT NULL CHECK (collection_searched IN ('raw','sterile','mixed')),
    generation_model     TEXT,
    sources_used         JSONB NOT NULL,
    retrieval_timestamp  TIMESTAMPTZ NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prov_mode ON provenance_records(query_mode);
CREATE INDEX IF NOT EXISTS idx_prov_time ON provenance_records(created_at DESC);
