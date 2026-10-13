-- Structured scrape pipeline: source registry + JSONB fallback table.
-- Per-adapter typed tables ship in their own migrations (031+).

CREATE TABLE IF NOT EXISTS scraped_sources (
    id              TEXT PRIMARY KEY,                          -- ULID
    url             TEXT NOT NULL UNIQUE,
    adapter_id      TEXT NOT NULL,
    last_status     TEXT NOT NULL DEFAULT 'ok'
        CHECK (last_status IN ('ok','fetch_failed','extract_failed','partial','quarantined')),
    last_scrape_at  TIMESTAMPTZ,
    schema_version  TEXT NOT NULL DEFAULT 'v1',
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS scraped_sources_adapter_id_idx
    ON scraped_sources (adapter_id);

CREATE INDEX IF NOT EXISTS scraped_sources_last_status_idx
    ON scraped_sources (last_status);

CREATE TABLE IF NOT EXISTS scraped_records_generic (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES scraped_sources(id) ON DELETE CASCADE,
    payload         JSONB NOT NULL,
    extracted_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS scraped_records_generic_source_id_idx
    ON scraped_records_generic (source_id);

CREATE INDEX IF NOT EXISTS scraped_records_generic_payload_gin
    ON scraped_records_generic USING GIN (payload);
