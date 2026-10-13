CREATE TABLE IF NOT EXISTS query_source_hit (
    id        TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    caller    TEXT NOT NULL,
    hit_at    TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qsh_source_caller_time
    ON query_source_hit(source_id, caller, hit_at DESC);
