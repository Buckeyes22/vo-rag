CREATE TABLE IF NOT EXISTS query_log (
    id          TEXT PRIMARY KEY,
    query_text  TEXT NOT NULL,
    mode        TEXT NOT NULL CHECK (mode IN ('internal','publication','partner')),
    collections TEXT[] NOT NULL,
    top_k       INT,
    caller      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_log_mode ON query_log(mode);
CREATE INDEX IF NOT EXISTS idx_query_log_time ON query_log(created_at DESC);
