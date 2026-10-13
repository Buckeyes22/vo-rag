CREATE TABLE IF NOT EXISTS backup_log (
    id              TEXT PRIMARY KEY,
    backup_type     TEXT NOT NULL,
    target          TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('success','failure','in_progress')),
    size_bytes      BIGINT,
    garage_key      TEXT,
    error_message   TEXT,
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backup_log_type_time
    ON backup_log(backup_type, started_at DESC);
