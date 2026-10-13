-- R2 staging audit log: who staged what, when, and when cleanup removed it.

CREATE TABLE IF NOT EXISTS r2_staging_log (
    r2_key           TEXT PRIMARY KEY,
    direction        TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    envelope_id      TEXT,
    bytes            BIGINT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    pulled_at        TIMESTAMPTZ,
    cleaned_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_r2_staging_created ON r2_staging_log(created_at DESC);

CREATE TABLE IF NOT EXISTS r2_staging_cleanup_run (
    id               BIGSERIAL PRIMARY KEY,
    ran_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    removed          INT NOT NULL
);
