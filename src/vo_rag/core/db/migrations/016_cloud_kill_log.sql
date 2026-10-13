CREATE TABLE IF NOT EXISTS cloud_kill_log (
    id                             BIGSERIAL PRIMARY KEY,
    triggered_at                   TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason                         TEXT NOT NULL,
    tailscale_acl_updated          BOOLEAN NOT NULL,
    devices_removed                INT NOT NULL,
    r2_credentials_rotated         BOOLEAN NOT NULL,
    skypilot_clusters_destroyed    INT NOT NULL,
    total_duration_ms              INT NOT NULL,
    errors                         JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_cloud_kill_log_triggered ON cloud_kill_log (triggered_at DESC);
