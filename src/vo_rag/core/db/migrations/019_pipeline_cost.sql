CREATE TABLE IF NOT EXISTS pipeline_cost (
    envelope_id      TEXT PRIMARY KEY,
    pipeline         TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('success', 'failed', 'partial')),
    cost_usd         NUMERIC(12, 6) NOT NULL,
    wall_clock_ms    BIGINT NOT NULL,
    step_breakdown   JSONB NOT NULL,
    token_breakdown  JSONB NOT NULL,
    started_at       TIMESTAMPTZ NOT NULL,
    finished_at      TIMESTAMPTZ NOT NULL,
    git_sha          TEXT,
    model_mix        TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[]
);

CREATE INDEX IF NOT EXISTS idx_pipeline_cost_pipeline_time
    ON pipeline_cost(pipeline, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_pipeline_cost_cost_usd
    ON pipeline_cost(cost_usd DESC);

CREATE INDEX IF NOT EXISTS idx_pipeline_cost_status
    ON pipeline_cost(status, started_at DESC);
