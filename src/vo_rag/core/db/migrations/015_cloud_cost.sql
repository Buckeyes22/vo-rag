-- vo-rag cloud cost ledger (static-router: text-gen / vlm / dsi / embed-batch / misc).

CREATE TABLE IF NOT EXISTS cloud_cost (
    job_id        TEXT PRIMARY KEY,
    provider      TEXT NOT NULL CHECK (provider IN ('runpod', 'vast', 'crusoe')),
    gpu_type      TEXT NOT NULL,
    workload      TEXT NOT NULL CHECK (workload IN ('re-embedding', 'ocr-burst', 'sft-generation', 'misc')),
    started_at    TIMESTAMPTZ NOT NULL,
    completed_at  TIMESTAMPTZ,
    duration_s    INT,
    cost_usd      NUMERIC(10, 4) NOT NULL DEFAULT 0,
    status        TEXT NOT NULL CHECK (status IN ('launching', 'running', 'completed', 'failed', 'killed'))
);
CREATE INDEX IF NOT EXISTS idx_cloud_cost_started ON cloud_cost(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_cloud_cost_status  ON cloud_cost(status);
