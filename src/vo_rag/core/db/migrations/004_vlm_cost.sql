CREATE TABLE IF NOT EXISTS vlm_cost (
    id            TEXT PRIMARY KEY,
    image_id      TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INT  NOT NULL DEFAULT 0,
    output_tokens INT  NOT NULL DEFAULT 0,
    cost_usd      NUMERIC(10, 6) NOT NULL DEFAULT 0.0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vlm_cost_model ON vlm_cost(model);
CREATE INDEX IF NOT EXISTS idx_vlm_cost_created ON vlm_cost(created_at);
