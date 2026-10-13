CREATE TABLE IF NOT EXISTS triage_cost (
    id            TEXT PRIMARY KEY,
    envelope_id   TEXT NOT NULL REFERENCES intake_envelopes(id),
    model         TEXT NOT NULL,
    input_tokens  INT  NOT NULL,
    output_tokens INT  NOT NULL,
    cost_usd      NUMERIC(10, 6) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_triage_cost_created ON triage_cost(created_at DESC);
