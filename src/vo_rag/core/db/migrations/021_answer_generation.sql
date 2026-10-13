CREATE TABLE IF NOT EXISTS answer_log (
    id                    TEXT PRIMARY KEY,
    query_log_id          TEXT REFERENCES query_log(id) ON DELETE SET NULL,
    query_text            TEXT NOT NULL,
    mode                  TEXT NOT NULL CHECK (mode IN ('internal','publication','partner')),
    profile               TEXT NOT NULL CHECK (profile IN ('quality','fast','kimi_benchmark')),
    model                 TEXT NOT NULL,
    prompt_name           TEXT,
    prompt_version        INT,
    collections           TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    context_scope         TEXT NOT NULL CHECK (context_scope IN ('sterile','raw')),
    retrieved_count       INT NOT NULL DEFAULT 0,
    source_ids            TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    latency_ms            DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_usd              DOUBLE PRECISION NOT NULL DEFAULT 0,
    compliance_outcome    TEXT NOT NULL,
    warnings              TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_answer_log_query_log ON answer_log(query_log_id);
CREATE INDEX IF NOT EXISTS idx_answer_log_mode_created ON answer_log(mode, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_answer_log_model_created ON answer_log(model, created_at DESC);

CREATE TABLE IF NOT EXISTS answer_feedback (
    id                    TEXT PRIMARY KEY,
    answer_id             TEXT NOT NULL REFERENCES answer_log(id) ON DELETE CASCADE,
    rating                TEXT NOT NULL CHECK (rating IN ('up','down')),
    reason                TEXT,
    selected_source_ids   TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    caller                TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_answer_feedback_answer ON answer_feedback(answer_id);
CREATE INDEX IF NOT EXISTS idx_answer_feedback_created ON answer_feedback(created_at DESC);
