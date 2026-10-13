CREATE TABLE IF NOT EXISTS agent_termination_log (
    id             TEXT PRIMARY KEY,
    envelope_id    TEXT NOT NULL,
    pipeline       TEXT NOT NULL,
    reason         TEXT NOT NULL CHECK (reason IN (
        'success',
        'max_steps',
        'max_cost_usd',
        'max_wall_clock',
        'max_sub_questions',
        'loop_detected',
        'tool_validation_failed',
        'upstream_error'
    )),
    step_count         INTEGER  NOT NULL,
    cost_usd           NUMERIC(12,6) NOT NULL,
    wall_clock_s       DOUBLE PRECISION NOT NULL,
    sub_questions      INTEGER  NOT NULL DEFAULT 0,
    detail             TEXT,
    extra              JSONB    NOT NULL DEFAULT '{}'::jsonb,
    occurred_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_termination_pipeline_time
    ON agent_termination_log(pipeline, occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_termination_reason
    ON agent_termination_log(reason, occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_termination_envelope
    ON agent_termination_log(envelope_id);
