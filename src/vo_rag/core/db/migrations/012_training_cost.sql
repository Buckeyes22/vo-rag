CREATE TABLE training_cost (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    persona TEXT NOT NULL CHECK (persona IN ('crew_chief','race_engineer','driving_coach')),
    stage TEXT NOT NULL CHECK (stage IN ('seed','bulk','raft','preference')),
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd NUMERIC(10, 6) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_training_cost_run ON training_cost(run_id);
CREATE INDEX idx_training_cost_persona_stage ON training_cost(persona, stage);
