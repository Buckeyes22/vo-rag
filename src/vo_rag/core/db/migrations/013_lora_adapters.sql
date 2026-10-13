CREATE TABLE IF NOT EXISTS lora_adapters (
    id                   TEXT PRIMARY KEY,
    persona              TEXT NOT NULL CHECK (persona IN ('crew_chief','race_engineer','driving_coach')),
    adapter_path         TEXT NOT NULL UNIQUE,
    base_model           TEXT NOT NULL DEFAULT 'Qwen/Qwen3-8B',
    training_date        TIMESTAMPTZ NOT NULL,
    eval_report          JSONB,
    memorization_report  JSONB,
    active               BOOLEAN NOT NULL DEFAULT false,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_lora_adapter_active_per_persona
    ON lora_adapters(persona) WHERE active = true;
