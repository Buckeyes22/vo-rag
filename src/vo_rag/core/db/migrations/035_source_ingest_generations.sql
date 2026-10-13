-- Generation-scoped book ingestion.  Qdrant points remain invisible while the
-- generation is staged and become queryable only after validation/smoke gates.

BEGIN;

CREATE TABLE IF NOT EXISTS source_ingest_generations (
    generation_id          TEXT PRIMARY KEY,
    source_id              TEXT NOT NULL,
    raw_collection         TEXT NOT NULL,
    envelope_id            TEXT NOT NULL REFERENCES intake_envelopes(id) ON DELETE RESTRICT,
    run_id                 TEXT,
    pipeline               TEXT NOT NULL DEFAULT 'book_rag',
    schema_version         TEXT NOT NULL,
    chunking_version       TEXT NOT NULL,
    sterilize_enabled      BOOLEAN NOT NULL DEFAULT FALSE,
    owner_token            TEXT,
    claim_expires_at       TIMESTAMPTZ,
    state                  TEXT NOT NULL DEFAULT 'staging',
    expected_raw_points    INTEGER,
    actual_raw_points      INTEGER NOT NULL DEFAULT 0,
    expected_sterile_points INTEGER,
    actual_sterile_points  INTEGER NOT NULL DEFAULT 0,
    expected_artifacts     INTEGER NOT NULL DEFAULT 1,
    actual_artifacts       INTEGER NOT NULL DEFAULT 0,
    validation_decision    JSONB,
    smoke_decision         TEXT,
    failure_reason         TEXT,
    cleanup_required       BOOLEAN NOT NULL DEFAULT FALSE,
    cleanup_error          TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_at           TIMESTAMPTZ,
    completed_at           TIMESTAMPTZ,

    CONSTRAINT source_ingest_generations_state_chk CHECK (
        state IN ('staging','validated','active','quarantined','failed','superseded')
    ),
    CONSTRAINT source_ingest_generations_smoke_chk CHECK (
        smoke_decision IS NULL OR smoke_decision IN ('pass','quarantine','fail','skipped')
    ),
    CONSTRAINT source_ingest_generations_counts_chk CHECK (
        COALESCE(expected_raw_points, 0) >= 0
        AND actual_raw_points >= 0
        AND COALESCE(expected_sterile_points, 0) >= 0
        AND actual_sterile_points >= 0
        AND expected_artifacts >= 0
        AND actual_artifacts >= 0
    ),
    CONSTRAINT source_ingest_generations_active_complete_chk CHECK (
        state <> 'active'
        OR (
            expected_raw_points IS NOT NULL
            AND expected_raw_points = actual_raw_points
            AND expected_sterile_points IS NOT NULL
            AND expected_sterile_points = actual_sterile_points
            AND expected_artifacts = actual_artifacts
            AND validation_decision IS NOT NULL
            AND validation_decision->>'passed' = 'true'
            AND smoke_decision IN ('pass', 'skipped')
        )
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_source_ingest_generations_active
    ON source_ingest_generations (source_id, raw_collection)
    WHERE state = 'active';
CREATE INDEX IF NOT EXISTS idx_source_ingest_generations_envelope
    ON source_ingest_generations (envelope_id);
CREATE INDEX IF NOT EXISTS idx_source_ingest_generations_source_created
    ON source_ingest_generations (source_id, raw_collection, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_source_ingest_generations_state
    ON source_ingest_generations (state);
CREATE INDEX IF NOT EXISTS idx_source_ingest_generations_cleanup
    ON source_ingest_generations (updated_at)
    WHERE cleanup_required;
CREATE INDEX IF NOT EXISTS idx_source_ingest_generations_claim_expiry
    ON source_ingest_generations (claim_expires_at)
    WHERE state IN ('staging', 'validated');

DROP TRIGGER IF EXISTS source_ingest_generations_updated_at ON source_ingest_generations;
CREATE TRIGGER source_ingest_generations_updated_at
    BEFORE UPDATE ON source_ingest_generations
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

COMMIT;
