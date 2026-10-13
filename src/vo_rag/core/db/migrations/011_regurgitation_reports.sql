CREATE TABLE IF NOT EXISTS regurgitation_reports (
    id                        TEXT PRIMARY KEY,
    total_tests               INT NOT NULL,
    extraction_tests          INT NOT NULL,
    reconstruction_tests      INT NOT NULL,
    verbatim_detected         INT NOT NULL,
    near_verbatim_detected    INT NOT NULL,
    max_verbatim_length       INT NOT NULL,
    max_similarity_score      REAL NOT NULL,
    passed                    BOOLEAN NOT NULL,
    failed_tests              JSONB NOT NULL,
    threshold_recommendations JSONB NOT NULL,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reg_time   ON regurgitation_reports(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reg_passed ON regurgitation_reports(passed);
