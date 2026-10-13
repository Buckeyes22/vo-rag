CREATE TABLE IF NOT EXISTS citation_assignments (
    id            TEXT PRIMARY KEY,
    output_id     TEXT NOT NULL,
    text_segment  TEXT NOT NULL,
    start_offset  INT  NOT NULL,
    end_offset    INT  NOT NULL,
    tier          TEXT NOT NULL CHECK (tier IN ('general_knowledge','author_associated','paraphrased','direct_quote')),
    source_id     TEXT,
    source_title  TEXT,
    confidence    REAL NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cit_output ON citation_assignments(output_id);
CREATE INDEX IF NOT EXISTS idx_cit_tier ON citation_assignments(tier);
