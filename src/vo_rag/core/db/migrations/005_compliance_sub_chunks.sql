CREATE TABLE IF NOT EXISTS compliance_sub_chunks (
    sub_chunk_id        TEXT PRIMARY KEY,
    parent_chunk_id     TEXT NOT NULL,
    content             TEXT NOT NULL,
    token_count         INT  NOT NULL,
    position_in_parent  INT  NOT NULL,
    source_id           TEXT NOT NULL,
    page_number         INT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_compliance_sub_parent ON compliance_sub_chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_compliance_sub_source ON compliance_sub_chunks(source_id);
