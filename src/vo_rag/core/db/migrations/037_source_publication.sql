-- Durable source revision ordering, publication pointers, and resumable jobs.
-- Qdrant payloads are staged by revision; this Postgres pointer is authoritative.

BEGIN;

CREATE TABLE IF NOT EXISTS source_publication_sequences (
    corpus          TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    last_revision   BIGINT NOT NULL DEFAULT 0 CHECK (last_revision >= 0),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (corpus, source_id)
);

CREATE TABLE IF NOT EXISTS source_publications (
    corpus          TEXT NOT NULL,
    collection      TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    active_revision BIGINT,
    source_sha256   TEXT,
    active_job_id   TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (corpus, collection, source_id),
    CHECK (active_revision IS NULL OR active_revision > 0),
    CHECK (source_sha256 IS NULL OR source_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS source_publication_jobs (
    job_id                  TEXT PRIMARY KEY,
    corpus                  TEXT NOT NULL,
    collection              TEXT NOT NULL,
    source_id               TEXT NOT NULL,
    revision                BIGINT NOT NULL CHECK (revision > 0),
    expected_active_revision BIGINT,
    source_sha256           TEXT NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    recipe                  JSONB NOT NULL DEFAULT '{}'::jsonb,
    state                   TEXT NOT NULL DEFAULT 'preparing',
    checkpoint              JSONB NOT NULL DEFAULT '{}'::jsonb,
    retention_disposition   TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    verified_at             TIMESTAMPTZ,
    published_at            TIMESTAMPTZ,
    UNIQUE (corpus, collection, source_id, revision),
    CHECK (expected_active_revision IS NULL OR expected_active_revision > 0),
    CHECK (state IN (
        'preparing', 'staged', 'verified', 'published', 'partial_preserved',
        'preserved_no_write', 'blocked_missing_evidence', 'failed_retryable', 'superseded'
    )),
    CHECK (retention_disposition IS NULL OR retention_disposition IN (
        'retained_old_revision', 'cleanup_complete'
    )),
    CHECK (state <> 'published' OR (
        published_at IS NOT NULL AND retention_disposition IS NOT NULL
    ))
);

CREATE INDEX IF NOT EXISTS idx_source_publication_jobs_source_state
    ON source_publication_jobs (corpus, source_id, state, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_source_publication_jobs_collection_source
    ON source_publication_jobs (corpus, collection, source_id, revision DESC);

DROP TRIGGER IF EXISTS source_publication_sequences_updated_at ON source_publication_sequences;
CREATE TRIGGER source_publication_sequences_updated_at
    BEFORE UPDATE ON source_publication_sequences
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS source_publications_updated_at ON source_publications;
CREATE TRIGGER source_publications_updated_at
    BEFORE UPDATE ON source_publications
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS source_publication_jobs_updated_at ON source_publication_jobs;
CREATE TRIGGER source_publication_jobs_updated_at
    BEFORE UPDATE ON source_publication_jobs
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

COMMIT;
