BEGIN;

CREATE TABLE IF NOT EXISTS portal_upload_sessions (
    id              TEXT PRIMARY KEY,
    owner_email     TEXT NOT NULL,
    filename        TEXT NOT NULL,
    content_type    TEXT NOT NULL,
    size_bytes      BIGINT NOT NULL,
    chunk_size      INTEGER NOT NULL,
    received_bytes  BIGINT NOT NULL DEFAULT 0,
    sha256          TEXT,
    status          TEXT NOT NULL DEFAULT 'created',
    staging_path    TEXT NOT NULL,
    garage_bucket   TEXT NOT NULL,
    garage_key      TEXT NOT NULL,
    envelope_id     TEXT REFERENCES intake_envelopes(id) ON DELETE SET NULL,
    error           TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_data     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,

    CONSTRAINT portal_upload_sessions_status_chk CHECK (
        status IN ('created','uploading','completed','cancelled','failed')
    )
);

CREATE TABLE IF NOT EXISTS portal_audit_log (
    id          BIGSERIAL PRIMARY KEY,
    actor_email TEXT NOT NULL,
    auth_source TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    entity_type TEXT,
    entity_id   TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_portal_upload_sessions_status
    ON portal_upload_sessions (status);
CREATE INDEX IF NOT EXISTS idx_portal_upload_sessions_owner
    ON portal_upload_sessions (owner_email, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_portal_upload_sessions_stale
    ON portal_upload_sessions (created_at)
    WHERE status IN ('created','uploading');
CREATE INDEX IF NOT EXISTS idx_portal_audit_log_created
    ON portal_audit_log (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_portal_audit_log_actor
    ON portal_audit_log (actor_email, created_at DESC);

DROP TRIGGER IF EXISTS portal_upload_sessions_updated_at ON portal_upload_sessions;
CREATE TRIGGER portal_upload_sessions_updated_at
    BEFORE UPDATE ON portal_upload_sessions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

COMMIT;
