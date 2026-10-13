-- Audit log of task executions.

CREATE TABLE IF NOT EXISTS task_audit (
    id            TEXT PRIMARY KEY,
    task_name     TEXT NOT NULL,
    envelope_id   TEXT,
    document_id   TEXT,
    worker_id     TEXT NOT NULL,
    worker_pool   TEXT NOT NULL CHECK (worker_pool IN ('cloud', 'homelab')),
    sovereignty   TEXT NOT NULL CHECK (sovereignty IN ('homelab_only', 'cloud_ok', 'cloud_ok_no_persist')),
    started_at    TIMESTAMPTZ NOT NULL,
    completed_at  TIMESTAMPTZ,
    status        TEXT NOT NULL CHECK (status IN ('started', 'completed', 'failed', 'sovereignty_violation'))
);

CREATE INDEX IF NOT EXISTS idx_task_audit_envelope ON task_audit(envelope_id);
CREATE INDEX IF NOT EXISTS idx_task_audit_sov ON task_audit(sovereignty, worker_pool);
