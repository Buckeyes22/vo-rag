-- Preserve and explicitly classify migration names inherited from an earlier
-- production database.  This migration never renames/deletes ledger rows and
-- never rewrites application data.  Each mapping is inserted only after the
-- restored-production DDL contract has been proved on the target database.

BEGIN;

CREATE TABLE IF NOT EXISTS schema_migration_legacy_mappings (
    legacy_version      TEXT PRIMARY KEY REFERENCES schema_migrations(version),
    canonical_version   TEXT,
    disposition         TEXT NOT NULL CHECK (disposition IN ('equivalent', 'retired')),
    evidence_review_id  TEXT NOT NULL,
    reconciled_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
DECLARE
    has_r2_legacy BOOLEAN := EXISTS (
        SELECT 1 FROM schema_migrations WHERE version = '014_r2_staging_log'
    );
    has_task_legacy BOOLEAN := EXISTS (
        SELECT 1 FROM schema_migrations WHERE version = '017_task_audit'
    );
    has_code_legacy BOOLEAN := EXISTS (
        SELECT 1 FROM schema_migrations
        WHERE version IN (
            '032_code_rag_initial',
            '033_code_rag_indexes',
            '034_code_rag_knowledge_artifacts_check'
        )
    );
BEGIN
    IF has_r2_legacy AND (
        to_regclass('public.r2_staging_log') IS NULL
        OR to_regclass('public.r2_staging_cleanup_run') IS NULL
        OR to_regclass('public.idx_r2_staging_created') IS NULL
    ) THEN
        RAISE EXCEPTION 'legacy 014_r2_staging_log DDL evidence is incomplete';
    END IF;

    IF has_task_legacy AND (
        to_regclass('public.task_audit') IS NULL
        OR to_regclass('public.idx_task_audit_envelope') IS NULL
        OR to_regclass('public.idx_task_audit_sov') IS NULL
    ) THEN
        RAISE EXCEPTION 'legacy 017_task_audit DDL evidence is incomplete';
    END IF;

    IF has_code_legacy AND (
        to_regnamespace('code_rag') IS NULL
        OR to_regclass('code_rag.repos') IS NULL
        OR to_regclass('code_rag.chunk_index') IS NULL
        OR to_regclass('code_rag.repo_summaries') IS NULL
        OR to_regclass('code_rag.ingestion_runs') IS NULL
        OR to_regclass('code_rag.collections') IS NULL
        OR to_regclass('code_rag.repo_collection_links') IS NULL
        OR to_regclass('code_rag.research_runs') IS NULL
        OR to_regclass('code_rag.research_steps') IS NULL
        OR to_regclass('code_rag.code_rag_repos_canonical_url_uq') IS NULL
        OR to_regclass('code_rag.idx_code_rag_chunk_repo_file') IS NULL
        OR NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conrelid = 'public.knowledge_artifacts'::regclass
              AND conname = 'knowledge_artifacts_artifact_type_chk'
              AND pg_get_constraintdef(oid) LIKE '%code_index%'
        )
    ) THEN
        RAISE EXCEPTION 'legacy code_rag 032-034 DDL evidence is incomplete';
    END IF;
END
$$;

INSERT INTO schema_migration_legacy_mappings (
    legacy_version,
    canonical_version,
    disposition,
    evidence_review_id
)
SELECT version, '027_r2_staging_log', 'equivalent', '20260815T183707Z-restored-copy'
FROM schema_migrations
WHERE version = '014_r2_staging_log'
ON CONFLICT (legacy_version) DO NOTHING;

INSERT INTO schema_migration_legacy_mappings (
    legacy_version,
    canonical_version,
    disposition,
    evidence_review_id
)
SELECT version, '028_task_audit', 'equivalent', '20260815T183707Z-restored-copy'
FROM schema_migrations
WHERE version = '017_task_audit'
ON CONFLICT (legacy_version) DO NOTHING;

INSERT INTO schema_migration_legacy_mappings (
    legacy_version,
    canonical_version,
    disposition,
    evidence_review_id
)
SELECT version, NULL, 'retired', '20260815T183707Z-restored-copy'
FROM schema_migrations
WHERE version IN (
    '032_code_rag_initial',
    '033_code_rag_indexes',
    '034_code_rag_knowledge_artifacts_check'
)
ON CONFLICT (legacy_version) DO NOTHING;

COMMIT;
