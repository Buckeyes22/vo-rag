-- 026_vce_knowledge_artifacts_check.sql
-- Extends knowledge_artifacts CHECK constraints to permit:
--   pipeline = 'vce'
--   artifact_type = 'coaching_record'
-- so that the VCE index step can insert KnowledgeArtifact rows.
-- (Idempotent: 001_initial.sql already includes these values,
-- so this migration is a no-op upgrade path for databases seeded from an
-- older 001.)

BEGIN;

ALTER TABLE knowledge_artifacts
    DROP CONSTRAINT IF EXISTS knowledge_artifacts_pipeline_chk;
ALTER TABLE knowledge_artifacts
    ADD CONSTRAINT knowledge_artifacts_pipeline_chk CHECK (
        pipeline IN (
            'book_rag','repo_analysis','research_agent','idea_capture',
            'article_extraction','tool_evaluation','url_snapshot',
            'structured_scrape','vce','code_rag'
        )
    );

ALTER TABLE knowledge_artifacts
    DROP CONSTRAINT IF EXISTS knowledge_artifacts_artifact_type_chk;
ALTER TABLE knowledge_artifacts
    ADD CONSTRAINT knowledge_artifacts_artifact_type_chk CHECK (
        artifact_type IN (
            'summary','evaluation','idea','research','extraction','snapshot',
            'coaching_record'
        )
    );

COMMIT;
