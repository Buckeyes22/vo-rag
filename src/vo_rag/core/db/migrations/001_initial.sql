-- vo-rag initial schema: intake envelopes, knowledge artifacts, ideas, compliance log, sources.
-- Applied once on first boot of vo-rag-postgres by the migration runner
-- (src/vo_rag/core/db/postgres.py).
-- Until the runner ships, apply manually with:
--   PGPASSWORD=... psql -h ... -U vo_rag -d vo_rag -f src/vo_rag/core/db/migrations/001_initial.sql

BEGIN;

-- ------------------------------------------------------------------
-- intake_envelopes: every input to vo-rag, from any source channel
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS intake_envelopes (
    id              TEXT PRIMARY KEY,
    source_type     TEXT NOT NULL,
    source_data     JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_content     JSONB NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    triage_result   JSONB,
    status          TEXT NOT NULL DEFAULT 'received',
    target_pipeline TEXT,
    priority        TEXT NOT NULL DEFAULT 'medium',
    error           TEXT,
    sovereignty     TEXT NOT NULL DEFAULT 'cloud_ok',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT intake_envelopes_source_type_chk CHECK (
        source_type IN ('github_star','file_drop','url','text_prompt','api')
    ),
    CONSTRAINT intake_envelopes_status_chk CHECK (
        status IN ('received','triaging','triaged','dispatched','processing','completed','failed','discarded')
    ),
    CONSTRAINT intake_envelopes_priority_chk CHECK (
        priority IN ('high','medium','low','noise')
    ),
    CONSTRAINT intake_envelopes_sovereignty_chk CHECK (
        sovereignty IN ('homelab_only','cloud_ok','cloud_ok_no_persist')
    ),
    CONSTRAINT intake_envelopes_target_pipeline_chk CHECK (
        target_pipeline IS NULL OR target_pipeline IN (
            'book_rag','repo_analysis','research_agent','idea_capture',
            'article_extraction','tool_evaluation','url_snapshot',
            'structured_scrape','vce','code_rag'
        )
    )
);

-- ------------------------------------------------------------------
-- sources: ingested source documents (books, articles, repos, webpages)
-- License class gates query-time routing in publication/partner modes.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sources (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    author          TEXT,
    isbn            TEXT,
    source_type     TEXT NOT NULL,
    license_class   TEXT NOT NULL DEFAULT 'unknown',
    file_path       TEXT,
    file_hash       TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT sources_source_type_chk CHECK (
        source_type IN ('book','article','webpage','repo','dataset','tool','idea','url','unknown')
    ),
    CONSTRAINT sources_license_class_chk CHECK (
        license_class IN ('cleared','pending','unlicensed','public_domain','unknown')
    )
);

-- ------------------------------------------------------------------
-- knowledge_artifacts: every pipeline output that becomes retrievable knowledge
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_artifacts (
    id                  TEXT PRIMARY KEY,
    envelope_id         TEXT NOT NULL REFERENCES intake_envelopes(id) ON DELETE RESTRICT,
    pipeline            TEXT NOT NULL,
    artifact_type       TEXT NOT NULL,
    title               TEXT NOT NULL,
    content             JSONB NOT NULL,
    tags                TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    priority            TEXT NOT NULL DEFAULT 'medium',
    status              TEXT NOT NULL DEFAULT 'active',
    license_class       TEXT NOT NULL DEFAULT 'unknown',
    qdrant_collection   TEXT,
    qdrant_point_ids    TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    source_provenance   JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT knowledge_artifacts_pipeline_chk CHECK (
        pipeline IN ('book_rag','repo_analysis','research_agent','idea_capture',
                     'article_extraction','tool_evaluation','url_snapshot',
                     'structured_scrape','vce','code_rag')
    ),
    CONSTRAINT knowledge_artifacts_artifact_type_chk CHECK (
        artifact_type IN ('summary','evaluation','idea','research','extraction',
                          'snapshot','coaching_record')
    ),
    CONSTRAINT knowledge_artifacts_priority_chk CHECK (
        priority IN ('high','medium','low','noise')
    ),
    CONSTRAINT knowledge_artifacts_license_class_chk CHECK (
        license_class IN ('cleared','pending','unlicensed','public_domain','unknown')
    )
);

-- ------------------------------------------------------------------
-- ideas: enrichments of raw-idea envelopes, plus optional project briefs
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ideas (
    id              TEXT PRIMARY KEY,
    artifact_id     TEXT REFERENCES knowledge_artifacts(id) ON DELETE SET NULL,
    core_concept    TEXT NOT NULL,
    domain          TEXT,
    applications    TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    open_questions  TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    related_ids     TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    status          TEXT NOT NULL DEFAULT 'captured',
    project_brief   JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT ideas_status_chk CHECK (
        status IN ('captured','exploring','building','parked','discarded')
    )
);

-- ------------------------------------------------------------------
-- compliance_log: every output-compliance check, enforcement or monitoring
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS compliance_log (
    id                    TEXT PRIMARY KEY,
    artifact_id           TEXT REFERENCES knowledge_artifacts(id) ON DELETE SET NULL,
    query_text            TEXT,
    output_text           TEXT,
    similarity_scores     JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_contributions  JSONB NOT NULL DEFAULT '{}'::jsonb,
    citation_tiers        JSONB NOT NULL DEFAULT '{}'::jsonb,
    passed                BOOLEAN NOT NULL,
    failure_reasons       TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    mode                  TEXT NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT compliance_log_mode_chk CHECK (
        mode IN ('internal','publication','partner')
    )
);

-- ------------------------------------------------------------------
-- Indexes
-- ------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_envelopes_status      ON intake_envelopes (status);
CREATE INDEX IF NOT EXISTS idx_envelopes_pipeline    ON intake_envelopes (target_pipeline);
CREATE INDEX IF NOT EXISTS idx_envelopes_created     ON intake_envelopes (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_envelopes_sourcetype  ON intake_envelopes (source_type);

CREATE INDEX IF NOT EXISTS idx_artifacts_envelope    ON knowledge_artifacts (envelope_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_pipeline    ON knowledge_artifacts (pipeline);
CREATE INDEX IF NOT EXISTS idx_artifacts_type        ON knowledge_artifacts (artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_tags        ON knowledge_artifacts USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_artifacts_license     ON knowledge_artifacts (license_class);
CREATE INDEX IF NOT EXISTS idx_artifacts_created     ON knowledge_artifacts (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_compliance_artifact   ON compliance_log (artifact_id);
CREATE INDEX IF NOT EXISTS idx_compliance_passed     ON compliance_log (passed);
CREATE INDEX IF NOT EXISTS idx_compliance_mode       ON compliance_log (mode);
CREATE INDEX IF NOT EXISTS idx_compliance_created    ON compliance_log (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_sources_license       ON sources (license_class);
CREATE INDEX IF NOT EXISTS idx_sources_hash          ON sources (file_hash);
CREATE INDEX IF NOT EXISTS idx_sources_source_type   ON sources (source_type);

CREATE INDEX IF NOT EXISTS idx_ideas_status          ON ideas (status);
CREATE INDEX IF NOT EXISTS idx_ideas_artifact        ON ideas (artifact_id);

-- ------------------------------------------------------------------
-- updated_at trigger for mutable rows
-- ------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS envelopes_updated_at ON intake_envelopes;
CREATE TRIGGER envelopes_updated_at
    BEFORE UPDATE ON intake_envelopes
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS artifacts_updated_at ON knowledge_artifacts;
CREATE TRIGGER artifacts_updated_at
    BEFORE UPDATE ON knowledge_artifacts
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS ideas_updated_at ON ideas;
CREATE TRIGGER ideas_updated_at
    BEFORE UPDATE ON ideas
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

COMMIT;
