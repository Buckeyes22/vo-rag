-- 023_vce_initial.sql
-- vo-rag-VCE: schema for video manifest, persona registry, graph,
-- stage run log, and signature point mapping side-table.

CREATE SCHEMA IF NOT EXISTS vce;

CREATE TABLE vce.video_manifest (
    video_id           TEXT PRIMARY KEY,
    source_url         TEXT,
    source_path        TEXT,
    source_platform    TEXT NOT NULL,
    file_sha256        TEXT NOT NULL,
    title              TEXT,
    channel            TEXT,
    channel_id         TEXT,
    coach_persona      TEXT,
    claimed_track      TEXT,
    claimed_discipline TEXT,
    claimed_telemetry  TEXT,
    duration_seconds   INTEGER,
    uploaded_at        TIMESTAMPTZ,
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stage_status       JSONB NOT NULL DEFAULT '{}'::jsonb,
    cost_usd           NUMERIC(8,4) NOT NULL DEFAULT 0,
    quality_grade      TEXT,
    license_posture    TEXT NOT NULL DEFAULT 'internal_research',
    notes              TEXT
);

CREATE TABLE vce.coach_persona (
    persona_id         TEXT PRIMARY KEY,
    display_name       TEXT NOT NULL,
    primary_channel    TEXT,
    style_tags         TEXT[],
    permission_status  TEXT NOT NULL DEFAULT 'unknown',
    style_notes        TEXT
);

CREATE TABLE vce.coaching_graph_nodes (
    node_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_type          TEXT NOT NULL,
    canonical_label    TEXT NOT NULL,
    description        TEXT,
    aliases            TEXT[],
    UNIQUE (node_type, canonical_label)
);

CREATE TABLE vce.coaching_graph_edges (
    edge_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_node_id     UUID NOT NULL REFERENCES vce.coaching_graph_nodes(node_id) ON DELETE CASCADE,
    target_node_id     UUID NOT NULL REFERENCES vce.coaching_graph_nodes(node_id) ON DELETE CASCADE,
    edge_type          TEXT NOT NULL,
    weight             NUMERIC(4,3) NOT NULL DEFAULT 1.0,
    evidence_record_ids TEXT[] NOT NULL DEFAULT '{}',
    UNIQUE (source_node_id, target_node_id, edge_type)
);

CREATE TABLE vce.record_graph_membership (
    record_id          TEXT NOT NULL,
    node_id            UUID NOT NULL REFERENCES vce.coaching_graph_nodes(node_id) ON DELETE CASCADE,
    PRIMARY KEY (record_id, node_id)
);

CREATE TABLE vce.record_signature_points (
    record_id          TEXT PRIMARY KEY,
    signature_point_id TEXT NOT NULL,
    signature_version  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE vce.stage_run_log (
    run_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id           TEXT NOT NULL REFERENCES vce.video_manifest(video_id) ON DELETE CASCADE,
    stage_name         TEXT NOT NULL,
    state              TEXT NOT NULL,
    started_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at        TIMESTAMPTZ,
    error              TEXT,
    metrics            JSONB
);
