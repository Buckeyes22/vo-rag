-- 024_vce_indexes.sql
-- vo-rag-VCE indexes.

CREATE INDEX idx_vce_manifest_persona ON vce.video_manifest(coach_persona);
CREATE INDEX idx_vce_manifest_grade ON vce.video_manifest(quality_grade);
CREATE INDEX idx_vce_manifest_sha ON vce.video_manifest(file_sha256);

CREATE INDEX idx_vce_graph_membership_node ON vce.record_graph_membership(node_id);

CREATE INDEX idx_vce_stage_run_video_stage ON vce.stage_run_log(video_id, stage_name, started_at DESC);
