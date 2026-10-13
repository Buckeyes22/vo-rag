"""Postgres query helpers grouped by table domain."""

from .agent_termination_log import list_terminations, record_termination
from .answer_generation import insert_answer_feedback, insert_answer_log
from .artifacts import (
    append_qdrant_point_ids,
    get_artifact,
    insert_artifact,
    list_artifacts,
)
from .backup_log import insert_log, list_recent
from .citation_assignments import insert_citation_assignments, load_citation_assignments
from .cloud_kill_log import persist
from .compliance import get_compliance_log, insert_compliance_log, recent_failures
from .compliance_sub_chunks import fetch_sub_chunks_for_parent_ids
from .envelopes import (
    fail_envelope,
    get_envelope,
    insert_envelope,
    list_envelopes,
    update_envelope_status,
    update_envelope_triage,
)
from .lora_adapters import (
    clear_active_for_persona,
    get_active_q,
    insert_adapter,
    list_adapters_q,
    set_active_q,
)
from .pii_subjects import delete_subject, lookup_subject, register_subject
from .provenance import (
    insert_provenance_record,
    load_provenance_record,
)
from .query_log import QueryLogRow, fetch_recent_queries, insert_query_log
from .r2_staging import record_cleanup_run, record_staged
from .rate_limit import count_hits, insert_hits, prune_hits_older_than_days
from .regurgitation_reports import insert_regurgitation_report
from .training_cost import record_cost, sum_run_cost
from .triage_cost import cost_last_n_days, insert_cost
from .vce_manifest import (
    get_video_manifest,
    insert_stage_run_log,
    set_stage_status,
    upsert_video_manifest,
)
from .vlm_cost import log_vlm_cost

__all__ = [
    "insert_envelope",
    "get_envelope",
    "fail_envelope",
    "update_envelope_status",
    "update_envelope_triage",
    "list_envelopes",
    "insert_artifact",
    "get_artifact",
    "list_artifacts",
    "append_qdrant_point_ids",
    "insert_compliance_log",
    "get_compliance_log",
    "recent_failures",
    "fetch_sub_chunks_for_parent_ids",
    "insert_answer_log",
    "insert_answer_feedback",
    "QueryLogRow",
    "insert_query_log",
    "fetch_recent_queries",
    "insert_citation_assignments",
    "load_citation_assignments",
    "insert_provenance_record",
    "load_provenance_record",
    "insert_regurgitation_report",
    "insert_log",
    "list_recent",
    "record_cleanup_run",
    "record_staged",
    "count_hits",
    "insert_hits",
    "prune_hits_older_than_days",
    "record_termination",
    "list_terminations",
    "insert_cost",
    "cost_last_n_days",
    "record_cost",
    "sum_run_cost",
    "log_vlm_cost",
    "insert_adapter",
    "list_adapters_q",
    "get_active_q",
    "clear_active_for_persona",
    "set_active_q",
    "persist",
    "register_subject",
    "lookup_subject",
    "delete_subject",
    "upsert_video_manifest",
    "get_video_manifest",
    "set_stage_status",
    "insert_stage_run_log",
]
