"""Layer 1A summary tree primitives + builder, retrieval, coverage, evaluation.

Payload shape, node IDs, prompt hashes, and BM25 tokenization are stable for
a given prompt version, so persisted points stay valid until the version is
bumped.
"""

from __future__ import annotations

from vo_rag.core.summary.builder import (
    BuildConfig,
    SummaryCorpusStats,
    SummaryTreeBuilder,
    _parse_summary_response,
    _strip_reasoning,
    build_summary_for_corpus,
)
from vo_rag.core.summary.coverage import (
    CoverageJudgeConfig,
    CoverageJudgment,
    CoverageResult,
    FaithfulnessClaim,
    FaithfulnessResult,
    KeyPoint,
    SummaryQualityScore,
    compute_summary_quality_score,
    extract_claims_from_summary,
    judge_coverage,
    judge_faithfulness,
)
from vo_rag.core.summary.evaluation import (
    SummaryEvalVerdict,
    SummaryMechanicalResult,
    SummaryPerQueryScore,
    emit_summary_eval_report,
    evaluate_summary_layer,
)
from vo_rag.core.summary.extract_support import (
    build_extract_support_prompt,
    select_evidence_snippets,
)
from vo_rag.core.summary.layer1a_retrieval import (
    SearchFn,
    build_layer1a_search_fn,
    layer1a_retrieve,
    search_summary_nodes,
)
from vo_rag.core.summary.phase2_aggregator import (
    CorpusQualityResult,
    Phase2FinalVerdict,
    aggregate_corpus_quality,
    decide_phase2,
    emit_phase2_report,
)
from vo_rag.core.summary.prompts import SUMMARY_PROMPT_VERSION
from vo_rag.core.summary.schemas import EvidenceSpan, NodeLevel, SummaryNode, node_id_for

__all__ = [
    "BuildConfig",
    "CorpusQualityResult",
    "CoverageJudgment",
    "CoverageJudgeConfig",
    "CoverageResult",
    "EvidenceSpan",
    "FaithfulnessClaim",
    "FaithfulnessResult",
    "KeyPoint",
    "NodeLevel",
    "Phase2FinalVerdict",
    "SUMMARY_PROMPT_VERSION",
    "SearchFn",
    "SummaryEvalVerdict",
    "SummaryCorpusStats",
    "SummaryMechanicalResult",
    "SummaryNode",
    "SummaryPerQueryScore",
    "SummaryQualityScore",
    "SummaryTreeBuilder",
    "_parse_summary_response",
    "_strip_reasoning",
    "aggregate_corpus_quality",
    "build_extract_support_prompt",
    "build_layer1a_search_fn",
    "build_summary_for_corpus",
    "compute_summary_quality_score",
    "decide_phase2",
    "emit_phase2_report",
    "emit_summary_eval_report",
    "evaluate_summary_layer",
    "extract_claims_from_summary",
    "judge_coverage",
    "judge_faithfulness",
    "layer1a_retrieve",
    "node_id_for",
    "search_summary_nodes",
    "select_evidence_snippets",
]
