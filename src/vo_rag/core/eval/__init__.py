"""Eval harness, statistical tests, and CI gates for vo-rag.

Public surface (the names callers reach for via
``from vo_rag.core.eval import ...``):

- :class:`Qrels` / :class:`Judgment` / :func:`load_qrels_tsv` /
  :func:`save_qrels_tsv` / :func:`qrels_hash` — qrels persistence +
  hashing for the harness contract.
- :class:`RetrievalVariant` / :class:`PerQueryResult` /
  :class:`HarnessReport` / :func:`run_harness` / :type:`SearchFn` /
  :class:`RetrievalRegression` / :func:`assert_no_regression` —
  retrieval harness primitives.
- :func:`mrr` / :func:`recall_at_k` / :func:`precision_at_k` /
  :func:`ndcg_at_k` / :func:`dcg_at_k` / :func:`average_precision` —
  ranking metrics shared across pipelines.
- :class:`NonInferiorityResult` / :func:`compute_non_inferiority` /
  :func:`compute_ndcg_at_k` / :func:`compute_recall_at_k` —
  statistical guards reused by the graph-quality gate.
- :class:`EvalGateResult` / :class:`EvaluatorEntry` /
  :func:`run_eval_gate` / :func:`build_eval_gate_evaluators` —
  the per-pipeline CI/operator gate entry points.

The private ``_normal_quantile`` Acklam helper is intentionally NOT in
the public surface; internal callers (and direct tests) import it from
``vo_rag.core.eval.judge_calibration``.
"""

from vo_rag.core.eval.gate import (
    EvalGateResult,
    EvaluatorEntry,
    run_eval_gate,
)
from vo_rag.core.eval.harness import (
    HarnessReport,
    PerQueryResult,
    RetrievalRegression,
    RetrievalVariant,
    SearchFn,
    assert_no_regression,
    run_harness,
)
from vo_rag.core.eval.metrics import (
    average_precision,
    dcg_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from vo_rag.core.eval.non_inferiority import (
    NonInferiorityResult,
    compute_ndcg_at_k,
    compute_non_inferiority,
    compute_recall_at_k,
)
from vo_rag.core.eval.passages import (
    GoldPassageProjectionFn,
    RelevanceMode,
    RetrievedDocument,
    RetrievedItem,
    doc_id_from_retrieved,
    is_passage_key,
    passage_key,
    passage_keys_from_gold_payload,
    passage_keys_from_payload,
    payload_from_retrieved,
    project_qrels_to_passage_keys,
    project_retrieved_to_doc_ids,
    project_retrieved_to_passage_keys,
)
from vo_rag.core.eval.qrels import (
    Judgment,
    Qrels,
    load_qrels_tsv,
    qrels_hash,
    save_qrels_tsv,
)
from vo_rag.core.eval.registered import (
    blocked_eval_gate_modes,
    build_eval_gate_evaluators,
    run_registered_eval_gate,
)

__all__ = [
    "EvalGateResult",
    "EvaluatorEntry",
    "GoldPassageProjectionFn",
    "HarnessReport",
    "Judgment",
    "NonInferiorityResult",
    "PerQueryResult",
    "Qrels",
    "RetrievedDocument",
    "RetrievedItem",
    "RetrievalRegression",
    "RetrievalVariant",
    "RelevanceMode",
    "SearchFn",
    "assert_no_regression",
    "average_precision",
    "blocked_eval_gate_modes",
    "build_eval_gate_evaluators",
    "compute_ndcg_at_k",
    "compute_non_inferiority",
    "compute_recall_at_k",
    "dcg_at_k",
    "doc_id_from_retrieved",
    "is_passage_key",
    "load_qrels_tsv",
    "mrr",
    "ndcg_at_k",
    "passage_key",
    "passage_keys_from_gold_payload",
    "passage_keys_from_payload",
    "payload_from_retrieved",
    "precision_at_k",
    "project_qrels_to_passage_keys",
    "project_retrieved_to_doc_ids",
    "project_retrieved_to_passage_keys",
    "qrels_hash",
    "recall_at_k",
    "run_eval_gate",
    "run_registered_eval_gate",
    "run_harness",
    "save_qrels_tsv",
]
