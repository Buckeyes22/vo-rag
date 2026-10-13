"""CI entry: run the eval gate against the SOT roster; exit non-zero on failure.

The gate runs the per-pipeline retrieval harness (queries.jsonl + qrels.tsv +
baseline.json + registered evaluator). Print summary:

  - `checked`  pipelines ran the harness and matched their baseline
  - `skipped`  pipelines that are not ACTIVE
  - `failed`   pipelines that are ACTIVE but either lack fixtures,
               lack a registered evaluator, or regressed on a metric

Exit 0 only when `passed is True`. Any `failed` pipeline -> exit 1.

Evaluator registration: ACTIVE pipelines must have a registered
``(RetrievalVariant, SearchFn)`` entry or the gate fails them (a missing
component cannot silently pass). book_rag registers an explicit evaluator mode
selected by ``VO_RAG_EVAL_MODE``: ``replay_smoke`` by default, or
``retrieval_gate`` when the live corpus snapshot has been re-grounded.
``VO_RAG_EVAL_RELEVANCE_MODE=passage`` switches the gate to
``qrels_passage.tsv``/``baseline_passage.json``; ``VO_RAG_EVAL_COLLECTION``
can target a suffixed collection for A/B arms.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vo_rag.core.config.pipelines import PipelineSpec, load_pipeline_registry
from vo_rag.core.eval.registered import (
    blocked_eval_gate_modes,
    run_registered_eval_gate,
)
from vo_rag.core.eval.registered import (
    build_eval_gate_evaluators as build_eval_gate_evaluators,
)

if TYPE_CHECKING:
    from vo_rag.pipelines.book_rag.eval import BookRagEvaluatorEntry

BLOCKED_EXIT_CODE = 3
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "eval"


def _build_evaluators(registry: dict[str, PipelineSpec]) -> dict[str, BookRagEvaluatorEntry]:
    """Register an explicit evaluator mode for each ACTIVE pipeline we gate.

    Pipelines that are not ACTIVE are skipped by the gate regardless.
    """
    return build_eval_gate_evaluators(registry, retriever_factory=_build_retriever_factory())


def _blocked_modes(evaluators: dict[str, BookRagEvaluatorEntry]) -> dict[str, str]:
    return blocked_eval_gate_modes(evaluators)


def _build_retriever_factory() -> Callable[[], Any] | None:
    if os.getenv("VO_RAG_EVAL_MODE", "replay_smoke") != "retrieval_gate":
        return None

    from vo_rag.pipelines.book_rag.eval import (
        BOOK_RAG_EVAL_CORPUS_SNAPSHOT,
        build_book_rag_snapshot_retriever,
    )

    snapshot_path = _FIXTURE_ROOT / "book_rag" / BOOK_RAG_EVAL_CORPUS_SNAPSHOT
    return lambda: build_book_rag_snapshot_retriever(snapshot_path)


def main() -> int:
    registry = load_pipeline_registry()
    result = run_registered_eval_gate(registry, retriever_factory=_build_retriever_factory())
    blocked = result.blocked_preconditions
    if blocked:
        print("eval gate BLOCKED - retrieval gate preconditions are not satisfied:")
        for pipeline, mode in blocked.items():
            print(f"  - {pipeline}: mode == {mode}")
        non_blocked_failed = [p for p in result.failed if p not in blocked]
        if not (
            non_blocked_failed
            or result.regressions
            or result.missing_fixtures
            or result.missing_evaluators
        ):
            return BLOCKED_EXIT_CODE
        # A blocked pipeline must never whitewash a genuine failure in another
        # pipeline: fall through, print the real failures, and exit 1 so the
        # ci.sh BLOCKED tolerance (exit 3 only) does not absorb them.
    if result.skipped:
        print(f"skipped inactive pipelines: {result.skipped}")
    if result.checked:
        print(f"checked pipelines (matched baseline): {result.checked}")
    if result.missing_fixtures:
        print(f"failed: active pipelines missing fixture set: {result.missing_fixtures}")
    if result.missing_evaluators:
        print(f"failed: active pipelines missing registered evaluator: {result.missing_evaluators}")
    if result.regressions:
        print("failed: retrieval regression detected:")
        for line in result.regressions:
            print(f"  - {line}")
    if result.failed:
        print(f"eval gate FAIL — failed pipelines: {result.failed}")
        return 1
    print(f"eval gate PASS (active pipelines checked: {result.checked or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
