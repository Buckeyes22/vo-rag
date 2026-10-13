"""Registered evaluator discovery for operator and CI eval gates."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from vo_rag.core.config.pipelines import PipelineSpec, PipelineStatus
from vo_rag.core.eval.gate import EvalGateResult, EvaluatorEntry, run_eval_gate
from vo_rag.core.eval.passages import RelevanceMode

if TYPE_CHECKING:
    from vo_rag.pipelines.book_rag.eval import BookRagEvaluatorEntry

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "eval"


def build_eval_gate_evaluators(
    registry: Mapping[str, PipelineSpec],
    *,
    fixture_root: Path = _FIXTURE_ROOT,
    retriever_factory: Callable[[], Any] | None = None,
    relevance_mode: RelevanceMode | None = None,
    collection: str | None = None,
) -> dict[str, BookRagEvaluatorEntry]:
    """Register evaluator entries for active pipelines with fixture sets."""
    evaluators: dict[str, BookRagEvaluatorEntry] = {}
    mode = os.getenv("VO_RAG_EVAL_MODE", "replay_smoke")
    selected_relevance_mode = relevance_mode or _relevance_mode_from_env()
    selected_collection = collection or os.getenv("VO_RAG_EVAL_COLLECTION")

    book_rag_spec = registry.get("book_rag")
    if book_rag_spec is not None and book_rag_spec.status is PipelineStatus.ACTIVE:
        book_dir = fixture_root / "book_rag"
        if book_dir.is_dir():
            from vo_rag.pipelines.book_rag.eval import (
                BOOK_RAG_EVAL_CORPUS_SNAPSHOT,
                book_rag_evaluator,
            )

            retriever = (
                retriever_factory()
                if (
                    mode == "retrieval_gate"
                    and retriever_factory
                    and (book_dir / BOOK_RAG_EVAL_CORPUS_SNAPSHOT).is_file()
                )
                else None
            )
            evaluators["book_rag"] = book_rag_evaluator(
                book_dir,
                mode=mode,
                retriever=retriever,
                relevance_mode=selected_relevance_mode,
                collection=selected_collection,
            )

    return evaluators


def blocked_eval_gate_modes(evaluators: Mapping[str, object]) -> dict[str, str]:
    """Return evaluator modes that indicate a precondition block."""
    from vo_rag.pipelines.book_rag.eval import BOOK_RAG_BLOCKED_MODES

    return {
        pipeline: str(mode)
        for pipeline, entry in evaluators.items()
        if (mode := getattr(entry, "mode", None)) in BOOK_RAG_BLOCKED_MODES
    }


def run_registered_eval_gate(
    registry: Mapping[str, PipelineSpec],
    *,
    fixture_root: Path = _FIXTURE_ROOT,
    retriever_factory: Callable[[], Any] | None = None,
    relevance_mode: RelevanceMode | None = None,
    collection: str | None = None,
) -> EvalGateResult:
    """Run the operator/CI eval gate with registered evaluator semantics."""
    selected_relevance_mode = relevance_mode or _relevance_mode_from_env()
    evaluators = build_eval_gate_evaluators(
        registry,
        fixture_root=fixture_root,
        retriever_factory=retriever_factory,
        relevance_mode=selected_relevance_mode,
        collection=collection,
    )
    blocked = blocked_eval_gate_modes(evaluators)
    runnable_registry = {name: spec for name, spec in registry.items() if name not in blocked}
    runnable_evaluators = {
        name: evaluator for name, evaluator in evaluators.items() if name not in blocked
    }
    result = run_eval_gate(
        runnable_registry,
        evaluators=cast(Mapping[str, EvaluatorEntry], runnable_evaluators),
        fixture_root=fixture_root,
        relevance_mode=selected_relevance_mode,
    )
    if blocked:
        result.blocked_preconditions.update(blocked)
        for pipeline in blocked:
            if pipeline not in result.failed:
                result.failed.append(pipeline)
    return result


def _relevance_mode_from_env() -> RelevanceMode:
    value = os.getenv("VO_RAG_EVAL_RELEVANCE_MODE", "chunk_id")
    if value == "chunk_id":
        return "chunk_id"
    if value == "passage":
        return "passage"
    msg = "VO_RAG_EVAL_RELEVANCE_MODE must be 'chunk_id' or 'passage'"
    raise ValueError(msg)


__all__ = [
    "blocked_eval_gate_modes",
    "build_eval_gate_evaluators",
    "run_registered_eval_gate",
]
