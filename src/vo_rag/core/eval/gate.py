"""Per-pipeline retrieval eval gate.

Contract: an active pipeline MUST have
  - a queries.jsonl + qrels.tsv + baseline.json fixture set under the
    pipeline's `fixture_root/<pipeline>/` directory,
  - a registered evaluator entry exposing `(RetrievalVariant, search_fn)`,

and the gate runs the harness against the pipeline's variant and calls
`assert_no_regression` against the frozen baseline. Missing fixture files,
missing evaluator, or a metric regression all flip `passed` to False and
cause scripts/run_eval_gate.py to exit non-zero.

Schema-versioned baselines also fail closed when the live `qrels_hash` differs,
when the `corpus_snapshot` differs from the supplied live corpus identity, or
when a legacy baseline is used without `VO_RAG_ALLOW_LEGACY_BASELINE=1`.
`VO_RAG_ALLOW_CORPUS_SNAPSHOT_MISMATCH=1` is the explicit corpus-mismatch escape
hatch.

A roster with no active pipelines passes trivially so the gate can be wired
into CI from day one.

This gate intentionally does not embed any specific threshold (such as
nDCG@10 >= 0.86 for book_rag); the fixture ships the thresholds per pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, computed_field

from vo_rag.core.config.pipelines import PipelineSpec, PipelineStatus
from vo_rag.core.eval.harness import (
    RetrievalRegression,
    RetrievalVariant,
    SearchFn,
    assert_no_regression,
    run_harness,
)
from vo_rag.core.eval.passages import RelevanceMode
from vo_rag.core.eval.qrels import load_qrels_tsv
from vo_rag.core.models.common import VoBaseModel

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "eval"

EvaluatorEntry = tuple[RetrievalVariant, SearchFn]


class EvalGateError(Exception):
    """Raised when eval gate preconditions fail (e.g. baseline provenance mismatch)."""


class BaselineV1(VoBaseModel):
    """Schema for frozen eval baselines consumed by the gate."""

    schema_version: Literal[1]
    pipeline: str
    qrels_hash: str
    corpus_snapshot: str
    created_at: str
    aggregate: dict[str, float]


class EvalGateResult(BaseModel):
    checked: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    blocked_preconditions: dict[str, str] = Field(default_factory=dict)
    missing_fixtures: list[str] = Field(default_factory=list)
    missing_evaluators: list[str] = Field(default_factory=list)
    missing_baselines: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return not (
            self.failed
            or self.blocked_preconditions
            or self.missing_fixtures
            or self.missing_evaluators
            or self.missing_baselines
            or self.regressions
        )


def _load_queries_jsonl(path: Path) -> dict[str, str]:
    """Return {query_id: text} from a JSONL file.

    Accepts either a ``text`` field (the harness-native name) or
    ``query_text`` (the field name used by the ported motorsport/book_rag
    benchmark fixtures). Without this fallback a verbatim-ported fixture
    resolves every query to the empty string, silently breaking the
    live-retriever gate.
    """
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        text = record.get("text") or record.get("query_text", "")
        out[str(record["query_id"])] = str(text)
    return out


def _build_query_text_fn(queries: dict[str, str]) -> Callable[[str], str]:
    def _lookup(query_id: str) -> str:
        return queries.get(query_id, query_id)

    return _lookup


def _run_pipeline(
    pipeline_name: str,
    *,
    variant: RetrievalVariant,
    search_fn: SearchFn,
    qrels_path: Path,
    queries_path: Path,
    baseline: dict[str, Any],
    corpus_snapshot: str | None,
    relevance_mode: RelevanceMode,
) -> None:
    if "schema_version" in baseline:
        try:
            baseline_v1 = BaselineV1.model_validate(baseline)
        except ValidationError as exc:
            raise EvalGateError(f"baseline schema invalid: {exc}") from exc
        if baseline_v1.pipeline != pipeline_name:
            raise EvalGateError(
                "baseline pipeline does not match active pipeline: "
                f"baseline={baseline_v1.pipeline!r}, pipeline={pipeline_name!r}"
            )
        aggregate = baseline_v1.aggregate
    else:
        if os.getenv("VO_RAG_ALLOW_LEGACY_BASELINE") != "1":
            raise EvalGateError(
                "legacy baseline without schema_version; "
                "set VO_RAG_ALLOW_LEGACY_BASELINE=1 to allow"
            )
        baseline_v1 = None
        aggregate = baseline

    qrels = load_qrels_tsv(qrels_path, pipeline=pipeline_name, collection="eval")
    queries = _load_queries_jsonl(queries_path)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise EvalGateError("run_eval_gate cannot run inside an active event loop")
    report = asyncio.run(
        run_harness(
            pipeline_name,
            qrels,
            variant,
            search_fn=search_fn,
            query_text_fn=_build_query_text_fn(queries),
            relevance_mode=relevance_mode,
        )
    )
    if baseline_v1 is not None:
        if baseline_v1.qrels_hash != report.qrels_hash:
            raise EvalGateError("baseline qrels_hash does not match current report")
        live_corpus_snapshot = corpus_snapshot or os.getenv("VO_RAG_LIVE_CORPUS_SNAPSHOT")
        if (
            live_corpus_snapshot
            and baseline_v1.corpus_snapshot != live_corpus_snapshot
            and os.getenv("VO_RAG_ALLOW_CORPUS_SNAPSHOT_MISMATCH") != "1"
        ):
            raise EvalGateError(
                "baseline corpus_snapshot does not match live corpus identity: "
                f"baseline={baseline_v1.corpus_snapshot!r}, live={live_corpus_snapshot!r}; "
                "set VO_RAG_ALLOW_CORPUS_SNAPSHOT_MISMATCH=1 to allow"
            )
    assert_no_regression(report, aggregate)


def run_eval_gate(
    registry: Mapping[str, PipelineSpec],
    *,
    evaluators: Mapping[str, EvaluatorEntry] | None = None,
    fixture_root: Path | str = _DEFAULT_FIXTURE_ROOT,
    baseline_exists: Callable[[str], bool] | None = None,
    corpus_snapshot: str | None = None,
    relevance_mode: RelevanceMode = "chunk_id",
) -> EvalGateResult:
    """Run the regression gate across the active pipelines in `registry`.

    Two modes are supported:

    - Harness mode (default): an ACTIVE pipeline must have a queries.jsonl +
      qrels.tsv + baseline.json fixture set under
      `fixture_root/<pipeline>/` and a registered evaluator entry. The
      gate runs the harness and calls `assert_no_regression`. Any failure
      (missing fixture, missing evaluator, metric regression) flips
      `passed` to False and records a reason in the corresponding detail
      list. Non-active pipelines are reported in `skipped`.

    - Callback mode (legacy, used by tests/core/eval/test_gate.py and any caller that
      still passes a `baseline_exists` callback): an ACTIVE pipeline must
      satisfy the callback; otherwise it is reported in
      `missing_baselines`. No harness is executed. This is preserved so
      day-one CI remains green for callers that have not yet registered
      an evaluator.
    """
    if baseline_exists is not None:
        active = [s.name for s in registry.values() if s.status is PipelineStatus.ACTIVE]
        missing = [name for name in active if not baseline_exists(name)]
        return EvalGateResult(
            checked=active,
            missing_baselines=missing,
        )

    evaluators_map: Mapping[str, EvaluatorEntry] = evaluators or {}
    root = Path(fixture_root)

    result = EvalGateResult()
    for spec in registry.values():
        if spec.status is not PipelineStatus.ACTIVE:
            result.skipped.append(spec.name)
            continue

        pipe_dir = root / spec.name
        qrels_filename, baseline_filename = _fixture_names(relevance_mode)
        qrels_path = pipe_dir / qrels_filename
        queries_path = pipe_dir / "queries.jsonl"
        baseline_path = pipe_dir / baseline_filename

        if not (qrels_path.is_file() and queries_path.is_file() and baseline_path.is_file()):
            result.missing_fixtures.append(spec.name)
            result.failed.append(spec.name)
            continue

        if spec.name not in evaluators_map:
            result.missing_evaluators.append(spec.name)
            result.failed.append(spec.name)
            continue

        variant, search_fn = evaluators_map[spec.name]
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        try:
            _run_pipeline(
                spec.name,
                variant=variant,
                search_fn=search_fn,
                qrels_path=qrels_path,
                queries_path=queries_path,
                baseline=baseline,
                corpus_snapshot=corpus_snapshot,
                relevance_mode=relevance_mode,
            )
        except RetrievalRegression as exc:
            result.regressions.append(f"{spec.name}: {exc}")
            result.failed.append(spec.name)
            continue
        except EvalGateError as exc:
            result.regressions.append(f"{spec.name}: {exc}")
            result.failed.append(spec.name)
            continue
        except Exception as exc:  # noqa: BLE001 - surface any harness error
            result.regressions.append(f"{spec.name}: harness error: {exc}")
            result.failed.append(spec.name)
            continue

        result.checked.append(spec.name)

    return result


def _fixture_names(relevance_mode: RelevanceMode) -> tuple[str, str]:
    if relevance_mode == "passage":
        return "qrels_passage.tsv", "baseline_passage.json"
    return "qrels.tsv", "baseline.json"
