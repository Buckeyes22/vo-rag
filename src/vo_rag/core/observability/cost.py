"""Per-envelope cost accumulator + DB persistence.

Two halves, one module so callers import from a single place:

1. `CostAccumulator` + `cost_context` (ContextVar) — in-process rollup.
   The LLM client calls `cost_accumulator.add(step, response)` on every
   successful call. At pipeline exit the workflow snapshots the accumulator
   and persists it to `pipeline_cost` via `append_step_cost`.

   `cost_context(envelope_id, pipeline)` is a context manager that installs
   a fresh accumulator on the current `ContextVar`; `current_cost_accumulator()`
   reads it. This lets downstream code (reranker, compliance gate) attach
   costs without threading the accumulator through every signature.

2. `append_step_cost` / `finalize_envelope_cost` — DB-backed rollup +
   Prometheus emission. These wrap `pipeline_cost` upserts in asyncpg.

Design note on Hatchet compatibility:
- Hatchet tasks are independent processes, so each step gets its own
  CostAccumulator bound to its own `cost_context`.
- `append_step_cost` uses an additive upsert so multiple steps on the
  same envelope merge into one row under (envelope_id) — the caller passes
  `status="partial"` for intermediate steps and the final step upgrades to
  `"success"` / `"failed"`.
- Cross-step aggregation still needs the final step to know about all prior
  steps (or to read the latest row). For v1, per-step `append_step_cost`
  with "partial" followed by a final-step write of "success" is enough.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import asyncpg  # type: ignore[import-untyped]
import structlog

from vo_rag.core.observability.metrics import (
    llm_usage_extraction_failed_total,
    pipeline_artifacts_total,
    pipeline_cost_usd,
)

_log = structlog.get_logger(__name__)

_ctx: ContextVar[CostAccumulator | None] = ContextVar("vo_rag_cost_ctx", default=None)


@dataclass
class StepCost:
    step: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


@dataclass(frozen=True)
class NormalizedUsage:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _extract_llm_usage(response: Any) -> NormalizedUsage:
    raw = _as_mapping(getattr(response, "raw", None))
    additional = _as_mapping(getattr(response, "additional_kwargs", None))
    usage = _as_mapping(raw.get("usage") or additional.get("usage"))

    return NormalizedUsage(
        model=str(
            getattr(response, "model", None)
            or raw.get("model")
            or additional.get("model")
            or "unknown"
        ),
        input_tokens=int(
            getattr(response, "input_tokens", 0)
            or raw.get("input_tokens", 0)
            or usage.get("prompt_tokens", 0)
            or usage.get("input_tokens", 0)
            or 0
        ),
        output_tokens=int(
            getattr(response, "output_tokens", 0)
            or raw.get("output_tokens", 0)
            or usage.get("completion_tokens", 0)
            or usage.get("output_tokens", 0)
            or 0
        ),
        cost_usd=float(
            getattr(response, "cost_usd", 0.0)
            or raw.get("cost_usd", 0.0)
            or additional.get("cost_usd", 0.0)
            or 0.0
        ),
    )


@dataclass
class CostAccumulator:
    envelope_id: str
    pipeline: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    by_key: dict[tuple[str, str], StepCost] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def add(self, step: str, response: Any) -> None:
        """Accept any object with the LLMResponse-ish shape. Duck-typed on
        purpose so tests and future providers can drop in their own."""
        usage = _extract_llm_usage(response)
        model = usage.model
        key = (step, model)
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_read = int(getattr(response, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(response, "cache_write_input_tokens", 0) or 0)
        cost = usage.cost_usd

        if input_tokens == 0 and output_tokens == 0:
            _log.warning(
                "llm_usage_shape_unknown_zero_tokens",
                step=step,
                model=model,
                response_type=type(response).__name__,
            )
            llm_usage_extraction_failed_total.inc()

        with self._lock:
            entry = self.by_key.get(key)
            if entry is None:
                entry = StepCost(step=step, model=model)
                self.by_key[key] = entry
            entry.input_tokens += input_tokens
            entry.output_tokens += output_tokens
            entry.cache_read_input_tokens += cache_read
            entry.cache_write_input_tokens += cache_write
            entry.cost_usd += cost
            entry.calls += 1

    def total_usd(self) -> float:
        with self._lock:
            return sum(e.cost_usd for e in self.by_key.values())

    def total_tokens(self) -> dict[str, int]:
        with self._lock:
            return {
                "total_input": sum(e.input_tokens for e in self.by_key.values()),
                "total_output": sum(e.output_tokens for e in self.by_key.values()),
                "total_cache_read": sum(e.cache_read_input_tokens for e in self.by_key.values()),
                "total_cache_write": sum(e.cache_write_input_tokens for e in self.by_key.values()),
            }

    def model_mix(self) -> list[str]:
        """Distinct models seen, sorted. Used as a Postgres array for filtering."""
        with self._lock:
            return sorted({e.model for e in self.by_key.values()})

    def step_breakdown(self) -> dict[str, dict[str, Any]]:
        """JSON-serialisable dict keyed on 'step|model' composite key."""
        with self._lock:
            return {
                f"{step}|{model}": {
                    "step": entry.step,
                    "model": entry.model,
                    "input_tokens": entry.input_tokens,
                    "output_tokens": entry.output_tokens,
                    "cache_read_input_tokens": entry.cache_read_input_tokens,
                    "cache_write_input_tokens": entry.cache_write_input_tokens,
                    "cost_usd": round(entry.cost_usd, 6),
                    "calls": entry.calls,
                }
                for (step, model), entry in self.by_key.items()
            }


@contextmanager
def cost_context(envelope_id: str, pipeline: str) -> Iterator[CostAccumulator]:
    """Install a fresh CostAccumulator on the ContextVar for the duration of
    the block. Guarantees the previous value is restored even on exception."""
    acc = CostAccumulator(envelope_id=envelope_id, pipeline=pipeline)
    token = _ctx.set(acc)
    try:
        yield acc
    finally:
        _ctx.reset(token)


def current_cost_accumulator() -> CostAccumulator | None:
    """Return the CostAccumulator installed by the nearest `cost_context`, or
    None when no context is active."""
    return _ctx.get()


# --------------------------------------------------------------------------- #
# DB persistence.
# --------------------------------------------------------------------------- #


def _git_sha() -> str | None:
    """Current git SHA from the environment (CI sets GIT_SHA / CI_COMMIT_SHA)."""
    return os.environ.get("GIT_SHA") or os.environ.get("CI_COMMIT_SHA") or None


async def append_step_cost(
    pool: asyncpg.Pool,
    *,
    envelope_id: str,
    pipeline: str,
    accumulator: CostAccumulator,
    finished_at: datetime | None = None,
    status: str = "partial",
) -> None:
    """Merge a step's CostAccumulator contents into the `pipeline_cost` row
    for `envelope_id`. First call creates the row; subsequent calls add to
    existing fields via jsonb || concat and numeric addition.

    Swallows DB errors — cost rollup is observability, not a hard requirement.
    """
    if not envelope_id or not pipeline:
        raise ValueError("envelope_id and pipeline are required")
    finished = finished_at or datetime.now(UTC)
    started = accumulator.started_at
    wall_clock_ms = int((finished - started).total_seconds() * 1000)

    breakdown = accumulator.step_breakdown()
    tokens = accumulator.total_tokens()
    total_cost = round(accumulator.total_usd(), 6)
    model_mix = accumulator.model_mix()

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO pipeline_cost(
                    envelope_id, pipeline, status, cost_usd, wall_clock_ms,
                    step_breakdown, token_breakdown, started_at, finished_at,
                    git_sha, model_mix
                )
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9, $10, $11)
                ON CONFLICT (envelope_id) DO UPDATE SET
                    -- status upgrades: partial -> success|failed wins; once
                    -- terminal, stays terminal.
                    status = CASE
                        WHEN pipeline_cost.status = 'partial' THEN EXCLUDED.status
                        ELSE pipeline_cost.status
                    END,
                    cost_usd = pipeline_cost.cost_usd + EXCLUDED.cost_usd,
                    wall_clock_ms = pipeline_cost.wall_clock_ms + EXCLUDED.wall_clock_ms,
                    step_breakdown = pipeline_cost.step_breakdown || EXCLUDED.step_breakdown,
                    token_breakdown = jsonb_build_object(
                        'total_input',
                            COALESCE((pipeline_cost.token_breakdown->>'total_input')::BIGINT, 0)
                            + COALESCE((EXCLUDED.token_breakdown->>'total_input')::BIGINT, 0),
                        'total_output',
                            COALESCE((pipeline_cost.token_breakdown->>'total_output')::BIGINT, 0)
                            + COALESCE((EXCLUDED.token_breakdown->>'total_output')::BIGINT, 0),
                        'total_cache_read',
                            COALESCE(
                                (pipeline_cost.token_breakdown->>'total_cache_read')::BIGINT, 0)
                            + COALESCE((EXCLUDED.token_breakdown->>'total_cache_read')::BIGINT, 0),
                        'total_cache_write',
                            COALESCE(
                                (pipeline_cost.token_breakdown->>'total_cache_write')::BIGINT, 0)
                            + COALESCE((EXCLUDED.token_breakdown->>'total_cache_write')::BIGINT, 0)
                    ),
                    finished_at = EXCLUDED.finished_at,
                    model_mix = (
                        SELECT ARRAY(SELECT DISTINCT unnest(
                            pipeline_cost.model_mix || EXCLUDED.model_mix
                        ))
                    )
                """,
                envelope_id,
                pipeline,
                status,
                total_cost,
                wall_clock_ms,
                json.dumps(breakdown),
                json.dumps(tokens),
                started,
                finished,
                _git_sha(),
                model_mix,
            )
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning(
            "pipeline_cost_persist_failed",
            envelope_id=envelope_id,
            pipeline=pipeline,
            error=str(exc),
        )
        return

    # Prometheus — emit even when the merge is partial. The histogram
    # observation is the step's contribution, not the envelope total.
    pipeline_cost_usd.labels(pipeline=pipeline).observe(total_cost)
    if status in {"success", "failed"}:
        pipeline_artifacts_total.labels(pipeline=pipeline, status=status).inc()


async def finalize_envelope_cost(
    pool: asyncpg.Pool,
    *,
    envelope_id: str,
    pipeline: str,
    status: str,
) -> None:
    """Mark an envelope's rollup as terminal (success / failed). Idempotent;
    uses the existing cost_usd / breakdowns accumulated by earlier
    `append_step_cost` calls. Emits the per-artifact counter."""
    if status not in {"success", "failed"}:
        raise ValueError("finalize status must be 'success' or 'failed'")
    try:
        async with pool.acquire() as conn:
            updated = await conn.fetchval(
                """
                UPDATE pipeline_cost
                SET status = $1, finished_at = NOW()
                WHERE envelope_id = $2 AND status = 'partial'
                RETURNING envelope_id
                """,
                status,
                envelope_id,
            )
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning(
            "pipeline_cost_finalize_failed",
            envelope_id=envelope_id,
            pipeline=pipeline,
            error=str(exc),
        )
        return

    if updated is not None:
        pipeline_artifacts_total.labels(pipeline=pipeline, status=status).inc()


__all__ = [
    "CostAccumulator",
    "StepCost",
    "append_step_cost",
    "cost_context",
    "current_cost_accumulator",
    "finalize_envelope_cost",
]
