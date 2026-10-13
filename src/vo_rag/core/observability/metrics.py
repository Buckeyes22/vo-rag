"""Prometheus metrics for vo-rag.

Architectural note: this module is a LEAF. It must not import
`vo_rag.core.storage` (or any sibling that depends on storage), because
`core/storage` depends on the same core primitives that this module
instruments.

The Qdrant upsert counter is wired up via `instrument_qdrant_store(cls)`,
which `core/storage` calls once at import time. Patching the storage class
from here at metric-import time would create an import cycle.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from prometheus_client.core import CounterMetricFamily

log = logging.getLogger("vo_rag.core.observability.metrics")

pipeline_runs_total = Counter(
    "vo_rag_pipeline_runs_total",
    "Total pipeline runs by outcome.",
    ["pipeline", "outcome"],
)

pipeline_duration_seconds = Histogram(
    "vo_rag_pipeline_duration_seconds",
    "Pipeline wall duration.",
    ["pipeline"],
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600),
)

embedding_latency_seconds = Histogram(
    "vo_rag_embedding_latency_seconds",
    "BGE-M3 embed call latency.",
    ["endpoint"],
)

triage_fallback_total = Counter(
    "vo_rag_triage_fallback_total",
    "Triage calls that fell back to Claude.",
    ["reason"],
)

qdrant_upsert_points_total = Counter(
    "vo_rag_qdrant_upsert_points_total",
    "Points upserted to Qdrant.",
    ["collection"],
)

source_publication_jobs = Gauge(
    "vo_rag_source_publication_jobs",
    "Durable source jobs by corpus and state, sampled from PostgreSQL.",
    ["corpus", "state"],
)
source_publication_pending_age_seconds = Gauge(
    "vo_rag_source_publication_pending_age_seconds",
    "Oldest pending source job age; zero when none, -1 when unavailable.",
    ["corpus"],
)
source_publication_pointer_mismatches = Gauge(
    "vo_rag_source_publication_pointer_mismatches",
    "Active pointers without the matching published source job; -1 when unavailable.",
    ["corpus"],
)
source_publication_dependency_mismatches = Gauge(
    "vo_rag_source_publication_dependency_mismatches",
    "Managed derived sources whose admitted raw dependency revision has advanced.",
    ["corpus"],
)

active_pipeline_runs = Gauge(
    "vo_rag_active_pipeline_runs",
    "Pipelines currently running.",
    ["pipeline"],
)

# ---------------------------------------------------------------------------
# LLM-call metrics. Populated by vo_rag.core.llm + retry.
# ---------------------------------------------------------------------------
llm_calls_total = Counter(
    "vo_rag_llm_calls_total",
    "LLM calls by outcome.",
    ["provider", "model", "prompt_name", "result"],
)

llm_retries_total = Counter(
    "vo_rag_llm_retries_total",
    "LLM retry attempts by reason.",
    ["provider", "model", "reason"],
)

llm_latency_seconds = Histogram(
    "vo_rag_llm_latency_seconds",
    "LLM call wall-clock latency.",
    ["provider", "model"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

llm_structured_parse_failures_total = Counter(
    "vo_rag_llm_structured_parse_failures_total",
    "Structured output parse retries exhausted.",
    ["prompt_name"],
)

llm_usage_extraction_failed_total = Counter(
    "vo_rag_llm_usage_extraction_failed_total",
    "LLM responses whose usage payload could not be attributed to tokens.",
)

llm_variant_selections_total = Counter(
    "vo_rag_llm_variant_selections_total",
    "Prompt variant selections.",
    ["prompt_name", "variant"],
)

# ---------------------------------------------------------------------------
# Agent safety metrics.
# ---------------------------------------------------------------------------
agent_termination_total = Counter(
    "vo_rag_agent_termination_total",
    "Bounded-agent runs by pipeline and termination reason.",
    ["pipeline", "reason"],
)

agent_steps = Histogram(
    "vo_rag_agent_steps",
    "Steps consumed by a bounded-agent run.",
    ["pipeline"],
    buckets=(1, 2, 4, 8, 12, 20, 40, 80),
)

agent_cost_usd = Histogram(
    "vo_rag_agent_cost_usd",
    "USD cost consumed by a bounded-agent run.",
    ["pipeline"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
)

agent_wall_clock_s = Histogram(
    "vo_rag_agent_wall_clock_s",
    "Wall-clock seconds consumed by a bounded-agent run.",
    ["pipeline"],
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800),
)

# ---------------------------------------------------------------------------
# Cost attribution.
# ---------------------------------------------------------------------------
pipeline_cost_usd = Histogram(
    "vo_rag_pipeline_cost_usd",
    "USD cost rolled up per pipeline step (observed at persist time).",
    ["pipeline"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
)

pipeline_artifacts_total = Counter(
    "vo_rag_pipeline_artifacts_total",
    "Terminal pipeline outcomes (success or failed).",
    ["pipeline", "status"],
)

# ---------------------------------------------------------------------------
# Prompt-injection defense.
# ---------------------------------------------------------------------------
injection_suspected_total = Counter(
    "vo_rag_injection_suspected_total",
    "validate_output fired a signal on an LLM response.",
    ["pipeline", "kind"],
)

injection_quarantined_total = Counter(
    "vo_rag_injection_quarantined_total",
    "Artifact upgraded to trust_level=quarantined by compliance gate.",
    ["pipeline"],
)

# ---------------------------------------------------------------------------
# Caching.
# ---------------------------------------------------------------------------
cache_requests_total = Counter(
    "vo_rag_cache_requests_total",
    "Cache lookups by cache name and result.",
    ["cache", "result"],  # cache: embedding|completion|prompt_prefix; result: hit|miss|skip
)

cache_size_bytes = Gauge(
    "vo_rag_cache_size_bytes",
    "On-disk cache size in bytes.",
    ["cache"],
)

cache_saved_cost_usd_total = Counter(
    "vo_rag_cache_saved_cost_usd_total",
    "Estimated USD saved by cache hits (completion + prompt_prefix).",
)

# ---------------------------------------------------------------------------
# Operational readiness and dispatcher useful-work signals.
# ---------------------------------------------------------------------------
dispatcher_heartbeat_age_seconds = Gauge(
    "vo_rag_dispatcher_heartbeat_age_seconds",
    "Age of the most recent dispatcher heartbeat; -1 when unavailable.",
)

intake_envelopes = Gauge(
    "vo_rag_intake_envelopes",
    "Current intake envelope count for bounded operational states.",
    ["status"],
)

oldest_claimable_age_seconds = Gauge(
    "vo_rag_oldest_claimable_age_seconds",
    "Age of the oldest dispatcher-claimable envelope; -1 when none or unknown.",
)

last_completion_age_seconds = Gauge(
    "vo_rag_last_completion_age_seconds",
    "Age of the last completed intake envelope; -1 when none or unknown.",
)

dependency_up = Gauge(
    "vo_rag_dependency_up",
    "Whether a required runtime dependency passed its active health check.",
    ["dependency"],
)

surface_readiness = Gauge(
    "vo_rag_readiness",
    "Whether a public surface has all required dependencies available.",
    ["surface"],
)

operational_sampler_errors_total = Counter(
    "vo_rag_operational_sampler_errors_total",
    "Operational sampler failures by bounded component name.",
    ["component"],
)

dispatcher_stop_requests_total = Counter(
    "vo_rag_dispatcher_stop_requests_total",
    "Dispatcher process stop requests observed from signal handlers.",
)

dispatcher_claims_total = Counter(
    "vo_rag_dispatcher_claims_total",
    "Dispatcher claims by whether work was new or reclaimed after staleness.",
    ["kind"],
)

dispatcher_active_ingest = Gauge(
    "vo_rag_dispatcher_active_ingest",
    "Dispatcher ingest operations currently executing.",
)

dispatcher_cancellations_total = Counter(
    "vo_rag_dispatcher_cancellations_total",
    "Dispatcher ingest operations cancelled during shutdown.",
)

dispatcher_heartbeat_failures_total = Counter(
    "vo_rag_dispatcher_heartbeat_failures_total",
    "Dispatcher heartbeat publication failures by target.",
    ["target"],
)

# ---------------------------------------------------------------------------
# Inference-node capacity and overload signals.
# ---------------------------------------------------------------------------
inference_requests_total = Counter(
    "vo_rag_inference_requests_total",
    "Inference requests by operation and bounded outcome.",
    ["operation", "outcome"],
)

inference_active_requests = Gauge(
    "vo_rag_inference_active_requests",
    "Inference requests currently holding a model execution slot.",
    ["operation"],
)

inference_overload_total = Counter(
    "vo_rag_inference_overload_total",
    "Inference requests rejected because all execution slots were occupied.",
    ["operation"],
)

inference_request_duration_seconds = Histogram(
    "vo_rag_inference_request_duration_seconds",
    "Inference request wall time by operation.",
    ["operation"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)

vlm_gateway_requests_total = Counter(
    "vo_rag_vlm_gateway_requests_total",
    "Resident VLM gateway requests by bounded outcome.",
    ["outcome"],
)

vlm_gateway_active_requests = Gauge(
    "vo_rag_vlm_gateway_active_requests",
    "Resident VLM requests currently forwarded to llama.cpp.",
)

vlm_gateway_queue_seconds = Histogram(
    "vo_rag_vlm_gateway_queue_seconds",
    "Time a resident VLM request waits for retrieval-idle admission.",
    buckets=(0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

vlm_gateway_duration_seconds = Histogram(
    "vo_rag_vlm_gateway_duration_seconds",
    "Resident VLM upstream request duration after admission.",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)

vlm_gateway_admission_timeouts_total = Counter(
    "vo_rag_vlm_gateway_admission_timeouts_total",
    "Resident VLM requests whose bounded retrieval-idle wait expired.",
)

vlm_gateway_up = Gauge(
    "vo_rag_vlm_gateway_up",
    "Whether the localhost llama.cpp VLM passed its latest health probe.",
)


# ---------------------------------------------------------------------------
# PII label guard for Prometheus metrics.
# ---------------------------------------------------------------------------
class UnsafeLabel(ValueError):
    """Raised when a Prometheus label value smells like PII."""


def assert_safe_labels(**labels: object) -> None:
    """Reject label values that look like PII before they ever reach /metrics.

    Prometheus labels are published verbatim on the scrape endpoint, so a
    user-provided value (email, phone, API key) leaks on every exposition.
    Call this from metric-emitting code when the label set is derived from
    any potentially-user-controlled input.
    """
    from vo_rag.core.observability.redaction import (  # local: avoid import cycle
        _API_KEY_PATTERNS,
        _EMAIL_RE,
        _PHONE_RE,
        _US_SSN_RE,
    )

    for name, value in labels.items():
        if not isinstance(value, str):
            continue
        if _EMAIL_RE.search(value):
            raise UnsafeLabel(f"label {name!r} contains an email address")
        if _US_SSN_RE.search(value):
            raise UnsafeLabel(f"label {name!r} contains a US SSN")
        if _PHONE_RE.search(value):
            raise UnsafeLabel(f"label {name!r} contains a phone number")
        for pat in _API_KEY_PATTERNS:
            if pat.search(value):
                raise UnsafeLabel(f"label {name!r} contains an API key")


_HOOKS_INSTALLED = False


def _embedding_endpoint_label(client: Any) -> str:
    base_url = str(getattr(getattr(client, "_client", None), "base_url", "")).lower()
    if "p520" in base_url or "localhost" in base_url or "query" in base_url:
        return "query"
    return "batch"


def track_pipeline_run(pipeline: str, outcome: str, duration_seconds: float) -> None:
    """Record pipeline outcome and wall-clock duration."""
    pipeline_runs_total.labels(pipeline=pipeline, outcome=outcome).inc()
    pipeline_duration_seconds.labels(pipeline=pipeline).observe(duration_seconds)


def instrument_pipeline_call(
    pipeline: str,
    call: Callable[[], Awaitable[Any]],
) -> Awaitable[Any]:
    """Wrap an async pipeline call with active gauge + duration counters."""

    async def _runner() -> Any:
        t0 = time.perf_counter()
        active_pipeline_runs.labels(pipeline=pipeline).inc()
        try:
            result = await call()
        except Exception:
            track_pipeline_run(
                pipeline=pipeline,
                outcome="failure",
                duration_seconds=time.perf_counter() - t0,
            )
            raise
        finally:
            active_pipeline_runs.labels(pipeline=pipeline).dec()

        outcome = "success"
        if isinstance(result, dict) and str(result.get("status", "")).lower() == "discarded":
            outcome = "discarded"
        track_pipeline_run(
            pipeline=pipeline,
            outcome=outcome,
            duration_seconds=time.perf_counter() - t0,
        )
        return result

    return _runner()


# ---------------------------------------------------------------------------
# Storage instrumentation.
#
# core/storage (built in a later task) will call `instrument_qdrant_store`
# once at import time. We deliberately do NOT import storage here — that
# would create a cycle: storage imports metrics, and metrics would import
# storage. Keeping the wire at the call site is what makes this safe to
# load before storage exists.
# ---------------------------------------------------------------------------
def instrument_qdrant_store(store_cls: type[Any]) -> type[Any]:
    """Monkey-patch `store_cls.upsert_vectors` to feed the
    `qdrant_upsert_points_total` counter. Idempotent.

    Args:
        store_cls: The Qdrant store class to instrument. Must expose an
            async `upsert_vectors(self, collection, points)` method.

    Returns:
        The same class, for call-site convenience.
    """
    original_upsert = store_cls.upsert_vectors

    async def instrumented_upsert(
        self: Any, collection: str, points: list[Any], **kwargs: Any
    ) -> Any:
        result = await original_upsert(self, collection, points, **kwargs)
        qdrant_upsert_points_total.labels(collection=collection).inc(len(points))
        return result

    store_cls.upsert_vectors = instrumented_upsert
    return store_cls


# ---------------------------------------------------------------------------
# Stuck / overloaded ingest.
#
# Dispatchers are separate processes from the API that serves /metrics, so
# nothing here relies on a dispatcher's in-process counters. Instead the ingest
# records what it did into its own envelope row, under
# ``intake_envelopes.metadata->'metrics'``, and the API's operational sampler
# derives the gauges/counters below from Postgres.
#
# ``metadata->'metrics'`` shape (every key optional; other metadata keys, e.g.
# ``quality`` or ``preflight``, are never touched):
#
#   claimed_at_epoch_s     number  set when work starts: every dispatcher claim,
#                                  and the start of a streaming ingest
#   stale_reclaims         number  consecutive stale reclaims since the last
#                                  fresh claim (a fresh claim resets it)
#   stale_reclaims_total   number  every stale reclaim ever (never reset)
#   active_stream          bool    true while a streaming (API/CLI/MCP) ingest
#                                  runs the envelope; false when it finishes or
#                                  a dispatcher claims it
#   stream_heartbeat_epoch_s number refreshed every flush interval by a live
#                                  stream; a stale one stops counting as in flight
#   failures               number  times the envelope was recorded failed
#   failed_at_epoch_s      number  when it was last recorded failed
#   current_stage          string  the stage running now (latest attempt)
#   pages                  number  page count, as soon as the parser knows it
#   stages                 object  {<stage>: {"seconds": number}}, for the
#                                  attempt named by ``stages_attempt``
#   stages_attempt         string  the attempt id ``stages`` belongs to
#   embed                  object  {"calls": number, "retries_429": number,
#                                  "attempts": {<attempt id>: {"calls",
#                                  "retries_429"}}}: per-attempt absolute totals;
#                                  calls/retries_429 are their sum (every attempt)
#   qdrant                 object  {"calls": number, "p95_s": number,
#                                  "max_s": number}, latest attempt
#
# Recording never fails a book: every write is best-effort (logged, swallowed).
# ---------------------------------------------------------------------------

ENVELOPE_METRICS_KEY = "metrics"

#: The per-envelope budget, ``max(15 min, pages * 6 s)`` -- the same formula
#: the canary's stage gate uses (``core/corpus/canary.stage_budget_s``; this
#: module is a leaf and can't import it, a unit test keeps the two equal).
ENVELOPE_BUDGET_FLOOR_S = 15 * 60.0
ENVELOPE_BUDGET_PER_PAGE_S = 6.0
#: Budget while pages are still unknown (neither ``metrics.pages`` nor
#: ``preflight.pages``) and the envelope hasn't finished parsing: a giant's
#: parse alone can pass 15 minutes, so the floor there is an hour.
ENVELOPE_BUDGET_UNKNOWN_PAGES_PARSE_S = 60 * 60.0

#: How often a running ingest flushes pending metrics (so a long embed stage's
#: 429s reach Postgres while it runs, not only when the stage ends).
INGEST_METRICS_FLUSH_INTERVAL_S = 60.0
#: How often the API recomputes the whole-table embed sums (cached between).
INGEST_TOTALS_INTERVAL_S = 300.0
#: A streaming ingest whose heartbeat is older than this is no longer treated
#: as in flight (e.g. the API crashed mid-stream and never cleared the flag).
STREAM_HEARTBEAT_STALE_S = 10 * 60.0
#: Qdrant latencies kept per attempt for the p95 (a uniform reservoir sample).
QDRANT_LATENCY_RESERVOIR = 4096

envelope_processing_seconds_max = Gauge(
    "vo_rag_envelope_processing_seconds_max",
    "Seconds since work started on the in-flight envelope furthest over its budget; "
    "-1 when none or unknown.",
)

envelope_budget_seconds = Gauge(
    "vo_rag_envelope_budget_seconds",
    "Budget of the envelope reported by vo_rag_envelope_processing_seconds_max: "
    "max(15 min, pages x 6 s) with pages from metrics or preflight, else 60 min "
    "before parsing ends, else 15 min; -1 when none or unknown.",
)

envelope_reclaims_max = Gauge(
    "vo_rag_envelope_reclaims_max",
    "Most stale reclaims (cumulative) of any processing envelope; -1 when unknown.",
)

intake_failures_last_hour = Gauge(
    "vo_rag_intake_failures_last_hour",
    "Failed envelopes whose recorded failure time is within the past hour; -1 when unknown.",
)

# The SQL below is spelled out literally (no string composition). Every jsonb
# read is type-guarded -- a non-object ``metrics``/``stages`` reads as ``{}``
# and a non-numeric count as 0 -- so a malformed row can never make a write or
# a sample raise. Writes only touch the ``metrics`` key, and only when
# ``metadata`` is an object.

#: $1 envelope id, $2 jsonb of keys to overwrite (pages, current_stage,
#: active_stream, qdrant), $3 jsonb of stages to set, $4/$5 this attempt's
#: *absolute* embed call / 429-retry totals, $6 the attempt id.
#:
#: ``embed.attempts`` maps every attempt id to its absolute totals and
#: ``embed.calls`` / ``embed.retries_429`` are recomputed as their sum under the
#: row lock, so replaying a flush, or two attempts interleaving (a zombie worker
#: after a stale reclaim), can never count an attempt twice. ``stages`` belongs
#: to one attempt (``stages_attempt``): the first write of a new attempt
#: replaces it wholesale, so stages an earlier attempt reached don't linger.
ENVELOPE_METRICS_MERGE_SQL = """
UPDATE intake_envelopes AS env
SET metadata = env.metadata || jsonb_build_object('metrics',
    (CASE WHEN jsonb_typeof(env.metadata->'metrics') = 'object'
          THEN env.metadata->'metrics' ELSE '{}'::jsonb END)
    || $2::jsonb
    || jsonb_build_object(
        'stages',
            (CASE WHEN env.metadata #>> '{metrics,stages_attempt}' = $6::text
                       AND jsonb_typeof(env.metadata #> '{metrics,stages}') = 'object'
                  THEN env.metadata #> '{metrics,stages}' ELSE '{}'::jsonb END)
            || $3::jsonb,
        'stages_attempt', $6::text,
        'embed', (
            SELECT jsonb_build_object(
                'calls', COALESCE(SUM(
                    CASE WHEN jsonb_typeof(e.value->'calls') = 'number'
                         THEN (e.value->>'calls')::numeric ELSE 0 END), 0),
                'retries_429', COALESCE(SUM(
                    CASE WHEN jsonb_typeof(e.value->'retries_429') = 'number'
                         THEN (e.value->>'retries_429')::numeric ELSE 0 END), 0),
                'attempts', a.attempts
            )
            FROM (
                SELECT (
                    CASE
                        WHEN jsonb_typeof(env.metadata #> '{metrics,embed,attempts}') = 'object'
                            THEN env.metadata #> '{metrics,embed,attempts}'
                        -- totals written without a per-attempt map: keep them as one entry
                        WHEN jsonb_typeof(env.metadata #> '{metrics,embed,calls}') = 'number'
                            THEN jsonb_build_object('legacy', jsonb_build_object(
                                'calls', env.metadata #> '{metrics,embed,calls}',
                                'retries_429',
                                    CASE WHEN jsonb_typeof(
                                              env.metadata #> '{metrics,embed,retries_429}'
                                          ) = 'number'
                                         THEN env.metadata #> '{metrics,embed,retries_429}'
                                         ELSE '0'::jsonb END))
                        ELSE '{}'::jsonb
                    END
                ) || jsonb_build_object(
                    $6::text,
                    jsonb_build_object('calls', $4::bigint, 'retries_429', $5::bigint)
                ) AS attempts
            ) AS a
            CROSS JOIN LATERAL jsonb_each(a.attempts) AS e
            GROUP BY a.attempts
        )
    )
)
WHERE env.id = $1 AND jsonb_typeof(env.metadata) = 'object'
"""

ENVELOPE_FAILURE_METRIC_SQL = """
UPDATE intake_envelopes
SET metadata = metadata || jsonb_build_object('metrics',
    (CASE WHEN jsonb_typeof(metadata->'metrics') = 'object'
          THEN metadata->'metrics' ELSE '{}'::jsonb END)
    || jsonb_build_object(
        'failures',
            (CASE WHEN jsonb_typeof(metadata #> '{metrics,failures}') = 'number'
                  THEN (metadata #> '{metrics,failures}')::numeric ELSE 0 END)
            + 1,
        'failed_at_epoch_s', EXTRACT(EPOCH FROM NOW())
    )
)
WHERE id = $1 AND jsonb_typeof(metadata) = 'object'
"""

#: A streaming ingest has no dispatcher claim: stamp the start the same way.
ENVELOPE_STREAM_START_SQL = """
UPDATE intake_envelopes
SET metadata = metadata || jsonb_build_object('metrics',
    (CASE WHEN jsonb_typeof(metadata->'metrics') = 'object'
          THEN metadata->'metrics' ELSE '{}'::jsonb END)
    || jsonb_build_object(
        'claimed_at_epoch_s', EXTRACT(EPOCH FROM NOW()),
        'stream_heartbeat_epoch_s', EXTRACT(EPOCH FROM NOW()),
        'active_stream', true,
        'current_stage', NULL
    )
)
WHERE id = $1 AND jsonb_typeof(metadata) = 'object'
"""

#: A live stream's heartbeat (database clock), sent every flush interval.
ENVELOPE_STREAM_HEARTBEAT_SQL = """
UPDATE intake_envelopes
SET metadata = metadata || jsonb_build_object('metrics',
    (CASE WHEN jsonb_typeof(metadata->'metrics') = 'object'
          THEN metadata->'metrics' ELSE '{}'::jsonb END)
    || jsonb_build_object('stream_heartbeat_epoch_s', EXTRACT(EPOCH FROM NOW()))
)
WHERE id = $1 AND jsonb_typeof(metadata) = 'object'
"""

#: Cheap, frequent sample. Only in-flight rows (``status IN ('received',
#: 'processing')``, served by ``idx_envelopes_status``) are unpacked, plus an
#: indexed count of recent failures (keyed on the recorded failure time, not
#: ``updated_at``, so later writes to a failed row don't re-count it). In
#: flight = dispatcher-processing, or a received envelope a streaming ingest
#: is running with a heartbeat newer than $4.
#: $1 floor seconds, $2 seconds per page, $3 unknown-pages budget before parse
#: ends, $4 stream heartbeat staleness bound (seconds).
INGEST_LIVE_SQL = """
WITH live AS (
    SELECT
        status,
        CASE WHEN jsonb_typeof(metadata->'metrics') = 'object'
             THEN metadata->'metrics' ELSE '{}'::jsonb END AS mx,
        metadata->'preflight' AS pf
    FROM intake_envelopes
    WHERE status IN ('received', 'processing')
),
timed AS (
    SELECT
        EXTRACT(EPOCH FROM NOW())::double precision
            - (mx->>'claimed_at_epoch_s')::double precision AS elapsed,
        COALESCE(
            CASE WHEN jsonb_typeof(mx->'pages') = 'number'
                      AND (mx->>'pages')::double precision > 0
                 THEN (mx->>'pages')::double precision END,
            CASE WHEN jsonb_typeof(pf->'pages') = 'number'
                      AND (pf->>'pages')::double precision > 0
                 THEN (pf->>'pages')::double precision
                 WHEN jsonb_typeof(pf->'pages') = 'string'
                      AND (pf->>'pages') ~ '^[0-9]{1,9}$'
                      AND (pf->>'pages')::double precision > 0
                 THEN (pf->>'pages')::double precision END
        ) AS pages,
        COALESCE(mx->>'current_stage', '') IN ('', 'register_source', 'parse') AS parsing
    FROM live
    WHERE jsonb_typeof(mx->'claimed_at_epoch_s') = 'number'
      AND (
            status = 'processing'
            OR (
                mx->'active_stream' = 'true'::jsonb
                AND jsonb_typeof(mx->'stream_heartbeat_epoch_s') = 'number'
                AND (mx->>'stream_heartbeat_epoch_s')::double precision
                    >= EXTRACT(EPOCH FROM NOW())::double precision - $4::double precision
            )
      )
),
budgeted AS (
    SELECT
        elapsed,
        CASE
            WHEN pages IS NOT NULL
                THEN GREATEST($1::double precision, pages * $2::double precision)
            WHEN parsing THEN GREATEST($1::double precision, $3::double precision)
            ELSE $1::double precision
        END AS budget
    FROM timed
),
worst AS (
    SELECT elapsed, budget FROM budgeted ORDER BY elapsed - budget DESC LIMIT 1
)
SELECT
    (SELECT elapsed FROM worst) AS processing_seconds,
    (SELECT budget FROM worst) AS budget_seconds,
    -- Processing rows only: a reclaimed envelope requeued
    -- to 'received' is waiting, not looping, and must not keep the
    -- VoRagEnvelopeReclaimLoop alert on until it runs again.
    (SELECT COALESCE(MAX(
        CASE WHEN jsonb_typeof(mx->'stale_reclaims_total') = 'number'
             THEN (mx->>'stale_reclaims_total')::numeric ELSE 0 END), 0)
        FROM live WHERE status = 'processing') AS reclaims_max,
    (SELECT COUNT(*) FROM intake_envelopes
        WHERE status = 'failed'
          AND jsonb_typeof(metadata #> '{metrics,failed_at_epoch_s}') = 'number'
          AND (metadata #>> '{metrics,failed_at_epoch_s}')::double precision
              >= EXTRACT(EPOCH FROM NOW())::double precision - 3600)
        AS failures_last_hour
"""

#: Whole-table embed sums: every row's metadata is unpacked, so this runs on
#: ``INGEST_TOTALS_INTERVAL_S`` and is cached between samples.
INGEST_TOTALS_SQL = """
SELECT
    COALESCE(SUM(
        CASE WHEN jsonb_typeof(metadata #> '{metrics,embed,calls}') = 'number'
             THEN (metadata #>> '{metrics,embed,calls}')::numeric ELSE 0 END), 0)
        AS embed_calls_total,
    COALESCE(SUM(
        CASE WHEN jsonb_typeof(metadata #> '{metrics,embed,retries_429}') = 'number'
             THEN (metadata #>> '{metrics,embed,retries_429}')::numeric ELSE 0 END), 0)
        AS embed_retries_429_total
FROM intake_envelopes
WHERE jsonb_typeof(metadata #> '{metrics,embed}') = 'object'
"""


class IngestMetricsRecorder:
    """What one ingest attempt did, accumulated until the next flush.

    Thread-safe (``asyncio.to_thread`` steps may record too). Stage times and
    the embed counters are *pending* until ``persist_ingest_metrics`` writes
    them; the overwrite keys (pages, current stage, stream flag, Qdrant
    summary) are re-sent only when they change. Qdrant latencies are kept as a
    bounded uniform reservoir (``QDRANT_LATENCY_RESERVOIR``) with the exact
    call count and max beside it.
    """

    def __init__(self, *, reservoir_size: int = QDRANT_LATENCY_RESERVOIR) -> None:
        self._lock = threading.Lock()
        self._flush_lock = asyncio.Lock()
        self._reservoir_size = reservoir_size
        self._rng = random.Random(0x7A1)  # noqa: S311 - sampling, not security
        self.stages: dict[str, float] = {}
        self.embed_calls = 0
        self.embed_retries_429 = 0
        self.qdrant_latencies: list[float] = []
        self.qdrant_calls = 0
        self.qdrant_max_s = 0.0
        self.pages: int | None = None
        self.current_stage: str | None = None
        self.active_stream: bool | None = None
        #: Identifies this attempt's absolute embed totals in the envelope row.
        self.attempt_id = uuid.uuid4().hex
        self._open_stage: tuple[str, float] | None = None
        self._flushed_overwrite: dict[str, Any] = {}
        self._flushed_embed: tuple[int, int] = (0, 0)

    def record_embed_call(self, retries_429: int) -> None:
        with self._lock:
            self.embed_calls += 1
            self.embed_retries_429 += max(0, int(retries_429))

    def record_qdrant_call(self, seconds: float) -> None:
        seconds = float(seconds)
        with self._lock:
            self.qdrant_calls += 1
            self.qdrant_max_s = max(self.qdrant_max_s, seconds)
            if len(self.qdrant_latencies) < self._reservoir_size:
                self.qdrant_latencies.append(seconds)
            else:
                slot = self._rng.randrange(self.qdrant_calls)
                if slot < self._reservoir_size:
                    self.qdrant_latencies[slot] = seconds

    def record_stage(self, stage: str, seconds: float) -> None:
        with self._lock:
            self.stages[stage] = float(seconds)

    def enter_stage(self, stage: str, *, now: float | None = None) -> None:
        """``stage`` starts now: the stage that was open ends here. For a
        pipeline (the llama-index path) whose steps can only mark their start."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._open_stage is not None and self._open_stage[0] == stage:
                return
            if self._open_stage is not None:
                name, started = self._open_stage
                self.stages[name] = max(0.0, now - started)
            self._open_stage = (stage, now)
            self.current_stage = stage

    def close_stage(self, *, now: float | None = None) -> None:
        """End the open stage (the run finished or failed)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._open_stage is not None:
                name, started = self._open_stage
                self.stages[name] = max(0.0, now - started)
                self._open_stage = None

    def qdrant_summary(self) -> dict[str, Any] | None:
        with self._lock:
            latencies = sorted(self.qdrant_latencies)
            calls, max_s = self.qdrant_calls, self.qdrant_max_s
        if not latencies:
            return None
        rank = max(1, math.ceil(0.95 * len(latencies)))
        return {"calls": calls, "p95_s": latencies[rank - 1], "max_s": max_s}

    def _overwrite(self) -> dict[str, Any]:
        overwrite: dict[str, Any] = {}
        if self.pages is not None:
            overwrite["pages"] = self.pages
        if self.current_stage is not None:
            overwrite["current_stage"] = self.current_stage
        if self.active_stream is not None:
            overwrite["active_stream"] = self.active_stream
        qdrant = self.qdrant_summary()
        if qdrant is not None:
            overwrite["qdrant"] = qdrant
        return overwrite

    def _take_pending(self) -> tuple[dict[str, float], int, int]:
        """Pending stage times (drained) and the attempt's absolute embed
        totals (never drained -- they are resent whole each time)."""
        with self._lock:
            pending = (dict(self.stages), self.embed_calls, self.embed_retries_429)
            self.stages.clear()
        return pending

    def _restore_pending(self, stages: dict[str, float]) -> None:
        # Stage times are overwrites, so re-sending them is harmless.
        with self._lock:
            for name, seconds in stages.items():
                self.stages.setdefault(name, seconds)


_current_recorder: ContextVar[IngestMetricsRecorder | None] = ContextVar(
    "vo_rag_ingest_metrics_recorder", default=None
)


def current_ingest_recorder() -> IngestMetricsRecorder | None:
    return _current_recorder.get()


def set_ingest_recorder(recorder: IngestMetricsRecorder | None) -> None:
    """Make ``recorder`` current in the running ``contextvars.Context``. For
    ``Context.run`` when the work runs in tasks started from a copied context
    (the streaming path); elsewhere use ``ingest_recording()``."""
    _current_recorder.set(recorder)


@contextmanager
def ingest_recording() -> Iterator[IngestMetricsRecorder]:
    """Make a fresh recorder current for this task (and tasks/threads it starts)."""
    recorder = IngestMetricsRecorder()
    token = _current_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _current_recorder.reset(token)


def record_embed_call(retries_429: int) -> None:
    """One logical embed request finished (or gave up) after ``retries_429`` 429s."""
    recorder = _current_recorder.get()
    if recorder is not None:
        recorder.record_embed_call(retries_429)


def record_qdrant_call(seconds: float) -> None:
    recorder = _current_recorder.get()
    if recorder is not None:
        recorder.record_qdrant_call(seconds)


def record_ingest_stage(stage: str, seconds: float) -> None:
    recorder = _current_recorder.get()
    if recorder is not None:
        recorder.record_stage(stage, seconds)


def enter_ingest_stage(stage: str) -> None:
    """The running ingest has started ``stage`` (ends the previous one)."""
    recorder = _current_recorder.get()
    if recorder is not None:
        recorder.enter_stage(stage)


def record_ingest_pages(pages: int | None) -> None:
    """The parser knows the page count (sent with the next flush)."""
    recorder = _current_recorder.get()
    if recorder is not None and pages and pages > 0:
        recorder.pages = int(pages)


async def persist_ingest_metrics(
    pool: Any | None, envelope_id: str, recorder: IngestMetricsRecorder
) -> None:
    """Merge the recorder's pending data into ``metadata->'metrics'``.

    Best-effort: a failure is logged and swallowed and the next flush re-sends
    (also after a cancel). Every write is idempotent -- the attempt's
    absolute embed totals and overwritten keys -- so a flush that committed
    before it was cancelled is never counted twice. Flushes of one recorder
    are serialized; nothing is written when there is nothing new.
    """
    if pool is None:
        return
    async with recorder._flush_lock:
        stages, calls, retries = recorder._take_pending()
        overwrite = recorder._overwrite()
        if (
            not stages
            and (calls, retries) == recorder._flushed_embed
            and overwrite == recorder._flushed_overwrite
        ):
            return
        try:
            await pool.execute(
                ENVELOPE_METRICS_MERGE_SQL,
                envelope_id,
                overwrite,
                {name: {"seconds": round(seconds, 3)} for name, seconds in stages.items()},
                calls,
                retries,
                recorder.attempt_id,
            )
            recorder._flushed_overwrite = overwrite
            recorder._flushed_embed = (calls, retries)
        except asyncio.CancelledError:
            # The write may or may not have committed; everything sent is
            # idempotent (absolute embed totals, overwritten stage times), so
            # re-sending it on the next flush can't double-count.
            recorder._restore_pending(stages)
            raise
        except Exception:  # noqa: BLE001 - recording must never fail a book
            recorder._restore_pending(stages)
            log.warning("ingest_metrics_persist_failed envelope_id=%s", envelope_id, exc_info=True)


async def _periodic_flush(
    pool: Any,
    envelope_id: str,
    recorder: IngestMetricsRecorder,
    interval_s: float,
    heartbeat: bool,
) -> None:
    while True:
        await asyncio.sleep(interval_s)
        if heartbeat:
            await _best_effort(pool, ENVELOPE_STREAM_HEARTBEAT_SQL, envelope_id, "heartbeat")
        await persist_ingest_metrics(pool, envelope_id, recorder)


def start_periodic_ingest_flush(
    pool: Any | None,
    envelope_id: str,
    recorder: IngestMetricsRecorder,
    interval_s: float = INGEST_METRICS_FLUSH_INTERVAL_S,
    *,
    heartbeat: bool = False,
) -> asyncio.Task[None] | None:
    """Flush ``recorder`` every ``interval_s`` until ``stop_periodic_ingest_flush``.
    With ``heartbeat`` (streaming ingests) each tick also refreshes
    ``stream_heartbeat_epoch_s``."""
    if pool is None or interval_s <= 0:
        return None
    return asyncio.create_task(
        _periodic_flush(pool, envelope_id, recorder, interval_s, heartbeat),
        name=f"vo-rag-ingest-metrics-flush-{envelope_id}",
    )


async def stop_periodic_ingest_flush(task: asyncio.Task[None] | None) -> None:
    """Cancel the flush task and wait for it. The task's own cancellation is
    expected and absorbed; a crash in it is logged; a cancellation aimed at the
    *caller* while it waits propagates (``gather`` raises it, it isn't
    swallowed)."""
    if task is None:
        return
    task.cancel()
    [outcome] = await asyncio.gather(task, return_exceptions=True)
    if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
        log.warning("ingest_metrics_flush_task_failed", exc_info=outcome)


async def record_envelope_failure(pool: Any | None, envelope_id: str) -> None:
    """Add one to ``metadata->'metrics'->'failures'``. Best-effort."""
    await _best_effort(pool, ENVELOPE_FAILURE_METRIC_SQL, envelope_id, "failure")


async def record_stream_start(pool: Any | None, envelope_id: str) -> None:
    """Stamp a streaming ingest's start (``claimed_at_epoch_s``, ``active_stream``)."""
    await _best_effort(pool, ENVELOPE_STREAM_START_SQL, envelope_id, "stream_start")


async def _best_effort(pool: Any | None, sql: str, envelope_id: str, what: str) -> None:
    if pool is None:
        return
    try:
        await pool.execute(sql, envelope_id)
    except Exception:  # noqa: BLE001 - recording must never fail a book
        log.warning("ingest_metric_%s_failed envelope_id=%s", what, envelope_id, exc_info=True)


# Postgres-derived embed totals, exposed as counters. Each is a sum over
# envelope rows, which can shrink when a row is deleted; a shrinking counter
# reads as a reset and would make ``increase()`` invent a burst, so each is
# clamped to its per-process high-water mark. Empty until the first sample.
_ingest_totals: dict[str, float] = {}
_ingest_totals_sampled_at: list[float] = []  # monotonic time of the last totals sample


class _IngestTotalsCollector:
    def _families(self) -> list[Any]:
        calls = CounterMetricFamily(
            "vo_rag_embed_calls",
            "Ingest embed requests recorded on envelopes (from Postgres, every 5 min).",
        )
        retries = CounterMetricFamily(
            "vo_rag_embed_retries",
            "Ingest embed retries by reason, recorded on envelopes (from Postgres, every 5 min).",
            labels=["reason"],
        )
        return [calls, retries]

    def describe(self) -> list[Any]:
        return self._families()

    def collect(self) -> Iterator[Any]:
        if not _ingest_totals:
            return
        calls, retries = self._families()
        calls.add_metric([], _ingest_totals["embed_calls"])
        retries.add_metric(["429"], _ingest_totals["embed_retries_429"])
        yield from (calls, retries)


REGISTRY.register(_IngestTotalsCollector())


def _update_total(key: str, value: float) -> None:
    _ingest_totals[key] = max(_ingest_totals.get(key, 0.0), value)


def _set_live_unknown() -> None:
    envelope_processing_seconds_max.set(-1)
    envelope_budget_seconds.set(-1)
    envelope_reclaims_max.set(-1)
    intake_failures_last_hour.set(-1)


def _row_number(row: Any, key: str) -> float | None:
    value = row[key]
    return None if value is None else float(value)


async def _fetch(pool: Any, sql: str, *args: Any) -> Any:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *args)
    if row is None:
        raise RuntimeError("ingest health aggregate returned no row")
    return row


async def sample_ingest_health(pool: Any | None, *, now: float | None = None) -> None:
    """One sample of the stuck/overloaded-ingest metrics.

    The in-flight gauges come from ``INGEST_LIVE_SQL`` every call; the embed
    totals from ``INGEST_TOTALS_SQL`` at most every ``INGEST_TOTALS_INTERVAL_S``
    (the counters keep their cached value between). Fails closed: on error
    the gauges read -1 and the error is counted under
    ``vo_rag_operational_sampler_errors_total{component="ingest_health"}``.
    """
    if pool is None:
        _set_live_unknown()
        return
    try:
        row = await _fetch(
            pool,
            INGEST_LIVE_SQL,
            ENVELOPE_BUDGET_FLOOR_S,
            ENVELOPE_BUDGET_PER_PAGE_S,
            ENVELOPE_BUDGET_UNKNOWN_PAGES_PARSE_S,
            STREAM_HEARTBEAT_STALE_S,
        )
        processing = _row_number(row, "processing_seconds")
        budget = _row_number(row, "budget_seconds")
        reclaims = _row_number(row, "reclaims_max") or 0
        failures = _row_number(row, "failures_last_hour") or 0
    except Exception:  # noqa: BLE001
        log.exception("ingest_health_sample_failed")
        operational_sampler_errors_total.labels(component="ingest_health").inc()
        _set_live_unknown()
    else:
        if processing is None or budget is None:
            envelope_processing_seconds_max.set(-1)
            envelope_budget_seconds.set(-1)
        else:
            envelope_processing_seconds_max.set(processing)
            envelope_budget_seconds.set(budget)
        envelope_reclaims_max.set(reclaims)
        intake_failures_last_hour.set(failures)

    clock = time.monotonic() if now is None else now
    if (
        _ingest_totals_sampled_at
        and clock - _ingest_totals_sampled_at[0] < INGEST_TOTALS_INTERVAL_S
    ):
        return
    try:
        totals = await _fetch(pool, INGEST_TOTALS_SQL)
        calls = _row_number(totals, "embed_calls_total") or 0
        retries = _row_number(totals, "embed_retries_429_total") or 0
    except Exception:  # noqa: BLE001
        log.exception("ingest_totals_sample_failed")
        operational_sampler_errors_total.labels(component="ingest_totals").inc()
        return
    _ingest_totals_sampled_at[:] = [clock]
    _update_total("embed_calls", calls)
    _update_total("embed_retries_429", retries)
