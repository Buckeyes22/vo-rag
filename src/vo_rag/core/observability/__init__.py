"""observability: metrics, redaction, cost rollup, persistence.

Architectural notes:

- This package is a LEAF. It must not import `vo_rag.core.storage` or any
  sibling that depends on storage, because storage imports observability.
  Storage wires up its own metric hooks by calling
  `vo_rag.core.observability.metrics.instrument_qdrant_store(QdrantStore)`
  at import time.
- The redaction policy env override is `VORAG_REDACTION_LEVEL`.
- The cost accumulator + asyncpg-backed persistence live together in
  `cost.py` — no dead modules.
"""

from vo_rag.core.observability.cost import (
    CostAccumulator,
    StepCost,
    append_step_cost,
    cost_context,
    current_cost_accumulator,
    finalize_envelope_cost,
)
from vo_rag.core.observability.redaction import (
    DEFAULT_ALWAYS_SCRUB,
    DEFAULT_ENTITIES_BY_LEVEL,
    RedactConfig,
    RedactionEntity,
    RedactionLevel,
    Redactor,
    get_default_redactor,
    redact,
    reset_default_redactor,
)

__all__ = [
    # redaction
    "DEFAULT_ALWAYS_SCRUB",
    "DEFAULT_ENTITIES_BY_LEVEL",
    "RedactConfig",
    "RedactionEntity",
    "RedactionLevel",
    "Redactor",
    "get_default_redactor",
    "redact",
    "reset_default_redactor",
    # cost
    "CostAccumulator",
    "StepCost",
    "append_step_cost",
    "cost_context",
    "current_cost_accumulator",
    "finalize_envelope_cost",
]
