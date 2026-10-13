"""core/storage: Qdrant vector store + Garage blob.

Public surface:

- `QdrantStore` — async wrapper, four-vector schema (dense / sparse / bm25 /
  colbert rerank-only), 19 named collections, idempotent `setup_collections`
  that raises on schema mismatch.
- `VoPoint` — Pydantic model for the write-side point contract.
- `deterministic_point_id` — F-024 point ID built from
  ``(source_id, chunk_index, stage[, version_hash])`` on SHA-256, so retries
  overwrite and a source's points carry a stable ID.
- `bge_sparse_to_qdrant` — BGE-M3 sparse map -> sorted Qdrant `SparseVector`.
- `setup_collections` / `COLLECTION_NAMES` — schema bootstrap.
- `GarageClient` — async S3-over-Garage blob.

Observability wiring: this package calls
`vo_rag.core.observability.metrics.instrument_qdrant_store(QdrantStore)` at
import time. That import direction (storage -> observability) is fine and
deliberately chosen — observability stays a leaf and must NOT import storage
(see `vo_rag/core/observability/metrics.py:1-12` and `:262-269`).
"""

from __future__ import annotations

from vo_rag.core.observability.metrics import instrument_qdrant_store
from vo_rag.core.storage.garage import (
    GarageClient,
    GarageObjectMetadata,
    GarageObjectTooLarge,
    VerifiedGarageObject,
    garage_from_env,
)
from vo_rag.core.storage.points import VoPoint
from vo_rag.core.storage.qdrant import (
    COLLECTION_NAMES,
    QdrantStore,
    bge_sparse_to_qdrant,
    deterministic_point_id,
    setup_collections,
)

# Wire Prometheus counter onto QdrantStore.upsert_vectors. Idempotent on
# repeat import per `metrics.instrument_qdrant_store` contract.
instrument_qdrant_store(QdrantStore)

__all__ = [
    "COLLECTION_NAMES",
    "GarageClient",
    "GarageObjectMetadata",
    "GarageObjectTooLarge",
    "QdrantStore",
    "VoPoint",
    "VerifiedGarageObject",
    "bge_sparse_to_qdrant",
    "deterministic_point_id",
    "garage_from_env",
    "setup_collections",
]
