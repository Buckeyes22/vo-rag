"""Worker roster — runtime surface for deployable vo-rag workers.

Re-exports the registry helpers so callers can do ``from vo_rag.workers
import load_worker_registry`` without reaching into ``registry``.
"""

from __future__ import annotations

from vo_rag.workers.registry import (
    WorkerSpec,
    WorkerStatus,
    load_worker_registry,
)

__all__ = ["WorkerSpec", "WorkerStatus", "load_worker_registry"]
