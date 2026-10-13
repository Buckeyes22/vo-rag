"""Smoke test: observability.metrics imports with zero vo_rag storage present.

This guards the architectural invariant that observability stays a leaf module
— `core/storage` (built in a later task) is responsible for calling
`instrument_qdrant_store(...)` once the QdrantStore class exists.
"""

from __future__ import annotations


def test_metrics_module_imports_without_storage() -> None:
    # Just import the module — the import path must not pull in
    # `vo_rag.core.storage` (which doesn't exist yet).
    import vo_rag.core.observability.metrics as m

    # Sanity: core prometheus objects are present.
    assert m.pipeline_runs_total is not None
    assert m.pipeline_duration_seconds is not None
    assert m.embedding_latency_seconds is not None
    assert m.qdrant_upsert_points_total is not None
    assert m.llm_calls_total is not None
    assert m.llm_retries_total is not None
    assert m.pipeline_cost_usd is not None
    assert m.pipeline_artifacts_total is not None
    assert m.assert_safe_labels is not None
    assert m.UnsafeLabel is not None


def test_instrument_qdrant_store_exposed_as_callable() -> None:
    """Later task `core/storage` will call this. Guard the symbol exists now."""
    from vo_rag.core.observability.metrics import instrument_qdrant_store

    assert callable(instrument_qdrant_store)
