"""Generation-scoped Qdrant mutation contracts."""

from __future__ import annotations

from typing import Any, cast

import pytest
from qdrant_client.models import FieldCondition, Filter, IsEmptyCondition

from vo_rag.core.storage.qdrant import QdrantStore


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def set_payload(self, **kwargs: Any) -> None:
        self.calls.append(("set_payload", kwargs))

    async def delete(self, **kwargs: Any) -> None:
        self.calls.append(("delete", kwargs))

    async def count(self, **kwargs: Any) -> Any:
        self.calls.append(("count", kwargs))
        return type("Count", (), {"count": 3})()


def _store() -> tuple[QdrantStore, _Client]:
    client = _Client()
    store = object.__new__(QdrantStore)
    store._client = cast(Any, client)  # noqa: SLF001
    return store, client


def _filter_values(value: object) -> dict[str, object]:
    assert isinstance(value, Filter)
    conditions = value.must or []
    return {
        condition.key: getattr(condition.match, "value", None)
        for condition in conditions
        if isinstance(condition, FieldCondition)
    }


@pytest.mark.asyncio
async def test_generation_state_update_targets_only_source_and_generation() -> None:
    store, client = _store()

    await store.set_generation_state("books", "source-1", "gen-1", "active")

    operation, kwargs = client.calls[0]
    assert operation == "set_payload"
    assert kwargs["payload"] == {"ingest_state": "active"}
    assert _filter_values(kwargs["points"]) == {
        "source_id": "source-1",
        "generation_id": "gen-1",
    }
    assert kwargs["wait"] is True


@pytest.mark.asyncio
async def test_generation_cleanup_never_deletes_other_source_points() -> None:
    store, client = _store()

    await store.delete_generation("books", "source-1", "gen-old")

    operation, kwargs = client.calls[0]
    assert operation == "delete"
    assert _filter_values(kwargs["points_selector"]) == {
        "source_id": "source-1",
        "generation_id": "gen-old",
    }


@pytest.mark.asyncio
async def test_legacy_cleanup_targets_only_source_points_without_generation() -> None:
    store, client = _store()

    await store.delete_legacy_source("books", "source-1")

    operation, kwargs = client.calls[0]
    assert operation == "delete"
    selector = kwargs["points_selector"]
    assert isinstance(selector, Filter)
    conditions = selector.must or []
    assert _filter_values(selector) == {"source_id": "source-1"}
    empty_conditions = [c for c in conditions if isinstance(c, IsEmptyCondition)]
    assert len(empty_conditions) == 1
    assert empty_conditions[0].is_empty.key == "generation_id"


@pytest.mark.asyncio
async def test_generation_count_is_exact_and_state_scoped() -> None:
    store, client = _store()

    count = await store.count_source(
        "books",
        "source-1",
        generation_id="gen-1",
        ingest_state="active",
    )

    assert count == 3
    _operation, kwargs = client.calls[0]
    assert kwargs["exact"] is True
    assert _filter_values(kwargs["count_filter"]) == {
        "source_id": "source-1",
        "generation_id": "gen-1",
        "ingest_state": "active",
    }


@pytest.mark.asyncio
async def test_legacy_duplicate_count_uses_active_or_missing_state() -> None:
    store, client = _store()

    count = await store.count_active_or_legacy_source("books", "source-1")

    assert count == 3
    _operation, kwargs = client.calls[0]
    count_filter = kwargs["count_filter"]
    assert isinstance(count_filter, Filter)
    assert _filter_values(count_filter) == {"source_id": "source-1"}
    should = count_filter.should
    if should is None:
        should = []
    elif not isinstance(should, list):
        should = [should]
    assert any(
        isinstance(condition, FieldCondition)
        and condition.key == "ingest_state"
        and getattr(condition.match, "value", None) == "active"
        for condition in should
    )
    assert any(isinstance(condition, IsEmptyCondition) for condition in should)
