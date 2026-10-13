"""Durable publication health stays source-scoped and honest on sample loss."""

from contextlib import asynccontextmanager
from typing import Any

import pytest

from vo_rag.api.operational_sampler import sample_source_publications
from vo_rag.core.observability.metrics import (
    source_publication_dependency_mismatches,
    source_publication_jobs,
    source_publication_pending_age_seconds,
    source_publication_pointer_mismatches,
)


class Connection:
    @asynccontextmanager
    async def transaction(self):  # noqa: ANN201
        yield

    async def execute(self, query: str) -> None:
        assert "REPEATABLE READ READ ONLY" in query

    async def fetch(self, query: str, corpora: list[str]) -> list[dict[str, Any]]:
        assert corpora == ["books", "reddit"]
        if "jsonb_each" in query:
            return [{"corpus": "books", "mismatches": 2}]
        if "LEFT JOIN" in query:
            return [{"corpus": "books", "mismatches": 1}]
        return [
            {"corpus": "books", "state": "partial_preserved", "job_count": 2, "oldest_age": 1900},
            {"corpus": "reddit", "state": "published", "job_count": 3, "oldest_age": 9000},
        ]


class Pool:
    @asynccontextmanager
    async def acquire(self):  # noqa: ANN201
        yield Connection()


@pytest.mark.asyncio
async def test_durable_partial_and_mismatch_counts_and_unknown_on_loss() -> None:
    await sample_source_publications(Pool())
    assert (
        source_publication_jobs.labels(corpus="books", state="partial_preserved")._value.get() == 2
    )  # noqa: SLF001
    assert source_publication_pending_age_seconds.labels(corpus="books")._value.get() == 1900  # noqa: SLF001
    assert source_publication_pending_age_seconds.labels(corpus="reddit")._value.get() == 0  # noqa: SLF001
    assert source_publication_pointer_mismatches.labels(corpus="books")._value.get() == 1  # noqa: SLF001
    assert source_publication_pointer_mismatches.labels(corpus="reddit")._value.get() == 0  # noqa: SLF001
    await sample_source_publications(None)
    assert (
        source_publication_jobs.labels(corpus="books", state="partial_preserved")._value.get() == -1
    )  # noqa: SLF001
    assert source_publication_pointer_mismatches.labels(corpus="books")._value.get() == -1  # noqa: SLF001


@pytest.mark.asyncio
async def test_dependency_health_is_distinct_and_becomes_unknown_on_sample_loss() -> None:
    await sample_source_publications(Pool())
    assert source_publication_dependency_mismatches.labels(corpus="books")._value.get() == 2  # noqa: SLF001
    assert source_publication_dependency_mismatches.labels(corpus="reddit")._value.get() == 0  # noqa: SLF001
    await sample_source_publications(None)
    assert source_publication_dependency_mismatches.labels(corpus="books")._value.get() == -1  # noqa: SLF001
