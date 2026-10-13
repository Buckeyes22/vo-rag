"""Unsupported cross-store erasure cannot commit SQL ahead of publication safety."""

from unittest.mock import AsyncMock

import pytest

from vo_rag.core.corpus.purge import purge_source


@pytest.mark.asyncio
async def test_publication_purge_apply_rejects_before_any_service_operation() -> None:
    store = AsyncMock()
    store.source_visibility_enabled = True
    pool = AsyncMock()
    with pytest.raises(RuntimeError, match="no PostgreSQL or Qdrant writes"):
        await purge_source(store, pool, "source-a", apply=True)
    assert store.mock_calls == []
    assert pool.mock_calls == []
