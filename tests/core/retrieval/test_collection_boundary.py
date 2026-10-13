"""Selected corpora survive degradation, injected providers and optional expansion."""

from __future__ import annotations

from typing import Any, cast

import pytest

from vo_rag.core.retrieval.execution import execute_retrieval
from vo_rag.core.retrieval.query_mode import QueryMode, resolve_collections
from vo_rag.core.retrieval.search import SearchResult

BRAIN = "vo_rag_brain_v1"
LIBRARY = "vo_rag_books_raw_v1"


def _hit(point: str, collection: str) -> SearchResult:
    return SearchResult(
        id=point, score=1.0, collection=collection, payload={"content": point, "source_id": point}
    )


class Retriever:
    def __init__(self, hits: list[SearchResult], error: Exception | None = None) -> None:
        self.hits = hits
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def cross_collection_search(self, **kwargs: Any) -> list[SearchResult]:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.hits


@pytest.mark.asyncio
async def test_rogue_retrieval_and_rerank_results_never_enter_selected_evidence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    retriever = Retriever(
        [_hit("foreign", LIBRARY), _hit("missing", ""), _hit("legitimate", BRAIN)]
    )
    seen = []

    def rerank(query: str, hits: list[SearchResult]) -> list[SearchResult]:
        seen.extend(hits)
        return [_hit("injected", LIBRARY), *hits]

    result = await execute_retrieval(
        retriever=cast(Any, retriever),
        query="retention?",
        requested_collections=[BRAIN],
        rerank_fn=rerank,
        top_k=8,
    )
    assert [hit.id for hit in seen] == ["legitimate"]
    assert [hit.id for hit in result] == ["legitimate"]
    assert retriever.calls[0]["collections"] == [BRAIN]
    assert retriever.calls[0]["top_k"] >= 8
    assert "retrieval_collection_boundary_rejected" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, TimeoutError("fixture timeout")])
async def test_empty_or_outage_never_retries_other_corpora(failure: Exception | None) -> None:
    retriever = Retriever([], failure)
    if failure:
        with pytest.raises(TimeoutError):
            await execute_retrieval(
                retriever=cast(Any, retriever), query="retention?", requested_collections=[BRAIN]
            )
    else:
        assert (
            await execute_retrieval(
                retriever=cast(Any, retriever), query="retention?", requested_collections=[BRAIN]
            )
            == []
        )
    assert len(retriever.calls) == 1 and retriever.calls[0]["collections"] == [BRAIN]


@pytest.mark.asyncio
async def test_explicit_empty_selection_refused_before_retrieval() -> None:
    retriever = Retriever([])
    with pytest.raises(ValueError, match="nonempty"):
        await execute_retrieval(
            retriever=cast(Any, retriever), query="retention?", requested_collections=[]
        )
    assert retriever.calls == []


@pytest.mark.asyncio
async def test_brain_unscoped_parent_fetch_is_not_invoked() -> None:
    child = _hit("child", BRAIN)
    child.payload.update(parent_id="library-parent", parent_child_count=1)
    retriever = Retriever([child])
    calls = []

    def fetch(point: str) -> SearchResult:
        calls.append(point)
        return _hit(point, LIBRARY)

    result = await execute_retrieval(
        retriever=cast(Any, retriever),
        query="retention?",
        requested_collections=[BRAIN],
        auto_merge=True,
        fetch_parent_fn=fetch,
    )
    assert result == [child] and calls == []


@pytest.mark.asyncio
async def test_intentional_library_and_combined_modes_preserve_selection() -> None:
    for mode, requested, effective in [
        (QueryMode.INTERNAL, [LIBRARY], [LIBRARY]),
        (QueryMode.PUBLICATION, [LIBRARY], ["vo_rag_books_sterile_v1"]),
        (QueryMode.INTERNAL, [BRAIN, LIBRARY], [BRAIN, LIBRARY]),
    ]:
        retriever = Retriever([_hit("allowed", effective[0])])
        result = await execute_retrieval(
            retriever=cast(Any, retriever),
            query="retention?",
            requested_collections=requested,
            mode=mode,
        )
        assert retriever.calls[0]["collections"] == effective
        assert [h.collection for h in result] == [effective[0]]
    assert resolve_collections(QueryMode.INTERNAL, None)[0] == LIBRARY
    assert BRAIN not in resolve_collections(QueryMode.INTERNAL, None)


@pytest.mark.asyncio
async def test_no_rerank_normal_depth_and_causal_same_collection_backfill() -> None:
    class BoundedRetriever(Retriever):
        async def cross_collection_search(self, **kwargs: Any) -> list[SearchResult]:
            self.calls.append(kwargs)
            return self.hits[: kwargs["top_k"]]

    healthy = BoundedRetriever([_hit("first", BRAIN), _hit("second", BRAIN)])
    result = await execute_retrieval(
        retriever=cast(Any, healthy), query="retention?", requested_collections=[BRAIN], top_k=2
    )
    assert len(result) == 2 and [c["top_k"] for c in healthy.calls] == [2]
    degraded = BoundedRetriever(
        [_hit("foreign", LIBRARY), _hit("first", BRAIN), _hit("second", BRAIN)]
    )
    result = await execute_retrieval(
        retriever=cast(Any, degraded), query="retention?", requested_collections=[BRAIN], top_k=2
    )
    assert [h.id for h in result] == ["first", "second"]
    assert [c["top_k"] for c in degraded.calls] == [2, 100]
    assert all(c["collections"] == [BRAIN] for c in degraded.calls)
