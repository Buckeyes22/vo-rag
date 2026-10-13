"""Framework-neutral ``execute_retrieval`` helper for the public API and ops.

The surface is an importable ``execute_retrieval(query, ...)`` callable plus a
``QueryExecution`` dataclass for callers that want a longer-lived
configuration object.

The flow:

  1. resolve collections via ``QueryMode`` + ``resolve_collections``
     (raw books in INTERNAL mode, sterile books in PUBLICATION /
     PARTNER).
  2. call ``HybridRetriever.cross_collection_search`` (which itself
     delegates to ``QdrantStore.search_hybrid``).
  3. optionally call a ``rerank_fn`` callable (the ``GteReranker``
     is an injected dependency so this module stays framework-neutral).
  4. truncate to ``top_k`` and return the list of ``SearchResult``.

No FastAPI, no auth, no DB. The helper is plumbed the same way in tests,
in scripts, and in the API surface.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vo_rag.core.chunking.models import brain_evidence_is_complete, brain_evidence_issue
from vo_rag.core.retrieval.auto_merge import FetchParentFn
from vo_rag.core.retrieval.auto_merge import auto_merge as _auto_merge
from vo_rag.core.retrieval.brain_redundancy import select_ranked_evidence
from vo_rag.core.retrieval.fusion import dedupe_exact_evidence
from vo_rag.core.retrieval.query_mode import QueryMode, resolve_collections
from vo_rag.core.retrieval.search import HybridRetriever, SearchResult
from vo_rag.core.storage.publication_read_view import publication_query

if TYPE_CHECKING:
    pass

#: Callable shape for an optional rerank hook. The in-process ``GteReranker`` /
#: ``GpuReranker`` implement the SYNC form; the remote ``RerankerClient.rerank``
#: is ASYNC — both are supported because ``execute_retrieval`` awaits the result
#: when it is awaitable. The test suite uses sync inline callables. Returning a
#: list shorter than the input is allowed — the helper respects the returned
#: length up to ``top_k``.
RerankFn = Callable[
    [str, list[SearchResult]],
    "list[SearchResult] | Awaitable[list[SearchResult]]",
]
PhaseContextFactory = Callable[[str], AbstractContextManager[Any]]


@dataclass
class QueryExecution:
    """Long-lived configuration object for an execute_retrieval call.

    Holds the ``HybridRetriever`` plus an optional ``rerank_fn``. Further
    fields (latency stages, per-stage timing, eval trace envelope) can be
    added without breaking the current narrow shape.
    """

    retriever: HybridRetriever
    rerank_fn: RerankFn | None = None
    # Reserved for eval harness integration; kept so the dataclass shape
    # is stable.
    metadata: dict[str, Any] = field(default_factory=dict)


@publication_query
async def execute_retrieval(
    *,
    retriever: HybridRetriever,
    query: str,
    mode: QueryMode = QueryMode.INTERNAL,
    requested_collections: list[str] | None = None,
    top_k: int = 20,
    filters: dict[str, Any] | None = None,
    visual_mode: str | None = None,
    rerank_fn: RerankFn | None = None,
    phase_context: PhaseContextFactory | None = None,
    auto_merge: bool = False,
    auto_merge_threshold: float = 0.5,
    fetch_parent_fn: FetchParentFn | None = None,
) -> list[SearchResult]:
    """Resolve collections → hybrid search → optional rerank → top-k.

    The helper is the single entry point used by the API surface, ops
    scripts, and tests. The visual-mode dispatcher, the license filter
    merge, the BM25 token-id persistence contract, and the two-pass RRF
    dedup are all owned by ``HybridRetriever`` (and its delegates).

    Returns at most ``top_k`` ``SearchResult`` objects, in the order
    produced by the (optional) rerank or — when ``rerank_fn`` is
    ``None`` — by the canonical RRF.
    """
    if len(query) > 4000:
        raise ValueError("query exceeds 4000-character retrieval budget")
    collections = resolve_collections(mode, requested_collections)

    # Candidate breadth is independent of caller output size and capped.
    # The cross-encoder cannot recover a passage discarded before reranking.
    pool_size = max(top_k, min(200, max(40, top_k * 4)))
    candidate_depth = min(400, max(100, pool_size * 2)) if rerank_fn else top_k
    with _phase_context(phase_context, "retrieval"):
        hits = await retriever.cross_collection_search(
            query=query,
            collections=collections,
            top_k=candidate_depth,
            filters=filters,
            mode=mode,
            visual_mode=visual_mode,
        )

    recalled_count = len(hits)
    hits = dedupe_exact_evidence(_selected_hits(hits, collections))
    if (
        rerank_fn is None
        and "vo_rag_brain_v1" in collections
        and len(hits) < top_k
        and recalled_count >= candidate_depth
    ):
        # Only degraded/duplicate packets need extra recall. Normal no-rerank
        # requests retain their original depth and compute cost.
        with _phase_context(phase_context, "retrieval_backfill"):
            hits = await retriever.cross_collection_search(
                query=query,
                collections=collections,
                top_k=min(400, max(100, top_k * 2)),
                filters=filters,
                mode=mode,
                visual_mode=visual_mode,
            )
        hits = dedupe_exact_evidence(_selected_hits(hits, collections))

    if auto_merge and fetch_parent_fn is not None and "vo_rag_brain_v1" not in collections:
        with _phase_context(phase_context, "auto_merge"):
            hits = _auto_merge(hits, fetch_parent_fn, threshold=auto_merge_threshold)

    if auto_merge and fetch_parent_fn is not None and "vo_rag_brain_v1" in collections:
        logging.getLogger(__name__).warning("brain_unscoped_parent_expansion_withheld")
    hits = _selected_hits(hits, collections)
    hits = dedupe_exact_evidence(hits)
    hook = rerank_fn
    if hook is not None:
        with _phase_context(phase_context, "rerank"):
            # A long source must not occupy the entire bounded rerank pool.
            # Round-robin sources while keeping each source's retrieval order.
            groups: dict[str, list[SearchResult]] = {}
            for hit in hits:
                source = str(hit.payload.get("source_id") or hit.id)
                groups.setdefault(source, []).append(hit)
            # Keep a relevance-ranked core, then use a source-balanced tail.
            # This preserves multiple strong passages from one source without
            # letting a long page consume every cross-encoder slot.
            diverse = hits[: pool_size // 2]
            core_ids = {hit.id for hit in diverse}
            groups = {
                source: [hit for hit in rows if hit.id not in core_ids]
                for source, rows in groups.items()
            }
            groups = {source: rows for source, rows in groups.items() if rows}
            while groups and len(diverse) < pool_size:
                for source in list(groups):
                    if len(diverse) == pool_size:
                        break
                    diverse.append(groups[source].pop(0))
                    if not groups[source]:
                        del groups[source]
            # Standard rerankers expose an optional output bound. Request the
            # whole scored pool before final truncation; custom two-argument
            # hooks retain their established contract.
            try:
                top_parameter = inspect.signature(hook).parameters.get("top_k")
            except (TypeError, ValueError):
                top_parameter = None
            if top_parameter is not None:
                bounded_hook: Any = hook
                if top_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                    result = bounded_hook(query, diverse, pool_size)
                else:
                    result = bounded_hook(query, diverse, top_k=pool_size)
            else:
                result = hook(query, diverse)
            # Support both the sync in-process reranker and the async RerankerClient.
            hits = await result if inspect.isawaitable(result) else result

    return select_ranked_evidence(dedupe_exact_evidence(_selected_hits(hits, collections)), top_k)


def _selected_hits(hits: list[SearchResult], collections: list[str]) -> list[SearchResult]:
    """Do not turn rogue/degraded provider results into out-of-scope evidence."""
    allowed = set(collections)
    for hit in hits:
        if hit.collection == "vo_rag_brain_v1" and (
            "governing_context_version" in hit.payload
            or "governing_context_complete" in hit.payload
        ):
            issue = brain_evidence_issue(hit.payload, hit.collection)
            if issue:
                logging.getLogger(__name__).warning(
                    "brain_governing_evidence_withheld",
                    extra={"reason": issue, "collection": hit.collection},
                )
    selected = [
        hit
        for hit in hits
        if hit.collection in allowed
        and not (
            hit.collection == "vo_rag_brain_v1"
            and (
                "governing_context_version" in hit.payload
                or "governing_context_complete" in hit.payload
            )
            and not brain_evidence_is_complete(hit.payload, hit.collection)
        )
    ]
    if len(selected) != len(hits):
        logging.getLogger(__name__).warning(
            "retrieval_collection_boundary_rejected",
            extra={
                "rejected_count": len(hits) - len(selected),
                "selected_collections": collections,
            },
        )
    return selected


def _phase_context(
    factory: PhaseContextFactory | None,
    phase_name: str,
) -> AbstractContextManager[Any]:
    if factory is None:
        return nullcontext()
    return factory(phase_name)


__all__ = [
    "QueryExecution",
    "PhaseContextFactory",
    "RerankFn",
    "execute_retrieval",
]
