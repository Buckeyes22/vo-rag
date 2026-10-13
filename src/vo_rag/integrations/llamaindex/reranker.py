"""VoRagReranker — the vo-rag reranker behind a LlamaIndex BaseNodePostprocessor.

Wraps a backend from ``build_reranker(settings)``. Backends must be SYNC
(GteReranker, GpuReranker); ASYNC backends (RerankerClient) are rejected
at construction time because the sync ``_postprocess_nodes`` contract
cannot bridge into a running event loop. Async hosts should wire the
reranker as a ``rerank_fn`` on ``VoRagRetriever`` instead, where it is
awaited natively in _aretrieve.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Protocol, cast

from vo_rag.core.config.settings import RerankerSettings
from vo_rag.core.reranking.factory import build_reranker
from vo_rag.core.retrieval.search import SearchResult
from vo_rag.integrations.llamaindex import import_llamaindex_symbol
from vo_rag.integrations.llamaindex.convert import (
    node_with_score_to_search_result,
    search_result_to_node_with_score,
)

if TYPE_CHECKING:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
    from pydantic import PrivateAttr
else:
    BaseNodePostprocessor = import_llamaindex_symbol(
        "llama_index.core.postprocessor.types", "BaseNodePostprocessor"
    )
    PrivateAttr = import_llamaindex_symbol("pydantic", "PrivateAttr")


_RERANKER_ASYNC_ERROR = (
    "VoRagReranker cannot wrap an async reranker backend as a sync node postprocessor; "
    "pass rerank_fn to VoRagRetriever instead"
)


class SyncRerankBackend(Protocol):
    """Synchronous backend shape accepted by ``VoRagReranker``."""

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        *,
        top_k: int,
    ) -> list[SearchResult]: ...


def _backend_is_async(reranker: Any) -> bool:
    """Return True if *reranker* exposes an async rerank seam."""
    rerank = getattr(reranker, "rerank", None)
    if rerank is not None and inspect.iscoroutinefunction(rerank):
        return True
    arerank = getattr(reranker, "arerank", None)
    return (
        arerank is not None
        and inspect.iscoroutinefunction(arerank)
        and (rerank is None or not callable(rerank))
    )


def _ensure_sync_backend(reranker: Any) -> SyncRerankBackend:
    if _backend_is_async(reranker):
        raise TypeError(_RERANKER_ASYNC_ERROR)
    if not callable(getattr(reranker, "rerank", None)):
        raise TypeError("VoRagReranker requires a sync reranker backend with rerank(...).")
    return cast(SyncRerankBackend, reranker)


class VoRagReranker(BaseNodePostprocessor):
    """Expose a vo-rag reranker backend as a LlamaIndex node postprocessor."""

    top_k: int = 5

    # BaseNodePostprocessor is a pydantic model; keep the non-serializable backend
    # off the schema as a private attr.
    _reranker: SyncRerankBackend = PrivateAttr()

    def __init__(
        self,
        *,
        settings: RerankerSettings | None = None,
        reranker: SyncRerankBackend | None = None,
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        if reranker is None:
            if settings is None:
                raise ValueError("VoRagReranker requires either `reranker` or `settings`.")
            reranker = build_reranker(settings)
        self._reranker = _ensure_sync_backend(reranker)

    @classmethod
    def class_name(cls) -> str:
        return "VoRagReranker"

    @property
    def backend_is_sync(self) -> bool:
        return not _backend_is_async(self._reranker) and callable(
            getattr(self._reranker, "rerank", None)
        )

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return []
        query = query_bundle.query_str if query_bundle is not None else ""
        candidates = [node_with_score_to_search_result(n) for n in nodes]

        reranked = _run_rerank_sync(self._reranker, query, candidates, self.top_k)

        return [search_result_to_node_with_score(r) for r in reranked]


def _run_rerank_sync(
    reranker: SyncRerankBackend,
    query: str,
    candidates: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Run a sync backend directly."""
    return reranker.rerank(query, candidates, top_k=top_k)
