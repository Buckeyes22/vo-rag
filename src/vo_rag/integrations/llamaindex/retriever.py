"""VoRagRetriever — the production retrieve path behind a LlamaIndex BaseRetriever.

SECURITY: query mode (raw vs sterile) and the license filter are enforced HERE, by
the retriever, not by the caller. A query engine wired to the wrong mode would
otherwise leak unredacted book text on a PUBLICATION path. ``mode`` is bound at
construction and defaults to the SAFE (PUBLICATION/sterile) collection set.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vo_rag.core.retrieval.execution import RerankFn, execute_retrieval
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.integrations.llamaindex import import_llamaindex_symbol
from vo_rag.integrations.llamaindex._async import run_coro_sync
from vo_rag.integrations.llamaindex.convert import (
    ChunkIdMissingError,
    search_result_to_node_with_score,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # heavy import kept off the default runtime path
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle

    from vo_rag.core.retrieval.search import HybridRetriever
else:
    BaseRetriever = import_llamaindex_symbol("llama_index.core.retrievers", "BaseRetriever")


class VoRagRetriever(BaseRetriever):
    """Wrap ``execute_retrieval`` as a LlamaIndex ``BaseRetriever``.

    The hybrid/RRF/visual-intent/license stack is hidden behind this one class.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        mode: QueryMode = QueryMode.PUBLICATION,  # SAFE default — sterile books
        top_k: int = 20,
        requested_collections: list[str] | None = None,
        filters: dict[str, object] | None = None,
        visual_mode: str | None = None,
        rerank_fn: RerankFn | None = None,
        allow_internal: bool = False,
        callback_manager: Any | None = None,
    ) -> None:
        # Refuse the unsafe (raw) mode unless the caller explicitly opts in.
        if mode == QueryMode.INTERNAL and not allow_internal:
            self._raise_unsafe_internal_mode()
        self._retriever = retriever
        self._mode = mode
        self._top_k = top_k
        self._requested_collections = requested_collections
        self._filters = filters
        self._visual_mode = visual_mode
        self._rerank_fn = rerank_fn
        self._allow_internal = allow_internal
        super().__init__(callback_manager=callback_manager)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if self._mode == QueryMode.INTERNAL and not self._allow_internal:
            self._raise_unsafe_internal_mode()
        results = await execute_retrieval(
            retriever=self._retriever,
            query=query_bundle.query_str,
            mode=self._mode,  # enforced HERE, not by the caller
            requested_collections=self._requested_collections,
            top_k=self._top_k,
            filters=self._filters,
            visual_mode=self._visual_mode,
            rerank_fn=self._rerank_fn,
        )
        # Fail-closed but resilient: a hit missing payload["chunk_id"] raises
        # ChunkIdMissingError (otherwise node.id_ would silently key on the index-
        # addressed Qdrant point id — the content-vs-index id mismatch). Quarantine that one hit
        # (warn + drop); never let a single corrupt point fail the whole query.
        nodes: list[NodeWithScore] = []
        quarantined_hits = 0
        for r in results:
            try:
                nodes.append(search_result_to_node_with_score(r))
            except ChunkIdMissingError as exc:
                quarantined_hits += 1
                logger.warning("dropping retrieval hit (quarantined): %s", exc)
        if results and quarantined_hits == len(results):
            logger.error(
                "all retrieval hits quarantined: %s/%s hits missing payload['chunk_id']",
                quarantined_hits,
                len(results),
            )
        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return run_coro_sync(
            lambda: self._aretrieve(query_bundle),
            component="VoRagRetriever._retrieve",
        )

    @staticmethod
    def _raise_unsafe_internal_mode() -> None:
        raise ValueError(
            "VoRagRetriever(mode=INTERNAL) returns UNREDACTED raw book text. "
            "Pass allow_internal=True to confirm this is an internal-only path."
        )
