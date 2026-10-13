"""retrieval: BM25 tokenizer, query mode, visual-intent leaves, fusion seam,
``HybridRetriever`` orchestrator, and framework-neutral execution helper.

The leaf contracts are BM25, code BM25, ``QueryMode``,
``resolve_collections`` and visual-intent. The public
retrieve path is ``HybridRetriever`` + ``SearchResult`` (in
``vo_rag.core.retrieval.search``), the canonical RRF/dedup fusion
module (``vo_rag.core.retrieval.fusion``), and the framework-neutral
``execute_retrieval`` helper (``vo_rag.core.retrieval.execution``).

The BM25 blake2b-4 token-id scheme is the persistence contract (see
``bm25_tokenizer`` docstring) — changing it means re-ingesting every
Qdrant point. The ``vo_rag_*_v1`` collection-name strings come
from the storage schema and are equally load-bearing.

The retriever delegates to ``QdrantStore.search_hybrid`` rather than
reaching past the store into ``qdrant._client.query_points``. License
filter merge + annotation are owned by
``vo_rag.core.compliance.license_filter`` and called by the retriever
in INTERNAL mode.
"""

from __future__ import annotations

from vo_rag.core.retrieval.bm25_tokenizer import (
    tokenize,
    tokens_to_bm25_sparse,
)
from vo_rag.core.retrieval.code_bm25_tokenizer import to_sparse as code_to_sparse
from vo_rag.core.retrieval.code_bm25_tokenizer import tokenize as code_tokenize
from vo_rag.core.retrieval.execution import (
    QueryExecution,
    RerankFn,
    execute_retrieval,
)
from vo_rag.core.retrieval.fusion import (
    RRF_DEFAULT_K,
    cross_variant_key,
    dedupe_key,
    rrf_merge_and_dedupe,
)
from vo_rag.core.retrieval.query_mode import (
    MODE_AGNOSTIC_COLLECTIONS,
    RAW_COLLECTIONS,
    RAW_TO_STERILE,
    STERILE_COLLECTIONS,
    QueryMode,
    resolve_collections,
)
from vo_rag.core.retrieval.search import (
    DEFAULT_VISUAL_MODE,
    HybridRetriever,
    SearchResult,
)
from vo_rag.core.retrieval.visual_intent import (
    TEXT_CHUNK_TYPES,
    VISUAL_CHUNK_TYPES,
    VISUAL_INTENT_PHRASES,
    VISUAL_INTENT_TOKENS,
    VISUAL_MODES,
    has_visual_intent,
    merge_chunk_type_filter,
    quotas_for_mode,
)

__all__ = [
    # bm25 tokenizer
    "tokenize",
    "tokens_to_bm25_sparse",
    # code bm25 tokenizer
    "code_tokenize",
    "code_to_sparse",
    # query mode
    "QueryMode",
    "RAW_COLLECTIONS",
    "STERILE_COLLECTIONS",
    "RAW_TO_STERILE",
    "MODE_AGNOSTIC_COLLECTIONS",
    "resolve_collections",
    # visual intent
    "VISUAL_INTENT_PHRASES",
    "VISUAL_INTENT_TOKENS",
    "VISUAL_CHUNK_TYPES",
    "TEXT_CHUNK_TYPES",
    "VISUAL_MODES",
    "has_visual_intent",
    "quotas_for_mode",
    "merge_chunk_type_filter",
    # fusion (RRF only)
    "RRF_DEFAULT_K",
    "dedupe_key",
    "cross_variant_key",
    "rrf_merge_and_dedupe",
    # search (HybridRetriever)
    "DEFAULT_VISUAL_MODE",
    "HybridRetriever",
    "SearchResult",
    # execution (framework-neutral helper)
    "QueryExecution",
    "RerankFn",
    "execute_retrieval",
]
