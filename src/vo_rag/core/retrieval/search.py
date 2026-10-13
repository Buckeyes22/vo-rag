"""``HybridRetriever`` + ``SearchResult`` — the public retrieve path.

``HybridRetriever`` does not reach past the store into
``self.qdrant._client.query_points``. It delegates the
4-vector Prefetch call to ``QdrantStore.search_hybrid`` (and per-
collection ``search_hybrid`` + canonical RRF for cross-collection
fan-out).

The two-pass RRF dedup lives in ``vo_rag.core.retrieval.fusion``.
The license filter / annotation seam lives in
``vo_rag.core.compliance.license_filter``. The BM25 token-id scheme
lives in ``vo_rag.core.retrieval.bm25_tokenizer`` — query and write
paths share the blake2b-4 contract.

Visual-mode dispatch:

  * ``intent_aware`` (default): auto-detect visual intent in the
    query and pick 70/30 visual/text or 30/70 quotas accordingly.
  * ``baseline``: original unfiltered hybrid path. No quotas.
  * ``visual_only``: restrict results to visual chunk_types.
  * ``visual_track``: fixed 60/40 text/visual quota.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client.models import (
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    PayloadField,
    Range,
    ScoredPoint,
)

from vo_rag.core.chunking.models import hit_payload
from vo_rag.core.compliance.license_filter import (
    annotate_unlicensed,
    license_filter_for_mode,
    merge_with_user_filter,
)
from vo_rag.core.dsi.routing import RoutingDecision
from vo_rag.core.retrieval.bm25_tokenizer import tokenize, tokens_to_bm25_sparse
from vo_rag.core.retrieval.fusion import rrf_merge_and_dedupe
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.visual_intent import (
    TEXT_CHUNK_TYPES,
    VISUAL_CHUNK_TYPES,
    VISUAL_MODES,
    merge_chunk_type_filter,
    quotas_for_mode,
)

DEFAULT_VISUAL_MODE = "intent_aware"

#: The bound names a caller may use in a filter dict value. Anything else is a
#: typo or a different dialect (e.g. ``between``) and is rejected rather than
#: handed to Qdrant as a silently-empty ``Range`` (VOR-1443).
_RANGE_BOUNDS: frozenset[str] = frozenset({"gte", "gt", "lte", "lt"})


def _with_generation_filter(
    collection: str,
    query_filter: Filter | None,
    *,
    ingest_state_override: str | None = None,
) -> Filter | None:
    del collection  # every collection is legacy-compatible and generation-safe
    if ingest_state_override is not None:
        state_filter = Filter(
            must=[
                FieldCondition(
                    key="ingest_state",
                    match=MatchValue(value=ingest_state_override),
                )
            ]
        )
    else:
        # Missing state is the legacy-active representation during rollout.
        # Every generated point carries an explicit state, so this branch never
        # admits a staged, failed, quarantined, or superseded generation.
        state_filter = Filter(
            should=[
                FieldCondition(key="ingest_state", match=MatchValue(value="active")),
                IsEmptyCondition(is_empty=PayloadField(key="ingest_state")),
            ]
        )
    return merge_with_user_filter(query_filter, state_filter)


@dataclass
class SearchResult:
    """The public retrieve-path return type. Intentionally narrow.

    ``id`` is the Qdrant point id (string form). ``score`` is the merged
    RRF score after the retriever's two-pass dedup. ``collection`` is
    the collection the hit was retrieved from. ``payload`` is the
    unparsed Qdrant payload (license annotation is appended in INTERNAL
    mode by ``annotate_unlicensed``).
    """

    id: str
    score: float
    collection: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VectorBundle:
    """Query-scoped vectors shared by every collection and visual track.

    A public query is embedded once, then the immutable bundle is reused for
    each Qdrant request.  Keeping BM25 in the same value object prevents its
    token-id contract from drifting from the dense/sparse/ColBERT query.
    """

    dense: list[float]
    sparse: dict[int, float]
    bm25: dict[int, float]
    colbert: list[list[float]]


class _QdrantStoreProtocol(Protocol):
    """Structural type for the QdrantStore seam we delegate to.

    Defined inline (instead of importing ``QdrantStore``) so callers can
    inject a fake in tests without paying the qdrant-client import
    cost. Production code passes the real ``QdrantStore`` instance.

    The ``filters`` parameter accepts the Qdrant ``Filter`` directly
    (not the raw user dict) so the license filter merge can produce a
    proper ``Filter`` before the retriever hands it to the store.
    """

    async def search_hybrid(
        self,
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        *,
        top_k: int = 20,
        filters: Filter | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]: ...


def _is_numeric_bound(value: Any) -> bool:
    """True for a real, finite number usable as a Qdrant range bound.

    ``bool`` is excluded even though it subclasses ``int``: ``Range`` rejects
    it, and ``True`` silently reading as ``1`` is never the caller's intent.
    ``nan``/``inf``/``-inf`` are excluded too: Qdrant accepts them on the wire
    but they never bound anything a caller could have meant (VOR-1443 r3).
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _range_condition(key: str, bounds: Mapping[str, Any]) -> FieldCondition:
    """Build one numeric ``Range`` clause from a filter's mapping value.

    Every entry must be a recognized bound name holding a real, finite
    number. An unknown name, an empty mapping, a ``None``, a non-numeric
    value, or a non-finite float would otherwise reach Qdrant as a ``Range``
    missing (or not really bounding) that bound — a clause that reads like a
    constraint while matching every point — so all raise the same
    ``ValueError`` (VOR-1443).

    The unknown-name message is built from ``str(name)`` for every key, not
    the raw key: ``bounds`` keys come straight off a filter dict, and
    ``sorted()`` cannot order a ``str`` against a non-``str`` key. Without the
    coercion, a non-``str`` key would raise ``TypeError`` while building the
    error message instead of the intended ``ValueError`` (VOR-1443 r3).
    """
    unknown = sorted(str(name) for name in set(bounds) - _RANGE_BOUNDS)
    non_numeric = sorted(
        str(name)
        for name, value in bounds.items()
        if name in _RANGE_BOUNDS and not _is_numeric_bound(value)
    )
    bad = unknown or non_numeric
    if not bounds or bad:
        raise ValueError(
            f"filter {key}: unsupported bound(s) {bad or ['<none>']}; use numeric gte/gt/lte/lt"
        )
    return FieldCondition(key=key, range=Range(**bounds))


def _build_filter(filters: dict[str, Any] | None) -> Filter | None:
    """Translate the user-provided filter dict into a Qdrant ``Filter``.

    List values become ``MatchAny``; scalar values become ``MatchValue``; a
    mapping of ``gte``/``gt``/``lte``/``lt`` becomes a ``Range`` (VOR-1443,
    e.g. ``{"tf_pub_year": {"gte": 1990}}``).
    Empty / ``None`` inputs return ``None`` (no filter). A mapping value that
    is not a usable range — unknown or non-numeric bound, or ``{}`` — raises
    ``ValueError``, because a ``Range`` with no bound matches every point.

    The ``must`` list is typed as ``list[Any]`` to match the
    qdrant-client ``Filter`` model's broad condition union — narrowing
    to ``list[FieldCondition]`` causes mypy variance errors against
    the union it expects.
    """
    if not filters:
        return None

    must: list[Any] = []
    for key, value in filters.items():
        if isinstance(value, Mapping):
            must.append(_range_condition(key, value))
        elif isinstance(value, list):
            must.append(FieldCondition(key=key, match=MatchAny(any=list(value))))
        else:
            must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=must)


def validate_filters(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return ``filters`` unchanged, or raise ``ValueError`` if it is malformed.

    The API boundary calls this so a clause the retriever cannot translate
    (``{"tf_pub_year": {"between": [1, 2]}}``, a null or non-numeric bound) is
    a 422 at the edge instead of a 500 raised from inside a query, while the
    accepted dialect stays defined in exactly one place (VOR-1443).
    """
    _build_filter(filters)
    return filters


class HybridRetriever:
    """4-vector hybrid retriever over the canonical ``QdrantStore`` seam.

    Responsibilities:

      1. Embed the query once (dense / sparse / ColBERT).
      2. Produce the BM25 sparse vector from the shared tokenizer
         (persistence contract: same ids as every Qdrant writer).
      3. Build a license-mode filter and merge it with the user's
         filters in the security-sensitive order documented in
         ``core.compliance.license_filter.merge_with_user_filter``.
      4. Delegate the 4-vector Prefetch to ``QdrantStore.search_hybrid``
         (the retriever must NOT reach past the store into
         ``qdrant._client.query_points``).
      5. Wrap ``ScoredPoint`` → ``SearchResult`` and apply license
         annotation in INTERNAL mode.

    The retriever does NOT hold a Qdrant client reference of its own;
    every store interaction goes through the ``QdrantStore`` API.
    """

    def __init__(self, qdrant: _QdrantStoreProtocol, embedding_client: Any) -> None:
        self.qdrant = qdrant
        self.embed = embedding_client

    async def prepare_query(self, query: str) -> VectorBundle:
        """Embed *query* exactly once and return its complete vector bundle."""
        emb = await self.embed.embed(
            [query], return_dense=True, return_sparse=True, return_colbert=True
        )
        return VectorBundle(
            dense=emb["dense"][0],
            sparse=dict(zip(emb["sparse"][0]["indices"], emb["sparse"][0]["values"], strict=True)),
            bm25=tokens_to_bm25_sparse(tokenize(query)),
            colbert=emb["colbert"][0],
        )

    async def hybrid_search(
        self,
        query: str,
        collection: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        prefetch_depth: int = 100,
        mode: QueryMode = QueryMode.INTERNAL,
        vector_bundle: VectorBundle | None = None,
        ingest_state_override: str | None = None,
    ) -> list[SearchResult]:
        bundle = vector_bundle or await self.prepare_query(query)

        mode_filter = license_filter_for_mode(mode)
        query_filter = merge_with_user_filter(mode_filter, _build_filter(filters))
        query_filter = _with_generation_filter(
            collection,
            query_filter,
            ingest_state_override=ingest_state_override,
        )

        # DELEGATE to the store. The retriever
        # must NOT call self.qdrant._client.query_points directly.
        scored_points = await self.qdrant.search_hybrid(
            collection=collection,
            query_dense=bundle.dense,
            query_sparse=bundle.sparse,
            query_bm25=bundle.bm25,
            query_colbert=bundle.colbert,
            top_k=top_k,
            filters=query_filter,
            prefetch_depth=max(prefetch_depth, top_k),
        )

        results = [
            SearchResult(
                id=str(point.id),
                score=point.score,
                collection=collection,
                payload=hit_payload(point.payload),
            )
            for point in scored_points
        ]
        return annotate_unlicensed(results, mode)

    async def cross_collection_search(
        self,
        query: str,
        collections: list[str],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        mode: QueryMode = QueryMode.INTERNAL,
        visual_mode: str | None = None,
    ) -> list[SearchResult]:
        """Hybrid search across collections, with optional visual-aware dispatch.

        ``visual_mode`` controls how visual chunks (figure_composite, figure,
        table) are treated relative to text chunks (text, body). Visual chunks
        are 5-40× outnumbered by text chunks per book, so without a quota
        they get crowded out of top-K on concept queries. Modes:

          * ``intent_aware`` (default): auto-detect visual intent in the
            query and pick 70/30 visual/text or 30/70 quotas accordingly.
          * ``baseline``: original unfiltered hybrid path. No quotas.
          * ``visual_only``: restrict results to visual chunk_types.
          * ``visual_track``: fixed 60/40 text/visual quota.
        """
        chosen_mode = visual_mode or DEFAULT_VISUAL_MODE
        if chosen_mode not in VISUAL_MODES:
            raise ValueError(f"visual_mode must be one of {VISUAL_MODES}; got {chosen_mode!r}")

        # Query vectors are request-scoped: all collections and both visual
        # tracks receive the same immutable bundle.
        vector_bundle = await self.prepare_query(query)

        if chosen_mode == "baseline":
            return await self._cross_collection_baseline(
                query,
                collections,
                top_k,
                filters,
                mode,
                vector_bundle,
            )
        if chosen_mode == "visual_only":
            visual_filter = merge_chunk_type_filter(filters, VISUAL_CHUNK_TYPES)
            return await self._cross_collection_baseline(
                query,
                collections,
                top_k,
                visual_filter,
                mode,
                vector_bundle,
            )

        # visual_track / intent_aware: parallel filtered searches with quotas
        if top_k < 2:
            return await self._cross_collection_baseline(
                query,
                collections,
                top_k,
                filters,
                mode,
                vector_bundle,
            )
        text_quota, visual_quota = quotas_for_mode(chosen_mode, query, top_k)
        text_filter = merge_chunk_type_filter(filters, TEXT_CHUNK_TYPES)
        visual_filter = merge_chunk_type_filter(filters, VISUAL_CHUNK_TYPES)
        text_results, visual_results = await asyncio.gather(
            self._cross_collection_baseline(
                query,
                collections,
                top_k,
                text_filter,
                mode,
                vector_bundle,
            ),
            self._cross_collection_baseline(
                query,
                collections,
                top_k,
                visual_filter,
                mode,
                vector_bundle,
            ),
        )
        # Visual chunks first so they're not buried at the bottom of the UI
        selected = list(visual_results[:visual_quota]) + list(text_results[:text_quota])
        # Quotas express preference, not permission to discard available hits.
        # Fetch a full pool per track so an empty/short track can be backfilled.
        seen = {result.id for result in selected}
        remaining = sorted(
            visual_results[visual_quota:] + text_results[text_quota:],
            key=lambda result: result.score,
            reverse=True,
        )
        for result in remaining:
            if result.id not in seen and len(selected) < top_k:
                selected.append(result)
                seen.add(result.id)
        return annotate_unlicensed(selected, mode)

    async def dsi_routed_search(
        self,
        query: str,
        *,
        routing: RoutingDecision,
        raw_collection: str,
        filters: dict[str, Any] | None = None,
        mode: QueryMode = QueryMode.INTERNAL,
        prefetch_depth: int = 100,
    ) -> list[SearchResult]:
        """Retrieve chunks from *raw_collection* honouring a per-book budget.

        This is the additive DSI-routed path.  It embeds the query once
        and then issues one ``search_hybrid`` call per ``BookRoute`` with
        ``total_budget > 0``, filtering by ``source_id = book_id``.
        Results are concatenated in route order and license-annotated.
        """
        vector_bundle = await self.prepare_query(query)

        mode_filter = license_filter_for_mode(mode)

        all_results: list[SearchResult] = []

        for route in routing.book_routes:
            if route.total_budget <= 0:
                continue

            per_book_filters: dict[str, Any] = {**filters} if filters else {}
            per_book_filters["source_id"] = route.book_id
            query_filter = merge_with_user_filter(mode_filter, _build_filter(per_book_filters))
            query_filter = _with_generation_filter(raw_collection, query_filter)

            scored_points = await self.qdrant.search_hybrid(
                collection=raw_collection,
                query_dense=vector_bundle.dense,
                query_sparse=vector_bundle.sparse,
                query_bm25=vector_bundle.bm25,
                query_colbert=vector_bundle.colbert,
                top_k=route.total_budget,
                filters=query_filter,
                prefetch_depth=prefetch_depth,
            )

            for point in scored_points:
                all_results.append(
                    SearchResult(
                        id=str(point.id),
                        score=point.score,
                        collection=raw_collection,
                        payload=hit_payload(point.payload),
                    )
                )

        return annotate_unlicensed(all_results, mode)

    async def _cross_collection_baseline(
        self,
        query: str,
        collections: list[str],
        top_k: int,
        filters: dict[str, Any] | None,
        mode: QueryMode,
        vector_bundle: VectorBundle,
    ) -> list[SearchResult]:
        """Unfiltered fan-out + RRF merge — extracted so the visual-mode
        dispatcher can call it twice (one per chunk_type filter).

        Public and store cross-collection paths share passage identity.
        The public path additionally collapses logical content variants.
        """
        per_collection = await asyncio.gather(
            *[
                self.hybrid_search(
                    query,
                    collection,
                    top_k=top_k,
                    filters=filters,
                    mode=mode,
                    vector_bundle=vector_bundle,
                )
                for collection in collections
            ]
        )
        results = rrf_merge_and_dedupe(per_collection, top_k=top_k)
        return annotate_unlicensed(results, mode)


__all__ = [
    "DEFAULT_VISUAL_MODE",
    "HybridRetriever",
    "SearchResult",
    "VectorBundle",
    "_build_filter",
    "validate_filters",
]
