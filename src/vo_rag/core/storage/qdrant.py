"""Qdrant collection schemas for vo-rag.

Collection names use the `vo_rag_` prefix and a `_v1` suffix.
`setup_collections` also creates the structured collection, so there is no
separate helper for it.

Creates 18 vo-rag collections, each with four named vectors:
  - dense    (1024d Cosine, BGE-M3)
  - sparse   (BGE-M3 learned sparse, modifier=idf)
  - bm25     (Qdrant-native BM25, modifier=idf)
  - colbert  (1024d Cosine, multivector max_sim, hnsw m=0 — rerank only)

`vo_rag_books_graph_edges_v1` is the only dense-only collection in the
schema (`_DENSE_ONLY_COLLECTIONS`).

`setup_collections()` is idempotent: on second call it no-ops for collections
with matching schema and raises ValueError on mismatch (no silent overwrite).

This module imports `vo_rag.core.idempotency` (a leaf), but observability
wiring is performed by `vo_rag.core.storage.__init__` calling
`vo_rag.core.observability.metrics.instrument_qdrant_store(QdrantStore)` —
that direction (storage -> observability) is fine; observability stays a leaf.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    HnswConfigDiff,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    Modifier,
    MultiVectorComparator,
    MultiVectorConfig,
    PayloadField,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from vo_rag.core.corpus import tf_schema
from vo_rag.core.idempotency import deterministic_uuid
from vo_rag.core.ingest.source_publication import SourceCoordinator, SourcePublicationJob
from vo_rag.core.observability.metrics import record_qdrant_call
from vo_rag.core.storage.passage_contract import (
    collapse_evidence_payloads,
    evidence_provenance,
    passage_identity,
    passage_payload,
)
from vo_rag.core.storage.points import VoPoint

#: Under heavy ingest load Qdrant answers an exact count with a 500
#: "read operations failed: Timeout" (or the client times out first). A read is
#: safe to repeat, so it is retried rather than failing the whole book (VOR-1427).
READ_RETRY_ATTEMPTS = 3
_READ_RETRY_BACKOFF_S = 5.0


def _is_read_timeout(exc: BaseException) -> bool:
    if isinstance(exc, ResponseHandlingException):
        return True
    if isinstance(exc, UnexpectedResponse) and exc.status_code == 500:
        return b"Timeout" in (exc.content or b"")
    return False


async def _timed[T](call: Awaitable[T]) -> T:
    """Await one Qdrant call, recording its latency (success or failure) onto the
    running ingest's envelope for the canary's p95 gate."""
    started = time.perf_counter()
    try:
        return await call
    finally:
        record_qdrant_call(time.perf_counter() - started)


async def retry_read[T](call: Callable[[], Awaitable[T]]) -> T:
    """Run a Qdrant read, retrying only its timeouts (bounded, growing backoff)."""
    for attempt in range(READ_RETRY_ATTEMPTS):
        try:
            return await call()
        except Exception as exc:
            if attempt == READ_RETRY_ATTEMPTS - 1 or not _is_read_timeout(exc):
                raise
            await asyncio.sleep(_READ_RETRY_BACKOFF_S * (attempt + 1))
    raise AssertionError("unreachable")  # pragma: no cover


COLLECTION_NAMES: list[str] = [
    "vo_rag_books_raw_v1",
    "vo_rag_books_sterile_v1",
    "vo_rag_books_summary_tree_v1",
    "vo_rag_books_raptor_v1",
    "vo_rag_books_contextualized_v1",
    "vo_rag_books_graph_nodes_v1",
    "vo_rag_books_graph_edges_v1",
    "vo_rag_books_dsi_v1",
    "vo_rag_repos_v1",
    "vo_rag_research_v1",
    "vo_rag_ideas_v1",
    "vo_rag_articles_v1",
    "vo_rag_tools_v1",
    "vo_rag_urls_v1",
    "vo_rag_structured_v1",
    "vo_rag_code_chunks_v1",
    "vo_rag_code_repo_summaries_v1",
    "vo_rag_code_collection_prototypes_v1",
    "vo_rag_reddit_v1",
]

# Collections that use dense-only vectors (no sparse, no ColBERT).
_DENSE_ONLY_COLLECTIONS: frozenset[str] = frozenset({"vo_rag_books_graph_edges_v1"})

DENSE_DIM = 1024
COLBERT_DIM = 1024

COMMON_PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "project_id": PayloadSchemaType.KEYWORD,
    "source_id": PayloadSchemaType.KEYWORD,
    "envelope_id": PayloadSchemaType.KEYWORD,
    "content_type": PayloadSchemaType.KEYWORD,
    "pipeline": PayloadSchemaType.KEYWORD,
    "generation_id": PayloadSchemaType.KEYWORD,
    "ingest_state": PayloadSchemaType.KEYWORD,
    "bm25_representation": PayloadSchemaType.KEYWORD,
}

BOOK_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "source_book": PayloadSchemaType.KEYWORD,
    "author": PayloadSchemaType.KEYWORD,
    "chapter": PayloadSchemaType.KEYWORD,
    "section_title": PayloadSchemaType.TEXT,
    "domain_tags": PayloadSchemaType.KEYWORD,
    "page_number": PayloadSchemaType.INTEGER,
    "has_image": PayloadSchemaType.BOOL,
    "license_class": PayloadSchemaType.KEYWORD,
}

# VOR-1443: track-and-field contextual tags (see core/corpus/tf_schema.py).
_TF_KIND: dict[str, PayloadSchemaType] = {
    "keyword": PayloadSchemaType.KEYWORD,
    "integer": PayloadSchemaType.INTEGER,
    "bool": PayloadSchemaType.BOOL,
}


def _tf_payload_kind(field: str, kind: str) -> PayloadSchemaType:
    """Map one ``tf_schema`` index kind onto its Qdrant payload type.

    Raises ``KeyError`` naming the field and the kind, so a kind added to
    ``TF_INDEXED_FIELDS`` without a mapping here fails at import time instead
    of silently leaving the ``tf_*`` field unindexed (VOR-1443).
    """
    if kind not in _TF_KIND:
        raise KeyError(f"tf_schema field {field!r} declares unknown index kind {kind!r}")
    return _TF_KIND[kind]


BOOK_EXTRA_INDEXES.update(
    {field: _tf_payload_kind(field, kind) for field, kind in tf_schema.TF_INDEXED_FIELDS.items()}
)

REPO_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "repo_name": PayloadSchemaType.KEYWORD,
    "language": PayloadSchemaType.KEYWORD,
    "topics": PayloadSchemaType.KEYWORD,
}

CODE_CHUNKS_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "repo_name": PayloadSchemaType.KEYWORD,
    "file_path": PayloadSchemaType.KEYWORD,
    "language": PayloadSchemaType.KEYWORD,
    "symbol_kind": PayloadSchemaType.KEYWORD,
    "symbol_name": PayloadSchemaType.KEYWORD,
    "sha": PayloadSchemaType.KEYWORD,
    "line_start": PayloadSchemaType.INTEGER,
    "line_end": PayloadSchemaType.INTEGER,
}

CODE_REPO_SUMMARIES_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "repo_name": PayloadSchemaType.KEYWORD,
    "host": PayloadSchemaType.KEYWORD,
    "primary_language": PayloadSchemaType.KEYWORD,
    "license_class": PayloadSchemaType.KEYWORD,
}

CODE_COLLECTION_PROTOTYPES_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "collection_slug": PayloadSchemaType.KEYWORD,
}

RESEARCH_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "confidence": PayloadSchemaType.FLOAT,
    "has_web_sources": PayloadSchemaType.BOOL,
}

SUMMARY_TREE_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "book_id": PayloadSchemaType.KEYWORD,
    "level": PayloadSchemaType.KEYWORD,
    "parent_node_id": PayloadSchemaType.KEYWORD,
    "canonical_path": PayloadSchemaType.KEYWORD,
}

RAPTOR_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "book_id": PayloadSchemaType.KEYWORD,
    "level": PayloadSchemaType.KEYWORD,
    "parent_node_id": PayloadSchemaType.KEYWORD,
}

CONTEXTUALIZED_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "book_id": PayloadSchemaType.KEYWORD,
    "section_path": PayloadSchemaType.KEYWORD,
    "chunk_id": PayloadSchemaType.KEYWORD,
}

GRAPH_NODES_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "node_type": PayloadSchemaType.KEYWORD,
    "book_ids": PayloadSchemaType.KEYWORD,
    "label": PayloadSchemaType.KEYWORD,
}

GRAPH_EDGES_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "edge_type": PayloadSchemaType.KEYWORD,
    "source_node_id": PayloadSchemaType.KEYWORD,
    "target_node_id": PayloadSchemaType.KEYWORD,
}

DSI_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "book_id": PayloadSchemaType.KEYWORD,
    "category_ids": PayloadSchemaType.KEYWORD,
    "technical_domains": PayloadSchemaType.KEYWORD,
}


BRAIN_EXTRA_INDEXES: dict[str, PayloadSchemaType] = {
    "tags": PayloadSchemaType.KEYWORD,
    "subject_type": PayloadSchemaType.KEYWORD,
    "page_area": PayloadSchemaType.KEYWORD,
    "writer_version": PayloadSchemaType.KEYWORD,
    "version_hash": PayloadSchemaType.KEYWORD,
}


def _extra_indexes_for(name: str) -> dict[str, PayloadSchemaType]:
    indexes = dict(_domain_indexes_for(name))
    if name.startswith("vo_rag_books_") or name == "vo_rag_reddit_v1":
        indexes.update(
            source_revision=PayloadSchemaType.INTEGER,
            publication_base_id=PayloadSchemaType.KEYWORD,
        )
    return indexes


def _domain_indexes_for(name: str) -> dict[str, PayloadSchemaType]:
    if name == "vo_rag_brain_v1":
        return BRAIN_EXTRA_INDEXES
    if name in ("vo_rag_books_raw_v1", "vo_rag_books_sterile_v1"):
        return BOOK_EXTRA_INDEXES
    if name == "vo_rag_repos_v1":
        return REPO_EXTRA_INDEXES
    if name == "vo_rag_code_chunks_v1":
        return CODE_CHUNKS_EXTRA_INDEXES
    if name == "vo_rag_code_repo_summaries_v1":
        return CODE_REPO_SUMMARIES_EXTRA_INDEXES
    if name == "vo_rag_code_collection_prototypes_v1":
        return CODE_COLLECTION_PROTOTYPES_EXTRA_INDEXES
    if name == "vo_rag_research_v1":
        return RESEARCH_EXTRA_INDEXES
    if name == "vo_rag_books_summary_tree_v1":
        return SUMMARY_TREE_EXTRA_INDEXES
    if name == "vo_rag_books_raptor_v1":
        return RAPTOR_EXTRA_INDEXES
    if name == "vo_rag_books_contextualized_v1":
        return CONTEXTUALIZED_EXTRA_INDEXES
    if name == "vo_rag_books_graph_nodes_v1":
        return GRAPH_NODES_EXTRA_INDEXES
    if name == "vo_rag_books_graph_edges_v1":
        return GRAPH_EDGES_EXTRA_INDEXES
    if name == "vo_rag_books_dsi_v1":
        return DSI_EXTRA_INDEXES
    return {}


def _vector_config() -> dict[str, VectorParams]:
    return {
        "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        "colbert": VectorParams(
            size=COLBERT_DIM,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
            hnsw_config=HnswConfigDiff(m=0),  # rerank only — NO ANN index
        ),
    }


def _dense_only_vector_config() -> dict[str, VectorParams]:
    """Dense-only 1024d config — used for edge collections that don't need sparse/ColBERT."""
    return {
        "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
    }


def _sparse_vector_config() -> dict[str, SparseVectorParams]:
    return {
        "sparse": SparseVectorParams(modifier=Modifier.IDF),
        "bm25": SparseVectorParams(modifier=Modifier.IDF),
    }


async def _ensure_collection(client: AsyncQdrantClient, name: str) -> bool:
    """Create collection if missing. Return True if created, False if pre-existing.

    Raises ValueError on schema mismatch.
    """
    existing = {c.name for c in (await client.get_collections()).collections}
    if name in existing:
        info = await client.get_collection(name)
        _assert_schema_matches(name, info)
        return False

    if name in _DENSE_ONLY_COLLECTIONS:
        await client.create_collection(
            collection_name=name,
            vectors_config=_dense_only_vector_config(),
        )
    else:
        await client.create_collection(
            collection_name=name,
            vectors_config=_vector_config(),
            sparse_vectors_config=_sparse_vector_config(),
        )
    return True


def _assert_schema_matches(name: str, info: Any) -> None:
    vectors = info.config.params.vectors or {}
    if "dense" not in vectors or vectors["dense"].size != DENSE_DIM:
        raise ValueError(f"{name}: dense vector missing or wrong size")

    if name in _DENSE_ONLY_COLLECTIONS:
        return

    if "colbert" not in vectors:
        raise ValueError(f"{name}: colbert multi-vector missing")
    if vectors["colbert"].hnsw_config is None or vectors["colbert"].hnsw_config.m != 0:
        raise ValueError(f"{name}: colbert hnsw.m must be 0 (rerank only)")

    sparse = info.config.params.sparse_vectors or {}
    for sname in ("sparse", "bm25"):
        if sname not in sparse:
            raise ValueError(f"{name}: sparse vector {sname!r} missing")
        if sparse[sname].modifier != Modifier.IDF:
            raise ValueError(f"{name}: sparse {sname!r} modifier must be IDF")


async def _ensure_payload_indexes(client: AsyncQdrantClient, name: str) -> None:
    indexes = {**COMMON_PAYLOAD_INDEXES, **_extra_indexes_for(name)}
    info = await client.get_collection(name)
    existing = set((info.payload_schema or {}).keys())
    for field, schema in indexes.items():
        if field in existing:
            continue
        await client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema,
        )


async def reconcile_payload_indexes(client: AsyncQdrantClient) -> dict[str, list[str]]:
    """Add declared payload indexes missing from collections that already exist.

    ``setup_collections`` only runs at bootstrap, so an index declared after a
    collection was created never reached production: the generation_id and
    ingest_state indexes were missing and the duplicate-ingest gate scanned
    every point per book (VOR-1427). Never creates collections. Returns
    ``{collection: [fields added]}``.
    """
    present = {c.name for c in (await client.get_collections()).collections}
    added: dict[str, list[str]] = {}
    for name in COLLECTION_NAMES:
        if name not in present:
            continue
        have = set(((await client.get_collection(name)).payload_schema or {}).keys())
        missing = [
            f for f in {**COMMON_PAYLOAD_INDEXES, **_extra_indexes_for(name)} if f not in have
        ]
        if missing:
            await _ensure_payload_indexes(client, name)
            added[name] = missing
    return added


async def setup_collections(client: AsyncQdrantClient) -> None:
    """Create all vo-rag Qdrant collections. Idempotent on re-run."""
    for name in COLLECTION_NAMES:
        await _ensure_collection(client, name)
        await _ensure_payload_indexes(client, name)


def deterministic_point_id(
    source_id: str, chunk_index: int, stage: str, version_hash: str | None = None
) -> str:
    """Build deterministic point ID for idempotent upserts.

    ``version_hash`` is optional. When None/empty the id is built from
    (source_id, chunk_index, stage) only — the unversioned book_rag raw/sterile
    scheme. Callers that pass a non-empty version (research, repo, code, raptor,
    dsi, summary) get the exact same id as before, so their stored points are
    unaffected.
    """

    parts = [source_id, str(chunk_index), stage]
    if version_hash:
        parts.append(version_hash)
    return deterministic_uuid(*parts)


def bge_sparse_to_qdrant(d: dict[int, float]) -> SparseVector:
    """Convert BGE-M3 sparse map to sorted Qdrant SparseVector."""

    items = sorted(d.items())
    return SparseVector(
        indices=[int(k) for k, _ in items],
        values=[float(v) for _, v in items],
    )


def _build_filter(filters: dict[str, Any] | Filter | None) -> Filter | None:
    if not filters:
        return None
    # Idempotent: HybridRetriever.hybrid_search merges the license mode filter
    # with the user filter into a Filter, then passes it here — re-building would
    # call .items() on a Filter and crash. An already-built Filter passes through.
    if isinstance(filters, Filter):
        return filters

    must: list[Any] = []
    for field, value in filters.items():
        if isinstance(value, list):
            must.append(FieldCondition(key=field, match=MatchAny(any=list(value))))
        else:
            must.append(FieldCondition(key=field, match=MatchValue(value=value)))

    return Filter(must=must)


_VISUAL_GEOMETRY_FIELDS = frozenset(
    {
        "original_bbox",
        "native_page_size",
        "page_rect",
        "media_box",
        "crop_box",
        "clip_bbox_top_left",
    }
)
_VISUAL_METADATA_FIELDS = frozenset(
    {
        "equation_visual",
        "table_visual",
        "source_page_visual",
        "source_page_visuals",
        "source_table_visuals",
    }
)


def _encode_source_geometry(payload: dict[str, Any]) -> dict[str, Any]:
    """Encode source visual floats losslessly before Qdrant JSON payload storage.

    Qdrant can change long decimal floats while storing payload JSON. Hex strings
    preserve the original IEEE754 values exactly; source text, pixels, integers
    and unrelated metadata retain their original values.
    """

    def encode(value: Any) -> Any:
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("source visual coordinates must be finite")
            return value.hex()
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [encode(item) for item in value]
        return value

    def validate_encoded(value: Any) -> None:
        if isinstance(value, bool | float):
            raise ValueError("marked source visual coordinates must use finite hex strings")
        if isinstance(value, int):
            return
        if isinstance(value, str):
            try:
                decoded = float.fromhex(value)
            except ValueError as exc:
                raise ValueError("invalid source visual hex coordinate") from exc
            if not math.isfinite(decoded) or decoded.hex() != value:
                raise ValueError("source visual hex coordinates must be finite and canonical")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "coord_origin" and isinstance(item, str):
                    continue
                validate_encoded(item)
            return
        if isinstance(value, list | tuple):
            for item in value:
                validate_encoded(item)
            return
        raise ValueError("invalid marked source visual coordinate shape")

    def visual(value: Any) -> Any:
        if isinstance(value, list):
            return [visual(item) for item in value]
        if not isinstance(value, dict) or value.get("status") != "source_bound":
            return value
        fields = _VISUAL_GEOMETRY_FIELDS.intersection(value)
        if not fields:
            return value
        marker = value.get("coordinate_encoding")
        if marker == "float-hex.v1":
            for key in fields:
                validate_encoded(value[key])
            return dict(value)
        if marker is not None:
            raise ValueError("unsupported source visual coordinate encoding")
        return {
            **value,
            **{key: encode(value[key]) for key in fields},
            "coordinate_encoding": "float-hex.v1",
        }

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return payload
    return {
        **payload,
        "metadata": {
            key: visual(value) if key in _VISUAL_METADATA_FIELDS else value
            for key, value in metadata.items()
        },
    }


def build_point_structs(points: list[VoPoint]) -> list[PointStruct]:
    """Construct the exact normal writer payload and named vector representations."""
    structs: list[PointStruct] = []
    for point in points:
        point_id = deterministic_point_id(
            point.source_id,
            point.chunk_index,
            point.stage,
            point.version_hash,
        )
        vectors: dict[str, Any] = {
            "dense": point.dense,
        }
        # Empty maps are valid sparse representations, including symbol-only text.
        vectors["sparse"] = bge_sparse_to_qdrant(point.sparse)
        vectors["bm25"] = bge_sparse_to_qdrant(point.bm25)
        if point.colbert:
            vectors["colbert"] = point.colbert
        structs.append(
            PointStruct(
                id=point_id,
                vector=vectors,
                payload=_encode_source_geometry(
                    passage_payload(
                        {
                            **point.payload,
                            "chunk_index": point.chunk_index,
                            "version_hash": point.version_hash,
                        },
                        point_id=point_id,
                        source_id=point.source_id,
                        dense=point.dense,
                        colbert=point.colbert,
                        bm25=point.bm25,
                    )
                ),
            )
        )
    return structs


class QdrantStore:
    """Async Qdrant wrapper for vo-rag."""

    def __init__(self, url: str, api_key: str | None = None, *, prefer_grpc: bool = True) -> None:
        self._publication_enabled = os.environ.get(
            "VORAG_SOURCE_PUBLICATION_ENABLED", "true"
        ).lower() in {"1", "true", "yes"}
        self._publication_writes_enabled = os.environ.get(
            "VORAG_SOURCE_PUBLICATION_WRITES_ENABLED", "false"
        ).lower() in {"1", "true", "yes"}
        if self._publication_writes_enabled and not self._publication_enabled:
            raise ValueError("source revision writes require publication readers")
        self._raw_client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            timeout=120,
        )
        self._client: Any = self._raw_client
        if self._publication_enabled:
            from vo_rag.core.storage.publication_client import PublicationClient

            self._client = PublicationClient(
                self._raw_client, self._publication_pointers, self._publication_mutation
            )

    async def aclose(self) -> None:
        """Close the async Qdrant client owned by this store."""
        await self._raw_client.close()

    async def _source_coordinator(self, collection: str) -> SourceCoordinator | None:
        from vo_rag.core.config.settings import get_settings
        from vo_rag.core.db.postgres import get_pool
        from vo_rag.core.ingest.source_publication import SourceCoordinator
        from vo_rag.core.storage.publication_filter import publication_corpus

        corpus = publication_corpus(collection)
        if corpus is None:
            return None
        settings = get_settings()
        pool = await get_pool(settings.database.url)
        return SourceCoordinator(pool, corpus=corpus)

    async def _publication_mutation(
        self, operation: str, collection: str, kwargs: dict[str, Any]
    ) -> Any:
        from vo_rag.core.storage.source_mutations import SourceMutationWriter

        coordinator = await self._source_coordinator(collection)
        if coordinator is None:
            raise ValueError("source mutation corpus scope mismatch")
        writer = SourceMutationWriter(self._raw_client, coordinator)
        if self.source_publication_enabled:
            return await writer.mutate(operation, collection, kwargs)
        return await writer.mutate_legacy(operation, collection, kwargs)

    async def _publication_pointers(self, corpus: str, collection: str) -> dict[str, int | None]:
        coordinator = await self._source_coordinator(collection)
        if coordinator is None or coordinator.corpus != corpus:
            raise ValueError("source publication corpus scope mismatch")
        from vo_rag.core.storage.publication_read_view import captured_pointers

        captured = captured_pointers(coordinator.pool, corpus, collection)
        if captured is not None:
            return captured
        result: dict[str, int | None] = await coordinator.list_active(collection)
        return result

    @property
    def source_publication_enabled(self) -> bool:
        """Return whether revision-producing writers are enabled."""
        return self._publication_enabled and self._publication_writes_enabled

    @property
    def source_visibility_enabled(self) -> bool:
        """Return whether readers enforce publication and dependency pointers."""
        return self._publication_enabled

    async def active_source_revision(self, collection: str, source_id: str) -> int | None:
        """Capture the authoritative revision before reading input for a derived producer."""
        if not self._publication_enabled:
            return None
        coordinator = await self._source_coordinator(collection)
        if coordinator is None:
            return None
        active = await coordinator.read_active(collection, source_id)
        return active.revision if active is not None else None

    async def admit_source_revision(
        self,
        collection: str,
        source_id: str,
        job_id: str,
        source_sha256: str,
        *,
        expected_active_revision: int | None | Literal["current"] = "current",
        expected_dependencies: Mapping[str, int | None] | None = None,
        producer_recipe: Mapping[str, Any] | None = None,
    ) -> SourcePublicationJob | None:
        """Register immutable source work before parser or embedding work starts."""
        if not self._publication_enabled:
            return None
        coordinator = await self._source_coordinator(collection)
        if coordinator is None:
            return None
        dependencies = dict(expected_dependencies or {})
        if dependencies and coordinator.corpus != "books":
            raise ValueError("raw dependency revisions require the Books corpus")
        if any(
            not isinstance(source, str)
            or not source.strip()
            or (revision is not None and (type(revision) is not int or revision <= 0))
            for source, revision in dependencies.items()
        ):
            raise ValueError("dependency revisions require source IDs and positive integers")
        if not self.source_publication_enabled:
            if any(revision is not None for revision in dependencies.values()):
                raise ValueError(
                    "reader-first legacy admission cannot bind a non-null raw revision"
                )
            async with coordinator.fence(
                source_id, dependency_sources=tuple(dependencies)
            ) as fence:
                await coordinator.assert_legacy_write_allowed(fence, source_id, tuple(dependencies))
            return None
        recipe: dict[str, Any] = {
            "writer": "normal-source-publication-v1",
            "collection": collection,
            "source_revision_order": "serialized immutable-source catalog admission",
        }
        if expected_dependencies is not None:
            recipe["expected_dependencies"] = dependencies
        if producer_recipe is not None:
            recipe["producer_recipe"] = dict(producer_recipe)
        async with coordinator.fence(
            source_id, job_id=job_id, dependency_sources=tuple(dependencies)
        ) as fence:
            for dependency, revision in dependencies.items():
                active = await coordinator.read_active("vo_rag_books_raw_v1", dependency)
                if (active.revision if active is not None else None) != revision:
                    raise ValueError("canonical raw dependency changed before admission")
            previous = await coordinator.read_active(collection, source_id)
            existing_job = await coordinator.read_job(job_id)
            expected = (
                (
                    existing_job.expected_active_revision
                    if existing_job is not None
                    else previous.revision
                    if previous is not None
                    else None
                )
                if expected_active_revision == "current"
                else expected_active_revision
            )
            return await coordinator.begin(
                fence,
                collection,
                source_id,
                job_id,
                source_sha256,
                expected,
                recipe,
            )

    async def abort_source_revision(self, job: SourcePublicationJob | None, reason: str) -> None:
        """Checkpoint retryable prewrite failures; retain every partial job for recovery."""
        if job is None:
            return
        from vo_rag.core.ingest.source_publication import PublicationState

        coordinator = await self._source_coordinator(job.collection)
        if coordinator is None:
            raise ValueError("publication job has an unsupported corpus")
        async with coordinator.fence(job.source_id, job_id=job.job_id) as fence:
            current = await coordinator.read_job(job.job_id)
            if current is None or current.state in {
                PublicationState.PUBLISHED,
                PublicationState.PRESERVED_NO_WRITE,
                PublicationState.SUPERSEDED,
            }:
                return
            checkpoint = dict(current.checkpoint)
            if not checkpoint.get("write_attempts") and not checkpoint.get("pending_batch"):
                checkpoint["no_write"] = {
                    "qdrant_write_attempts": 0,
                    "acknowledgement_unknown": False,
                    "reason": reason,
                }
                checkpoint["owner"] = "normal_source_producer"
                checkpoint["recovery"] = "retry identical admitted job; no Qdrant write attempted"
                await coordinator.checkpoint(
                    fence, job.job_id, PublicationState.FAILED_RETRYABLE, checkpoint
                )

    async def upsert_vectors(
        self,
        collection: str,
        points: list[VoPoint],
        *,
        publication_job: SourcePublicationJob | None = None,
        expected_before: Mapping[str, str] | None = None,
        expected_before_receipts: Mapping[str, Any] | None = None,
        expected_dependencies: Mapping[str, int | None] | None = None,
    ) -> list[str]:
        structs = build_point_structs(points)

        if self.source_publication_enabled:
            from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter

            coordinator = await self._source_coordinator(collection)
            if coordinator is not None and structs:
                sources = {point.source_id for point in points}
                if len(sources) != 1:
                    raise ValueError("source replacement must contain exactly one complete source")
                if publication_job is None:
                    raise ValueError(
                        "source writer requires admission before representation generation"
                    )
                source_id = sources.pop()
                if (
                    publication_job.source_id != source_id
                    or publication_job.collection != collection
                ):
                    raise ValueError("source writer admission scope mismatch")
                if publication_job.recipe.get("expected_dependencies", {}) != dict(
                    expected_dependencies or {}
                ):
                    raise ValueError("writer dependencies differ from admitted producer inputs")
                return await SourceRevisionWriter(self._raw_client, coordinator).replace(
                    collection,
                    source_id,
                    structs,
                    job_id=publication_job.job_id,
                    source_sha256=publication_job.source_sha256,
                    expected_active_revision=publication_job.expected_active_revision,
                    expected_before=expected_before,
                    expected_before_receipts=expected_before_receipts,
                    expected_dependencies=expected_dependencies,
                    recipe=publication_job.recipe
                    or {
                        "writer": "normal-source-publication-v1",
                        "collection": collection,
                        "source_revision_order": "serialized immutable-source catalog admission",
                    },
                )

        if self.source_visibility_enabled and structs:
            from vo_rag.core.storage.source_mutations import SourceMutationWriter

            coordinator = await self._source_coordinator(collection)
            if coordinator is not None:
                if publication_job is not None:
                    raise ValueError("revision job cannot use compatibility writes")
                await SourceMutationWriter(self._raw_client, coordinator).mutate_legacy(
                    "upsert", collection, {"points": structs, "wait": True}
                )
                return [str(point.id) for point in structs]

        # Batch the upsert: ColBERT multivectors are large (~768 KB/point), so a
        # whole book in one request (hundreds of points) exceeds the HTTP body
        # limit and the client raises ResponseHandlingException. Cap each request.
        _BATCH = 8  # 192-row ColBERT ~786 KB/pt; >8/req trips httpx ReadError on wait=True
        for start in range(0, len(structs), _BATCH):
            batch = structs[start : start + _BATCH]
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    await _timed(
                        self._client.upsert(collection_name=collection, points=batch, wait=True)
                    )
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))
            if last_error is not None:
                raise last_error

        return [str(point.id) for point in structs]

    async def delete_source(self, collection: str, source_id: str) -> None:
        await _timed(
            self._client.delete(
                collection_name=collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_id",
                            match=MatchValue(value=source_id),
                        )
                    ]
                ),
                wait=True,
            )
        )

    async def delete_source_except(
        self, collection: str, source_id: str, version_hash: str | None
    ) -> None:
        """Remove old source versions after its replacement was fully upserted."""
        if not version_hash:
            raise ValueError("replacement version is required")
        await _timed(
            self._client.delete(
                collection_name=collection,
                points_selector=Filter(
                    must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))],
                    must_not=[
                        FieldCondition(key="version_hash", match=MatchValue(value=version_hash))
                    ],
                ),
                wait=True,
            )
        )

    @staticmethod
    def _generation_filter(
        source_id: str,
        *,
        generation_id: str | None = None,
        ingest_state: str | None = None,
    ) -> Filter:
        must: list[Any] = [FieldCondition(key="source_id", match=MatchValue(value=source_id))]
        if generation_id is not None:
            must.append(FieldCondition(key="generation_id", match=MatchValue(value=generation_id)))
        if ingest_state is not None:
            must.append(FieldCondition(key="ingest_state", match=MatchValue(value=ingest_state)))
        return Filter(must=must)

    async def delete_generation(
        self,
        collection: str,
        source_id: str,
        generation_id: str,
    ) -> None:
        """Delete exactly one superseded/failed generation, never all source points."""
        await _timed(
            self._client.delete(
                collection_name=collection,
                points_selector=self._generation_filter(
                    source_id,
                    generation_id=generation_id,
                ),
                wait=True,
            )
        )

    @staticmethod
    def _legacy_source_filter(source_id: str) -> Filter:
        """Select pre-generation points whose payload has no generation id."""
        return Filter(
            must=[
                FieldCondition(key="source_id", match=MatchValue(value=source_id)),
                IsEmptyCondition(is_empty=PayloadField(key="generation_id")),
            ]
        )

    async def delete_legacy_source(self, collection: str, source_id: str) -> None:
        """Delete only pre-generation points for a migrated source."""
        await _timed(
            self._client.delete(
                collection_name=collection,
                points_selector=self._legacy_source_filter(source_id),
                wait=True,
            )
        )

    async def set_legacy_source_state(
        self,
        collection: str,
        source_id: str,
        state: str,
    ) -> None:
        """Hide or restore pre-generation points during first activation."""
        await _timed(
            self._client.set_payload(
                collection_name=collection,
                payload={"ingest_state": state},
                points=self._legacy_source_filter(source_id),
                wait=True,
            )
        )

    async def set_generation_state(
        self,
        collection: str,
        source_id: str,
        generation_id: str,
        state: str,
    ) -> None:
        """Atomically update publication state for every point in a generation."""
        await _timed(
            self._client.set_payload(
                collection_name=collection,
                payload={"ingest_state": state},
                points=self._generation_filter(source_id, generation_id=generation_id),
                wait=True,
            )
        )

    async def set_source_quality_warnings(
        self,
        collection: str,
        source_id: str,
        warnings: list[dict[str, Any]],
        *,
        generation_id: str | None = None,
    ) -> None:
        """Attach honest quality warnings to the exact ingested source generation."""
        raw_pointer_before = None
        coordinator = None
        if self.source_publication_enabled and collection == "vo_rag_books_raw_v1":
            coordinator = await self._source_coordinator(collection)
            if coordinator is not None:
                raw_pointer_before = await coordinator.read_active(collection, source_id)
                if raw_pointer_before is not None and raw_pointer_before.revision is not None:
                    from vo_rag.core.storage.source_quality_dependencies import (
                        copy_forward_quality_dependencies,
                    )

                    await copy_forward_quality_dependencies(
                        self._raw_client,
                        coordinator,
                        source_id,
                        new_revision=raw_pointer_before.revision,
                    )
                    from vo_rag.core.storage.source_revision_writer import SourceRevisionWriter

                    async with coordinator.fence(source_id):
                        current_pointer = await coordinator.read_active(collection, source_id)
                        if (
                            current_pointer is not None
                            and current_pointer.revision == raw_pointer_before.revision
                        ):
                            active_rows = await SourceRevisionWriter(
                                self._raw_client, coordinator
                            ).census(collection, source_id, current_pointer.revision)
                            selected_rows = [
                                row
                                for row in active_rows
                                if generation_id is None
                                or (row.payload or {}).get("generation_id") == generation_id
                            ]
                            if selected_rows and all(
                                (row.payload or {}).get("quality_warnings") == warnings
                                and (row.payload or {}).get("quality_disposition") == "warning"
                                for row in selected_rows
                            ):
                                return
        await _timed(
            self._client.set_payload(
                collection_name=collection,
                payload={"quality_warnings": warnings, "quality_disposition": "warning"},
                points=self._generation_filter(source_id, generation_id=generation_id),
                wait=True,
            )
        )
        if coordinator is not None and raw_pointer_before is not None:
            raw_pointer_after = await coordinator.read_active(collection, source_id)
            if (
                raw_pointer_after is not None
                and raw_pointer_before.revision is not None
                and raw_pointer_after.revision is not None
                and raw_pointer_after.revision != raw_pointer_before.revision
            ):
                from vo_rag.core.storage.source_quality_dependencies import (
                    copy_forward_quality_dependencies,
                )

                await copy_forward_quality_dependencies(
                    self._raw_client,
                    coordinator,
                    source_id,
                    raw_pointer_before.revision,
                    raw_pointer_after.revision,
                )

    async def resume_source_quality_dependencies(self, source_id: str) -> dict[str, list[str]]:
        """Resume stale quality-only derived dependencies for one raw Books source.

        Args:
            source_id: Canonical raw Books source ID to recover.

        Returns:
            Successfully published derived source IDs grouped by collection.

        Raises:
            ValueError: If source publication writes are disabled or the source
                has no active raw Books revision.
        """
        if not source_id:
            raise ValueError("source_id must be non-empty")
        if not self.source_publication_enabled:
            raise ValueError("quality dependency recovery requires source publication writes")
        coordinator = await self._source_coordinator("vo_rag_books_raw_v1")
        if coordinator is None:
            raise ValueError("quality dependency recovery requires the Books corpus")
        pointer = await coordinator.read_active("vo_rag_books_raw_v1", source_id)
        if pointer is None or pointer.revision is None:
            raise ValueError("quality dependency recovery requires an active raw revision")
        from vo_rag.core.storage.source_quality_dependencies import (
            copy_forward_quality_dependencies,
        )

        return await copy_forward_quality_dependencies(
            self._raw_client, coordinator, source_id, new_revision=pointer.revision
        )

    async def count_source(
        self,
        collection: str,
        source_id: str,
        *,
        generation_id: str | None = None,
        ingest_state: str | None = None,
    ) -> int:
        """Exact number of points already stored for ``source_id``.

        Used as the duplicate-ingest gate (VOR-1403), so it counts exactly:
        an approximate 0 would silently re-embed a book that is already indexed.
        """
        result = await retry_read(
            lambda: _timed(
                self._client.count(
                    collection_name=collection,
                    count_filter=self._generation_filter(
                        source_id,
                        generation_id=generation_id,
                        ingest_state=ingest_state,
                    ),
                    exact=True,
                )
            )
        )
        return int(result.count)

    async def count_active_or_legacy_source(self, collection: str, source_id: str) -> int:
        """Count only query-visible active or pre-generation source points."""
        result = await retry_read(
            lambda: _timed(
                self._client.count(
                    collection_name=collection,
                    count_filter=Filter(
                        must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))],
                        should=[
                            FieldCondition(key="ingest_state", match=MatchValue(value="active")),
                            IsEmptyCondition(is_empty=PayloadField(key="ingest_state")),
                        ],
                    ),
                    exact=True,
                )
            )
        )
        return int(result.count)

    async def search_hybrid(
        self,
        collection: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        prefetch_depth: int = 100,
    ) -> list[ScoredPoint]:
        if collection in _DENSE_ONLY_COLLECTIONS:
            if not query_dense:
                return []
            return await self.search_single_collection(
                collection, query_dense, top_k=top_k, filters=filters
            )
        query_filter = _build_filter(filters)
        bm25_filter = query_filter
        if collection in {
            "vo_rag_books_summary_tree_v1",
            "vo_rag_books_raptor_v1",
            "vo_rag_books_graph_nodes_v1",
        }:
            # Historical derived-node writers copied learned sparse token IDs
            # into BM25. Only migrated genuine hashed-token vectors may vote.
            clauses: list[Any] = [
                FieldCondition(
                    key="bm25_representation", match=MatchValue(value="hashed_tokens_v1")
                )
            ]
            if query_filter:
                clauses.append(query_filter)
            bm25_filter = Filter(must=clauses)

        prefetch: list[Prefetch] = []
        if query_dense:
            prefetch.append(
                Prefetch(
                    query=query_dense, using="dense", limit=prefetch_depth, filter=query_filter
                )
            )
        if query_sparse:
            prefetch.append(
                Prefetch(
                    query=bge_sparse_to_qdrant(query_sparse),
                    using="sparse",
                    limit=prefetch_depth,
                    filter=query_filter,
                )
            )
        if query_bm25:
            prefetch.append(
                Prefetch(
                    query=bge_sparse_to_qdrant(query_bm25),
                    using="bm25",
                    limit=prefetch_depth,
                    filter=bm25_filter,
                )
            )
        if not prefetch:
            return []
        response = await _timed(
            self._client.query_points(
                collection_name=collection,
                prefetch=prefetch,
                # Token vectors and dense substitutes share dimensions, not a
                # representation. Fuse genuine retrieval channels for every
                # corpus; optional stored ColBERT is not implicitly rescored.
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        )
        points: list[ScoredPoint] = response.points
        return points

    async def search_single_collection(
        self,
        collection: str,
        query_dense: list[float],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredPoint]:
        response = await _timed(
            self._client.query_points(
                collection_name=collection,
                query=query_dense,
                using="dense",
                limit=top_k,
                query_filter=_build_filter(filters),
                with_payload=True,
            )
        )
        points: list[ScoredPoint] = response.points
        return points

    async def search_cross_collection(
        self,
        collections: list[str],
        query_dense: list[float],
        query_sparse: dict[int, float],
        query_bm25: dict[int, float],
        query_colbert: list[list[float]],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        k_rrf: int = 60,
    ) -> list[ScoredPoint]:
        per_collection = await asyncio.gather(
            *[
                self.search_hybrid(
                    collection,
                    query_dense,
                    query_sparse,
                    query_bm25,
                    query_colbert,
                    top_k=top_k,
                    filters=filters,
                )
                for collection in collections
            ]
        )

        rrf_scores: dict[tuple[str, str], float] = {}
        representatives: dict[tuple[str, str], ScoredPoint] = {}
        origins: dict[tuple[str, str], str] = {}
        citations: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = {}

        for collection, points in zip(collections, per_collection, strict=True):
            seen_in_track: set[tuple[str, str]] = set()
            for rank, point in enumerate(points, start=1):
                payload = point.payload if isinstance(point.payload, dict) else {}
                key = passage_identity(payload, str(point.id))

                for citation in evidence_provenance(payload, str(point.id), collection):
                    citations.setdefault(key, {})[(citation["collection"], citation["id"])] = (
                        citation
                    )
                if key not in seen_in_track:
                    rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k_rrf + rank))
                    seen_in_track.add(key)
                if key not in representatives:
                    representatives[key] = point
                    origins[key] = collection

        merged = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        results: list[ScoredPoint] = []
        for key, score in merged:
            representative = representatives[key]
            payload = representative.payload or {}
            if len(citations[key]) > 1:
                payload = {**payload, "duplicate_provenance": list(citations[key].values())}
            results.append(representative.model_copy(update={"score": score, "payload": payload}))

        selected = collapse_evidence_payloads(
            [
                (result.payload or {}, str(result.id), origins[key])
                for result, (key, _) in zip(results, merged, strict=True)
            ]
        )
        return [
            results[index].model_copy(update={"payload": payload})
            for index, payload in selected[:top_k]
        ]
