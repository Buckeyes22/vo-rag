"""Book-reader API routes backed by Postgres source metadata and Qdrant layers."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, status
from qdrant_client.models import FieldCondition, Filter, MatchValue

from vo_rag.api.dependencies import get_services
from vo_rag.api.schemas import (
    BookDsiSummary,
    BookGraphNodeSummary,
    BookGraphSummary,
    BookLayers,
    BookOverviewResponse,
    BookPageChunk,
    BookPageResponse,
    BookRaptorSummary,
    BookStats,
    BookSummariesResponse,
    BookSummaryTreeNode,
    BookSummaryTreeSummary,
)
from vo_rag.api.singletons import SurfaceServices
from vo_rag.core.retrieval.search import _with_generation_filter

router = APIRouter()
ServicesDep = Annotated[SurfaceServices, Depends(get_services)]

RAW_COLLECTION = "vo_rag_books_raw_v1"
SUMMARY_TREE_COLLECTION = "vo_rag_books_summary_tree_v1"
RAPTOR_COLLECTION = "vo_rag_books_raptor_v1"
DSI_COLLECTION = "vo_rag_books_dsi_v1"
GRAPH_NODES_COLLECTION = "vo_rag_books_graph_nodes_v1"


def _require_pool(services: SurfaceServices) -> Any:
    if services.db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="db_pool_unavailable",
        )
    return services.db_pool


def _require_qdrant_client(services: SurfaceServices) -> Any:
    if services.qdrant is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="qdrant_unavailable",
        )
    return services.qdrant._client


@router.get("/api/v1/books/{source_id}", response_model=BookOverviewResponse)
async def book_overview(
    source_id: str,
    services: ServicesDep,
) -> BookOverviewResponse:
    pool = _require_pool(services)
    client = _require_qdrant_client(services)

    source_row = await _fetch_source_row(pool, source_id)
    artifact_row = await _fetch_artifact_row(pool, source_id)
    raw_count = await _count(client, RAW_COLLECTION, "source_id", source_id)
    if source_row is None and raw_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="book_not_found")

    return BookOverviewResponse(
        source_id=source_id,
        title=_optional_str(_row_get(source_row, "title")),
        author=_optional_str(_row_get(source_row, "author")),
        license_class=_optional_str(_row_get(source_row, "license_class")),
        artifact_id=_optional_str(_row_get(artifact_row, "id")),
        stats=_artifact_stats(artifact_row),
        layers=BookLayers(
            summary_tree=await _count(client, SUMMARY_TREE_COLLECTION, "book_id", source_id),
            raptor=await _count(client, RAPTOR_COLLECTION, "book_id", source_id),
            dsi=await _count(client, DSI_COLLECTION, "book_id", source_id),
            graph_nodes=await _count(client, GRAPH_NODES_COLLECTION, "book_ids", source_id),
        ),
    )


@router.get("/api/v1/books/{source_id}/pages/{page}", response_model=BookPageResponse)
async def book_page(
    source_id: str,
    page: Annotated[int, Path(ge=1)],
    services: ServicesDep,
) -> BookPageResponse:
    pool = _require_pool(services)
    client = _require_qdrant_client(services)

    payloads = await _scroll_all(
        client,
        RAW_COLLECTION,
        _raw_page_filter(source_id, page),
        with_payload=[
            "chunk_id",
            "section_title",
            "content",
            "has_image",
            "image_id",
            "vlm_description",
            "token_count",
        ],
    )
    chunks = sorted(
        (_page_chunk(payload) for payload in payloads), key=lambda chunk: chunk.chunk_id
    )
    stats = _artifact_stats(await _fetch_artifact_row(pool, source_id))
    return BookPageResponse(
        page=page,
        max_page=stats.pages if stats is not None else None,
        chunks=chunks,
    )


@router.get("/api/v1/books/{source_id}/summaries", response_model=BookSummariesResponse)
async def book_summaries(
    source_id: str,
    services: ServicesDep,
) -> BookSummariesResponse:
    client = _require_qdrant_client(services)

    dsi_payloads = await _scroll_all(
        client,
        DSI_COLLECTION,
        _book_filter(source_id),
        limit=1,
        with_payload=[
            "chapter_outline",
            "key_concepts",
            "key_methods",
            "key_equations_or_models",
            "prerequisite_topics",
            "figure_heavy_topics",
        ],
    )
    summary_payloads = await _scroll_all(
        client,
        SUMMARY_TREE_COLLECTION,
        _book_filter(source_id),
        with_payload=["level", "display_path", "metadata"],
    )
    graph_count = await _count(client, GRAPH_NODES_COLLECTION, "book_ids", source_id)
    graph_payloads = (
        await _scroll_all(
            client,
            GRAPH_NODES_COLLECTION,
            _graph_filter(source_id),
            limit=200,
            with_payload=["label", "node_type", "chunk_ids"],
        )
        if graph_count > 0
        else []
    )
    raptor_count = await _count(client, RAPTOR_COLLECTION, "book_id", source_id)

    return BookSummariesResponse(
        dsi=_dsi_summary(dsi_payloads[0]) if dsi_payloads else None,
        summary_tree=_summary_tree(summary_payloads) if summary_payloads else None,
        graph=_graph_summary(graph_count, graph_payloads) if graph_count > 0 else None,
        raptor=BookRaptorSummary(node_count=raptor_count) if raptor_count > 0 else None,
    )


async def _fetch_source_row(pool: Any, source_id: str) -> Any | None:
    return await pool.fetchrow(
        """
        SELECT id, title, author, license_class
        FROM sources
        WHERE id = $1
        """,
        source_id,
    )


async def _fetch_artifact_row(pool: Any, source_id: str) -> Any | None:
    return await pool.fetchrow(
        """
        SELECT id, content
        FROM knowledge_artifacts
        WHERE pipeline = 'book_rag'
          AND content->>'source_id' = $1
          AND status = 'active'
          AND COALESCE(content->>'raw_collection', 'vo_rag_books_raw_v1')
              = 'vo_rag_books_raw_v1'
        ORDER BY updated_at DESC, created_at DESC, id ASC
        LIMIT 1
        """,
        source_id,
    )


async def _count(client: Any, collection: str, key: str, value: str) -> int:
    response = await client.count(
        collection_name=collection,
        count_filter=_with_generation_filter(collection, _match_filter(key, value)),
        exact=True,
    )
    return int(response.count)


async def _scroll_all(
    client: Any,
    collection: str,
    scroll_filter: Filter,
    *,
    limit: int = 256,
    with_payload: bool | list[str] = True,
) -> list[dict[str, Any]]:
    offset: Any | None = None
    payloads: list[dict[str, Any]] = []
    while True:
        records, next_offset = await client.scroll(
            collection_name=collection,
            scroll_filter=_with_generation_filter(collection, scroll_filter),
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
        )
        for record in records:
            payload = getattr(record, "payload", None)
            if isinstance(payload, dict):
                payloads.append(payload)
        if next_offset is None:
            return payloads
        offset = next_offset


def _match_filter(key: str, value: str) -> Filter:
    must: list[Any] = [FieldCondition(key=key, match=MatchValue(value=value))]
    return Filter(must=must)


def _book_filter(source_id: str) -> Filter:
    return _match_filter("book_id", source_id)


def _graph_filter(source_id: str) -> Filter:
    return _match_filter("book_ids", source_id)


def _raw_page_filter(source_id: str, page: int) -> Filter:
    must: list[Any] = [
        FieldCondition(key="source_id", match=MatchValue(value=source_id)),
        FieldCondition(key="page_number", match=MatchValue(value=page)),
    ]
    return Filter(must=must)


def _page_chunk(payload: dict[str, Any]) -> BookPageChunk:
    return BookPageChunk(
        chunk_id=str(payload.get("chunk_id") or ""),
        section_title=_optional_str(payload.get("section_title")),
        text=str(payload.get("content") or ""),
        has_image=bool(payload.get("has_image", False)),
        image_id=_optional_str(payload.get("image_id")),
        vlm_description=_optional_str(payload.get("vlm_description")),
        token_count=_optional_int(payload.get("token_count")),
    )


def _artifact_stats(row: Any | None) -> BookStats | None:
    content = _as_dict(_row_get(row, "content"))
    stats = _as_dict(content.get("stats"))
    if not stats:
        return None
    return BookStats(
        text_chunks=_int_or_zero(stats.get("text_chunks")),
        figure_chunks=_int_or_zero(stats.get("figure_chunks")),
        pages=_optional_int(stats.get("pages")),
        domain_tags=_str_list(stats.get("domain_tags")),
    )


def _dsi_summary(payload: dict[str, Any]) -> BookDsiSummary:
    return BookDsiSummary(
        chapter_outline=_optional_str(payload.get("chapter_outline")),
        key_concepts=_flatten_value_list(payload.get("key_concepts")),
        key_methods=_flatten_value_list(payload.get("key_methods")),
        key_equations_or_models=_flatten_value_list(payload.get("key_equations_or_models")),
        prerequisite_topics=_str_list(payload.get("prerequisite_topics")),
        figure_heavy_topics=_str_list(payload.get("figure_heavy_topics")),
    )


def _summary_tree(payloads: list[dict[str, Any]]) -> BookSummaryTreeSummary:
    book_nodes: list[BookSummaryTreeNode] = []
    chapter_nodes: list[BookSummaryTreeNode] = []
    section_count = 0
    for payload in payloads:
        level = payload.get("level")
        if level == "section":
            section_count += 1
            continue
        if level not in {"book", "chapter"}:
            continue
        node = BookSummaryTreeNode(
            display_path=_optional_str(payload.get("display_path")),
            key_points=_summary_key_points(payload),
        )
        if level == "book":
            book_nodes.append(node)
        else:
            chapter_nodes.append(node)
    chapter_nodes.sort(key=lambda node: node.display_path or "")
    return BookSummaryTreeSummary(
        book=book_nodes, chapter=chapter_nodes, section_count=section_count
    )


def _summary_key_points(payload: dict[str, Any]) -> list[str]:
    metadata = _as_dict(payload.get("metadata"))
    summary_extract = _as_dict(metadata.get("summary_extract"))
    return _str_list(summary_extract.get("key_points"))


def _graph_summary(node_count: int, payloads: list[dict[str, Any]]) -> BookGraphSummary:
    nodes = [
        BookGraphNodeSummary(
            label=str(payload.get("label") or ""),
            node_type=str(payload.get("node_type") or ""),
            mentions=len(chunk_ids)
            if isinstance((chunk_ids := payload.get("chunk_ids")), list)
            else 0,
        )
        for payload in payloads
    ]
    nodes.sort(key=lambda node: (-node.mentions, node.label))
    return BookGraphSummary(node_count=node_count, top_nodes=nodes[:10])


def _flatten_value_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and isinstance(item.get("value"), str):
            out.append(item["value"])
    return out


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _row_get(row: Any | None, key: str) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, TypeError):
        return getattr(row, key, None)


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _int_or_zero(value: Any) -> int:
    return _optional_int(value) or 0
