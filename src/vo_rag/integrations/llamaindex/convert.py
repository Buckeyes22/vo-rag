"""LlamaIndex <-> vo-rag type converters.

This module is the only *genuinely new* code in the adapter
layer: it maps vo-rag schema objects (``RetrievalChunk``,
``SearchResult``, ``GraphNode``, ``GraphEdge``) to the LlamaIndex
schema objects (``TextNode``, ``NodeWithScore``, ``EntityNode``,
``Relation``) the rest of the adapter tier consumes.

The load-bearing invariant is **ID fidelity** — a ``TextNode`` derived
from a ``RetrievalChunk`` must carry that chunk's ``chunk_id`` as its
``node.id_``, or LlamaIndex's index/storage layer will re-key the node
and break vo-rag's deterministic-write idempotency. The contract tests
pin this behavior.

Implemented surface
-------------------
Chunk and search-result converters live in this module. Graph converters
delegate to the property-graph adapter so the public ``__all__`` surface
matches the lazy package re-exports without carrying duplicate mapping code.
"""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core.schema import MetadataMode as MetadataMode
from llama_index.core.schema import NodeWithScore as NodeWithScore
from llama_index.core.schema import TextNode

from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.retrieval.search import SearchResult as SearchResult

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkIdMissingError",
    "chunk_to_text_node",
    "text_node_to_chunk",
    "search_result_to_node_with_score",
    "node_with_score_to_search_result",
    "graph_node_to_entity_node",
    "graph_edge_to_relation",
]


class ChunkIdMissingError(ValueError):
    """A Qdrant hit lacked the indexed ``chunk_id`` payload field.

    Raised by ``search_result_to_node_with_score`` instead of silently keying
    ``node.id_`` on the Qdrant point id (``result.id``). The retriever adapter
    catches this, logs a warning, and drops the hit (quarantine) so one corrupt
    point cannot fail a whole query.
    """


# Metadata keys we project onto the TextNode. section_path is the embed-only
# prefix that reproduces RetrievalChunk.embedding_text; everything else is
# carried for filtering / citation but kept OUT of the embed + LLM text.
_EMBED_PREFIX_KEY = "section_path"
_VO_RAG_CHUNK_METADATA_KEY = "_vo_rag_chunk_metadata"
_VO_RAG_EMBEDDING_TEXT_KEY = "_vo_rag_embedding_text"
_CARRIED_KEYS = (
    "chunk_type",
    "document_id",
    "source_id",
    "chapter_path",
    "section_title",
    "page_number",
    "page_range",
    "token_count",
    "has_image",
    "image_id",
    "image_ref",
    "vlm_description",
)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _embedding_text_from(section_path: str, content: str) -> str:
    return f"{section_path}\n\n{content}" if section_path else content


def chunk_to_text_node(chunk: RetrievalChunk) -> TextNode:
    """Convert a vo-rag RetrievalChunk into a LlamaIndex TextNode.

    ID FIDELITY: node.id_ MUST equal chunk.chunk_id. vo-rag idempotency
    depends on this round-tripping unchanged.

    EMBED/LLM SPLIT: RetrievalChunk.embedding_text == section_path + "\n\n" +
    content. We reproduce that split *without* duplicating text: node.text
    holds the bare content; section_path rides in metadata and is included in
    the EMBED view but excluded from the LLM view via MetadataMode.
    """
    metadata: dict[str, Any] = {_EMBED_PREFIX_KEY: chunk.section_path}
    for key in _CARRIED_KEYS:
        metadata[key] = getattr(chunk, key)
    if chunk.metadata:
        metadata[_VO_RAG_CHUNK_METADATA_KEY] = dict(chunk.metadata)

    reconstructed_embedding_text = _embedding_text_from(chunk.section_path, chunk.content)
    if chunk.embedding_text != reconstructed_embedding_text:
        metadata[_VO_RAG_EMBEDDING_TEXT_KEY] = chunk.embedding_text

    node = TextNode(text=chunk.content, id_=chunk.chunk_id, metadata=metadata)

    # The embed model SEES section_path (reproduces embedding_text); the LLM
    # does NOT (it gets clean content + whatever the synthesizer adds).
    node.excluded_embed_metadata_keys = [k for k in metadata if k != _EMBED_PREFIX_KEY]
    node.excluded_llm_metadata_keys = list(metadata.keys())
    return node


def text_node_to_chunk(node: TextNode) -> RetrievalChunk:
    """Inverse of chunk_to_text_node.

    chunk_id is recovered from node.id_ — NOT recomputed — so identity holds
    even if the content was normalized differently downstream.
    """
    md = node.metadata
    section_path = str(md.get(_EMBED_PREFIX_KEY, ""))
    content = node.text
    raw_chunk_metadata = md.get(_VO_RAG_CHUNK_METADATA_KEY)
    if raw_chunk_metadata is None:
        chunk_metadata: dict[str, Any] = {}
    elif isinstance(raw_chunk_metadata, dict):
        chunk_metadata = dict(raw_chunk_metadata)
    else:
        logger.warning(
            "llamaindex_text_node_invalid_vo_rag_chunk_metadata",
            extra={"metadata_type": type(raw_chunk_metadata).__name__},
        )
        raise ValueError("_vo_rag_chunk_metadata must be a mapping when present")
    embedding_text = (
        str(md[_VO_RAG_EMBEDDING_TEXT_KEY])
        if _VO_RAG_EMBEDDING_TEXT_KEY in md
        else _embedding_text_from(section_path, content)
    )
    return RetrievalChunk(
        chunk_id=node.id_,
        chunk_type=str(md.get("chunk_type", "text")),
        document_id=str(md.get("document_id", "")),
        source_id=str(md.get("source_id", "")),
        page_number=_optional_int(md.get("page_number")),
        page_range=_optional_str(md.get("page_range")),
        section_path=section_path,
        section_title=str(md.get("section_title", "")),
        chapter_path=str(md.get("chapter_path", "")),
        content=content,
        embedding_text=embedding_text,
        token_count=int(md.get("token_count", 0) or 0),
        has_image=bool(md.get("has_image", False)),
        image_id=_optional_str(md.get("image_id")),
        image_ref=_optional_str(md.get("image_ref")),
        vlm_description=_optional_str(md.get("vlm_description")),
        metadata=chunk_metadata,
    )


def search_result_to_node_with_score(result: SearchResult) -> NodeWithScore:
    """Wrap a vo-rag SearchResult as a LlamaIndex NodeWithScore.

    ``result.id`` is the Qdrant point id, so it is stored as
    ``metadata["qdrant_point_id"]`` while ``node.id_`` stays on the logical
    content-addressed ``payload["chunk_id"]`` used for citation and dedup.
    """
    payload = result.payload
    raw_chunk_id = payload.get("chunk_id")
    if not raw_chunk_id:
        raise ChunkIdMissingError(
            f"SearchResult id={result.id!r} (collection={result.collection!r}) "
            "has no payload['chunk_id']; refusing to key node.id_ on the Qdrant "
            "point id (would reintroduce the content-vs-index id mismatch)."
        )

    chunk_id = str(raw_chunk_id)
    content = str(payload.get("content", payload.get("text", "")))
    section_path = str(payload.get("section_path", ""))

    metadata: dict[str, Any] = dict(payload)
    metadata[_EMBED_PREFIX_KEY] = section_path
    for key in _CARRIED_KEYS:
        if key in payload:
            metadata[key] = payload[key]
    metadata["collection"] = result.collection
    metadata["qdrant_point_id"] = result.id

    node = TextNode(text=content, id_=chunk_id, metadata=metadata)
    node.excluded_embed_metadata_keys = [k for k in metadata if k != _EMBED_PREFIX_KEY]
    node.excluded_llm_metadata_keys = list(metadata.keys())
    return NodeWithScore(node=node, score=result.score)


def node_with_score_to_search_result(nws: NodeWithScore) -> SearchResult:
    """Rebuild a vo-rag SearchResult from a LlamaIndex NodeWithScore."""
    node = nws.node
    metadata = dict(node.metadata)
    qdrant_point_id = str(metadata.get("qdrant_point_id") or node.id_)
    collection = str(metadata.get("collection", ""))
    payload = dict(metadata)
    payload["chunk_id"] = node.id_
    payload["content"] = node.get_content(metadata_mode=MetadataMode.NONE)
    return SearchResult(
        id=qdrant_point_id,
        score=nws.score if nws.score is not None else 0.0,
        collection=collection,
        payload=payload,
    )


def graph_node_to_entity_node(node: Any) -> Any:  # vo_rag GraphNode -> li EntityNode
    """Map GraphNode to LlamaIndex EntityNode via the property-graph adapter."""
    from vo_rag.integrations.llamaindex.graph_store import _graphnode_to_entity

    return _graphnode_to_entity(node)


def graph_edge_to_relation(edge: Any) -> Any:  # vo_rag GraphEdge -> li Relation
    """Map GraphEdge to LlamaIndex Relation via the property-graph adapter."""
    from vo_rag.integrations.llamaindex.graph_store import _edge_to_relation

    return _edge_to_relation(edge)
