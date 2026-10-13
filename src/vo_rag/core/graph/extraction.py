from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import Field

from vo_rag.core.chunking import RetrievalChunk
from vo_rag.core.graph.prompts import (
    GRAPH_EXTRACTION_SYSTEM_PROMPT,
    GRAPH_EXTRACTION_USER_TEMPLATE,
)
from vo_rag.core.graph.schemas import (
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    edge_id_for,
    node_id_for,
)
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.common import VoBaseModel

log = structlog.get_logger(__name__)

# Cap chunk content fed into the extraction prompt. Graph extraction needs the
# key terms/relations, not the full prose — full chunks (up to ~2.6k chars each)
# x30 bloated the prompt and, with an oversized max_tokens, made each call take
# 8-18 min. Mirrors DSI's 600-char cap.
_CHUNK_CONTENT_MAX_CHARS = 600


class GraphExtractionConfig(VoBaseModel):
    summary_model: str = "qwen3.6-27b-fp8"
    temperature: float = 0.2
    # Per-call output scales with node/edge COUNT (hence chunk count + density),
    # NOT with chunk char-size. Measured on the pilot (6-chunk batches): a 2500
    # cap truncated ~40% of batches mid-edge (→ 0 nodes, silent data loss); 3500
    # still left a ~3% tail (one batch wanted >3500); 4000 covers the observed
    # max (3163) plus that tail with headroom, still well under the wasteful 6000
    # we started from. (output is bounded by node count, not the 600-char chunk.)
    max_tokens: int = 4000
    prompt_hash: str
    max_chunks_per_call: int = 30


class ExtractedNodeSpec(VoBaseModel):
    node_type: GraphNodeType
    label: str
    aliases: list[str] = Field(default_factory=list)
    chunk_ids: list[str]


class ExtractedEdgeSpec(VoBaseModel):
    source_label: str
    source_type: GraphNodeType
    target_label: str
    target_type: GraphNodeType
    edge_type: GraphEdgeType
    evidence_chunk_ids: list[str]


class GraphExtractionResult(VoBaseModel):
    book_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    validation_errors: list[str]
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    wall_time_ms: int


# Node types the MODEL is allowed to emit. GraphNodeType also has BOOK, but the
# book node is synthesized by the pipeline, not extracted — so it's excluded from
# the guided-decoding enum to stop the model minting duplicate book nodes.
_EXTRACTABLE_NODE_TYPES: tuple[GraphNodeType, ...] = (
    GraphNodeType.CONCEPT,
    GraphNodeType.METHOD,
    GraphNodeType.EQUATION,
    GraphNodeType.PERSON,
    GraphNodeType.COMPONENT,
)


def graph_response_json_schema() -> dict[str, Any]:
    """JSON Schema for the extraction response, passed to vLLM as ``guided_json``.

    The serverless worker (xgrammar backend) constrains generation to this shape
    token-by-token, so the model *cannot* emit markdown fences, preamble, invalid
    node_type/edge_type enums, or chunk refs outside the compact ``cN`` labels. This
    uses structured output (``response_format`` + ``schema``)
    and removes the malformed-JSON failure mode at the source rather than scrubbing
    it after the fact. (xgrammar can't prevent *length* truncation, so max_tokens
    still matters — see GraphExtractionConfig.)
    """
    node_types = [t.value for t in _EXTRACTABLE_NODE_TYPES]
    edge_types = [t.value for t in GraphEdgeType]
    chunk_ref = {"type": "string", "pattern": "^c[0-9]+$"}
    return {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "node_type": {"type": "string", "enum": node_types},
                        "label": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "chunk_ids": {"type": "array", "items": chunk_ref},
                    },
                    "required": ["node_type", "label", "chunk_ids"],
                },
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_label": {"type": "string"},
                        "source_type": {"type": "string", "enum": node_types},
                        "target_label": {"type": "string"},
                        "target_type": {"type": "string", "enum": node_types},
                        "edge_type": {"type": "string", "enum": edge_types},
                        "evidence_chunk_ids": {"type": "array", "items": chunk_ref},
                    },
                    "required": [
                        "source_label",
                        "source_type",
                        "target_label",
                        "target_type",
                        "edge_type",
                        "evidence_chunk_ids",
                    ],
                },
            },
        },
        "required": ["nodes", "edges"],
    }


def build_extraction_prompt(
    *, book_id: str, title: str, author: str, chunks: list[RetrievalChunk]
) -> tuple[str, str, dict[str, str]]:
    """Build the (system, user) extraction prompt plus a ``chunk_index_map``.

    Chunks are labeled with compact batch-local ids (``[c0]``, ``[c1]``, …) rather
    than their 36-char UUIDs: the model only has to echo ``c0`` in its node/edge
    chunk references, which collapses output the slow 27B endpoint must decode.
    The returned map (``{"c0": <real chunk_id>, …}``) is the single source of
    truth for translating those labels back to real ids in ``parse_graph_response``.
    """
    chunk_parts: list[str] = []
    chunk_index_map: dict[str, str] = {}
    for i, c in enumerate(chunks):
        label = f"c{i}"
        chunk_index_map[label] = c.chunk_id
        section = c.section_path if c.section_path else ""
        content = c.content
        if len(content) > _CHUNK_CONTENT_MAX_CHARS:
            content = content[:_CHUNK_CONTENT_MAX_CHARS] + "…"
        chunk_parts.append(f"[{label}] {section}\n{content}")

    chunk_contents = "\n\n---\n\n".join(chunk_parts)
    user = GRAPH_EXTRACTION_USER_TEMPLATE.format(
        title=title,
        author=author,
        chunk_contents=chunk_contents,
    )
    return GRAPH_EXTRACTION_SYSTEM_PROMPT, user, chunk_index_map


def _extract_json_object(text: str) -> str:
    """Pull a JSON object out of an LLM response that may wrap it in markdown
    code fences or surrounding prose/reasoning, despite the prompt asking for raw
    JSON. Strips ```/```json fences, then returns the substring from the first
    ``{`` to the last ``}``. Falls back to the stripped text. (qwen3-6-27b often
    fences/preambles its output — a raw json.loads then loses the whole batch.)
    """
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def parse_graph_response(
    *,
    book_id: str,
    llm_response_json: str,
    chunk_index_map: dict[str, str],
    prompt_hash: str,
) -> GraphExtractionResult:
    """Parse + validate the LLM graph JSON.

    ``chunk_index_map`` maps the compact labels the model was shown (``c0``, ``c1``,
    …) to the real chunk_ids. Every chunk reference in the response is validated
    against the map's keys and then translated to the real id before being stored,
    so persisted nodes/edges carry real chunk provenance, not the wire labels.
    """
    nodes_out: list[GraphNode] = []
    edges_out: list[ExtractedEdgeSpec] = []
    validation_errors: list[str] = []

    try:
        data: dict[str, Any] = json.loads(_extract_json_object(llm_response_json))
    except json.JSONDecodeError as exc:
        validation_errors.append(f"JSON decode error: {exc}")
        return GraphExtractionResult(
            book_id=book_id,
            nodes=[],
            edges=[],
            validation_errors=validation_errors,
            prompt_hash=prompt_hash,
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    if not isinstance(data, dict):
        validation_errors.append("Response root must be a JSON object")
        return GraphExtractionResult(
            book_id=book_id,
            nodes=[],
            edges=[],
            validation_errors=validation_errors,
            prompt_hash=prompt_hash,
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])

    if not isinstance(raw_nodes, list):
        validation_errors.append("nodes must be an array")
        return GraphExtractionResult(
            book_id=book_id,
            nodes=[],
            edges=[],
            validation_errors=validation_errors,
            prompt_hash=prompt_hash,
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    valid_node_specs: list[ExtractedNodeSpec] = []
    extracted_labels: set[str] = set()

    for i, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            validation_errors.append(f"nodes[{i}]: must be an object")
            continue
        node_type_val = raw.get("node_type", "")
        label_val = raw.get("label", "")
        aliases_val = raw.get("aliases", [])
        chunk_ids_val = raw.get("chunk_ids", [])

        if not label_val or not isinstance(label_val, str):
            validation_errors.append(f"nodes[{i}]: label missing or not a string")
            continue

        try:
            node_type = GraphNodeType(node_type_val)
        except (ValueError, TypeError):
            validation_errors.append(f"nodes[{i}]: invalid node_type '{node_type_val}'")
            continue

        if not isinstance(chunk_ids_val, list):
            validation_errors.append(f"nodes[{i}]: chunk_ids must be an array")
            continue

        bad_chunk_ids = [cid for cid in chunk_ids_val if cid not in chunk_index_map]
        if bad_chunk_ids:
            validation_errors.append(f"nodes[{i}]: unknown chunk_ids {bad_chunk_ids}")
            continue
        mapped_chunk_ids = [chunk_index_map[cid] for cid in chunk_ids_val]

        if not isinstance(aliases_val, list):
            aliases_val = []

        valid_node_specs.append(
            ExtractedNodeSpec(
                node_type=node_type,
                label=label_val,
                aliases=aliases_val,
                chunk_ids=mapped_chunk_ids,
            )
        )
        extracted_labels.add(label_val)

    raw_edges = data.get("edges", [])
    if not isinstance(raw_edges, list):
        validation_errors.append("edges must be an array")
        return GraphExtractionResult(
            book_id=book_id,
            nodes=[],
            edges=[],
            validation_errors=validation_errors,
            prompt_hash=prompt_hash,
            input_tokens=0,
            output_tokens=0,
            wall_time_ms=0,
        )

    for i, raw in enumerate(raw_edges):
        if not isinstance(raw, dict):
            validation_errors.append(f"edges[{i}]: must be an object")
            continue

        source_label = raw.get("source_label", "")
        source_type_val = raw.get("source_type", "")
        target_label = raw.get("target_label", "")
        target_type_val = raw.get("target_type", "")
        edge_type_val = raw.get("edge_type", "")
        evidence_chunk_ids = raw.get("evidence_chunk_ids", [])

        if source_label not in extracted_labels:
            validation_errors.append(
                f"edges[{i}]: source_label '{source_label}' not in extracted nodes"
            )
            continue

        if target_label not in extracted_labels:
            validation_errors.append(
                f"edges[{i}]: target_label '{target_label}' not in extracted nodes"
            )
            continue

        if not isinstance(evidence_chunk_ids, list):
            validation_errors.append(f"edges[{i}]: evidence_chunk_ids must be an array")
            continue

        bad_chunk_ids = [cid for cid in evidence_chunk_ids if cid not in chunk_index_map]
        if bad_chunk_ids:
            validation_errors.append(f"edges[{i}]: unknown evidence_chunk_ids {bad_chunk_ids}")
            continue
        mapped_evidence_chunk_ids = [chunk_index_map[cid] for cid in evidence_chunk_ids]

        try:
            source_type = GraphNodeType(source_type_val)
        except (ValueError, TypeError):
            validation_errors.append(f"edges[{i}]: invalid source_type '{source_type_val}'")
            continue

        try:
            target_type = GraphNodeType(target_type_val)
        except (ValueError, TypeError):
            validation_errors.append(f"edges[{i}]: invalid target_type '{target_type_val}'")
            continue

        try:
            edge_type = GraphEdgeType(edge_type_val)
        except (ValueError, TypeError):
            validation_errors.append(f"edges[{i}]: invalid edge_type '{edge_type_val}'")
            continue

        edges_out.append(
            ExtractedEdgeSpec(
                source_label=source_label,
                source_type=source_type,
                target_label=target_label,
                target_type=target_type,
                edge_type=edge_type,
                evidence_chunk_ids=mapped_evidence_chunk_ids,
            )
        )

    for spec in valid_node_specs:
        n_id = node_id_for(node_type=spec.node_type, canonical_label=spec.label)
        nodes_out.append(
            GraphNode(
                node_id=n_id,
                node_type=spec.node_type,
                label=spec.label,
                aliases=spec.aliases,
                book_ids=[book_id],
                chunk_ids=spec.chunk_ids,
                created_at=datetime.now(UTC),
                prompt_hash=prompt_hash,
            )
        )

    edges_resolved: list[GraphEdge] = []
    for edge_spec in edges_out:
        src_id = node_id_for(
            node_type=edge_spec.source_type, canonical_label=edge_spec.source_label
        )
        tgt_id = node_id_for(
            node_type=edge_spec.target_type, canonical_label=edge_spec.target_label
        )
        e_id = edge_id_for(
            source_node_id=src_id,
            target_node_id=tgt_id,
            edge_type=edge_spec.edge_type,
        )
        edges_resolved.append(
            GraphEdge(
                edge_id=e_id,
                source_node_id=src_id,
                target_node_id=tgt_id,
                edge_type=edge_spec.edge_type,
                book_ids=[book_id],
                evidence_chunk_ids=edge_spec.evidence_chunk_ids,
                created_at=datetime.now(UTC),
                prompt_hash=prompt_hash,
            )
        )

    return GraphExtractionResult(
        book_id=book_id,
        nodes=nodes_out,
        edges=edges_resolved,
        validation_errors=validation_errors,
        prompt_hash=prompt_hash,
        input_tokens=0,
        output_tokens=0,
        wall_time_ms=0,
    )


async def extract_graph_from_book(
    *,
    book_id: str,
    title: str,
    author: str,
    chunks: list[RetrievalChunk],
    config: GraphExtractionConfig,
    llm_call: Callable[[str, str, int, float], Awaitable[LLMResponse]],
) -> GraphExtractionResult:
    t0 = time.monotonic()

    system, user, chunk_index_map = build_extraction_prompt(
        book_id=book_id,
        title=title,
        author=author,
        chunks=chunks,
    )

    llm_response: LLMResponse = await llm_call(
        system,
        user,
        config.max_tokens,
        config.temperature,
    )

    wall_time_ms = int((time.monotonic() - t0) * 1000)

    result = parse_graph_response(
        book_id=book_id,
        llm_response_json=llm_response.text,
        chunk_index_map=chunk_index_map,
        prompt_hash=config.prompt_hash,
    )

    result.input_tokens = llm_response.input_tokens
    result.output_tokens = llm_response.output_tokens
    result.wall_time_ms = wall_time_ms

    log.info(
        "graph.extract_book.complete",
        book_id=book_id,
        node_count=len(result.nodes),
        edge_count=len(result.edges),
        validation_error_count=len(result.validation_errors),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        wall_time_ms=wall_time_ms,
    )

    return result


__all__ = [
    "GraphExtractionConfig",
    "ExtractedNodeSpec",
    "ExtractedEdgeSpec",
    "GraphExtractionResult",
    "graph_response_json_schema",
    "build_extraction_prompt",
    "parse_graph_response",
    "extract_graph_from_book",
]
