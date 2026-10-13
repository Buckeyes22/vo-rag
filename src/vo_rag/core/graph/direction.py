from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, Final

import structlog
from pydantic import Field

from vo_rag.core.graph.schemas import GraphEdge, GraphEdgeType, GraphNode, GraphNodeType
from vo_rag.core.models.common import VoBaseModel

log = structlog.get_logger(__name__)

_DIRECTED_EDGE_TYPES: Final = {
    GraphEdgeType.PART_OF,
    GraphEdgeType.IS_A,
    GraphEdgeType.USED_BY,
}
_PARENT_SCORE_EDGE_TYPES: Final = {
    GraphEdgeType.PART_OF,
    GraphEdgeType.IS_A,
}
_TOOL_TYPES: Final = {
    GraphNodeType.COMPONENT,
    GraphNodeType.EQUATION,
    GraphNodeType.PERSON,
}
_PROCESS_TYPES: Final = {
    GraphNodeType.CONCEPT,
    GraphNodeType.METHOD,
}
_CATEGORY_HEAD_LABELS: Final = {
    "parts",
    "spare parts",
    "components",
    "accessories",
    "tools",
    "hardware",
    "fluids",
    "consumables",
}
_CATEGORY_HEAD_SUFFIXES: Final = {
    "system",
    "assembly",
    "kit",
}
_PROCESS_PHRASE_SUFFIXES: Final = {
    "adjustment",
    "analysis",
    "calibration",
    "compensation",
    "control",
    "estimation",
    "estimator",
    "filtering",
    "identification",
    "management",
    "measurement",
    "modeling",
    "observer",
    "optimization",
    "prediction",
    "prevention",
    "regulation",
    "setup",
    "technique",
    "testing",
    "tuning",
}
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+")
_LOW_CONFIDENCE_REORIENTATION_SYSTEM_PROMPT: Final = (
    "You are a technical graph edge direction reviewer. Return only valid JSON."
)


class DirectionCoverage(VoBaseModel):
    n_directed: int
    n_high: int
    n_medium: int
    n_low: int
    coverage: float
    low_confidence_edge_ids: list[str] = Field(default_factory=list)


def orient_edges(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> tuple[list[GraphEdge], DirectionCoverage]:
    """Orient deterministic directed graph edges and report coverage.

    Only ``part_of``, ``is_a``, and ``used_by`` participate in directional
    coverage. Other edge types are returned unchanged and are not counted.
    """
    node_map = {node.node_id: node for node in nodes}
    parent_scores = _parent_scores(edges)

    oriented_edges: list[GraphEdge] = []
    n_high = 0
    n_medium = 0
    n_low = 0
    low_confidence_edge_ids: list[str] = []

    for edge in edges:
        if edge.edge_type not in _DIRECTED_EDGE_TYPES:
            oriented_edges.append(edge)
            continue

        confidence, should_swap = _orientation_decision(
            edge=edge,
            node_map=node_map,
            parent_scores=parent_scores,
        )
        if confidence == "high":
            n_high += 1
        elif confidence == "medium":
            n_medium += 1
        else:
            n_low += 1
            low_confidence_edge_ids.append(edge.edge_id)

        if should_swap:
            oriented_edges.append(
                edge.model_copy(
                    update={
                        "source_node_id": edge.target_node_id,
                        "target_node_id": edge.source_node_id,
                    }
                )
            )
        else:
            oriented_edges.append(edge)

    n_directed = n_high + n_medium + n_low
    coverage = (n_high + n_medium) / n_directed if n_directed else 1.0
    return oriented_edges, DirectionCoverage(
        n_directed=n_directed,
        n_high=n_high,
        n_medium=n_medium,
        n_low=n_low,
        coverage=coverage,
        low_confidence_edge_ids=low_confidence_edge_ids,
    )


async def orient_low_confidence_via_llm(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    low_confidence_edge_ids: list[str],
    llm_call: Callable[[str, str, int, float], Awaitable[Any]],
) -> list[GraphEdge]:
    """Use an LLM to re-orient deterministic low-confidence directed edges.

    The function is deliberately side-effect free: it returns a full edge list,
    preserving order and leaving any unparseable or unmentioned edge unchanged.
    """
    low_confidence_ids = set(low_confidence_edge_ids)
    if not low_confidence_ids:
        return edges

    node_map = {node.node_id: node for node in nodes}
    indexed_edges: dict[str, tuple[int, GraphEdge, str, str]] = {}
    prompt_lines: list[str] = []

    for edge in edges:
        if edge.edge_id not in low_confidence_ids:
            continue
        source = node_map.get(edge.source_node_id)
        target = node_map.get(edge.target_node_id)
        if source is None or target is None:
            log.warning(
                "graph.direction.fallback_missing_node",
                edge_id=edge.edge_id,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
            )
            continue
        index = len(prompt_lines) + 1
        indexed_edges[edge.edge_id] = (index, edge, source.label, target.label)
        prompt_lines.append(
            f'{index}: "{source.label}"  --{edge.edge_type.value}--> "{target.label}"'
        )

    if not prompt_lines:
        return edges

    user_prompt = _low_confidence_reorientation_prompt(prompt_lines)
    response = await llm_call(
        _LOW_CONFIDENCE_REORIENTATION_SYSTEM_PROMPT,
        user_prompt,
        _low_confidence_reorientation_max_tokens(len(prompt_lines)),
        0.0,
    )
    items_by_index = _parse_reorientation_response(str(getattr(response, "text", "")))
    if items_by_index is None:
        return edges

    corrected_edges: list[GraphEdge] = []
    for edge in edges:
        indexed = indexed_edges.get(edge.edge_id)
        if indexed is None:
            corrected_edges.append(edge)
            continue

        index, _candidate_edge, source_label, target_label = indexed
        item = items_by_index.get(index)
        if item is None:
            log.warning(
                "graph.direction.fallback_missing_item",
                edge_id=edge.edge_id,
                index=index,
            )
            corrected_edges.append(edge)
            continue

        should_swap = _reorientation_item_requests_swap(
            item,
            current_source_label=source_label,
            current_target_label=target_label,
            edge_id=edge.edge_id,
            index=index,
        )
        if should_swap:
            corrected_edges.append(
                edge.model_copy(
                    update={
                        "source_node_id": edge.target_node_id,
                        "target_node_id": edge.source_node_id,
                    }
                )
            )
        else:
            corrected_edges.append(edge)

    return corrected_edges


def _low_confidence_reorientation_prompt(prompt_lines: list[str]) -> str:
    numbered_edges = "\n".join(prompt_lines)
    return f"""Review the direction of these low-confidence directed graph edges.

Relationship definitions:
- part_of: source is a part of target
- is_a: source is a kind/instance of target
- used_by: source is a tool used by target

For each numbered edge, choose the correct source and target labels. If the current
direction is reversed, either set "swap": true or echo the current target label as
"source_label".

Edges:
{numbered_edges}

Return only JSON in this shape:
{{"items": [{{"i": 1, "source_label": "correct source", "target_label": "correct target"}}]}}
"""


def _low_confidence_reorientation_max_tokens(n_edges: int) -> int:
    return max(512, n_edges * 96)


def _parse_reorientation_response(text: str) -> dict[int, dict[str, Any]] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning(
            "graph.direction.fallback_parse_failed",
            error=str(exc),
        )
        return None

    raw_items: Any
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        raw_items = data.get("items") or data.get("edges") or data.get("results")
    else:
        raw_items = None

    if not isinstance(raw_items, list):
        log.warning("graph.direction.fallback_parse_failed", error="missing items list")
        return None

    items_by_index: dict[int, dict[str, Any]] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        raw_index = raw_item.get("i", raw_item.get("index", raw_item.get("edge_index")))
        if raw_index is None:
            log.warning(
                "graph.direction.fallback_parse_item_failed",
                raw_index=raw_index,
            )
            continue
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            log.warning(
                "graph.direction.fallback_parse_item_failed",
                raw_index=raw_index,
            )
            continue
        items_by_index[index] = raw_item

    return items_by_index


def _reorientation_item_requests_swap(
    item: dict[str, Any],
    *,
    current_source_label: str,
    current_target_label: str,
    edge_id: str,
    index: int,
) -> bool:
    swap = item.get("swap")
    if isinstance(swap, bool):
        return swap

    source_label = _item_string(item, "source_label", "source", "correct_source")
    if source_label is None:
        log.warning(
            "graph.direction.fallback_item_missing_direction",
            edge_id=edge_id,
            index=index,
        )
        return False

    source_norm = _normalize_label(source_label)
    if source_norm == _normalize_label(current_target_label):
        return True
    if source_norm == _normalize_label(current_source_label):
        return False

    log.warning(
        "graph.direction.fallback_item_unknown_label",
        edge_id=edge_id,
        index=index,
        source_label=source_label,
    )
    return False


def _item_string(item: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _orientation_decision(
    *,
    edge: GraphEdge,
    node_map: dict[str, GraphNode],
    parent_scores: dict[str, int],
) -> tuple[str, bool]:
    source_node = node_map.get(edge.source_node_id)
    target_node = node_map.get(edge.target_node_id)

    if edge.edge_type in (GraphEdgeType.PART_OF, GraphEdgeType.IS_A):
        containment_signal = _containment_swap_signal(
            source_label=source_node.label if source_node is not None else "",
            target_label=target_node.label if target_node is not None else "",
        )
        if containment_signal is not None:
            return "high", containment_signal

        head_noun_signal = _head_noun_swap_signal(
            source_label=source_node.label if source_node is not None else "",
            target_label=target_node.label if target_node is not None else "",
        )
        if head_noun_signal is not None:
            return "high", head_noun_signal

        category_head_signal = _category_head_swap_signal(
            source_label=source_node.label if source_node is not None else "",
            target_label=target_node.label if target_node is not None else "",
        )
        if category_head_signal is not None:
            return "medium", category_head_signal

        source_parent_score = parent_scores.get(edge.source_node_id, 0)
        target_parent_score = parent_scores.get(edge.target_node_id, 0)
        if source_parent_score > target_parent_score:
            return "medium", True
        if source_parent_score < target_parent_score:
            return "medium", False
        return "low", False

    if edge.edge_type == GraphEdgeType.USED_BY:
        source_type = source_node.node_type if source_node is not None else None
        target_type = target_node.node_type if target_node is not None else None

        if source_type in _PROCESS_TYPES and target_type in _TOOL_TYPES:
            return "medium", True
        if source_type in _TOOL_TYPES and target_type in _PROCESS_TYPES:
            return "medium", False

        process_phrase_signal = _process_phrase_swap_signal(
            source_label=source_node.label if source_node is not None else "",
            target_label=target_node.label if target_node is not None else "",
        )
        if process_phrase_signal is not None:
            return "medium", process_phrase_signal

    return "low", False


def _containment_swap_signal(*, source_label: str, target_label: str) -> bool | None:
    source_norm = _normalize_label(source_label)
    target_norm = _normalize_label(target_label)
    if not source_norm or not target_norm or source_norm == target_norm:
        return None
    if source_norm in target_norm:
        return True
    if target_norm in source_norm:
        return False
    return None


def _head_noun_swap_signal(*, source_label: str, target_label: str) -> bool | None:
    source_tokens = _label_tokens(source_label)
    target_tokens = _label_tokens(target_label)
    if not source_tokens or not target_tokens:
        return None
    if source_tokens[-1] != target_tokens[-1]:
        return None
    if len(source_tokens) == len(target_tokens):
        return None
    return len(source_tokens) < len(target_tokens)


def _category_head_swap_signal(*, source_label: str, target_label: str) -> bool | None:
    source_is_category = _is_category_head(source_label)
    target_is_category = _is_category_head(target_label)
    if source_is_category == target_is_category:
        return None
    return source_is_category


def _process_phrase_swap_signal(*, source_label: str, target_label: str) -> bool | None:
    source_is_process = _is_process_phrase(source_label)
    target_is_process = _is_process_phrase(target_label)
    if source_is_process == target_is_process:
        return None
    return source_is_process


def _is_category_head(label: str) -> bool:
    label_norm = _normalize_label(label)
    if label_norm in _CATEGORY_HEAD_LABELS:
        return True
    tokens = _label_tokens(label)
    return bool(tokens and tokens[-1] in _CATEGORY_HEAD_SUFFIXES)


def _is_process_phrase(label: str) -> bool:
    label_norm = _normalize_label(label)
    if any(label_norm.endswith(suffix) for suffix in _PROCESS_PHRASE_SUFFIXES):
        return True
    tokens = _label_tokens(label)
    return bool(tokens and (tokens[-1].endswith("ing") or tokens[0].endswith("ing")))


def _label_tokens(label: str) -> list[str]:
    return _TOKEN_RE.findall(label.casefold())


def _normalize_label(label: str) -> str:
    return _WHITESPACE_RE.sub(" ", label.casefold().strip())


def _parent_scores(edges: list[GraphEdge]) -> dict[str, int]:
    parent_edge_keys: dict[str, set[tuple[str, str, GraphEdgeType]]] = defaultdict(set)
    for edge in edges:
        if edge.edge_type in _PARENT_SCORE_EDGE_TYPES:
            parent_edge_keys[edge.target_node_id].add(
                (edge.source_node_id, edge.target_node_id, edge.edge_type)
            )
    return {node_id: len(edge_keys) for node_id, edge_keys in parent_edge_keys.items()}


__all__ = ["DirectionCoverage", "orient_edges", "orient_low_confidence_via_llm"]
