from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum

from pydantic import Field

from vo_rag.core.models.common import VoBaseModel


class GraphNodeType(StrEnum):
    BOOK = "book"
    CONCEPT = "concept"
    METHOD = "method"
    EQUATION = "equation"
    PERSON = "person"
    COMPONENT = "component"


class GraphEdgeType(StrEnum):
    MENTIONS = "mentions"
    AUTHORED_BY = "authored_by"
    RELATES_TO = "relates_to"
    USED_BY = "used_by"
    PART_OF = "part_of"
    IS_A = "is_a"
    INTRODUCES = "introduces"


class GraphNode(VoBaseModel):
    node_id: str
    node_type: GraphNodeType
    label: str
    aliases: list[str] = Field(default_factory=list)
    book_ids: list[str]
    chunk_ids: list[str]
    attributes: dict[str, str] = Field(default_factory=dict)
    created_at: datetime
    prompt_hash: str | None = None


class GraphEdge(VoBaseModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: GraphEdgeType
    book_ids: list[str]
    weight: float = 1.0
    evidence_chunk_ids: list[str]
    created_at: datetime
    prompt_hash: str | None = None


def node_id_for(*, node_type: GraphNodeType, canonical_label: str) -> str:
    """Return a deterministic node ID for (node_type, canonical_label).

    The ID is the first 32 hex characters of the SHA-256 hash of the
    lowercased, stripped canonical label prefixed by the node type value.
    """
    return hashlib.sha256(
        (node_type.value + "::" + canonical_label.lower().strip()).encode()
    ).hexdigest()[:32]


def edge_id_for(*, source_node_id: str, target_node_id: str, edge_type: GraphEdgeType) -> str:
    """Return a deterministic edge ID for (source, target, edge_type).

    Edges are directed; the source-to-target order is preserved as given.
    The ID is the first 32 hex characters of the SHA-256 hash of the
    source ID, a directed-arrow separator, the target ID, and the edge type.
    """
    return hashlib.sha256(
        (source_node_id + "→" + target_node_id + "::" + edge_type.value).encode()
    ).hexdigest()[:32]


class GraphSmokeTestConfig(VoBaseModel):
    min_node_count_pilot: int = 100
    min_edge_count_pilot: int = 200
    judge_min_improvement_pp: float = 1.0
    metrics_to_test: list[str] = Field(
        default_factory=lambda: ["recall_at_10", "ndcg_at_10", "answer_quality_judge"]
    )


__all__ = [
    "GraphNodeType",
    "GraphEdgeType",
    "GraphNode",
    "GraphEdge",
    "node_id_for",
    "edge_id_for",
    "GraphSmokeTestConfig",
]
