from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

import structlog
from pydantic import Field

from vo_rag.core.models.common import VoBaseModel
from vo_rag.core.storage.qdrant import deterministic_point_id

log = structlog.get_logger(__name__)


class ClusterLevel(StrEnum):
    LEVEL_0 = "level_0"
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"


class ClusterReproducibilityConfig(VoBaseModel):
    # Frozen clustering controls: fixed so a RAPTOR build is reproducible.
    umap_n_components: int = 5
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    umap_random_state: int
    gmm_max_components: int = 50
    gmm_covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    gmm_random_state: int
    gmm_cluster_selection_criterion: Literal["bic", "aic"] = "bic"
    min_cluster_size: int = 5
    max_cluster_size: int = 100
    chunk_overlap_policy: Literal["hard_assignment", "soft_assignment"] = "hard_assignment"
    summary_temperature: float = 0.2
    summary_max_tokens: int = 800


class ClusterNode(VoBaseModel):
    # Deterministic semantic cluster identity.
    node_id: str
    book_id: str
    level: ClusterLevel
    cluster_index_within_level: int
    parent_node_id: str | None
    child_node_ids: list[str] = Field(default_factory=list)

    # Raw chunk provenance and direct cluster structure.
    source_chunk_ids: list[str]
    direct_child_chunk_ids: list[str]

    # Extract-Support evidence from raw chunks only.
    evidence_spans: list[dict[str, str | float]] = Field(default_factory=list)

    # Clustering diagnostics.
    cluster_centroid_umap: list[float] = Field(default_factory=list)
    gmm_cluster_probability: float = 1.0

    # Summary content.
    summary_text: str
    summary_model: str
    prompt_hash: str
    prompt_version: str

    # Embedding metadata. BGE-M3 dense vectors are 1024d per vo_rag.embedding.schemas.
    embedding_model: str
    embedding_version: str

    # Frozen clustering controls used for this node.
    umap_random_state: int
    gmm_random_state: int
    gmm_cluster_selection_criterion: str

    # Audit.
    created_at: datetime
    source_collection_version: str

    metadata: dict[str, Any] = Field(default_factory=dict)


def cluster_node_id_for(
    book_id: str,
    level: ClusterLevel,
    member_chunk_ids: Iterable[str],
    prompt_hash: str,
) -> str:
    """Return the deterministic Layer 1B semantic-cluster node ID.

    The Qdrant point-id helper accepts an integer chunk index. Cluster members
    are sorted, joined, SHA-256 hashed, and truncated to a stable 64-bit integer
    so retries produce byte-identical node IDs regardless of input order.
    """

    sorted_member_chunk_ids = sorted(member_chunk_ids)
    if not sorted_member_chunk_ids:
        raise ValueError("member_chunk_ids must not be empty")

    chunk_set_digest = hashlib.sha256(
        "\n".join(sorted_member_chunk_ids).encode("utf-8")
    ).hexdigest()
    chunk_index = int(chunk_set_digest[:16], 16)
    return deterministic_point_id(
        source_id=book_id,
        chunk_index=chunk_index,
        stage=f"raptor_{level.value}",
        version_hash=prompt_hash,
    )
