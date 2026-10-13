"""Optional LlamaIndex node postprocessors for vo-rag.

Both are config-gated and default OFF so they never change current behaviour
unless explicitly enabled.

- ``DiversityPostprocessor`` — MMR re-ranking over the post-reranker top-k.
- ``DomainTagFilterPostprocessor`` — drop nodes whose payload ``domain_tags``
  do not intersect a requested allowlist.

These are pure post-processing: they operate on the already-retrieved
``NodeWithScore`` list, make no new LLM or service calls, and preserve
quarantine/trust semantics by never stripping metadata.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vo_rag.integrations.llamaindex import import_llamaindex_symbol

if TYPE_CHECKING:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
    from pydantic import PrivateAttr
else:
    BaseNodePostprocessor = import_llamaindex_symbol(
        "llama_index.core.postprocessor.types", "BaseNodePostprocessor"
    )
    PrivateAttr = import_llamaindex_symbol("pydantic", "PrivateAttr")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Token-overlap Jaccard similarity (case-insensitive)."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union


def _node_similarity(a: NodeWithScore, b: NodeWithScore) -> float:
    """Similarity between two nodes.

    Uses embeddings when present on both nodes; falls back to lexical
    (token-overlap Jaccard) otherwise.
    """
    emb_a = a.node.embedding
    emb_b = b.node.embedding
    if emb_a is not None and emb_b is not None:
        return _cosine_similarity(emb_a, emb_b)
    return _jaccard_similarity(a.node.get_content() or "", b.node.get_content() or "")


class DiversityPostprocessor(BaseNodePostprocessor):
    """MMR-style diversity re-ranking of retrieved nodes.

    Re-ranks the incoming node list for maximal marginal relevance:
    each selected node maximises ``lambda_mult * relevance - (1 -
    lambda_mult) * max_similarity_to_already_selected``.  This penalises
    near-duplicate chunks (e.g. consecutive chunks from the same section).

    Parameters
    ----------
    enabled:
        When ``False`` (default) the postprocessor is a pass-through no-op.
    lambda_mult:
        Trade-off between relevance and diversity.  ``0.0`` = pure diversity,
        ``1.0`` = pure relevance (original ordering).  Default ``0.5``.
    top_k:
        Maximum number of nodes to return.  Default ``5``.

    Determinism
    -----------
    Ties are broken by ``chunk_id`` (``node.id_``) lexicographically.  Deterministic
    content-addressed ids guarantee stable ordering.
    """

    enabled: bool = False
    lambda_mult: float = 0.5
    top_k: int = 5

    _similarity_fn = PrivateAttr()

    def __init__(
        self,
        *,
        enabled: bool = False,
        lambda_mult: float = 0.5,
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.lambda_mult = lambda_mult
        self.top_k = top_k
        self._similarity_fn = _node_similarity

    @classmethod
    def class_name(cls) -> str:
        return "DiversityPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not self.enabled or not nodes:
            return list(nodes)

        if self.lambda_mult < 0.0 or self.lambda_mult > 1.0:
            raise ValueError("lambda_mult must be in [0.0, 1.0]")

        remaining = list(nodes)
        selected: list[NodeWithScore] = []

        while remaining and len(selected) < self.top_k:

            def _mmr_score(n: NodeWithScore) -> tuple[float, str]:
                rel = n.score if n.score is not None else 0.0
                if selected:
                    max_sim = max(self._similarity_fn(n, s) for s in selected)
                    score = self.lambda_mult * rel - (1.0 - self.lambda_mult) * max_sim
                else:
                    # Seed: when nothing is selected yet, diversity term is
                    # zero so we fall back to pure relevance (standard MMR).
                    score = rel
                # Deterministic tie-breaker: the content-addressed chunk_id (stable).
                return (score, n.node.id_)

            best = max(remaining, key=_mmr_score)
            selected.append(best)
            remaining.remove(best)

        return selected


class DomainTagFilterPostprocessor(BaseNodePostprocessor):
    """Filter nodes by payload ``domain_tags``.

    Drops nodes whose ``domain_tags`` do not intersect the requested
    ``allowed_tags`` allowlist.  A node missing ``domain_tags`` is dropped
    only when a non-empty allowlist is set **and** ``drop_untagged=True``.

    Parameters
    ----------
    enabled:
        When ``False`` (default) the postprocessor is a pass-through no-op.
    allowed_tags:
        Allowlist of domain tags.  Empty/absent (default) means pass-through
        (no filtering).
    drop_untagged:
        If ``True``, nodes that carry no ``domain_tags`` metadata are dropped
        when ``allowed_tags`` is non-empty.  Default ``False`` (untagged nodes
        pass).
    """

    enabled: bool = False
    allowed_tags: list[str] = []
    drop_untagged: bool = False

    def __init__(
        self,
        *,
        enabled: bool = False,
        allowed_tags: list[str] | None = None,
        drop_untagged: bool = False,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.allowed_tags = list(allowed_tags) if allowed_tags is not None else []
        self.drop_untagged = drop_untagged

    @classmethod
    def class_name(cls) -> str:
        return "DomainTagFilterPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not self.enabled or not nodes:
            return list(nodes)

        if not self.allowed_tags:
            return list(nodes)

        allowset = set(self.allowed_tags)
        out: list[NodeWithScore] = []
        for n in nodes:
            node_tags = n.node.metadata.get("domain_tags")
            if node_tags is None:
                if not self.drop_untagged:
                    out.append(n)
                continue
            if isinstance(node_tags, str):
                node_tags = [node_tags]
            tagset = set(node_tags)
            if tagset & allowset:
                out.append(n)
        return out
