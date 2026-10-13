"""Contract tests for optional node postprocessors.

Tests the two adopted pieces:

- ``DiversityPostprocessor`` — MMR diversity re-ranking.
- ``DomainTagFilterPostprocessor`` — domain-tag allowlist filtering.

The three REJECTED pieces (recency postprocessor, auto-merging, ingestion
 cache) are NOT in scope and must NOT be built.

Landmine: this module imports ``llama_index`` symbols at module top, so it
must be collected in the ``test-llamaindex`` CI job (or be silently dropped
by the light-job ``collect_ignore`` guard).  Do NOT add it to the light job.
"""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index")

from typing import Any

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from vo_rag.integrations.llamaindex.postprocessors import (
    DiversityPostprocessor,
    DomainTagFilterPostprocessor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    chunk_id: str,
    text: str = "",
    score: float = 0.0,
    embedding: list[float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> NodeWithScore:
    n = TextNode(text=text, id_=chunk_id, metadata=metadata or {})
    if embedding is not None:
        n.embedding = embedding
    return NodeWithScore(node=n, score=score)


# ---------------------------------------------------------------------------
# DiversityPostprocessor — disabled-by-default contract
# ---------------------------------------------------------------------------


def test_diversity_disabled_by_default_is_pass_through():
    """Default ``enabled=False`` must be a no-op."""
    pp = DiversityPostprocessor()
    nodes = [
        _node("c1", text="aaa", score=0.9),
        _node("c2", text="bbb", score=0.8),
        _node("c3", text="ccc", score=0.7),
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert [n.node.id_ for n in out] == ["c1", "c2", "c3"]
    assert all(n.node.metadata == {} for n in out)


def test_diversity_empty_input():
    pp = DiversityPostprocessor(enabled=True, top_k=2)
    assert pp._postprocess_nodes([]) == []


# ---------------------------------------------------------------------------
# DiversityPostprocessor — MMR logic
# ---------------------------------------------------------------------------


def test_diversity_mmr_pure_relevance_lambda_1():
    """lambda_mult=1.0 must return the top-k by original score (relevance only)."""
    pp = DiversityPostprocessor(enabled=True, lambda_mult=1.0, top_k=2)
    nodes = [
        _node("c1", score=0.3),
        _node("c2", score=0.9),
        _node("c3", score=0.6),
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert [n.node.id_ for n in out] == ["c2", "c3"]


def test_diversity_mmr_pure_diversity_lambda_0():
    """lambda_mult=0.0 maximises diversity; first pick is still highest score
    (max_sim=0), then the most dissimilar remaining node is chosen."""
    pp = DiversityPostprocessor(enabled=True, lambda_mult=0.0, top_k=2)
    # Embeddings: c1 and c2 are identical, c3 is orthogonal.
    nodes = [
        _node("c1", score=0.9, embedding=[1.0, 0.0]),
        _node("c2", score=0.8, embedding=[1.0, 0.0]),
        _node("c3", score=0.7, embedding=[0.0, 1.0]),
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    # First pick: c1 (highest score, no similarity penalty yet).
    # Second pick: c3 is most dissimilar to c1, so it wins over c2.
    assert [n.node.id_ for n in out] == ["c1", "c3"]


def test_diversity_mmr_balanced():
    """Default lambda_mult=0.5 balances relevance and diversity."""
    pp = DiversityPostprocessor(enabled=True, lambda_mult=0.5, top_k=2)
    nodes = [
        _node("c1", score=1.0, embedding=[1.0, 0.0]),
        _node("c2", score=0.9, embedding=[1.0, 0.0]),  # very similar to c1
        _node("c3", score=0.5, embedding=[0.0, 1.0]),  # dissimilar to c1
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    # c1 is picked first (score 1.0).
    # c2 MMR = 0.5*0.9 - 0.5*1.0 = -0.05
    # c3 MMR = 0.5*0.5 - 0.5*0.0 = 0.25
    # c3 wins.
    assert [n.node.id_ for n in out] == ["c1", "c3"]


def test_diversity_falls_back_to_lexical_when_embeddings_absent():
    """Without embeddings, similarity is token-overlap Jaccard."""
    pp = DiversityPostprocessor(enabled=True, lambda_mult=0.0, top_k=2)
    nodes = [
        _node("c1", score=0.9, text="foo bar baz"),
        _node("c2", score=0.8, text="foo bar qux"),
        _node("c3", score=0.7, text="completely different words here"),
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    # c1 picked first.  c3 has lower lexical overlap with c1 than c2 does,
    # so c3 should win the diversity round.
    assert [n.node.id_ for n in out] == ["c1", "c3"]


def test_diversity_preserves_metadata_and_scores():
    """Quarantine/trust semantics: metadata and scores must survive."""
    pp = DiversityPostprocessor(enabled=True, lambda_mult=1.0, top_k=2)
    nodes = [
        _node("c1", score=0.9, metadata={"trust": "untrusted", "domain_tags": ["a"]}),
        _node("c2", score=0.8, metadata={"trust": "trusted"}),
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert len(out) == 2
    assert out[0].node.metadata["trust"] == "untrusted"
    assert out[0].node.metadata["domain_tags"] == ["a"]
    assert out[1].node.metadata["trust"] == "trusted"
    assert out[0].score == 0.9
    assert out[1].score == 0.8


def test_diversity_top_k_truncation():
    pp = DiversityPostprocessor(enabled=True, lambda_mult=1.0, top_k=2)
    nodes = [_node(f"c{i}", score=float(i)) for i in range(5)]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert len(out) == 2


def test_diversity_deterministic_tie_break():
    """Equal MMR scores must be resolved by chunk_id for determinism."""
    pp = DiversityPostprocessor(enabled=True, lambda_mult=1.0, top_k=2)
    nodes = [
        _node("c_a", score=0.5),
        _node("c_b", score=0.5),
        _node("c_c", score=0.5),
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    # All scores equal; max picks by id lexicographically: c_c > c_b > c_a
    assert [n.node.id_ for n in out] == ["c_c", "c_b"]


def test_diversity_invalid_lambda_raises():
    pp = DiversityPostprocessor(enabled=True, lambda_mult=-0.1)
    with pytest.raises(ValueError, match="lambda_mult must be in"):
        pp._postprocess_nodes([_node("c1")])

    pp2 = DiversityPostprocessor(enabled=True, lambda_mult=1.1)
    with pytest.raises(ValueError, match="lambda_mult must be in"):
        pp2._postprocess_nodes([_node("c1")])


# ---------------------------------------------------------------------------
# DomainTagFilterPostprocessor — disabled-by-default contract
# ---------------------------------------------------------------------------


def test_domain_tag_filter_disabled_by_default_is_pass_through():
    pp = DomainTagFilterPostprocessor()
    nodes = [
        _node("c1", metadata={"domain_tags": ["weather_remote_sensing"]}),
        _node("c2", metadata={"domain_tags": ["security_netsec"]}),
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert [n.node.id_ for n in out] == ["c1", "c2"]


def test_domain_tag_filter_empty_allowlist_is_pass_through():
    pp = DomainTagFilterPostprocessor(enabled=True, allowed_tags=[])
    nodes = [
        _node("c1", metadata={"domain_tags": ["weather_remote_sensing"]}),
        _node("c2", metadata={"domain_tags": ["security_netsec"]}),
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert [n.node.id_ for n in out] == ["c1", "c2"]


def test_domain_tag_filter_empty_input():
    pp = DomainTagFilterPostprocessor(enabled=True, allowed_tags=["a"])
    assert pp._postprocess_nodes([]) == []


# ---------------------------------------------------------------------------
# DomainTagFilterPostprocessor — filtering logic
# ---------------------------------------------------------------------------


def test_domain_tag_filter_keeps_matching_nodes():
    pp = DomainTagFilterPostprocessor(
        enabled=True, allowed_tags=["weather_remote_sensing", "vehicle_dynamics"]
    )
    nodes = [
        _node("c1", metadata={"domain_tags": ["weather_remote_sensing"]}),
        _node("c2", metadata={"domain_tags": ["security_netsec"]}),
        _node("c3", metadata={"domain_tags": ["vehicle_dynamics", "guidance_navigation"]}),
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert [n.node.id_ for n in out] == ["c1", "c3"]


def test_domain_tag_filter_accepts_string_tag():
    """Some payloads may store a single tag as a string rather than list."""
    pp = DomainTagFilterPostprocessor(enabled=True, allowed_tags=["weather_remote_sensing"])
    nodes = [
        _node("c1", metadata={"domain_tags": "weather_remote_sensing"}),
        _node("c2", metadata={"domain_tags": "security_netsec"}),
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert [n.node.id_ for n in out] == ["c1"]


def test_domain_tag_filter_untagged_passes_when_drop_untagged_false():
    """Default drop_untagged=False lets untagged nodes through."""
    pp = DomainTagFilterPostprocessor(enabled=True, allowed_tags=["a"])
    nodes = [
        _node("c1", metadata={"domain_tags": ["a"]}),
        _node("c2", metadata={}),  # no domain_tags
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert [n.node.id_ for n in out] == ["c1", "c2"]


def test_domain_tag_filter_untagged_dropped_when_drop_untagged_true():
    """With drop_untagged=True, untagged nodes are dropped."""
    pp = DomainTagFilterPostprocessor(enabled=True, allowed_tags=["a"], drop_untagged=True)
    nodes = [
        _node("c1", metadata={"domain_tags": ["a"]}),
        _node("c2", metadata={}),  # no domain_tags
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert [n.node.id_ for n in out] == ["c1"]


def test_domain_tag_filter_preserves_metadata_and_scores():
    """Quarantine/trust semantics: metadata must survive filtering."""
    pp = DomainTagFilterPostprocessor(enabled=True, allowed_tags=["weather_remote_sensing"])
    nodes = [
        _node(
            "c1",
            score=0.9,
            metadata={
                "domain_tags": ["weather_remote_sensing"],
                "trust": "untrusted",
                "quarantine": True,
            },
        ),
        _node("c2", score=0.8, metadata={"domain_tags": ["security_netsec"]}),
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert len(out) == 1
    assert out[0].node.id_ == "c1"
    assert out[0].node.metadata["trust"] == "untrusted"
    assert out[0].node.metadata["quarantine"] is True
    assert out[0].score == 0.9


def test_domain_tag_filter_no_overlap_drops_all():
    pp = DomainTagFilterPostprocessor(enabled=True, allowed_tags=["x"])
    nodes = [
        _node("c1", metadata={"domain_tags": ["a"]}),
        _node("c2", metadata={"domain_tags": ["b", "c"]}),
    ]
    out = pp._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert out == []


def test_domain_tag_filter_allows_list_mutation_safe():
    """Mutating the list passed to the constructor must not affect the instance."""
    tags = ["a", "b"]
    pp = DomainTagFilterPostprocessor(enabled=True, allowed_tags=tags)
    tags.append("c")
    assert pp.allowed_tags == ["a", "b"]
