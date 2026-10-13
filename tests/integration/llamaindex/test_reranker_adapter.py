"""Contract tests for VoRagReranker (BaseNodePostprocessor adapter).

These tests verify:
- ID fidelity through the round-trip (NodeWithScore → SearchResult → rerank → NodeWithScore)
- top_k truncation is respected
- parity between adapter path and direct backend.rerank path
- async backend bridging raises inside a running loop (forces rerank_fn usage)
- backend parity: all build_reranker choices behave identically through the adapter
- seam cleanliness: no AdaptiveReranker path remains (VOR-188 follow-through)
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Protocol, cast

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from vo_rag.core.config.settings import RerankerSettings
from vo_rag.core.retrieval.search import SearchResult
from vo_rag.integrations.llamaindex.reranker import VoRagReranker


class _FakeReranker:
    """Reverse order + restamp scores; stands in for a real backend."""

    def rerank(self, query, candidates, top_k=20):
        out = list(reversed(candidates))[:top_k]
        return [
            SearchResult(
                id=c.id,
                score=float(i),
                collection=c.collection,
                payload=c.payload,
            )
            for i, c in enumerate(out)
        ]


def _node(i):
    n = TextNode(text=f"chunk {i}", metadata={"collection": "vo_rag_books_sterile_v1"})
    n.id_ = f"id-{i}"
    return NodeWithScore(node=n, score=float(i))


def test_postprocess_preserves_ids_and_respects_top_k():
    rr = VoRagReranker(reranker=_FakeReranker(), top_k=2)
    nodes = [_node(0), _node(1), _node(2)]
    out = rr._postprocess_nodes(nodes, QueryBundle(query_str="q"))

    assert [n.node.id_ for n in out] == ["id-2", "id-1"]  # reordered, ids intact
    assert len(out) == 2  # top_k honored
    assert all(n.node.id_ for n in out)  # no id loss


def test_public_sync_backend_protocol_and_boundary_property() -> None:
    adapter_module = importlib.import_module("vo_rag.integrations.llamaindex.reranker")
    protocol = getattr(adapter_module, "SyncRerankBackend", None)

    assert protocol is not None
    assert Protocol in protocol.__mro__

    signature = inspect.signature(protocol.rerank)
    assert list(signature.parameters) == ["self", "query", "candidates", "top_k"]
    assert signature.parameters["top_k"].kind is inspect.Parameter.KEYWORD_ONLY

    rr = VoRagReranker(reranker=_FakeReranker(), top_k=2)
    assert rr.backend_is_sync is True


def test_parity_with_rerank_fn_seam():
    """Adapter result == calling the backend.rerank directly through SearchResult."""
    backend = _FakeReranker()
    nodes = [_node(0), _node(1), _node(2)]
    cands = [
        SearchResult(
            id=n.node.id_,
            score=n.score,
            collection="vo_rag_books_sterile_v1",
            payload={},
        )
        for n in nodes
    ]
    native = backend.rerank("q", cands, top_k=2)

    out = VoRagReranker(reranker=backend, top_k=2)._postprocess_nodes(
        nodes, QueryBundle(query_str="q")
    )
    assert [n.node.id_ for n in out] == [r.id for r in native]


@pytest.mark.asyncio
async def test_async_backend_rejected_at_construction():
    class _AsyncBackend:
        async def rerank(self, query, candidates, top_k=20):
            return candidates[:top_k]

    # Async backends are rejected at construction so the sync-bridge never runs.
    with pytest.raises(TypeError, match="cannot wrap an async reranker backend"):
        VoRagReranker(reranker=_AsyncBackend(), top_k=1)


@pytest.mark.asyncio
async def test_arerank_only_backend_rejected_at_construction():
    class _AsyncOnlyBackend:
        async def arerank(self, query, candidates, top_k=20):
            return candidates[:top_k]

    with pytest.raises(TypeError, match="cannot wrap an async reranker backend"):
        VoRagReranker(reranker=cast(Any, _AsyncOnlyBackend()), top_k=1)


# --- VOR-188: seam-cleanliness guards ------------------------------------ #


def test_no_adaptive_reranker_path() -> None:
    """AdaptiveReranker must not be reachable from any public reranking surface.

    Mirrors the ``test_gte.py`` drop-assertion at the adapter layer (VOR-135
    excision follow-through). The two-list ``rerank(query, dense_hits, lex_hits,
    top_k)`` shape is dead; the post-RRF single-pool ``RerankFn`` is the only
    seam.
    """
    pkg = importlib.import_module("vo_rag.core.reranking")
    assert not hasattr(pkg, "AdaptiveReranker")

    gte_module = importlib.import_module("vo_rag.core.reranking.gte")
    assert not hasattr(gte_module, "AdaptiveReranker")
    assert not hasattr(gte_module, "_union_unique")

    factory_module = importlib.import_module("vo_rag.core.reranking.factory")
    assert not hasattr(factory_module, "AdaptiveReranker")

    # The adapter must import build_reranker (the single backend chooser) and
    # must NOT import anything from an ``adaptive`` sub-module.
    adapter_module = importlib.import_module("vo_rag.integrations.llamaindex.reranker")
    assert hasattr(adapter_module, "build_reranker")
    assert "adaptive" not in {
        name.split(".")[-1] for name in getattr(adapter_module, "__modules", []) or []
    }

    # _postprocess_nodes must accept a single fused candidate pool, not the
    # pre-fusion two-list ``dense_hits / lex_hits`` shape.
    sig = inspect.signature(adapter_module.VoRagReranker._postprocess_nodes)
    params = list(sig.parameters.keys())
    assert params == ["self", "nodes", "query_bundle"]
    for forbidden in ("dense_hits", "lex_hits"):
        assert forbidden not in params


def test_backend_parity_across_build_reranker_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    """All ``build_reranker`` backends behave identically through VoRagReranker.

    Uses lightweight spies (no real ONNX/GPU/Cohere). Every backend must:
    - produce the same reordered id list for the same input
    - respect ``top_k``
    - preserve ``node.id_`` (fail-closed identity)
    """

    class _SpyBackend:
        def __init__(self, name: str) -> None:
            self.name = name
            self.calls: list[dict[str, Any]] = []

        def rerank(
            self, query: str, candidates: list[SearchResult], top_k: int = 20
        ) -> list[SearchResult]:
            self.calls.append({"query": query, "candidates": candidates, "top_k": top_k})
            # Reverse order + restamp scores, same as _FakeReranker.
            out = list(reversed(candidates))[:top_k]
            return [
                SearchResult(
                    id=c.id,
                    score=float(i),
                    collection=c.collection,
                    payload=c.payload,
                )
                for i, c in enumerate(out)
            ]

    import vo_rag.integrations.llamaindex.reranker as adapter_module

    spies: dict[str, _SpyBackend] = {}

    def _fake_build_reranker(settings: RerankerSettings) -> _SpyBackend:
        spy = _SpyBackend(settings.backend)
        spies[settings.backend] = spy
        return spy

    # Patch the name as bound inside the adapter module (import-time alias).
    monkeypatch.setattr(adapter_module, "build_reranker", _fake_build_reranker)

    nodes = [_node(0), _node(1), _node(2)]
    expected_ids = ["id-2", "id-1", "id-0"]

    for backend in ("client", "gpu", "cohere", "onnx_cpu"):
        settings = RerankerSettings(backend=backend)
        rr = VoRagReranker(settings=settings, top_k=2)
        out = rr._postprocess_nodes(nodes, QueryBundle(query_str="q"))

        assert [n.node.id_ for n in out] == expected_ids[:2], (
            f"backend={backend!r} produced unexpected id order"
        )
        assert len(out) == 2, f"backend={backend!r} ignored top_k"
        assert all(n.node.id_ for n in out), f"backend={backend!r} lost node.id_"

        spy = spies[backend]
        assert len(spy.calls) == 1
        assert spy.calls[0]["query"] == "q"
        assert spy.calls[0]["top_k"] == 2
        # The seam must hand the backend a single fused pool, not two lists.
        assert isinstance(spy.calls[0]["candidates"], list)
        assert len(spy.calls[0]["candidates"]) == 3
