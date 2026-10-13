"""build_reranker backend selection (VOR-8)."""

from __future__ import annotations

from vo_rag.core.config.settings import RerankerSettings
from vo_rag.core.reranking.client import RerankerClient
from vo_rag.core.reranking.cohere import CohereReranker
from vo_rag.core.reranking.factory import build_reranker


def test_builds_client_when_backend_client() -> None:
    r = build_reranker(RerankerSettings(url="http://x/rerank", backend="client"))
    assert isinstance(r, RerankerClient)


def test_auto_resolves_to_client_with_url() -> None:
    r = build_reranker(RerankerSettings(url="http://x/rerank"))  # auto → client
    assert isinstance(r, RerankerClient)


def test_no_url_resolves_onnx_cpu_backend() -> None:
    # Don't instantiate GteReranker (needs the onnx artifact); assert the branch.
    assert RerankerSettings().backend == "onnx_cpu"


def test_builds_cohere_when_backend_cohere(monkeypatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "test-key")

    r = build_reranker(RerankerSettings(backend="cohere"))

    assert isinstance(r, CohereReranker)
