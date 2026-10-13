"""Defer-LLM ingest: the runner tolerates an unconfigured serverless LLM and the
retrieval-smoke gate is then skipped, so parse/extract-images/chunk/embed runs
without a RunPod/serverless endpoint (VOR-285)."""

from __future__ import annotations

from types import SimpleNamespace

from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.runner import _build_optional_llm_client


def test_build_optional_llm_client_returns_none_when_serverless_unconfigured() -> None:
    def _raise(**_kwargs: object) -> object:
        raise ValueError("serverless role 'text' has no endpoint_id configured")

    llm_mod = SimpleNamespace(LLMClient=_raise)
    assert _build_optional_llm_client(llm_mod) is None


def test_build_optional_llm_client_returns_client_when_constructible() -> None:
    marker = object()

    def _ok(**_kwargs: object) -> object:
        return marker

    llm_mod = SimpleNamespace(LLMClient=_ok)
    assert _build_optional_llm_client(llm_mod) is marker


def test_retrieval_smoke_enabled_defaults_true() -> None:
    # Prod behaviour unchanged: the smoke gate is on by default; defer-LLM
    # ingest turns it off (or it auto-skips when there is no LLM client).
    assert BookRagConfig().retrieval_smoke_enabled is True
