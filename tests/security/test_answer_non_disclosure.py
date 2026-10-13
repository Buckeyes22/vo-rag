from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from vo_rag.api.singletons import SurfaceServices
from vo_rag.core.answer.synthesizer import AnswerDraft, AnswerProfile, AnswerSynthesis
from vo_rag.core.llm.output_validator import InjectionSignal, ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.retrieval.search import SearchResult

_API_KEY = "answer-nondisclosure-test-key"  # pragma: allowlist secret
_RAW_TEXT = "RAW SECRET SOURCE TEXT 123"
_CREDENTIAL = "sk-live-do-not-disclose"  # pragma: allowlist secret


class QuarantineSynthesizer:
    async def synthesize(
        self,
        *,
        query: str,
        context: list[SearchResult],
        profile: AnswerProfile,
    ) -> AnswerSynthesis:
        return AnswerSynthesis(
            draft=AnswerDraft(
                answer="Suspicious [S1].",
                confidence=0.4,
                cited_chunk_ids=["chunk-secret"],
            ),
            llm_response=LLMResponse(text='{"answer":"x"}', model="test-model"),
            validation=ValidationResult(
                ok=False,
                signals=[
                    InjectionSignal(
                        kind="verbatim_untrusted_echo",
                        confidence=0.95,
                        evidence=f"{_RAW_TEXT} {_CREDENTIAL}",
                    )
                ],
            ),
            should_quarantine=True,
            model="test-model",
        )


class ExplodingSynthesizer:
    async def synthesize(
        self,
        *,
        query: str,
        context: list[SearchResult],
        profile: AnswerProfile,
    ) -> AnswerSynthesis:
        raise RuntimeError(f"provider leaked {_RAW_TEXT} {_CREDENTIAL}")


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        api=SimpleNamespace(api_key=_API_KEY, api_keys_json="", cors_origins=["*"]),
        portal=SimpleNamespace(csrf_secret="csrf-secret"),
        answer=SimpleNamespace(
            allow_hosted=True,
            allow_raw_hosted_egress=False,
            max_context_chunks=8,
            max_output_tokens=1200,
        ),
    )


def _route_module(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name in {"vo_rag.api.routes", name}:
            return None
        raise


def _client(
    monkeypatch: Any,
    services: SurfaceServices,
    *,
    raise_server_exceptions: bool = True,
) -> TestClient:
    settings = services.settings
    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: settings)

    from vo_rag.api.main import create_app

    return TestClient(
        create_app(settings=settings, services=services),
        raise_server_exceptions=raise_server_exceptions,
    )


def _results() -> list[SearchResult]:
    return [
        SearchResult(
            id="point-secret",
            score=1.0,
            collection="vo_rag_books_sterile_v1",
            payload={
                "chunk_id": "chunk-secret",
                "source_id": "source-secret",
                "content": _RAW_TEXT,
                "api_key": _CREDENTIAL,
            },
        )
    ]


def test_answer_quarantine_payload_redacts_raw_text_and_credentials(
    monkeypatch: Any,
) -> None:
    services = SurfaceServices(
        settings=_settings(),
        retriever=object(),
        answer_synthesizer=QuarantineSynthesizer(),
    )
    route = _route_module("vo_rag.api.routes.answer")
    if route is not None:

        async def fake_execute_retrieval(**kwargs: Any) -> list[SearchResult]:
            return _results()

        monkeypatch.setattr(route, "execute_retrieval", fake_execute_retrieval)

    client = _client(monkeypatch, services)
    response = client.post(
        "/api/v1/answer",
        headers={"X-API-Key": _API_KEY},
        json={"query": "answer secret"},
    )

    assert response.status_code == 200
    body_text = response.text
    assert "prompt_injection_suspected" in response.json()["warnings"]
    assert _RAW_TEXT not in body_text
    assert _CREDENTIAL not in body_text


def test_answer_generation_exception_detail_does_not_disclose_context_or_credentials(
    monkeypatch: Any,
) -> None:
    services = SurfaceServices(
        settings=_settings(),
        retriever=object(),
        answer_synthesizer=ExplodingSynthesizer(),
    )
    route = _route_module("vo_rag.api.routes.answer")
    if route is not None:

        async def fake_execute_retrieval(**kwargs: Any) -> list[SearchResult]:
            return _results()

        monkeypatch.setattr(route, "execute_retrieval", fake_execute_retrieval)

    client = _client(monkeypatch, services, raise_server_exceptions=False)
    response = client.post(
        "/api/v1/answer",
        headers={"X-API-Key": _API_KEY},
        json={"query": "answer secret"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "answer_generation_failed"
    assert _RAW_TEXT not in response.text
    assert _CREDENTIAL not in response.text
