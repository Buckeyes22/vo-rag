from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from vo_rag.api.singletons import SurfaceServices
from vo_rag.core.answer.context import ContextScope
from vo_rag.core.answer.synthesizer import AnswerDraft, AnswerProfile, AnswerSynthesis
from vo_rag.core.llm.output_validator import InjectionSignal, ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import SearchResult

_API_KEY = "answer-route-test-key"  # pragma: allowlist secret


class FakeReranker:
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 20,
    ) -> list[SearchResult]:
        return list(candidates[:top_k])


class FakeSynthesizer:
    def __init__(self, synthesis: AnswerSynthesis | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.synthesis = synthesis or _synthesis(answer="Answer [S1].")

    async def synthesize(
        self,
        *,
        query: str,
        context: list[SearchResult],
        profile: AnswerProfile,
    ) -> AnswerSynthesis:
        self.calls.append({"query": query, "context": context, "profile": profile})
        return self.synthesis


def _settings(
    *,
    allow_raw_hosted_egress: bool = False,
    allow_hosted: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        api=SimpleNamespace(api_key=_API_KEY, api_keys_json="", cors_origins=["*"]),
        portal=SimpleNamespace(csrf_secret="csrf-secret"),
        answer=SimpleNamespace(
            allow_hosted=allow_hosted,
            allow_raw_hosted_egress=allow_raw_hosted_egress,
            max_context_chunks=8,
            max_output_tokens=1200,
        ),
    )


def _synthesis(
    *,
    answer: str,
    confidence: float = 0.82,
    cited_chunk_ids: list[str] | None = None,
    should_quarantine: bool = False,
    signals: list[InjectionSignal] | None = None,
) -> AnswerSynthesis:
    return AnswerSynthesis(
        draft=AnswerDraft(
            answer=answer,
            confidence=confidence,
            cited_chunk_ids=cited_chunk_ids or ["chunk-1"],
        ),
        llm_response=LLMResponse(
            text='{"answer":"ok"}',
            model="test-model",
            provider="test-provider",
            latency_ms=12.5,
            cost_usd=0.01,
            prompt_name="answer",
            prompt_version=1,
        ),
        validation=ValidationResult(ok=not should_quarantine, signals=signals or []),
        should_quarantine=should_quarantine,
        model="test-model",
    )


def _results() -> list[SearchResult]:
    return [
        SearchResult(
            id="point-1",
            score=0.9,
            collection="vo_rag_books_sterile_v1",
            payload={
                "chunk_id": "chunk-1",
                "source_id": "source-1",
                "content": "raw source text must stay inside synthesis context",
            },
        )
    ]


def _client(monkeypatch: Any, services: SurfaceServices) -> TestClient:
    settings = services.settings
    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: settings)

    from vo_rag.api.main import create_app

    return TestClient(create_app(settings=settings, services=services))


def _route_module(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name in {"vo_rag.api.routes", name}:
            return None
        raise


def test_answer_route_resolves_sterile_context_before_retrieval_and_synthesizes(
    monkeypatch: Any,
) -> None:
    settings = _settings()
    synthesizer = FakeSynthesizer()
    reranker = FakeReranker()
    services = SurfaceServices(
        settings=settings,
        db_pool=object(),
        retriever=object(),
        reranker=reranker,
        answer_synthesizer=synthesizer,
    )
    captured: dict[str, Any] = {"order": [], "answer_logs": [], "citations": []}
    route = _route_module("vo_rag.api.routes.answer")
    if route is not None:
        real_resolve = route.resolve_answer_collections

        def wrapped_resolve_answer_collections(**kwargs: Any) -> list[str]:
            captured["order"].append("resolve")
            captured["resolve"] = kwargs
            return cast(list[str], real_resolve(**kwargs))

        async def fake_execute_retrieval(**kwargs: Any) -> list[SearchResult]:
            captured["order"].append("retrieve")
            captured["retrieval"] = kwargs
            return _results()

        async def fake_insert_answer_log(*args: Any, **kwargs: Any) -> None:
            captured["answer_logs"].append({"args": args, "kwargs": kwargs})

        async def fake_insert_citation_assignments(*args: Any, **kwargs: Any) -> None:
            captured["citations"].append({"args": args, "kwargs": kwargs})

        monkeypatch.setattr(route, "resolve_answer_collections", wrapped_resolve_answer_collections)
        monkeypatch.setattr(route, "execute_retrieval", fake_execute_retrieval)
        monkeypatch.setattr(route, "insert_answer_log", fake_insert_answer_log)
        monkeypatch.setattr(route, "insert_citation_assignments", fake_insert_citation_assignments)
        monkeypatch.setattr(route, "ulid_new", lambda: "01ARZ3NDEKTSV4RRFFQ69G5FAV")

    client = _client(monkeypatch, services)
    response = client.post(
        "/api/v1/answer",
        headers={"X-API-Key": _API_KEY, "X-VO-RAG-Caller": "cli"},
        json={
            "query": "answer apex",
            "collections": ["vo_rag_books_raw_v1"],
            "top_k": 5,
            "filters": {"source_id": "source-1"},
            "visual_mode": "baseline",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert body["answer"] == "Answer [S1]."
    assert body["confidence"] == 0.82
    assert body["citations"] == ["chunk-1"]
    assert body["sources"] == ["source-1"]
    assert body["retrieval_results"][0]["id"] == "point-1"
    assert body["compliance"]["status"] == "not_configured"
    assert body["compliance"]["should_quarantine"] is False

    assert captured["order"] == ["resolve", "retrieve"]
    assert captured["resolve"] == {
        "mode": QueryMode.INTERNAL,
        "requested": ["vo_rag_books_raw_v1"],
        "context_scope": ContextScope.STERILE,
    }
    retrieval = captured["retrieval"]
    assert retrieval["retriever"] is services.retriever
    assert retrieval["query"] == "answer apex"
    assert retrieval["mode"] == QueryMode.INTERNAL
    assert retrieval["requested_collections"] == ["vo_rag_books_sterile_v1"]
    assert retrieval["top_k"] == 5
    assert retrieval["filters"] == {"source_id": "source-1"}
    assert retrieval["visual_mode"] == "baseline"
    assert retrieval["rerank_fn"].__self__ is reranker

    assert synthesizer.calls == [
        {
            "query": "answer apex",
            "context": _results(),
            "profile": AnswerProfile.QUALITY,
        }
    ]
    assert captured["answer_logs"][0]["kwargs"]["source_ids"] == ["source-1"]


def test_answer_include_retrieval_false_only_suppresses_response_hits(
    monkeypatch: Any,
) -> None:
    settings = _settings()
    synthesizer = FakeSynthesizer()
    services = SurfaceServices(
        settings=settings,
        retriever=object(),
        answer_synthesizer=synthesizer,
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
        json={"query": "answer apex", "include_retrieval": False},
    )

    assert response.status_code == 200
    assert response.json()["retrieval_results"] == []
    assert synthesizer.calls[0]["context"] == _results()


def test_answer_raw_context_with_hosted_generation_requires_explicit_opt_in(
    monkeypatch: Any,
) -> None:
    settings = _settings(allow_raw_hosted_egress=False, allow_hosted=True)
    services = SurfaceServices(
        settings=settings,
        retriever=object(),
        answer_synthesizer=FakeSynthesizer(),
    )
    client = _client(monkeypatch, services)
    response = client.post(
        "/api/v1/answer",
        headers={"X-API-Key": _API_KEY},
        json={
            "query": "raw answer",
            "context_scope": "raw",
            "collections": ["vo_rag_books_raw_v1"],
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "raw_context_hosted_egress_forbidden"


def test_answer_quarantine_is_surfaced_not_converted_to_success(
    monkeypatch: Any,
) -> None:
    signal = InjectionSignal(
        kind="canary_echoed",
        confidence=1.0,
        evidence="canary token echoed",
    )
    synthesizer = FakeSynthesizer(
        _synthesis(
            answer="Suspicious [S1].",
            should_quarantine=True,
            signals=[signal],
        )
    )
    services = SurfaceServices(
        settings=_settings(),
        retriever=object(),
        answer_synthesizer=synthesizer,
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
        json={"query": "answer apex"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "prompt_injection_suspected" in body["warnings"]
    assert "canary_echoed" in body["warnings"]
    assert body["compliance"]["should_quarantine"] is True
    assert body["compliance"]["reason"] == "prompt_injection_suspected"


def test_empty_answer_returns_null_confidence_zero_and_no_citations(
    monkeypatch: Any,
) -> None:
    synthesizer = FakeSynthesizer(
        _synthesis(answer="", confidence=0.7, cited_chunk_ids=["chunk-1"])
    )
    services = SurfaceServices(
        settings=_settings(),
        retriever=object(),
        answer_synthesizer=synthesizer,
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
        json={"query": "missing answer"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["confidence"] == 0.0
    assert body["citations"] == []


def test_answer_route_rejects_a_malformed_range_filter_with_422(
    monkeypatch: Any,
) -> None:
    """``/answer`` shares the filter dialect with ``/query``, so an
    untranslatable range dict is a 422 at the boundary, never a 500 raised from
    inside the retrieval call (VOR-1443)."""
    synthesizer = FakeSynthesizer()
    services = SurfaceServices(
        settings=_settings(),
        retriever=object(),
        answer_synthesizer=synthesizer,
    )
    client = _client(monkeypatch, services)
    response = client.post(
        "/api/v1/answer",
        headers={"X-API-Key": _API_KEY},
        json={"query": "answer apex", "filters": {"tf_pub_year": {"between": [1, 2]}}},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert [error["loc"][-1] for error in detail] == ["filters"], detail
    assert synthesizer.calls == []


def test_public_answer_mode_fails_closed_without_compliance_gate(
    monkeypatch: Any,
) -> None:
    synthesizer = FakeSynthesizer()
    services = SurfaceServices(
        settings=_settings(),
        retriever=object(),
        answer_synthesizer=synthesizer,
    )
    client = _client(monkeypatch, services)
    response = client.post(
        "/api/v1/answer",
        headers={"X-API-Key": _API_KEY},
        json={"query": "publish answer", "mode": "publication"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "answer_compliance_gate_not_configured"
    assert synthesizer.calls == []


def test_answer_route_does_not_call_protected_llm_directly(monkeypatch: Any) -> None:
    route = _route_module("vo_rag.api.routes.answer")
    if route is None:
        services = SurfaceServices(settings=_settings(), retriever=object())
        client = _client(monkeypatch, services)
        response = client.post(
            "/api/v1/answer",
            headers={"X-API-Key": _API_KEY},
            json={"query": "answer apex"},
        )
        assert response.status_code == 200
        return

    source = inspect.getsource(route)
    assert "complete_json_protected" not in source
    assert "LLMClient" not in source


def test_brain_answer_evidence_budget_is_actionable_and_precedes_model_calls(
    monkeypatch: Any,
) -> None:
    from vo_rag.core.answer.synthesizer import AnswerSynthesizer
    from vo_rag.core.config.settings import AnswerSettings

    synthesizer = AnswerSynthesizer(AnswerSettings())
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("model")
        raise AssertionError("over-budget evidence must not reach provider selection")

    monkeypatch.setattr(synthesizer, "select_model", forbidden)
    services = SurfaceServices(
        settings=_settings(), db_pool=object(), retriever=object(), answer_synthesizer=synthesizer
    )
    route = _route_module("vo_rag.api.routes.answer")
    assert route is not None

    async def retrieve(**kwargs: Any) -> list[SearchResult]:
        assert kwargs["requested_collections"] == ["vo_rag_brain_v1"]
        return [
            SearchResult(
                id="point",
                score=1.0,
                collection="vo_rag_brain_v1",
                payload={
                    "content": "Complete source.\n" * 5000 + "Only laboratory firmware 3.",
                    "source_id": "wiki/device.md",
                    "commit": "a" * 40,
                    "governing_context": [],
                    "governing_context_version": 1,
                    "governing_context_complete": True,
                },
            )
        ]

    monkeypatch.setattr(route, "execute_retrieval", retrieve)
    response = _client(monkeypatch, services).post(
        "/api/v1/answer",
        headers={"X-API-Key": _API_KEY, "X-VO-RAG-Caller": "cli"},
        json={"query": "retention?", "collections": ["vo_rag_brain_v1"], "mode": "internal"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "brain_answer_evidence_exceeds_65536_character_budget"
    assert calls == []


def test_brain_authority_and_qualifications_reach_protected_synthesis(
    monkeypatch: Any,
) -> None:
    from tests.core.answer.test_synthesizer import _capture_protected
    from tests.core.answer.test_synthesizer import _settings as model_settings
    from vo_rag.core.answer.synthesizer import AnswerDraft, AnswerSynthesizer
    from vo_rag.core.llm.output_validator import ValidationResult
    from vo_rag.core.llm.response import LLMResponse

    captured: list[Any] = []
    monkeypatch.setattr(
        "vo_rag.core.answer.synthesizer.complete_json_protected",
        _capture_protected(
            captured,
            AnswerDraft(answer="This is a proposal [S1].", confidence=0.9),
            LLMResponse(text="{}", model="fixture"),
            ValidationResult(ok=True, signals=[]),
            False,
        ),
    )
    services = SurfaceServices(
        settings=_settings(),
        db_pool=None,
        retriever=object(),
        answer_synthesizer=AnswerSynthesizer(model_settings()),
    )
    route = _route_module("vo_rag.api.routes.answer")
    assert route is not None

    async def retrieve(**kwargs: Any) -> list[SearchResult]:
        assert kwargs["requested_collections"] == ["vo_rag_brain_v1"]
        return [
            SearchResult(
                id="proposal",
                score=1,
                collection="vo_rag_brain_v1",
                payload={
                    "content": (
                        "The device retains seven days of records for enterprise firmware 3 only."
                    ),
                    "source_id": "inbox/device-proposal.md",
                    "path": "inbox/device-proposal.md",
                    "commit": "a" * 40,
                    "source_revision": "a" * 40,
                    "page_area": "inbox",
                    "status": "proposed",
                    "kind": "source-reference",
                    "url": "https://example.test/device",
                    "fetched_at": "2026-09-01",
                    "source_origin": "operator",
                    "updated": "2026-09-01",
                    "source_lineage": [
                        {"url": "https://example.test/device", "fetched_at": "2026-09-01"}
                    ],
                    "edition": "enterprise",
                    "version": "3",
                    "governing_context_version": 1,
                    "governing_context_complete": True,
                    "governing_context": [],
                },
            )
        ]

    monkeypatch.setattr(route, "execute_retrieval", retrieve)
    response = _client(monkeypatch, services).post(
        "/api/v1/answer",
        headers={"X-API-Key": _API_KEY},
        json={"query": "retention?", "collections": ["vo_rag_brain_v1"], "mode": "internal"},
    )
    assert response.status_code == 200 and len(captured) == 1
    frame = captured[0].untrusted_content
    for value in (
        "vo_rag_brain_v1",
        "inbox/device-proposal.md",
        "a" * 40,
        '"status": "proposed"',
        '"page_area": "inbox"',
        '"source_origin": "operator"',
        "source_lineage",
        "source-reference",
        "https://example.test/device",
        "fetched_at",
        "2026-09-01",
        "enterprise firmware 3 only",
        "does not promote inbox proposals",
        "currently running",
    ):
        assert value in frame


class RecordingGate:
    def __init__(self, *, allowed: bool = True, reason: str = "ok") -> None:
        self.calls: list[dict[str, Any]] = []
        self._allowed = allowed
        self._reason = reason

    def check(self, *, mode: QueryMode, hits: Any, caller: str) -> Any:
        from vo_rag.core.compliance.answer_gate import AnswerGateDecision

        self.calls.append({"mode": mode, "hits": list(hits), "caller": caller})
        return AnswerGateDecision(allowed=self._allowed, reason=self._reason)


class RaisingGate:
    def check(self, *, mode: QueryMode, hits: Any, caller: str) -> Any:
        raise RuntimeError("secret-policy-detail")


def _gated_answer(monkeypatch: Any, gate: Any, mode: str) -> tuple[Any, FakeSynthesizer]:
    synthesizer = FakeSynthesizer()
    services = SurfaceServices(
        settings=_settings(),
        retriever=object(),
        answer_synthesizer=synthesizer,
    )
    services.extra["answer_compliance_gate"] = gate
    route = _route_module("vo_rag.api.routes.answer")
    assert route is not None

    async def fake_execute_retrieval(**kwargs: Any) -> list[SearchResult]:
        return _results()

    monkeypatch.setattr(route, "execute_retrieval", fake_execute_retrieval)
    client = _client(monkeypatch, services)
    response = client.post(
        "/api/v1/answer",
        headers={"X-API-Key": _API_KEY, "X-VO-RAG-Caller": "cli"},
        json={"query": "publish answer", "mode": mode},
    )
    return response, synthesizer


def test_gate_allow_returns_200_and_gate_called_once_with_mode_hits_caller(
    monkeypatch: Any,
) -> None:
    gate = RecordingGate(allowed=True)
    response, synthesizer = _gated_answer(monkeypatch, gate, "publication")

    assert response.status_code == 200
    assert len(synthesizer.calls) == 1
    assert len(gate.calls) == 1
    assert gate.calls[0]["mode"] == QueryMode.PUBLICATION
    assert gate.calls[0]["hits"] == _results()
    assert gate.calls[0]["caller"] == "cli"


def test_gate_deny_returns_403_denied_and_skips_synthesis(monkeypatch: Any, caplog: Any) -> None:
    gate = RecordingGate(allowed=False, reason="license_not_cleared")
    with caplog.at_level("WARNING", logger="vo_rag.api.routes.answer"):
        response, synthesizer = _gated_answer(monkeypatch, gate, "partner")

    assert response.status_code == 403
    assert response.json()["detail"] == "answer_compliance_gate_denied"
    assert "license_not_cleared" not in response.text
    assert synthesizer.calls == []
    record = next(r for r in caplog.records if r.getMessage() == "answer_compliance_gate_denied")
    assert record.mode == "partner"
    assert record.reason == "license_not_cleared"


def test_gate_raising_returns_403_error_and_logs_type_only(monkeypatch: Any, caplog: Any) -> None:
    with caplog.at_level("WARNING", logger="vo_rag.api.routes.answer"):
        response, synthesizer = _gated_answer(monkeypatch, RaisingGate(), "publication")

    assert response.status_code == 403
    assert response.json()["detail"] == "answer_compliance_gate_error"
    assert synthesizer.calls == []
    assert "secret-policy-detail" not in caplog.text
    record = next(r for r in caplog.records if r.getMessage() == "answer_compliance_gate_error")
    assert record.exc_type == "RuntimeError"


def test_invalid_gate_object_returns_403_invalid(monkeypatch: Any) -> None:
    response, synthesizer = _gated_answer(monkeypatch, object(), "publication")

    assert response.status_code == 403
    assert response.json()["detail"] == "answer_compliance_gate_invalid"
    assert synthesizer.calls == []


def test_gate_with_non_callable_check_is_invalid(monkeypatch: Any) -> None:
    gate = type("NotCallable", (), {"check": "nope"})()
    response, synthesizer = _gated_answer(monkeypatch, gate, "publication")

    assert response.status_code == 403
    assert response.json()["detail"] == "answer_compliance_gate_invalid"
    assert synthesizer.calls == []


def test_gate_returning_a_non_decision_is_invalid(monkeypatch: Any) -> None:
    class BadReturn:
        def check(self, *, mode: QueryMode, hits: Any, caller: str) -> Any:
            return True

    response, synthesizer = _gated_answer(monkeypatch, BadReturn(), "publication")

    assert response.status_code == 403
    assert response.json()["detail"] == "answer_compliance_gate_invalid"
    assert synthesizer.calls == []


def test_internal_mode_does_not_consult_a_denying_gate(monkeypatch: Any) -> None:
    gate = RecordingGate(allowed=False, reason="deny")
    response, synthesizer = _gated_answer(monkeypatch, gate, "internal")

    assert response.status_code == 200
    assert gate.calls == []
    assert len(synthesizer.calls) == 1


@pytest.mark.parametrize("allowed", ["false", 1, [0], "", None])
def test_gate_with_non_bool_allowed_is_invalid_and_never_allows(
    monkeypatch: Any, allowed: Any
) -> None:
    from vo_rag.core.compliance.answer_gate import AnswerGateDecision

    class LooseGate:
        def check(self, *, mode: QueryMode, hits: Any, caller: str) -> Any:
            return AnswerGateDecision(allowed=allowed, reason="x")

    response, synthesizer = _gated_answer(monkeypatch, LooseGate(), "publication")

    assert response.status_code == 403
    assert response.json()["detail"] == "answer_compliance_gate_invalid"
    assert synthesizer.calls == []


def test_gate_with_non_string_reason_is_invalid(monkeypatch: Any) -> None:
    from vo_rag.core.compliance.answer_gate import AnswerGateDecision

    class BadReason:
        def check(self, *, mode: QueryMode, hits: Any, caller: str) -> Any:
            return AnswerGateDecision(allowed=False, reason=["x"])  # type: ignore[arg-type]

    response, synthesizer = _gated_answer(monkeypatch, BadReason(), "publication")

    assert response.status_code == 403
    assert response.json()["detail"] == "answer_compliance_gate_invalid"
    assert synthesizer.calls == []


def test_async_gate_is_invalid_without_never_awaited_warning(monkeypatch: Any) -> None:
    import warnings

    from vo_rag.core.compliance.answer_gate import AnswerGateDecision

    class AsyncGate:
        async def check(self, *, mode: QueryMode, hits: Any, caller: str) -> Any:
            return AnswerGateDecision(allowed=True, reason="ok")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        response, synthesizer = _gated_answer(monkeypatch, AsyncGate(), "publication")
        import gc

        gc.collect()

    assert response.status_code == 403
    assert response.json()["detail"] == "answer_compliance_gate_invalid"
    assert synthesizer.calls == []


def test_gate_reason_is_sanitized_before_logging(monkeypatch: Any, caplog: Any) -> None:
    gate = RecordingGate(allowed=False, reason="line1\nline2\r\n" + "x" * 500)
    with caplog.at_level("WARNING", logger="vo_rag.api.routes.answer"):
        response, _ = _gated_answer(monkeypatch, gate, "publication")

    assert response.status_code == 403
    record = next(r for r in caplog.records if r.getMessage() == "answer_compliance_gate_denied")
    assert "\n" not in record.reason and "\r" not in record.reason
    assert len(record.reason) <= 200
    assert record.reason.startswith("line1 line2")


def test_gate_cannot_mutate_the_evidence_passed_to_synthesis(monkeypatch: Any) -> None:
    class MutatingGate:
        def check(self, *, mode: QueryMode, hits: Any, caller: str) -> Any:
            from vo_rag.core.compliance.answer_gate import AnswerGateDecision

            assert isinstance(hits, tuple)
            return AnswerGateDecision(allowed=True, reason="ok")

    response, synthesizer = _gated_answer(monkeypatch, MutatingGate(), "publication")

    assert response.status_code == 200
    assert synthesizer.calls[0]["context"] == _results()


def test_allowed_gate_is_reported_in_compliance_status(monkeypatch: Any) -> None:
    response, _ = _gated_answer(monkeypatch, RecordingGate(allowed=True), "publication")

    assert response.status_code == 200
    assert response.json()["compliance"]["status"] == "allowed"


def test_internal_mode_compliance_status_stays_not_configured(monkeypatch: Any) -> None:
    response, _ = _gated_answer(monkeypatch, RecordingGate(allowed=True), "internal")

    assert response.json()["compliance"]["status"] == "not_configured"
