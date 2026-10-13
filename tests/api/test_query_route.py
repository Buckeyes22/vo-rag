from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from vo_rag.api.singletons import SurfaceServices
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import SearchResult

_API_KEY = "query-route-test-key"  # pragma: allowlist secret


class FakeReranker:
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 20,
    ) -> list[SearchResult]:
        return list(candidates[:top_k])


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        api=SimpleNamespace(api_key=_API_KEY, api_keys_json="", cors_origins=["*"]),
        portal=SimpleNamespace(csrf_secret="csrf-secret"),
        answer=SimpleNamespace(allow_hosted=True, allow_raw_hosted_egress=False),
        reranker=SimpleNamespace(auto_merge_enabled=False, auto_merge_threshold=0.5),
    )


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


def test_query_route_calls_landed_retrieval_and_preserves_hits(monkeypatch: Any) -> None:
    settings = _settings()
    reranker = FakeReranker()
    services = SurfaceServices(
        settings=settings,
        db_pool=object(),
        retriever=object(),
        reranker=reranker,
    )
    captured: dict[str, Any] = {"logs": [], "scopes": []}
    route = _route_module("vo_rag.api.routes.query")
    if route is not None:

        async def fake_execute_retrieval(**kwargs: Any) -> list[SearchResult]:
            captured["retrieval"] = kwargs
            return [
                SearchResult(
                    id="point-1",
                    score=0.875,
                    collection="vo_rag_books_raw_v1",
                    payload={
                        "chunk_id": "chunk-1",
                        "source_id": "source-1",
                        "nested": {"kept": True},
                    },
                )
            ]

        def fake_enforce_query_scope(**kwargs: Any) -> None:
            captured["scopes"].append(kwargs)

        async def fake_insert_query_log(*args: Any, **kwargs: Any) -> None:
            captured["logs"].append({"args": args, "kwargs": kwargs})

        monkeypatch.setattr(route, "execute_retrieval", fake_execute_retrieval)
        monkeypatch.setattr(route, "enforce_query_scope", fake_enforce_query_scope)
        monkeypatch.setattr(route, "insert_query_log", fake_insert_query_log)
        monkeypatch.setattr(route, "ulid_new", lambda: "01ARZ3NDEKTSV4RRFFQ69G5FAV")

    client = _client(monkeypatch, services)
    response = client.post(
        "/api/v1/query",
        headers={"X-API-Key": _API_KEY, "X-VO-RAG-Caller": "mcp"},
        json={
            "query": "find apex references",
            "collections": ["vo_rag_books_raw_v1"],
            "top_k": 7,
            "filters": {"source_id": "source-1"},
            "visual_mode": "baseline",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "results": [
            {
                "id": "point-1",
                "score": 0.875,
                "collection": "vo_rag_books_raw_v1",
                "payload": {
                    "chunk_id": "chunk-1",
                    "source_id": "source-1",
                    "nested": {"kept": True},
                },
            }
        ]
    }

    retrieval = captured["retrieval"]
    assert retrieval["retriever"] is services.retriever
    assert retrieval["query"] == "find apex references"
    assert retrieval["mode"] == QueryMode.INTERNAL
    assert retrieval["requested_collections"] == ["vo_rag_books_raw_v1"]
    assert retrieval["top_k"] == 7
    assert retrieval["filters"] == {"source_id": "source-1"}
    assert retrieval["visual_mode"] == "baseline"
    assert retrieval["rerank_fn"].__self__ is reranker

    scope = captured["scopes"][0]
    assert scope["identity"].name == "legacy"
    assert scope["caller"] == "mcp"
    assert scope["mode"] == "internal"
    assert scope["collections"] == ["vo_rag_books_raw_v1"]
    assert captured["logs"][0]["kwargs"]["caller"] == "mcp"


def test_query_route_validation_errors(monkeypatch: Any) -> None:
    settings = _settings()
    services = SurfaceServices(settings=settings, retriever=object(), reranker=None)
    client = _client(monkeypatch, services)

    invalid_payloads = [
        {"query": "   "},
        {"query": "ok", "collections": []},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": 201},
        {"query": "ok", "visual_mode": "nope"},
        # A bound dialect the retriever cannot translate is a bad request, not
        # a 500 from inside the query (VOR-1443).
        {"query": "ok", "filters": {"tf_pub_year": {"between": [1, 2]}}},
        {"query": "ok", "filters": {"tf_pub_year": {"gte": None}}},
        {"query": "ok", "filters": {"tf_pub_year": {}}},
    ]

    for payload in invalid_payloads:
        response = client.post(
            "/api/v1/query",
            headers={"X-API-Key": _API_KEY},
            json=payload,
        )
        assert response.status_code == 422, payload


def test_query_logging_failure_does_not_block_response(monkeypatch: Any) -> None:
    settings = _settings()
    services = SurfaceServices(settings=settings, db_pool=object(), retriever=object())
    route = _route_module("vo_rag.api.routes.query")
    if route is not None:

        async def fake_execute_retrieval(**kwargs: Any) -> list[SearchResult]:
            return [
                SearchResult(
                    id="point-2",
                    score=1.0,
                    collection="vo_rag_research_v1",
                    payload={"source_id": "source-2"},
                )
            ]

        async def failing_insert_query_log(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("query log optional failure")

        monkeypatch.setattr(route, "execute_retrieval", fake_execute_retrieval)
        monkeypatch.setattr(route, "insert_query_log", failing_insert_query_log)

    client = _client(monkeypatch, services)
    response = client.post(
        "/api/v1/query",
        headers={"X-API-Key": _API_KEY},
        json={"query": "still return results"},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "point-2"
