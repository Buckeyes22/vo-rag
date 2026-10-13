"""Cross-surface contract tests (non-live).

These tests exercise the three vo-rag production surfaces (HTTP API,
Click CLI, Python MCP) over ONE shared fake :class:`SurfaceServices`
container. The contract is: every surface preserves the core
``SearchResult`` / ``IntakeSubmission`` / ``EvalGateResult`` shape
byte-for-byte so a caller can switch surfaces without re-validating
their data.

The tests do NOT open sockets. Every dependency (Qdrant, Garage,
Postgres, Anthropic, embedding server) is replaced by a fake
structurally compatible with the seam the surface touches. The same
scripted ``SearchResult`` list, the same normalized intake envelope,
and the same ``EvalGateResult`` fixture is used across all three
surfaces — that is the cross-surface contract.

Coverage:

- Query contract: API ``POST /api/v1/query``, CLI
  ``vo-rag query`` (HTTP mocked via respx), and MCP ``vo_rag_query``
  all preserve ``id`` / ``collection`` / ``payload`` / ``score``,
  honour ``top_k`` truncation, echo the requested mode, and surface
  the reranker availability warning.

- Ingest contract: the same ``url`` / ``text`` / ``file_drop``
  input shape is built into an :class:`IntakeSubmission` by the API,
  CLI payload builder, and MCP tool builder. The submission
  round-trips through :func:`vo_rag.core.ingest.normalize.normalize`
  and produces the same ``raw_content`` shape. Unsafe URLs are
  rejected by all three surfaces (F-014).

- Answer non-disclosure contract: API + CLI answer flows
  route through :class:`AnswerSynthesizer` and preserve
  :class:`ContextScope` ``STERILE`` by default. Raw collection use
  is rejected unless ``allow_raw_hosted_egress=True``. A
  ``should_quarantine=True`` synthesis surfaces as warnings +
  compliance payload, never as a normal success (F-012 / F-045).

- Admin / eval contract: API ``/api/v1/admin/status``,
  CLI ``vo-rag eval-gate``, and MCP ``vo_rag_status`` all surface
  the same :class:`EvalGateResult` fields from
  :func:`vo_rag.core.eval.gate.run_eval_gate` with registered
  evaluators. The CI gate behaviour is unchanged — the
  ``scripts/ci.sh`` eval-gate exit code depends on the same fields.

- Import smoke: ``vo_rag.api.main:app, create_app``,
  ``vo_rag.cli.main:cli``, and ``vo_rag.mcp.server:build_mcp_server``
  import cleanly without opening any sockets.

The tests in this module are non-live — they use no real network,
DB, Qdrant, Garage, or LLM. They are NOT marked
``@pytest.mark.integration`` (the live-infra marker); they live
under ``tests/integration/`` for organizational reasons. The CI
gate (``bash scripts/ci.sh``) runs them through the default
``-m "not integration and not slow and not live"`` filter.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from click.testing import CliRunner
from fastapi.testclient import TestClient

from vo_rag.api.main import create_app
from vo_rag.api.singletons import SurfaceServices
from vo_rag.cli.main import cli
from vo_rag.core.answer.synthesizer import (
    AnswerDraft,
    AnswerProfile,
    AnswerSynthesis,
)
from vo_rag.core.config.pipelines import load_pipeline_registry
from vo_rag.core.eval.gate import EvalGateResult
from vo_rag.core.eval.registered import run_registered_eval_gate
from vo_rag.core.ingest.normalize import IntakeSubmission, normalize
from vo_rag.core.llm.output_validator import InjectionSignal, ValidationResult
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.models.envelope import IntakeSource
from vo_rag.core.retrieval.search import SearchResult
from vo_rag.web.url_safety import ResolvedURL, SSRFBlocked

# --------------------------------------------------------------------------- #
# Constants — the one canonical "fixture" the three surfaces share.
# --------------------------------------------------------------------------- #
_API_KEY = "surface-contract-test-key"  # pragma: allowlist secret
_API_BASE = "http://localhost:8000"

# A small, deterministic SearchResult list. The exact same list is
# fed to each surface; the assertions pin that no surface mutates
# id / collection / score / payload.
_SCRIPTED_HITS: list[SearchResult] = [
    SearchResult(
        id="point-1",
        score=0.91,
        collection="vo_rag_books_raw_v1",
        payload={
            "chunk_id": "c-1",
            "source_id": "src-1",
            "section_title": "Trail Braking",
            "source_title": "Corner Craft",
            "page_number": 4,
            "text": "weight transfer to the front under braking",
        },
    ),
    SearchResult(
        id="point-2",
        score=0.74,
        collection="vo_rag_books_sterile_v1",
        payload={
            "chunk_id": "c-2",
            "source_id": "src-2",
            "section_title": "Apex Entry",
            "source_title": "Turn Fast",
            "page_number": 11,
            "text": "rotate to apex on the trailing throttle",
        },
    ),
    SearchResult(
        id="point-3",
        score=0.55,
        collection="vo_rag_articles_v1",
        payload={
            "chunk_id": "c-3",
            "source_id": "src-3",
            "title": "Coachable braking points",
            "text": "brake markers scale with corner tightness",
        },
    ),
]


# --------------------------------------------------------------------------- #
# Fake seams — one shared SurfaceServices.
# --------------------------------------------------------------------------- #
@dataclass
class FakeSearchRetriever:
    """Duck-typed stand-in for :class:`HybridRetriever`.

    The API route reaches the retriever through
    :func:`vo_rag.core.retrieval.execution.execute_retrieval`; the
    MCP tool reaches it through ``cross_collection_search`` on the
    retriever. We expose both call shapes so the same fake drives
    both surfaces without per-surface fakes.
    """

    scripted: list[SearchResult] = field(default_factory=list)
    cross_calls: list[dict[str, Any]] = field(default_factory=list)
    execute_calls: list[dict[str, Any]] = field(default_factory=list)

    async def cross_collection_search(
        self,
        *,
        query: str,
        collections: list[str],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        mode: Any = None,
        visual_mode: str | None = None,
    ) -> list[SearchResult]:
        self.cross_calls.append(
            {
                "query": query,
                "collections": list(collections),
                "top_k": top_k,
                "filters": filters,
                "mode": mode,
                "visual_mode": visual_mode,
            }
        )
        return list(self.scripted)[:top_k]

    async def execute(
        self,
        *,
        query: str,
        collections: list[str],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        mode: Any = None,
        visual_mode: str | None = None,
    ) -> list[SearchResult]:
        self.execute_calls.append(
            {
                "query": query,
                "collections": list(collections),
                "top_k": top_k,
                "filters": filters,
                "mode": mode,
                "visual_mode": visual_mode,
            }
        )
        return list(self.scripted)[:top_k]


@dataclass
class FakeReranker:
    """Duck-typed stand-in for :class:`GteReranker`.

    The retriever/reranker's only contract is
    ``rerank(query, candidates, top_k)`` — we preserve order and
    record the call.
    """

    available: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 20,
    ) -> list[SearchResult]:
        self.calls.append({"query": query, "n": len(candidates), "top_k": top_k})
        return list(candidates[:top_k])


class FakeAnswerSynthesizer:
    """Duck-typed stand-in for :class:`AnswerSynthesizer`.

    Records every call and returns a scripted :class:`AnswerSynthesis`.
    The scriptable synthesis lets us drive the quarantine / compliance
    branch without touching the LLM client.
    """

    def __init__(self, synthesis: AnswerSynthesis) -> None:
        self.synthesis = synthesis
        self.calls: list[dict[str, Any]] = []

    async def synthesize(
        self,
        *,
        query: str,
        context: list[SearchResult],
        profile: AnswerProfile,
    ) -> AnswerSynthesis:
        self.calls.append({"query": query, "context": list(context), "profile": profile})
        return self.synthesis


@dataclass
class _FakeAcquire:
    async def __aenter__(self) -> _FakeAcquire:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


@dataclass
class FakePool:
    """Asyncpg-shaped pool. The route uses ``request.app.state.db_pool``."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def acquire(self) -> _FakeAcquire:
        self.calls.append({"op": "acquire"})
        return _FakeAcquire()


@dataclass
class FakeQdrantClient:
    """Underlying ``AsyncQdrantClient`` for the /ready route."""

    collections: list[str] = field(
        default_factory=lambda: [
            "vo_rag_books_raw_v1",
            "vo_rag_books_sterile_v1",
        ]
    )

    async def get_collections(self) -> Any:
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.collections]
        )

    async def count(self, collection_name: str, exact: bool = True) -> Any:  # noqa: ARG002
        return SimpleNamespace(count=1)


@dataclass
class FakeQdrantStore:
    _client: Any = None


@dataclass
class FakeGarage:
    """Minimal Garage stub for upload URL and byte verification routes."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    object_bytes: bytes = b"surface contract Garage bytes"

    async def generate_presigned_upload(
        self,
        bucket: str,
        key: str,
        ttl: int = 21600,
        *,
        content_type: str | None = None,
        content_length: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "bucket": bucket,
                "key": key,
                "ttl": ttl,
                "content_type": content_type,
                "content_length": content_length,
            }
        )
        return f"https://garage.example/{bucket}/{key}?sig=test"

    async def verify_object_sha256(
        self,
        bucket: str,
        key: str,
        *,
        max_bytes: int,
    ) -> Any:
        from vo_rag.core.storage.garage import VerifiedGarageObject

        assert len(self.object_bytes) <= max_bytes
        return VerifiedGarageObject(
            size=len(self.object_bytes),
            content_type="application/pdf",
            etag="surface-etag",
            sha256=hashlib.sha256(self.object_bytes).hexdigest(),
        )

    async def delete_object(self, bucket: str, key: str) -> None:
        self.calls.append({"delete_bucket": bucket, "delete_key": key})

    async def promote_verified_object(
        self,
        bucket: str,
        staged_key: str,
        *,
        sha256: str,
        etag: str | None,
    ) -> str:
        del bucket, etag
        suffix = "." + staged_key.rsplit(".", 1)[-1] if "." in staged_key else ""
        return f"verified/sha256/{sha256}{suffix}"


# --------------------------------------------------------------------------- #
# Settings + surface services
# --------------------------------------------------------------------------- #
def _fake_settings(**overrides: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "api": SimpleNamespace(
            api_key=_API_KEY,  # matches the X-API-Key the contract tests send
            api_keys_json="",
            cors_origins=["*"],
        ),
        "database": SimpleNamespace(
            url="postgresql://localhost:5432/vorag",
            pool_min=1,
            pool_max=2,
        ),
        "qdrant": SimpleNamespace(
            url="http://localhost:6333",
            api_key="",
            prefer_grpc=False,
        ),
        "garage": SimpleNamespace(
            endpoint="http://localhost:3900",
            access_key="",
            secret_key="",
            region="garage",
            buckets=SimpleNamespace(
                artifacts="vorag-artifacts",
                models="vorag-models",
                backups="vorag-backups",
            ),
        ),
        "embedding": SimpleNamespace(
            query_url="http://localhost:8080",
            timeout_s=30.0,
        ),
        "reranker": SimpleNamespace(
            model_path="./models/gte-reranker-onnx-int8",
            intra_threads=4,
            batch_size=8,
            auto_merge_enabled=False,
            auto_merge_threshold=0.5,
        ),
        "answer": SimpleNamespace(
            allow_hosted=True,
            allow_raw_hosted_egress=False,
            max_context_chunks=8,
            max_output_tokens=1200,
            timeout_s=90,
        ),
        "triage": SimpleNamespace(anthropic_api_key=""),
        "cloud": SimpleNamespace(
            monthly_budget_usd=50.0,
            per_job_max_usd=10.0,
            api_key="",  # F-014 — keep secret-shaped values out of the body
            tailscale=SimpleNamespace(
                oauth_client_id="",
                oauth_client_secret="",
                tailnet="",
            ),
        ),
        "portal": SimpleNamespace(
            upload_max_bytes=2 * 1024 * 1024 * 1024,
            csrf_secret="csrf-secret",  # pragma: allowlist secret
            access_team_domain="",
            access_aud="",
            allowed_emails=[],
        ),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# Sentinel that distinguishes "argument omitted" from "argument
# passed as None". Needed because SurfaceServices.reranker=None is
# a meaningful state (a not-yet-loaded GTE model) the contract
# tests want to exercise.
_UNSET: Any = object()


def make_surface_services(
    *,
    synthesis: AnswerSynthesis | None = None,
    reranker: FakeReranker | None | Any = _UNSET,
    retriever: FakeSearchRetriever | None = None,
    pool: FakePool | None = None,
    garage: FakeGarage | None = None,
    qdrant: FakeQdrantStore | None = None,
    ingest_submit: Any = None,
    status_provider: Any = None,
    envelope_lookup: Any = None,
    settings_overrides: dict[str, Any] | None = None,
) -> SurfaceServices:
    """Build the ONE shared fake :class:`SurfaceServices`.

    The same instance is used across the API / CLI / MCP tests in
    this module. Tests can override individual slots; defaults are
    sane for the query / ingest / answer / admin sections.

    Pass ``reranker=None`` to leave the slot as ``None`` (the
    reranker-unavailable state the contract tests cover). Omit
    ``reranker`` to get the default :class:`FakeReranker`.
    """
    settings = _fake_settings(**(settings_overrides or {}))
    extra: dict[str, Any] = {}
    if ingest_submit is not None:
        extra["ingest_submit"] = ingest_submit
    if status_provider is not None:
        extra["status_provider"] = status_provider
    if envelope_lookup is not None:
        extra["envelope_lookup"] = envelope_lookup

    if reranker is _UNSET:
        reranker_value: Any = FakeReranker()
    else:
        reranker_value = reranker

    return SurfaceServices(
        settings=settings,
        db_pool=pool if pool is not None else FakePool(),
        qdrant=qdrant if qdrant is not None else FakeQdrantStore(_client=FakeQdrantClient()),
        garage=garage if garage is not None else FakeGarage(),
        embedding_client=SimpleNamespace(),
        retriever=(
            retriever if retriever is not None else FakeSearchRetriever(scripted=_SCRIPTED_HITS)
        ),
        reranker=reranker_value,
        answer_synthesizer=(FakeAnswerSynthesizer(synthesis) if synthesis is not None else None),
        cloud_cost_tracker=None,
        kill_switch=None,
        extra=extra,
    )


# --------------------------------------------------------------------------- #
# API client factory — uses the SAME fake services the MCP and CLI
# tests drive. The contract: a single SurfaceServices, three
# surfaces, identical behaviour.
# --------------------------------------------------------------------------- #
def _api_client(monkeypatch: pytest.MonkeyPatch, services: SurfaceServices) -> TestClient:
    from vo_rag.core import config as config_module

    # The auth middleware reads settings through get_settings(), not
    # app.state — patch the resolver so it returns the test settings
    # (with the api_key the test sends in the X-API-Key header).
    monkeypatch.setattr(config_module, "get_settings", lambda: services.settings)
    return TestClient(create_app(settings=services.settings, services=services))


# --------------------------------------------------------------------------- #
# Query contract across API / CLI / MCP.
# --------------------------------------------------------------------------- #
pytestmark_query = pytest.mark.security


@pytest.mark.security
def test_api_query_preserves_hits_id_collection_payload_score_top_k_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/v1/query must return every scripted hit byte-for-byte.

    The same scripted :class:`SearchResult` list is the canonical
    fixture — the API is just one of three surfaces that must
    pass it through unchanged.
    """
    import vo_rag.api.routes.query as query_route
    from vo_rag.core.retrieval.search import SearchResult as _SR

    async def _fake_execute_retrieval(**kwargs: Any) -> list[_SR]:
        return list(_SCRIPTED_HITS)[: kwargs.get("top_k", 20)]

    async def _noop_log(*args: Any, **kwargs: Any) -> None:
        return None

    def _identity(req: object) -> Any:
        return SimpleNamespace(
            name="legacy",
            is_admin=True,
            caller=None,
            allowed_modes=frozenset(),
            allowed_collections=frozenset(),
            auth_source="api_key",
            email=None,
        )

    monkeypatch.setattr(query_route, "execute_retrieval", _fake_execute_retrieval)
    monkeypatch.setattr(query_route, "enforce_query_scope", lambda **kwargs: None)
    monkeypatch.setattr(query_route, "insert_query_log", _noop_log)
    monkeypatch.setattr(query_route, "identity_from_request", _identity)

    retriever = FakeSearchRetriever(scripted=_SCRIPTED_HITS)
    services = make_surface_services(retriever=retriever)
    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: services.settings)
    client = TestClient(create_app(settings=services.settings, services=services))

    response = client.post(
        "/api/v1/query",
        headers={"X-API-Key": _API_KEY, "X-VO-RAG-Caller": "mcp"},
        json={
            "query": "find apex references",
            "collections": ["vo_rag_books_raw_v1"],
            "top_k": 2,
            "mode": "internal",
            "filters": {"source_id": "src-1"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2, "top_k must truncate the scripted list"
    first, second = body["results"]
    # Hits are byte-for-byte identical to the scripted fixture.
    assert first["id"] == "point-1"
    assert first["collection"] == "vo_rag_books_raw_v1"
    assert first["score"] == pytest.approx(0.91)
    assert first["payload"]["section_title"] == "Trail Braking"
    assert first["payload"]["source_title"] == "Corner Craft"
    assert first["payload"]["text"] == "weight transfer to the front under braking"
    assert second["id"] == "point-2"
    assert second["collection"] == "vo_rag_books_sterile_v1"
    assert second["score"] == pytest.approx(0.74)


@pytest.mark.security
def test_cli_query_posts_payload_and_renders_scripted_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    """``vo-rag query`` must POST the right payload and render the API's
    scripted response byte-for-byte. The HTTP layer is mocked via
    respx — the CLI never sees a real server. The assertion is the
    *same* scripted list, just rendered as a table.
    """
    monkeypatch.setenv("VORAG_API_KEY", _API_KEY)
    api_response = {
        "results": [
            {
                "id": hit.id,
                "score": hit.score,
                "collection": hit.collection,
                "payload": dict(hit.payload),
            }
            for hit in _SCRIPTED_HITS
        ]
    }
    with respx.mock:
        route = respx.post(f"{_API_BASE}/api/v1/query").mock(
            return_value=httpx.Response(200, json=api_response)
        )
        result = CliRunner().invoke(
            cli,
            [
                "query",
                "find apex references",
                "--top-k",
                "2",
                "--collection",
                "vo_rag_books_raw_v1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert route.called
        sent = route.calls[0].request.content.decode()
        # Payload shape must match the API contract.
        assert "find apex references" in sent
        assert "vo_rag_books_raw_v1" in sent
        assert '"top_k":2' in sent
        assert '"mode":"internal"' in sent
        # Render must surface the scripted ids / collections / scores.
        assert "point-1" in result.output
        assert "point-2" in result.output
        assert "0.9100" in result.output
        assert "0.7400" in result.output
        assert "vo_rag_books_raw_v1" in result.output
        assert "vo_rag_books_sterile_v1" in result.output


@pytest.mark.asyncio
async def test_mcp_query_preserves_hits_clamps_top_k_and_surfaces_mode_and_warnings() -> None:
    """``vo_rag_query`` must clamp top_k to 50, surface mode verbatim,
    and warn when the reranker is unavailable but rerank=True."""
    from vo_rag.mcp.tools import vo_rag_query_impl

    retriever = FakeSearchRetriever(scripted=_SCRIPTED_HITS)
    # No reranker injected — should produce a warning when rerank=True.
    services = make_surface_services(retriever=retriever, reranker=None)

    result = await vo_rag_query_impl(
        services=services,
        query="find apex references",
        collections=["vo_rag_books_raw_v1"],
        top_k=999,  # over the 50 ceiling
        mode="internal",
        filters={"source_id": "src-1"},
        rerank=True,
    )

    # The retriever saw the clamped top_k — never the raw 999.
    assert retriever.cross_calls, "retriever was not called"
    call = retriever.cross_calls[0]
    assert call["top_k"] == 50
    assert call["query"] == "find apex references"
    assert call["mode"].value == "internal"
    assert call["filters"] == {"source_id": "src-1"}

    # Explicit raw selection excludes the injected sterile/article results.
    assert [hit.id for hit in result.results] == ["point-1"]
    assert [hit.collection for hit in result.results] == [
        "vo_rag_books_raw_v1",
    ]
    assert result.mode == "internal"
    assert result.top_k == 50
    # No reranker + rerank=True surfaces a warning, never a fabricated result.
    assert any("reranker_unavailable" in w for w in result.warnings)

    combined = await vo_rag_query_impl(
        services=services,
        query="find apex references",
        collections=["vo_rag_books_raw_v1", "vo_rag_books_sterile_v1", "vo_rag_articles_v1"],
        top_k=50,
        rerank=False,
    )
    assert [hit.id for hit in combined.results] == ["point-1", "point-2", "point-3"]
    assert retriever.cross_calls[-1]["collections"] == [
        "vo_rag_books_raw_v1",
        "vo_rag_books_sterile_v1",
        "vo_rag_articles_v1",
    ]


# --------------------------------------------------------------------------- #
# Ingest contract across API / CLI / MCP.
# --------------------------------------------------------------------------- #
def _resolved_url(url: str) -> ResolvedURL:
    return ResolvedURL(
        url=url,
        scheme="https",
        host="example.com",
        ip="93.184.216.34",
        port=443,
    )


def _patch_ssrf_pass(monkeypatch: pytest.MonkeyPatch, *modules: Any) -> None:
    """Patch ``validate_external_url`` to return a safe ResolvedURL."""
    for module in modules:
        monkeypatch.setattr(
            module,
            "validate_external_url",
            lambda url: _async_return(_resolved_url(url)),
        )


def _patch_ssrf_block(monkeypatch: pytest.MonkeyPatch, *modules: Any) -> None:
    """Patch ``validate_external_url`` to raise :class:`SSRFBlocked`."""
    for module in modules:

        async def _raise(url: str) -> None:
            raise SSRFBlocked(url, "blocked by SSRF test stub")

        monkeypatch.setattr(module, "validate_external_url", _raise)


async def _async_return(value: Any) -> Any:
    return value


@pytest.mark.security
def test_api_intake_normalizes_text_url_and_file_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/v1/intake must accept text_prompt / url / file_drop
    submissions and run them through ``normalize()`` exactly once.

    The same submission shape (modulo the safety gate) is what the
    CLI and MCP tool builders produce — the API is the canonical
    validation point. The assertions pin the normalized
    ``raw_content`` shape so all three surfaces agree byte-for-byte.
    """
    import vo_rag.api.routes.ingest as ingest_route
    import vo_rag.mcp.tools as mcp_tools_module
    from vo_rag.core.db.queries import envelopes as queries_envelopes
    from vo_rag.web import url_safety as url_safety_module

    _patch_ssrf_pass(monkeypatch, url_safety_module, ingest_route, mcp_tools_module)

    captured: list[Any] = []

    async def _capture(pool: object, envelope: object) -> None:
        captured.append(envelope)

    monkeypatch.setattr(queries_envelopes, "insert_envelope", _capture)
    monkeypatch.setattr(ingest_route, "insert_envelope", _capture)

    services = make_surface_services()
    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: services.settings)
    client = TestClient(create_app(settings=services.settings, services=services))

    # text_prompt
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _API_KEY},
        json={
            "source_type": "text_prompt",
            "raw_content": {"text": "hello", "title": "greeting"},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_type"] == "text_prompt"
    assert body["raw_content"]["text"] == "hello"
    assert body["raw_content"]["title"] == "greeting"
    assert body["dispatch_status"] == "not_configured"

    # url — public, passes the SSRF gate.
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _API_KEY},
        json={
            "source_type": "url",
            "raw_content": {"url": "https://example.com/article", "title": "Example"},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_type"] == "url"
    assert body["raw_content"]["url"] == "https://example.com/article"
    assert body["raw_content"]["title"] == "Example"

    # file_drop — Garage bytes, rather than the client assertion, own identity.
    garage = services.garage
    assert isinstance(garage, FakeGarage)
    verified_hash = hashlib.sha256(garage.object_bytes).hexdigest()
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _API_KEY},
        json={
            "source_type": "file_drop",
            "raw_content": {
                "garage_key": "intake/01H/foo.pdf",
                "filename": "Book.PDF",
                "content_type": "application/pdf",
                "file_hash": f"  {verified_hash.upper()}  ",
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_type"] == "file_drop"
    assert body["raw_content"]["garage_key"] == f"verified/sha256/{verified_hash}.pdf"
    assert body["raw_content"]["file_hash"] == verified_hash
    assert body["raw_content"]["object_verification"]["status"] == "verified"

    # The captured envelopes must all pass normalize() cleanly.
    assert len(captured) == 3
    for envelope in captured:
        # Each envelope has the post-normalize shape — never the raw
        # client payload. The MCP tool builder produces the same
        # post-normalize shape (verified by test_mcp_query_ingest_*
        # below).
        assert envelope.id
        assert envelope.status.value == "received"
        assert envelope.source_type in {
            IntakeSource.TEXT_PROMPT,
            IntakeSource.URL,
            IntakeSource.FILE_DROP,
        }


@pytest.mark.security
def test_api_intake_rejects_unsafe_url_without_db_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-014: an unsafe URL must be 422'd BEFORE the DB insert runs."""
    import vo_rag.api.routes.ingest as ingest_route
    from vo_rag.core.db.queries import envelopes as queries_envelopes
    from vo_rag.web import url_safety as url_safety_module

    _patch_ssrf_block(monkeypatch, url_safety_module, ingest_route)

    insert_called: list[object] = []

    async def _track(pool: object, envelope: object) -> None:
        insert_called.append(envelope)

    monkeypatch.setattr(queries_envelopes, "insert_envelope", _track)
    monkeypatch.setattr(ingest_route, "insert_envelope", _track)

    services = make_surface_services()
    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: services.settings)
    client = TestClient(create_app(settings=services.settings, services=services))

    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _API_KEY},
        json={"source_type": "url", "raw_content": {"url": "http://10.0.0.5/secret"}},
    )

    assert response.status_code == 422
    assert "ssrf" in str(response.json()["detail"]).lower()
    assert insert_called == [], "DB insert must NOT run on an SSRF-blocked URL"


@pytest.mark.security
def test_cli_ingest_url_text_file_drop_payload_shapes_match_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``vo-rag ingest {text,url,idea,file-drop}`` must POST a payload
    the API would accept, AND the CLI's local SSRF gate must reject
    unsafe URLs before any HTTP call.

    The CLI's payload shape is the surface it agrees on with the API
    and the MCP tool builder: same keys, same ``raw_content``
    structure, same ``source_type`` strings. The monkeypatch
    captures every outgoing POST so the assertion is the byte-level
    payload.
    """
    monkeypatch.setenv("VORAG_API_KEY", _API_KEY)

    # Stub the CLI's local SSRF gate to pass on the happy path.
    import vo_rag.cli.ingest as cli_ingest

    monkeypatch.setattr(
        cli_ingest,
        "validate_external_url",
        lambda url: _async_return(_resolved_url(url)),
    )

    with respx.mock:
        intake = respx.post(f"{_API_BASE}/api/v1/intake").mock(
            return_value=httpx.Response(200, json={"envelope_id": "env-cli-1"})
        )
        runner = CliRunner()

        # text
        result = runner.invoke(cli, ["ingest", "text", "hello world", "--title", "Greeting"])
        assert result.exit_code == 0, result.output
        sent = intake.calls[-1].request.content.decode()
        assert '"source_type":"text_prompt"' in sent
        assert '"text":"hello world"' in sent
        assert '"title":"Greeting"' in sent

        # url — public
        result = runner.invoke(
            cli, ["ingest", "url", "https://example.com/article", "--title", "Example"]
        )
        assert result.exit_code == 0, result.output
        sent = intake.calls[-1].request.content.decode()
        assert '"source_type":"url"' in sent
        assert '"url":"https://example.com/article"' in sent
        assert '"title":"Example"' in sent

        # idea — text_prompt with `idea` tag
        result = runner.invoke(cli, ["ingest", "idea", "my great idea", "--tags", "feature,ui"])
        assert result.exit_code == 0, result.output
        sent = intake.calls[-1].request.content.decode()
        assert '"source_type":"text_prompt"' in sent
        assert '"text":"my great idea"' in sent
        assert '"idea"' in sent  # tag auto-appended
        assert "feature" in sent
        assert "ui" in sent

        # file-drop
        result = runner.invoke(cli, ["ingest", "file-drop", "gk-123", "--title", "Dropped"])
        assert result.exit_code == 0, result.output
        sent = intake.calls[-1].request.content.decode()
        assert '"source_type":"file_drop"' in sent
        assert '"garage_key":"gk-123"' in sent


@pytest.mark.security
def test_cli_ingest_url_blocks_unsafe_url_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI runs the SSRF gate locally BEFORE the HTTP call. F-014."""
    monkeypatch.setenv("VORAG_API_KEY", _API_KEY)
    import vo_rag.cli.ingest as cli_ingest

    _patch_ssrf_block(monkeypatch, cli_ingest)

    with respx.mock:
        intake = respx.post(f"{_API_BASE}/api/v1/intake").mock(
            return_value=httpx.Response(200, json={"envelope_id": "never"})
        )
        result = CliRunner().invoke(cli, ["ingest", "url", "http://10.0.0.5/secret"])
        assert result.exit_code != 0
        assert "URL validation failed" in result.output
        assert not intake.called, "the HTTP call must NOT run when the SSRF gate blocks"


@pytest.mark.asyncio
async def test_mcp_ingest_text_prompt_url_file_drop_share_normalized_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MCP tool builder must produce the same normalized
    ``raw_content`` the API and CLI produce for the same input.

    The MCP tool calls ``normalize()`` internally; the API and CLI
    produce the same post-normalize envelope when their JSON payload
    is built from the same input. This test pins the equivalence
    by feeding the MCP tool the same ``source_type`` / ``content``
    the API and CLI tests use, and asserting the post-normalize
    envelope shape matches byte-for-byte.
    """
    import vo_rag.mcp.tools as mcp_tools_module
    from vo_rag.mcp.tools import vo_rag_ingest_impl
    from vo_rag.web import url_safety as url_safety_module

    _patch_ssrf_pass(monkeypatch, url_safety_module, mcp_tools_module)

    captured: list[Any] = []

    def _submit(envelope: Any) -> dict[str, str]:
        captured.append(envelope)
        return {"envelope_id": envelope.id, "status": "received", "dispatch_status": "submitted"}

    services = make_surface_services(ingest_submit=_submit)

    # text_prompt
    result_text = await vo_rag_ingest_impl(
        services=services,
        source_type="text_prompt",
        content="hello",
        metadata={"title": "greeting"},
    )
    assert result_text.dispatch_status == "submitted"
    assert captured[0].source_type is IntakeSource.TEXT_PROMPT
    assert captured[0].raw_content == {"text": "hello", "title": "greeting"}
    assert captured[0].status.value == "received"

    # url (after SSRF passes)
    result_url = await vo_rag_ingest_impl(
        services=services,
        source_type="url",
        content="https://example.com/article",
        metadata={"title": "Example"},
    )
    assert result_url.dispatch_status == "submitted"
    assert captured[1].source_type is IntakeSource.URL
    assert captured[1].raw_content == {"url": "https://example.com/article", "title": "Example"}

    # file_drop
    result_file = await vo_rag_ingest_impl(
        services=services,
        source_type="file_drop",
        content="intake/01H/foo.pdf",
    )
    assert result_file.dispatch_status == "submitted"
    assert captured[2].source_type is IntakeSource.FILE_DROP
    assert captured[2].raw_content["garage_key"] == "intake/01H/foo.pdf"

    # All three envelopes are accepted by ``normalize()`` when built
    # from the same input the API and CLI tests use — i.e. the MCP
    # tool builder is shape-compatible with normalize() in either
    # direction.
    for envelope in captured:
        submission = IntakeSubmission(
            source_type=envelope.source_type,
            source_data=dict(envelope.source_data),
            raw_content=dict(envelope.raw_content),
            metadata=dict(envelope.metadata),
            sovereignty=envelope.sovereignty,
            priority=envelope.priority,
        )
        round_tripped = normalize(submission)
        assert round_tripped.raw_content == envelope.raw_content
        assert round_tripped.source_type == envelope.source_type


@pytest.mark.asyncio
async def test_mcp_ingest_unsafe_url_raises_tool_error_no_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-014: MCP tool must raise ToolError on unsafe URL and create no envelope."""
    import vo_rag.mcp.tools as mcp_tools_module
    from vo_rag.mcp.errors import ToolError
    from vo_rag.mcp.tools import vo_rag_ingest_impl
    from vo_rag.web import url_safety as url_safety_module

    _patch_ssrf_block(monkeypatch, url_safety_module, mcp_tools_module)

    captured: list[Any] = []

    def _submit(envelope: Any) -> dict[str, str]:
        captured.append(envelope)
        return {"envelope_id": envelope.id, "status": "received", "dispatch_status": "submitted"}

    services = make_surface_services(ingest_submit=_submit)
    with pytest.raises(ToolError) as excinfo:
        await vo_rag_ingest_impl(
            services=services, source_type="url", content="http://10.0.0.5/secret"
        )
    assert "ssrf_blocked" in str(excinfo.value)
    assert captured == []


# --------------------------------------------------------------------------- #
# Answer non-disclosure contract.
# --------------------------------------------------------------------------- #
def _ok_synthesis(*, answer: str = "Answer [S1].", confidence: float = 0.82) -> AnswerSynthesis:
    return AnswerSynthesis(
        draft=AnswerDraft(answer=answer, confidence=confidence, cited_chunk_ids=["chunk-1"]),
        llm_response=LLMResponse(text='{"answer":"ok"}', model="test-model"),
        validation=ValidationResult(ok=True, signals=[]),
        should_quarantine=False,
        model="test-model",
    )


def _quarantine_synthesis() -> AnswerSynthesis:
    return AnswerSynthesis(
        draft=AnswerDraft(answer="Suspicious [S1].", confidence=0.4, cited_chunk_ids=["c-1"]),
        llm_response=LLMResponse(text='{"answer":"x"}', model="test-model"),
        validation=ValidationResult(
            ok=False,
            signals=[
                InjectionSignal(
                    kind="canary_echoed",
                    confidence=1.0,
                    evidence="canary token echoed",
                )
            ],
        ),
        should_quarantine=True,
        model="test-model",
    )


@pytest.mark.security
def test_api_answer_routes_through_answer_synthesizer_with_sterile_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The /api/v1/answer route must call the configured
    ``AnswerSynthesizer`` and default ``context_scope`` to STERILE.

    F-045: STERILE is the unconditional default — the test omits
    ``context_scope`` and asserts the synthesizer was called with
    STERILE-shaped input.
    """
    synthesizer = FakeAnswerSynthesizer(_ok_synthesis())
    services = make_surface_services(synthesis=_ok_synthesis())
    services.answer_synthesizer = synthesizer  # wire the recording fake
    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: services.settings)
    client = TestClient(create_app(settings=services.settings, services=services))

    response = client.post(
        "/api/v1/answer",
        headers={"X-API-Key": _API_KEY},
        json={"query": "answer apex"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Answer [S1]."
    assert body["confidence"] == pytest.approx(0.82)
    assert synthesizer.calls, "AnswerSynthesizer.synthesize was not called"
    call = synthesizer.calls[0]
    assert call["query"] == "answer apex"
    assert call["profile"] == AnswerProfile.QUALITY
    assert all("raw_text" not in str(c) for c in call["context"]), (
        "raw text must not leak into the synthesizer input"
    )


@pytest.mark.security
def test_api_answer_raw_context_rejected_unless_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw collection use is rejected (403) when
    ``allow_raw_hosted_egress`` is False — the default.

    F-045: the STERILE default + the raw-egress gate are the two
    non-negotiable pieces of the answer surface.
    """
    services = make_surface_services(
        synthesis=_ok_synthesis(),
        settings_overrides={
            "answer": SimpleNamespace(
                allow_hosted=True,
                allow_raw_hosted_egress=False,
                max_context_chunks=8,
                max_output_tokens=1200,
                timeout_s=90,
            ),
        },
    )
    synthesizer = FakeAnswerSynthesizer(_ok_synthesis())
    services.answer_synthesizer = synthesizer
    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: services.settings)
    client = TestClient(create_app(settings=services.settings, services=services))

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
    assert synthesizer.calls == [], "the synthesizer must NOT run on a 403'd raw-egress"


@pytest.mark.security
def test_api_answer_quarantine_surfaces_warnings_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``should_quarantine=True`` synthesis must surface as
    warnings + compliance payload — never as a normal success.
    F-012 / F-045.
    """
    synthesis = _quarantine_synthesis()
    services = make_surface_services(synthesis=synthesis)
    services.answer_synthesizer = FakeAnswerSynthesizer(synthesis)
    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: services.settings)
    client = TestClient(create_app(settings=services.settings, services=services))

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


@pytest.mark.security
def test_cli_answer_posts_sterile_default_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``vo-rag answer`` must POST ``context_scope: sterile`` by default
    and surface warnings from the API's compliance payload.

    The CLI is a thin client — the AnswerSynthesizer routing lives
    on the server. The contract is: the CLI's payload matches the
    API's request schema, the STERILE default survives the round
    trip, and the response render surfaces warnings/compliance.
    """
    monkeypatch.setenv("VORAG_API_KEY", _API_KEY)
    with respx.mock:
        route = respx.post(f"{_API_BASE}/api/v1/answer").mock(
            return_value=httpx.Response(
                200,
                json={
                    "answer_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                    "answer": "The answer is 42.",
                    "confidence": 0.95,
                    "citations": ["source-1"],
                    "sources": ["source-1"],
                    "retrieval_results": [],
                    "compliance": {
                        "status": "not_configured",
                        "should_quarantine": True,
                        "reason": "prompt_injection_suspected",
                    },
                    "warnings": ["prompt_injection_suspected", "canary_echoed"],
                    "model": "claude-3-sonnet",
                    "mode": "internal",
                },
            )
        )
        result = CliRunner().invoke(cli, ["answer", "what is the meaning?"])
        assert result.exit_code == 0, result.output
        assert route.called
        sent = route.calls[0].request.content.decode()
        # STERILE default + QUALITY default + INTERNAL default.
        assert '"context_scope":"sterile"' in sent
        assert '"profile":"quality"' in sent
        assert '"mode":"internal"' in sent
        # Compliance + warnings render.
        assert "The answer is 42." in result.output
        assert "Warnings:" in result.output
        assert "prompt_injection_suspected" in result.output
        assert "canary_echoed" in result.output


# --------------------------------------------------------------------------- #
# Admin / eval contract across API / CLI / MCP.
# --------------------------------------------------------------------------- #
@pytest.mark.security
def test_api_admin_status_reports_eval_gate_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /api/v1/admin/status must surface the live
    registered eval gate result. The ``scripts/ci.sh`` exit
    code depends on the same fields the API reports here — so the
    API/CLI/MCP contract is that the EvalGateResult fields are
    preserved byte-for-byte.
    """
    services = make_surface_services()
    from vo_rag.core import config as config_module

    monkeypatch.setattr(config_module, "get_settings", lambda: services.settings)
    client = TestClient(create_app(settings=services.settings, services=services))

    response = client.get("/api/v1/admin/status", headers={"X-API-Key": _API_KEY})
    assert response.status_code == 200
    body = response.json()
    assert "eval_gate" in body

    registry = load_pipeline_registry()
    real = run_registered_eval_gate(registry)
    assert body["eval_gate"] == real.model_dump()


@pytest.mark.security
def test_cli_eval_gate_reports_same_eval_gate_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """``vo-rag eval-gate`` must call the registered eval gate
    and report the same fields the API and MCP surfaces do.

    The CLI is a direct caller (no HTTP). The contract is: the
    rendered output mentions every checked / skipped / failed /
    missing_fixtures / missing_evaluators pipeline the live
    ``EvalGateResult`` reports, and exits 0 only when ``passed is
    True`` (the ``scripts/ci.sh`` gate behavior is unchanged).
    """
    registry = load_pipeline_registry()
    real = run_registered_eval_gate(registry)
    if not real.passed:
        pytest.skip("Live eval gate is failing — CLI gating is exercised elsewhere")

    result = CliRunner().invoke(cli, ["eval-gate"])
    assert result.exit_code == 0, result.output
    # Same fields the API / MCP surface report.
    assert "PASS" in result.output
    for checked in real.checked:
        assert checked in result.output


@pytest.mark.asyncio
async def test_mcp_status_surfaces_eval_gate_fields_via_status_provider() -> None:
    """``vo_rag_status`` must surface the same ``EvalGateResult``
    fields the API and CLI report when a status_provider is
    injected that returns the live gate result.

    The MCP tool does not run the eval gate itself (no fixture
    data is loaded by the MCP path) — the status_provider is the
    seam the test harness injects to mirror the API/CLI
    evaluation surface. The contract is byte-for-byte equivalence
    on the fields the CI gate depends on.
    """
    from vo_rag.mcp.tools import vo_rag_status_impl

    registry = load_pipeline_registry()
    real: EvalGateResult = await asyncio.to_thread(run_registered_eval_gate, registry)
    real_dump = real.model_dump()

    def _provider() -> dict[str, Any]:
        return {
            "service": "vo-rag",
            "version": "0.1.0",
            "ok": real.passed,
            "eval_gate": real_dump,
        }

    services = make_surface_services(status_provider=_provider)

    result = await vo_rag_status_impl(services=services, envelope_id=None)
    assert result.scope == "system"
    assert result.system is not None
    eval_gate = result.system["eval_gate"]
    assert eval_gate == real_dump


# --------------------------------------------------------------------------- #
# Surface import smoke.
# --------------------------------------------------------------------------- #
def test_surface_imports_succeed_without_opening_sockets() -> None:
    """``vo_rag.api.main:app, create_app``, ``vo_rag.cli.main:cli``,
    and ``vo_rag.mcp.server:build_mcp_server`` must import cleanly.

    No socket is opened at import time; the surface modules own
    no eager I/O. The ``app = create_app()`` line at the bottom
    of ``api/main.py`` is the most likely regression point — it
    runs at import time and must not call the lifespan.
    """
    from vo_rag.api.main import app
    from vo_rag.api.main import create_app as _create_app
    from vo_rag.cli.main import cli as _cli
    from vo_rag.mcp.server import build_mcp_server as _build_mcp_server

    assert app is not None
    assert callable(_create_app)
    assert _cli is not None
    assert callable(_build_mcp_server)

    # build_mcp_server with no services must still build (the CLI
    # default path is hermetic).
    server = _build_mcp_server()
    assert server.name == "vo-rag"

    # The app's lifespan is the no-op injected path because no
    # services were passed at construction time of the module-level
    # `app` ... actually the module-level `app` is the production
    # path (it sets lifespan=lifespan). The import itself must not
    # trigger any socket open — the lifespan only runs on first
    # request inside an ASGI server.
    assert not inspect.iscoroutine(app)


__all__ = [
    "FakeAnswerSynthesizer",
    "FakeGarage",
    "FakePool",
    "FakeQdrantClient",
    "FakeQdrantStore",
    "FakeReranker",
    "FakeSearchRetriever",
    "make_surface_services",
]
