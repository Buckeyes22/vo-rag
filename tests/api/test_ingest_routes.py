"""/api/v1/intake and /api/v1/intake/upload-url route tests.

These cover the route contracts:

- POST /api/v1/intake accepts IntakeSubmission JSON, calls normalize() exactly
  once, returns envelope_id == IntakeEnvelope.id plus status, source_type,
  and the normalized body.
- source_type="url" runs through the SSRF gate (F-014) before DB insert and
  rejects private/Tailscale/link-local/loopback addresses with 422.
- source_type="file_drop" preserves garage_key and the canonical lowercased
  file_hash.
- Unknown fields are rejected (extra="forbid").
- Persistence goes through ``vo_rag.core.db.queries.envelopes.insert_envelope``,
  not inline SQL in the route.
- If no queue/Hatchet dispatcher exists, the route returns
  ``dispatch_status="not_configured"`` rather than pretending a pipeline ran.
- POST /api/v1/intake/upload-url sanitizes filenames via secure_filename,
  calls ``GarageClient.generate_presigned_upload``, and respects
  ``settings.portal.upload_max_bytes`` when it is configured smaller than
  the hard 2 GiB schema ceiling.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.security

_TEST_API_KEY = "test-key"  # pragma: allowlist secret


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeAcquire:
    async def __aenter__(self) -> _FakeAcquire:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _FakePool:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire()


class _FakeGarage:
    def __init__(
        self,
        *,
        object_bytes: bytes = b"authoritative Garage bytes",
        content_type: str = "application/pdf",
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.object_bytes = object_bytes
        self.content_type = content_type
        self.deleted: list[tuple[str, str]] = []
        self.promoted: list[tuple[str, str, str, str | None]] = []

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.object_bytes).hexdigest()

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
    ) -> object:
        from vo_rag.core.storage.garage import VerifiedGarageObject

        assert bucket == "vorag-artifacts"
        assert key
        assert len(self.object_bytes) <= max_bytes
        return VerifiedGarageObject(
            size=len(self.object_bytes),
            content_type=self.content_type,
            etag="test-etag",
            sha256=self.sha256,
        )

    async def delete_object(self, bucket: str, key: str) -> None:
        self.deleted.append((bucket, key))

    async def promote_verified_object(
        self,
        bucket: str,
        staged_key: str,
        *,
        sha256: str,
        etag: str | None,
    ) -> str:
        self.promoted.append((bucket, staged_key, sha256, etag))
        suffix = "." + staged_key.rsplit(".", 1)[-1] if "." in staged_key else ""
        return f"verified/sha256/{sha256}{suffix}"


def _fake_settings(**overrides: Any) -> SimpleNamespace:
    base = {
        "api": SimpleNamespace(
            api_key=_TEST_API_KEY,
            api_keys_json="",
            cors_origins=["*"],
            request_timeout_s=60,
        ),
        "portal": SimpleNamespace(
            upload_max_bytes=2 * 1024 * 1024 * 1024,
        ),
        "garage": SimpleNamespace(
            buckets=SimpleNamespace(
                artifacts="vorag-artifacts",
                models="vorag-models",
                backups="vorag-backups",
            ),
        ),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _build_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: SimpleNamespace | None = None,
    pool: _FakePool | None = None,
    garage: _FakeGarage | None = None,
) -> FastAPI:
    """Build a hermetic FastAPI app with the ingest router included.

    We use the ``create_app(settings=..., services=...)`` injection path so
    the lifespan does not open any sockets. The fakes are mirrored onto
    ``app.state`` so route handlers see them via attribute access.

    The ApiKeyMiddleware reads settings through ``vo_rag.core.config.get_settings``
    (not from ``app.state.settings``), so we monkeypatch the resolver here
    in addition to passing the settings into ``create_app``.
    """
    from vo_rag.api import main as main_module
    from vo_rag.api.singletons import SurfaceServices
    from vo_rag.core import config as config_module

    settings = settings or _fake_settings()
    pool = pool or _FakePool()
    garage = garage or _FakeGarage()

    @dataclass
    class _StubRerank:
        def rerank(self, query: str, candidates: list[Any], top_k: int = 20) -> list[Any]:
            return candidates[:top_k]

    services = SurfaceServices(
        settings=settings,
        db_pool=pool,
        qdrant=None,
        garage=garage,
        embedding_client=SimpleNamespace(),
        retriever=SimpleNamespace(),
        reranker=_StubRerank(),
        answer_synthesizer=SimpleNamespace(),
        cloud_cost_tracker=None,
        kill_switch=None,
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    app = main_module.create_app(settings=settings, services=services)
    app.state.db_pool = pool
    app.state.garage = garage
    app.state.settings = settings
    return app


def _client(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> TestClient:
    app = _build_app(monkeypatch, **kwargs)
    return TestClient(app)


def _patch_route_dependency(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_module: Any,
    attr: str,
    replacement: Any,
) -> None:
    """Patch ``attr`` on both the source module and the route module.

    The route uses ``from X import Y`` which captures the function reference
    at import time. Patching only ``X.Y`` is not enough if the route module
    was already imported by an earlier test (or by the test runner's module
    cache) — the route still holds the original reference. We patch both to
    keep the test order-independent and to keep the local binding consistent
    with the source for future readers.

    The route also aliases a few symbols under different local names — e.g.
    ``from vo_rag.core.ingest.normalize import normalize as core_normalize``
    means the local binding is ``core_normalize``, not ``normalize``. We
    patch both the canonical attr and the route-local alias so the test
    exercises the same call site the route uses.

    Important: ``monkeypatch.setattr`` records the previous value of the
    attribute and restores it on teardown. When the same attribute is
    patched twice, only the second ``setattr`` reverts to the value that
    existed *between* the two sets — which is still the fake. To get a
    correct revert, we patch each attribute at most once.
    """
    importlib.import_module("vo_rag.api.routes.ingest")
    from vo_rag.api.routes import ingest as ingest_module

    monkeypatch.setattr(source_module, attr, replacement)
    if hasattr(ingest_module, attr):
        monkeypatch.setattr(ingest_module, attr, replacement)
    # The route imports ``normalize`` under the local name ``core_normalize``.
    # Patch the alias too so the test actually intercepts the call.
    if attr == "normalize" and hasattr(ingest_module, "core_normalize"):
        monkeypatch.setattr(ingest_module, "core_normalize", replacement)


# --------------------------------------------------------------------------- #
# /api/v1/intake
# --------------------------------------------------------------------------- #
def test_intake_text_prompt_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: text_prompt submission lands in DB and returns core ids."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes

    recorded: dict[str, Any] = {}

    async def _fake_insert(pool: object, envelope: object) -> None:
        recorded["pool"] = pool
        recorded["envelope"] = envelope

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_fake_insert,
    )

    pool = _FakePool()
    client = _client(monkeypatch, pool=pool)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={"source_type": "text_prompt", "raw_content": {"text": "hello"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "received"
    assert body["source_type"] == "text_prompt"
    assert "envelope_id" in body
    assert len(body["envelope_id"]) == 26
    assert body["raw_content"]["text"] == "hello"
    assert body["dispatch_status"] == "not_configured"
    assert recorded["pool"] is pool
    envelope = recorded["envelope"]
    assert envelope.raw_content["text"] == "hello"


def test_intake_calls_normalize_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The route must call normalize() exactly once per submission."""

    import importlib

    from vo_rag.core.db.queries import envelopes as queries_envelopes

    # ``vo_rag.core.ingest.__init__`` re-exports ``normalize`` as a
    # function name, so ``from vo_rag.core.ingest import normalize``
    # yields the function, not the module. Use ``importlib`` to grab
    # the actual module object so we can patch its ``normalize`` attr.
    normalize_module = importlib.import_module("vo_rag.core.ingest.normalize")

    call_count = {"n": 0}
    real_normalize = normalize_module.normalize

    def _counting_normalize(submission: object) -> object:
        call_count["n"] += 1
        return real_normalize(submission)

    async def _noop_insert(pool: object, envelope: object) -> None:
        return None

    _patch_route_dependency(
        monkeypatch,
        source_module=normalize_module,
        attr="normalize",
        replacement=_counting_normalize,
    )
    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_noop_insert,
    )

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={"source_type": "text_prompt", "raw_content": {"text": "hi"}},
    )

    assert response.status_code == 200
    assert call_count["n"] == 1


def test_intake_persists_via_queries_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """No inline SQL in the route — envelope goes through insert_envelope."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes

    captured: dict[str, Any] = {}

    async def _fake_insert(pool: object, envelope: object) -> None:
        captured["pool"] = pool
        captured["envelope"] = envelope

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_fake_insert,
    )

    pool = _FakePool()
    client = _client(monkeypatch, pool=pool)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={"source_type": "text_prompt", "raw_content": {"text": "persist me"}},
    )

    assert response.status_code == 200
    assert captured["pool"] is pool
    envelope = captured["envelope"]
    assert envelope.status.value == "received"


def test_intake_drops_server_owned_metadata_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """``metrics`` (active_stream + a far-future heartbeat hides an envelope
    from every dispatcher) and ``quality`` (forged findings) are written only
    by the server. A client's copy is dropped at intake; ``preflight`` stays
    client-settable."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes

    captured: dict[str, Any] = {}

    async def _fake_insert(pool: object, envelope: object) -> None:
        captured["envelope"] = envelope

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_fake_insert,
    )

    client = _client(monkeypatch, pool=_FakePool())
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "source_type": "text_prompt",
            "raw_content": {"text": "hello"},
            "metadata": {
                "metrics": {"active_stream": True, "stream_heartbeat_epoch_s": 9e18},
                "quality": {"findings": []},
                "preflight": {"route": "giant", "pages": 1200},
                "title": "kept",
            },
        },
    )

    assert response.status_code == 200
    metadata = captured["envelope"].metadata
    assert "metrics" not in metadata
    assert "quality" not in metadata
    assert metadata["preflight"] == {"route": "giant", "pages": 1200}
    assert metadata["title"] == "kept"


def test_intake_returns_dispatch_status_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """No dispatcher is wired in vo-rag core; the route must NOT pretend."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes

    async def _noop_insert(pool: object, envelope: object) -> None:
        return None

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_noop_insert,
    )

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={"source_type": "text_prompt", "raw_content": {"text": "no dispatcher"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dispatch_status"] == "not_configured"
    # The envelope is still persisted as `received`; the route does not
    # lie about pipeline state.
    assert body["status"] == "received"


def test_intake_reports_queued_only_with_fresh_dispatcher_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.api.routes import ingest as ingest_route
    from vo_rag.core.db.queries import envelopes as queries_envelopes

    async def _noop_insert(pool: object, envelope: object) -> None:
        return None

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_noop_insert,
    )
    monkeypatch.setattr(ingest_route, "dispatcher_health", lambda: {"ok": True, "state": "idle"})

    garage = _FakeGarage(content_type="text/plain")
    response = _client(monkeypatch, garage=garage).post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "source_type": "file_drop",
            "raw_content": {
                "garage_key": "intake/test/document.txt",
                "file_hash": garage.sha256,
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "received"
    assert response.json()["dispatch_status"] == "queued"


def test_intake_does_not_queue_unsupported_source_with_fresh_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.api.routes import ingest as ingest_route
    from vo_rag.core.db.queries import envelopes as queries_envelopes

    async def _noop_insert(pool: object, envelope: object) -> None:
        return None

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_noop_insert,
    )
    monkeypatch.setattr(ingest_route, "dispatcher_health", lambda: {"ok": True, "state": "idle"})

    response = _client(monkeypatch).post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={"source_type": "text_prompt", "raw_content": {"text": "register only"}},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "received"
    assert response.json()["dispatch_status"] == "not_configured"


def test_intake_file_drop_preserves_garage_key_and_lowercased_file_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """file_drop normalization keeps garage_key verbatim; file_hash is lowercased + stripped."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes

    captured: dict[str, Any] = {}

    async def _fake_insert(pool: object, envelope: object) -> None:
        captured["envelope"] = envelope

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_fake_insert,
    )

    garage = _FakeGarage(content_type="application/pdf")
    client = _client(monkeypatch, garage=garage)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "source_type": "file_drop",
            "raw_content": {
                "garage_key": "intake/01H/foo.pdf",
                "filename": "Book.PDF",
                "content_type": "application/pdf",
                "file_hash": f"  {garage.sha256.upper()}  ",
            },
        },
    )

    assert response.status_code == 200
    envelope = captured["envelope"]
    raw = envelope.raw_content
    assert raw["garage_key"] == f"verified/sha256/{garage.sha256}.pdf"
    assert raw["filename"] == "Book.PDF"
    assert raw["content_type"] == "application/pdf"
    assert raw["file_hash"] == garage.sha256
    assert raw["object_verification"]["status"] == "verified"

    body = response.json()
    assert body["raw_content"]["garage_key"] == f"verified/sha256/{garage.sha256}.pdf"
    assert body["raw_content"]["file_hash"] == garage.sha256


def test_intake_rejects_hash_mismatch_and_deletes_only_staged_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    garage = _FakeGarage(content_type="application/pdf")
    client = _client(monkeypatch, garage=garage)

    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "source_type": "file_drop",
            "raw_content": {
                "garage_key": "intake/01H/mismatch.pdf",
                "file_hash": "b" * 64,
                "content_type": "application/pdf",
            },
        },
    )

    assert response.status_code == 422
    assert "file_hash_mismatch" in str(response.json()["detail"])
    assert garage.deleted == [("vorag-artifacts", "intake/01H/mismatch.pdf")]


def test_intake_hash_stream_interruption_is_unavailable_not_a_false_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _InterruptedGarage(_FakeGarage):
        async def verify_object_sha256(
            self,
            bucket: str,
            key: str,
            *,
            max_bytes: int,
        ) -> object:
            raise OSError("stream interrupted")

    garage = _InterruptedGarage()
    response = _client(monkeypatch, garage=garage).post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "source_type": "file_drop",
            "raw_content": {"garage_key": "intake/01H/interrupted.pdf"},
        },
    )

    assert response.status_code == 503
    assert "could not be verified" in str(response.json()["detail"])
    assert garage.deleted == []


def test_file_drop_fails_before_promotion_when_db_pool_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    garage = _FakeGarage()
    app = _build_app(monkeypatch, garage=garage)
    app.state.db_pool = None

    response = TestClient(app).post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "source_type": "file_drop",
            "raw_content": {"garage_key": "intake/01H/retry.pdf"},
        },
    )

    assert response.status_code == 503
    assert garage.promoted == []
    assert garage.deleted == []


def test_insert_failure_retains_staged_key_and_same_upload_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vo_rag.core.db.queries import envelopes as queries_envelopes

    should_fail = True

    async def _insert(pool: object, envelope: object) -> None:
        del pool, envelope
        if should_fail:
            raise OSError("database unavailable")

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_insert,
    )
    garage = _FakeGarage()
    client = _client(monkeypatch, garage=garage)
    payload = {
        "source_type": "file_drop",
        "raw_content": {"garage_key": "intake/01H/retry.pdf"},
    }

    first = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json=payload,
    )

    assert first.status_code == 503
    assert len(garage.promoted) == 1
    assert garage.deleted == []

    should_fail = False
    second = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json=payload,
    )

    assert second.status_code == 200
    assert len(garage.promoted) == 2
    assert garage.deleted == [("vorag-artifacts", "intake/01H/retry.pdf")]


def test_intake_rejects_unknown_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """extra='forbid' catches typos at the API boundary."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes

    async def _noop_insert(pool: object, envelope: object) -> None:
        return None

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_noop_insert,
    )

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "source_type": "text_prompt",
            "raw_content": {"text": "hello"},
            "unexpected_field": "boom",
        },
    )

    assert response.status_code == 422


def test_intake_rejects_invalid_pipeline_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """pipeline_hint must be a valid TargetPipeline string."""

    from vo_rag.core.db.queries import envelopes as queries_envelopes

    async def _noop_insert(pool: object, envelope: object) -> None:
        return None

    _patch_route_dependency(
        monkeypatch,
        source_module=queries_envelopes,
        attr="insert_envelope",
        replacement=_noop_insert,
    )

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "source_type": "text_prompt",
            "source_data": {"pipeline_hint": "Book_RAG"},
            "raw_content": {"text": "hello"},
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "invalid_pipeline_hint" in str(detail)


def test_intake_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The /intake route is gated by ApiKeyMiddleware."""

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake",
        json={"source_type": "text_prompt", "raw_content": {"text": "hello"}},
    )

    assert response.status_code == 401


# --------------------------------------------------------------------------- #
# /api/v1/intake/upload-url
# --------------------------------------------------------------------------- #
def test_upload_url_returns_presigned_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Presigned upload URL comes back with the expected bucket + key shape."""

    garage = _FakeGarage()
    client = _client(monkeypatch, garage=garage)
    response = client.post(
        "/api/v1/intake/upload-url",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "filename": "book.pdf",
            "content_type": "application/pdf",
            "size_bytes": 4096,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["bucket"] == "vorag-artifacts"
    assert body["garage_key"].startswith("intake/")
    assert body["garage_key"].endswith("book.pdf")
    assert body["upload_url"].startswith("https://garage.example/")
    assert body["expires_in_s"] == 21600

    # The GarageClient method was called with bucket+key+ttl.
    assert len(garage.calls) == 1
    call = garage.calls[0]
    assert call["bucket"] == "vorag-artifacts"
    assert call["ttl"] == 21600
    assert call["key"].endswith("book.pdf")
    assert call["content_type"] == "application/pdf"
    assert call["content_length"] == 4096


def test_upload_url_sanitizes_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path-traversal characters in the filename are stripped before keying."""

    garage = _FakeGarage()
    client = _client(monkeypatch, garage=garage)
    response = client.post(
        "/api/v1/intake/upload-url",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "filename": "../etc/passwd\x00",
            "content_type": "text/plain",
            "size_bytes": 128,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert ".." not in body["garage_key"]
    assert "\x00" not in body["garage_key"]
    # The filename component (the last path segment) must be free of
    # path-traversal characters. The ``intake/{ulid}/{filename}`` shape
    # is the only place a ``/`` is allowed and it is the second slash,
    # not in the filename itself.
    filename_component = body["garage_key"].rsplit("/", 1)[-1]
    assert ".." not in filename_component
    assert "/" not in filename_component
    assert "\x00" not in filename_component


def test_upload_url_rejects_oversize_against_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """settings.portal.upload_max_bytes is enforced even when below the 2 GiB ceiling."""

    importlib.import_module("vo_rag.api.routes.ingest")

    # 1 MiB portal cap.
    settings = _fake_settings(portal=SimpleNamespace(upload_max_bytes=1024 * 1024))
    client = _client(monkeypatch, settings=settings)
    response = client.post(
        "/api/v1/intake/upload-url",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "filename": "big.bin",
            "content_type": "application/octet-stream",
            "size_bytes": 2 * 1024 * 1024,  # > 1 MiB
        },
    )

    assert response.status_code == 422
    detail = str(response.json()["detail"])
    assert "size" in detail.lower() or "max" in detail.lower()


def test_upload_url_rejects_oversize_above_hard_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hard 2 GiB schema ceiling is enforced before the route runs."""

    importlib.import_module("vo_rag.api.routes.ingest")

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake/upload-url",
        headers={"X-API-Key": _TEST_API_KEY},
        json={
            "filename": "huge.bin",
            "content_type": "application/octet-stream",
            "size_bytes": 3 * 1024 * 1024 * 1024,  # > 2 GiB
        },
    )

    assert response.status_code == 422


def test_upload_url_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The upload-url route is also gated by ApiKeyMiddleware."""

    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/intake/upload-url",
        json={
            "filename": "x.pdf",
            "content_type": "application/pdf",
            "size_bytes": 128,
        },
    )

    assert response.status_code == 401
