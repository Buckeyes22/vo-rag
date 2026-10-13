from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import respx
from click.testing import CliRunner

from vo_rag.cli.main import cli


def test_ingest_text_posts_to_api(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with respx.mock:
        route = respx.post("http://localhost:8000/api/v1/intake").mock(
            return_value=httpx.Response(200, json={"envelope_id": "env-123"})
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "text", "hello world", "--title", "Test"])
        assert result.exit_code == 0
        assert route.called
        sent = route.calls[0].request.content.decode()
        assert "text_prompt" in sent
        assert "hello world" in sent
        assert "Test" in sent
        assert "env-123" in result.output


def test_ingest_url_rejects_unsafe_url(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "url", "http://192.168.1.1/secret"])
    assert result.exit_code != 0
    assert "URL validation failed" in result.output


def test_ingest_url_posts_to_api(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with respx.mock:
        route = respx.post("http://localhost:8000/api/v1/intake").mock(
            return_value=httpx.Response(200, json={"envelope_id": "env-456"})
        )
        runner = CliRunner()
        result = runner.invoke(
            cli, ["ingest", "url", "https://example.com/article", "--title", "Example"]
        )
        assert result.exit_code == 0
        assert route.called
        sent = route.calls[0].request.content.decode()
        assert "url" in sent
        assert "https://example.com/article" in sent
        assert "env-456" in result.output


def test_ingest_idea_posts_to_api(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with respx.mock:
        route = respx.post("http://localhost:8000/api/v1/intake").mock(
            return_value=httpx.Response(200, json={"envelope_id": "env-789"})
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "idea", "my great idea", "--tags", "feature,ui"])
        assert result.exit_code == 0
        assert route.called
        sent = route.calls[0].request.content.decode()
        assert "text_prompt" in sent
        assert "my great idea" in sent
        assert "idea" in sent
        assert "feature" in sent
        assert "ui" in sent
        assert "env-789" in result.output


def test_ingest_file_drop_posts_to_api(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with respx.mock:
        route = respx.post("http://localhost:8000/api/v1/intake").mock(
            return_value=httpx.Response(200, json={"envelope_id": "env-abc"})
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "file-drop", "gk-123", "--title", "Dropped"])
        assert result.exit_code == 0
        assert route.called
        sent = route.calls[0].request.content.decode()
        assert "file_drop" in sent
        assert "gk-123" in sent
        assert "env-abc" in result.output


def test_ingest_file_uploads_and_posts(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("file content here")
        tmp_path = Path(f.name)

    try:
        with respx.mock:
            upload_route = respx.post("http://localhost:8000/api/v1/intake/upload-url").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "upload_url": "https://storage.example.com/upload",
                        "garage_key": "gk-file-1",
                        "bucket": "test-bucket",
                        "expires_in_s": 3600,
                    },
                )
            )
            put_route = respx.put("https://storage.example.com/upload").mock(
                return_value=httpx.Response(200)
            )
            intake_route = respx.post("http://localhost:8000/api/v1/intake").mock(
                return_value=httpx.Response(200, json={"envelope_id": "env-file"})
            )
            runner = CliRunner()
            result = runner.invoke(cli, ["ingest", "file", str(tmp_path), "--title", "My File"])
            assert result.exit_code == 0
            assert upload_route.called
            assert put_route.called
            assert intake_route.called
            sent = intake_route.calls[0].request.content.decode()
            assert "file_drop" in sent
            assert "gk-file-1" in sent
            payload = json.loads(sent)
            assert (
                payload["raw_content"]["file_hash"]
                == hashlib.sha256(b"file content here").hexdigest()
            )
            assert payload["raw_content"]["size_bytes"] == len(b"file content here")
            assert put_route.calls[0].request.headers["Content-Length"] == str(
                len(b"file content here")
            )
            assert "env-file" in result.output
    finally:
        tmp_path.unlink(missing_ok=True)


def test_ingest_run_loads_envelope_and_calls_runner(monkeypatch) -> None:
    import vo_rag.core.config as config_mod
    import vo_rag.core.db.postgres as postgres_mod
    import vo_rag.core.db.queries.envelopes as envelopes_mod
    import vo_rag.pipelines.book_rag.runner as runner_mod
    from vo_rag.core.models.envelope import EnvelopeStatus, IntakeEnvelope, IntakeSource
    from vo_rag.pipelines.book_rag.workflow import PipelineOutput

    envelope = IntakeEnvelope(
        id="01KT6XE8M1MW42M0Y0JJKRZSVP",  # pragma: allowlist secret
        source_type=IntakeSource.FILE_DROP,
        raw_content={"garage_key": "drops/book.pdf", "file_hash": "a" * 64},
        status=EnvelopeStatus.RECEIVED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    settings = SimpleNamespace(
        database=SimpleNamespace(url="postgres://db", pool_min=1, pool_max=2)
    )
    pool = object()
    calls: dict[str, object] = {}

    async def _fake_get_pool(dsn: str, *, min_size: int, max_size: int) -> object:
        calls["pool_args"] = (dsn, min_size, max_size)
        return pool

    async def _fake_close_pool() -> None:
        calls["closed"] = True

    async def _fake_get_envelope(pool_arg: object, envelope_id: str) -> IntakeEnvelope:
        calls["lookup"] = (pool_arg, envelope_id)
        return envelope

    async def _fake_run_book_rag_ingest(
        envelope_arg: IntakeEnvelope, *, settings: object, pg_pool: object | None = None
    ) -> PipelineOutput:
        calls["runner"] = (envelope_arg, settings, pg_pool)
        return PipelineOutput(
            source_id="source-1",
            status=EnvelopeStatus.COMPLETED,
            qdrant_point_ids=["p1"],
            raw_point_ids=["p1"],
            sterile_point_ids=[],
            smoke_decision=None,
            chunk_count=1,
        )

    # Base install (no `llamaindex` extra): `vo-rag ingest run` must fall back to
    # the non-streaming core runner. Force the fallback deterministically so this
    # light-job test asserts the base-install path regardless of the worktree's
    # current extra state. The streamed path is covered in the gated adapter
    # suite (tests/integration/llamaindex/test_ingest_streaming.py).
    import vo_rag.cli.ingest as cli_ingest_mod

    monkeypatch.setattr(cli_ingest_mod, "_llamaindex_extra_present", lambda: False)
    monkeypatch.setattr(config_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(postgres_mod, "get_pool", _fake_get_pool)
    monkeypatch.setattr(postgres_mod, "close_pool", _fake_close_pool)
    monkeypatch.setattr(envelopes_mod, "get_envelope", _fake_get_envelope)
    monkeypatch.setattr(runner_mod, "run_book_rag_ingest", _fake_run_book_rag_ingest)

    result = CliRunner().invoke(
        cli,
        ["ingest", "run", "--envelope-id", "01KT6XE8M1MW42M0Y0JJKRZSVP"],
    )

    assert result.exit_code == 0
    assert "source-1" in result.output
    assert "p1" in result.output
    assert calls["pool_args"] == ("postgres://db", 1, 2)
    assert calls["lookup"] == (pool, "01KT6XE8M1MW42M0Y0JJKRZSVP")
    assert calls["runner"] == (envelope, settings, pool)
    assert calls["closed"] is True


def test_ingest_run_chunking_strategy_overrides_effective_settings(monkeypatch) -> None:
    import vo_rag.cli.ingest as cli_ingest_mod
    import vo_rag.core.config as config_mod
    import vo_rag.core.db.postgres as postgres_mod
    import vo_rag.core.db.queries.envelopes as envelopes_mod
    import vo_rag.pipelines.book_rag.runner as runner_mod
    from vo_rag.core.config.settings import ChunkingSettings
    from vo_rag.core.models.envelope import EnvelopeStatus, IntakeEnvelope, IntakeSource
    from vo_rag.pipelines.book_rag.workflow import PipelineOutput

    envelope = IntakeEnvelope(
        id="01KT6XE8M1MW42M0Y0JJKRZSVP",  # pragma: allowlist secret
        source_type=IntakeSource.FILE_DROP,
        raw_content={"garage_key": "drops/book.pdf", "file_hash": "a" * 64},
        status=EnvelopeStatus.RECEIVED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    settings = SimpleNamespace(
        database=SimpleNamespace(url="postgres://db", pool_min=1, pool_max=2),
        chunking=ChunkingSettings(strategy="hierarchical"),
    )
    pool = object()
    calls: dict[str, object] = {}

    async def _fake_get_pool(dsn: str, *, min_size: int, max_size: int) -> object:
        return pool

    async def _fake_close_pool() -> None:
        calls["closed"] = True

    async def _fake_get_envelope(pool_arg: object, envelope_id: str) -> IntakeEnvelope:
        return envelope

    async def _fake_run_book_rag_ingest(
        envelope_arg: IntakeEnvelope, *, settings: Any, pg_pool: object | None = None
    ) -> PipelineOutput:
        calls["runner_strategy"] = settings.chunking.strategy
        return PipelineOutput(
            source_id="source-1",
            status=EnvelopeStatus.COMPLETED,
            qdrant_point_ids=["p1"],
            raw_point_ids=["p1"],
            sterile_point_ids=[],
            smoke_decision=None,
            chunk_count=1,
        )

    monkeypatch.setattr(cli_ingest_mod, "_llamaindex_extra_present", lambda: False)
    monkeypatch.setattr(config_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(postgres_mod, "get_pool", _fake_get_pool)
    monkeypatch.setattr(postgres_mod, "close_pool", _fake_close_pool)
    monkeypatch.setattr(envelopes_mod, "get_envelope", _fake_get_envelope)
    monkeypatch.setattr(runner_mod, "run_book_rag_ingest", _fake_run_book_rag_ingest)

    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            "run",
            "--envelope-id",
            "01KT6XE8M1MW42M0Y0JJKRZSVP",
            "--chunking-strategy",
            "semantic",
        ],
    )

    assert result.exit_code == 0
    assert calls["runner_strategy"] == "semantic"
    assert settings.chunking.strategy == "hierarchical"
    assert calls["closed"] is True


def test_ingest_run_persists_failed_on_workflow_failure(monkeypatch) -> None:
    """A workflow failure must persist envelope FAILED, not leave it `received`.

    A parse quality-gate failure on the CLI `ingest run` direct path must not
    leave the envelope at `received`: terminal-status persistence has to cover
    this surface as well as the SSE route and MCP tool.
    """
    import vo_rag.cli.ingest as cli_ingest_mod
    import vo_rag.core.config as config_mod
    import vo_rag.core.db.postgres as postgres_mod
    import vo_rag.core.db.queries.envelopes as envelopes_mod
    import vo_rag.pipelines.book_rag.runner as runner_mod
    from vo_rag.core.models.envelope import EnvelopeStatus, IntakeEnvelope, IntakeSource

    envelope = IntakeEnvelope(
        id="01KT6XE8M1MW42M0Y0JJKRZSVP",  # pragma: allowlist secret
        source_type=IntakeSource.FILE_DROP,
        raw_content={"garage_key": "drops/book.pdf", "file_hash": "a" * 64},
        status=EnvelopeStatus.RECEIVED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    settings = SimpleNamespace(
        database=SimpleNamespace(url="postgres://db", pool_min=1, pool_max=2)
    )
    pool = object()
    failed: list[tuple[object, str, str]] = []

    async def _fake_get_pool(dsn: str, *, min_size: int, max_size: int) -> object:
        return pool

    async def _fake_close_pool() -> None:
        return None

    async def _fake_get_envelope(pool_arg: object, envelope_id: str) -> IntakeEnvelope:
        return envelope

    async def _raise_ingest(
        envelope_arg: IntakeEnvelope, *, settings: object, pg_pool: object | None = None
    ) -> object:
        raise RuntimeError("Error in step 'parse': quality gate failed")

    async def _fake_fail_envelope(pool_arg: object, envelope_id: str, error: str) -> None:
        failed.append((pool_arg, envelope_id, error))

    monkeypatch.setattr(cli_ingest_mod, "_llamaindex_extra_present", lambda: False)
    monkeypatch.setattr(config_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(postgres_mod, "get_pool", _fake_get_pool)
    monkeypatch.setattr(postgres_mod, "close_pool", _fake_close_pool)
    monkeypatch.setattr(envelopes_mod, "get_envelope", _fake_get_envelope)
    monkeypatch.setattr(envelopes_mod, "fail_envelope", _fake_fail_envelope)
    monkeypatch.setattr(runner_mod, "run_book_rag_ingest", _raise_ingest)

    result = CliRunner().invoke(
        cli,
        ["ingest", "run", "--envelope-id", "01KT6XE8M1MW42M0Y0JJKRZSVP"],
    )

    assert result.exit_code != 0
    assert failed == [
        (pool, "01KT6XE8M1MW42M0Y0JJKRZSVP", "Error in step 'parse': quality gate failed")
    ]


def test_admin_status_posts_to_api(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")
    with respx.mock:
        route = respx.get("http://localhost:8000/api/v1/admin/status").mock(
            return_value=httpx.Response(200, json={"ok": True, "services": {}})
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["admin", "status"])
        assert result.exit_code == 0
        assert route.called
        assert "ok" in result.output


def test_health_no_auth(monkeypatch) -> None:
    monkeypatch.delenv("VORAG_API_KEY", raising=False)
    with respx.mock:
        route = respx.get("http://localhost:8000/health").mock(
            return_value=httpx.Response(200, json={"status": "ok", "service": "vo-rag"})
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["--url", "http://localhost:8000", "health"])
        assert result.exit_code == 0
        assert route.called
        assert "status" in result.output


def test_health_fails_on_not_ok(monkeypatch) -> None:
    monkeypatch.delenv("VORAG_API_KEY", raising=False)
    with respx.mock:
        respx.get("http://localhost:8000/health").mock(
            return_value=httpx.Response(200, json={"status": "degraded", "service": "vo-rag"})
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["--url", "http://localhost:8000", "health"])
        assert result.exit_code != 0
        assert "health checks reported failures" in result.output


def test_status_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("VORAG_API_KEY", raising=False)
    result = CliRunner().invoke(cli, ["status"])

    assert result.exit_code != 0
    assert "VORAG_API_KEY is required" in result.output


def test_eval_gate_pass(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")

    class MockResult:
        passed = True
        checked = ["book_rag"]
        skipped: list[str] = []
        failed: list[str] = []
        blocked_preconditions: dict[str, str] = {}
        missing_fixtures: list[str] = []
        missing_evaluators: list[str] = []
        regressions: list[str] = []

    monkeypatch.setattr("vo_rag.cli.eval.run_registered_eval_gate", lambda _registry: MockResult())
    runner = CliRunner()
    result = runner.invoke(cli, ["eval-gate"])
    assert result.exit_code == 0
    assert "PASS" in result.output
    assert "book_rag" in result.output


def test_eval_gate_fail(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")

    class MockResult:
        passed = False
        checked: list[str] = []
        skipped: list[str] = []
        failed = ["book_rag"]
        blocked_preconditions: dict[str, str] = {}
        missing_fixtures = ["book_rag"]
        missing_evaluators: list[str] = []
        regressions: list[str] = []

    monkeypatch.setattr("vo_rag.cli.eval.run_registered_eval_gate", lambda _registry: MockResult())
    runner = CliRunner()
    result = runner.invoke(cli, ["eval-gate"])
    assert result.exit_code != 0
    assert "FAIL" in result.output
    assert "book_rag" in result.output
    assert "missing fixture set" in result.output


def test_eval_gate_blocked(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "test-key")

    class MockResult:
        passed = False
        checked: list[str] = []
        skipped: list[str] = []
        failed = ["book_rag"]
        blocked_preconditions = {"book_rag": "blocked_missing_corpus_snapshot"}
        missing_fixtures: list[str] = []
        missing_evaluators: list[str] = []
        regressions: list[str] = []

    monkeypatch.setattr("vo_rag.cli.eval.run_registered_eval_gate", lambda _registry: MockResult())
    runner = CliRunner()
    result = runner.invoke(cli, ["eval-gate"])
    assert result.exit_code != 0
    assert "eval gate BLOCKED" in result.output
    assert "blocked_missing_corpus_snapshot" in result.output


def test_upload_progress_never_writes_to_stdout(tmp_path: Path, capfd: Any) -> None:
    """VOR-1430: stdout carries a single JSON envelope (books backlog run,
    download --ingest); the upload progress bar belongs on stderr."""
    from vo_rag.cli.ingest import _upload_file_sync

    book = tmp_path / "book.pdf"
    book.write_bytes(b"%PDF-1.4 " + b"x" * 4096)
    with respx.mock:
        respx.put("http://garage.test/upload").mock(return_value=httpx.Response(200))
        digest = _upload_file_sync(book, "http://garage.test/upload", "application/pdf")

    out, _err = capfd.readouterr()
    assert out == ""
    assert digest == hashlib.sha256(book.read_bytes()).hexdigest()
