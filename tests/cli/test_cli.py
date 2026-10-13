from __future__ import annotations

import json

from click.testing import CliRunner

from vo_rag.cli.main import cli


def test_help_lists_commands() -> None:
    r = CliRunner().invoke(cli, ["--help"])
    assert r.exit_code == 0
    expected = (
        "pipeline",
        "query",
        "answer",
        "ingest",
        "eval-gate",
        "admin",
        "health",
        "status",
        "config",
        "version",
    )
    for cmd in expected:
        assert cmd in r.output


def test_version_command_prints_version() -> None:
    result = CliRunner().invoke(cli, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_pipelines_command_lists_roster_with_status() -> None:
    result = CliRunner().invoke(cli, ["pipeline", "list"])
    assert result.exit_code == 0
    assert "book_rag" in result.output
    assert "active" in result.output  # book_rag shipped to production (VOR-7)


def test_roster_alias_lists_pipeline_roster() -> None:
    result = CliRunner().invoke(cli, ["roster"])
    assert result.exit_code == 0
    assert "book_rag" in result.output
    assert "active" in result.output


def test_pipeline_status_json() -> None:
    result = CliRunner().invoke(cli, ["pipeline", "status", "--json"])
    assert result.exit_code == 0
    assert "planned" in result.output
    assert "next" in result.output
    assert "active" in result.output


def test_config_show_masks_environment_key(monkeypatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "secret-token-xyz")
    r = CliRunner().invoke(cli, ["config", "show"])
    assert r.exit_code == 0
    assert "secret-token-xyz" not in r.output


def test_config_show_recursively_masks_file_secrets(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """api_url: http://127.0.0.1:8000
api_key: file-secret-value
provider:
  access_token: nested-token-value
  endpoint: https://example.invalid
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("VORAG_API_KEY", raising=False)

    result = CliRunner().invoke(cli, ["--config", str(config_path), "config", "show"])

    assert result.exit_code == 0, result.output
    assert "file-secret-value" not in result.output
    assert "nested-token-value" not in result.output
    payload = json.loads(result.output)
    assert payload["api_key"].endswith("alue")
    assert payload["file"]["api_key"] == "***REDACTED***"
    assert payload["file"]["provider"]["access_token"] == "***REDACTED***"
    assert payload["file"]["provider"]["endpoint"] == "https://example.invalid"
    assert result.stderr == ""


def test_config_show_environment_precedence_redacts_both_sources(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "api_url: http://127.0.0.1:8000\napi_key: lower-priority-file-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VORAG_API_KEY", "higher-priority-environment-secret")

    result = CliRunner().invoke(cli, ["--config", str(config_path), "config", "show"])

    assert result.exit_code == 0, result.output
    assert "lower-priority-file-secret" not in result.output
    assert "higher-priority-environment-secret" not in result.output
    payload = json.loads(result.output)
    assert payload["api_key"].endswith("cret")
    assert payload["file"]["api_key"] == "***REDACTED***"


def test_config_show_absent_key_is_safe(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("api_url: http://127.0.0.1:8000\n", encoding="utf-8")
    monkeypatch.delenv("VORAG_API_KEY", raising=False)

    result = CliRunner().invoke(cli, ["--config", str(config_path), "config", "show"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["api_key"] == "(not set)"
    assert "api_key" not in payload["file"]


def test_ingest_file_help() -> None:
    r = CliRunner().invoke(cli, ["ingest", "file", "--help"])
    assert r.exit_code == 0
    for flag in ("--title", "--author", "--tags", "--sovereignty"):
        assert flag in r.output


def test_query_flags() -> None:
    r = CliRunner().invoke(cli, ["query", "--help"])
    assert "--collection" in r.output
    assert "--top-k" in r.output
    assert "--format" in r.output
    assert "--mode" in r.output


def test_answer_flags() -> None:
    r = CliRunner().invoke(cli, ["answer", "--help"])
    assert "--profile" in r.output
    assert "--context-scope" in r.output
    assert "--include-retrieval" in r.output
    assert "--mode" in r.output
    assert "--collection" in r.output
    assert "--top-k" in r.output
    assert "--rerank" in r.output
    assert "--visual-mode" in r.output
    assert "--format" in r.output


def test_admin_status_help() -> None:
    r = CliRunner().invoke(cli, ["admin", "status", "--help"])
    assert r.exit_code == 0
    assert "admin" in r.output


def test_health_help() -> None:
    r = CliRunner().invoke(cli, ["health", "--help"])
    assert r.exit_code == 0
    assert "health" in r.output


def test_status_watch_help() -> None:
    r = CliRunner().invoke(cli, ["status", "--help"])
    assert r.exit_code == 0
    assert "--watch" in r.output


def test_eval_gate_help() -> None:
    r = CliRunner().invoke(cli, ["eval-gate", "--help"])
    assert r.exit_code == 0
    assert "eval-gate" in r.output
