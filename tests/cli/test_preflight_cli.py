"""`vo-rag preflight` — top-level command contract."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from vo_rag.cli.main import cli


def test_preflight_empty_directory_emits_ok_envelope(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["preflight", str(tmp_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["schema_version"] == "2"
    assert payload["command"] == "preflight"
    assert payload["results"] == []


def test_preflight_without_catalog_skips_duplicate_check(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-catalog.json"

    result = CliRunner().invoke(cli, ["preflight", str(tmp_path), "--catalog", str(missing)])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["duplicate_groups"] in ([], {})


def test_books_group_no_longer_exists() -> None:
    result = CliRunner().invoke(cli, ["books", "--help"])

    assert result.exit_code != 0
    assert "No such command" in result.output
