"""Unit contracts for the read-only migration deployment preflight."""

from __future__ import annotations

import json
from typing import Any

from scripts import check_migrations as preflight


def test_preflight_returns_zero_for_clean_status(monkeypatch, capsys) -> None:
    async def fake_check() -> dict[str, Any]:
        return {
            "ok": True,
            "pending": [],
            "unknown": [],
            "checksum_drifted": [],
            "checksum_untracked": ["001_initial"],
        }

    monkeypatch.setattr(preflight, "check_migrations", fake_check)

    assert preflight.main([]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_preflight_blocks_rollout_for_drift(monkeypatch, capsys) -> None:
    async def fake_check() -> dict[str, Any]:
        return {
            "ok": False,
            "pending": ["032_cloud_cost_static_workloads"],
            "unknown": ["032_code_rag_initial"],
            "checksum_drifted": ["001_initial"],
            "checksum_untracked": [],
        }

    monkeypatch.setattr(preflight, "check_migrations", fake_check)

    assert preflight.main([]) == 1
    status = json.loads(capsys.readouterr().out)
    assert status["pending"] == ["032_cloud_cost_static_workloads"]
    assert status["unknown"] == ["032_code_rag_initial"]
    assert status["checksum_drifted"] == ["001_initial"]


def test_preflight_redacts_connection_error_details(monkeypatch, capsys) -> None:
    async def fake_check() -> dict[str, Any]:
        raise RuntimeError(
            "postgresql://user:secret@example.invalid/database"  # pragma: allowlist secret
        )

    monkeypatch.setattr(preflight, "check_migrations", fake_check)

    assert preflight.main([]) == 2
    output = capsys.readouterr().out
    assert "secret" not in output
    assert json.loads(output) == {"error_type": "RuntimeError", "ok": False}


def test_preflight_atomically_replaces_requested_output(tmp_path, monkeypatch, capsys) -> None:
    async def fake_check() -> dict[str, Any]:
        return {
            "ok": False,
            "pending": [],
            "unknown": [],
            "checksum_drifted": ["001_initial"],
            "checksum_untracked": [],
        }

    monkeypatch.setattr(preflight, "check_migrations", fake_check)
    output_path = tmp_path / "migration-status.json"
    output_path.write_text("stale", encoding="utf-8")

    assert preflight.main(["--output", str(output_path)]) == 1
    assert capsys.readouterr().out == ""
    assert json.loads(output_path.read_text(encoding="utf-8"))["checksum_drifted"] == [
        "001_initial"
    ]
    assert list(tmp_path.glob(".migration-status.json.*.tmp")) == []
