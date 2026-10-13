"""Fail-closed dispatcher heartbeat tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from vo_rag.pipelines.book_rag.dispatcher_health import (
    dispatcher_health,
    write_dispatcher_heartbeat,
)


def test_fresh_heartbeat_is_ready_and_content_free(tmp_path) -> None:
    path = tmp_path / "dispatcher.json"
    now = datetime(2026, 7, 15, tzinfo=UTC)
    write_dispatcher_heartbeat("idle", path=path, observed_at=now)

    assert dispatcher_health(path=path, now=now, stale_after_s=120) == {
        "ok": True,
        "state": "idle",
        "age_seconds": 0.0,
    }
    text = path.read_text(encoding="utf-8")
    assert "envelope" not in text
    assert "source" not in text


def test_missing_stale_and_symlink_heartbeat_fail_closed(tmp_path) -> None:
    path = tmp_path / "dispatcher.json"
    now = datetime(2026, 7, 15, tzinfo=UTC)
    assert dispatcher_health(path=path, now=now)["error"] == "not_configured"

    write_dispatcher_heartbeat("processing", path=path, observed_at=now - timedelta(minutes=5))
    stale = dispatcher_health(path=path, now=now, stale_after_s=120)
    assert stale["ok"] is False
    assert stale["error"] == "stale_heartbeat"

    link = tmp_path / "link.json"
    link.symlink_to(path)
    assert dispatcher_health(path=link, now=now)["error"] == "unsafe_heartbeat_path"
