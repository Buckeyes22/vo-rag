"""Content-free heartbeat contract for the book-rag dispatcher."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

DEFAULT_HEARTBEAT_PATH: Final = Path("/run/vo-rag/dispatcher-heartbeat.json")
DEFAULT_STALE_AFTER_S: Final = 130.0
MAX_HEARTBEAT_BYTES: Final = 4096
DispatcherState = Literal["idle", "processing", "stopping"]
_STATES: Final = frozenset({"idle", "processing", "stopping"})


def configured_heartbeat_path() -> Path:
    """Return the operator-configured heartbeat path without resolving symlinks."""
    raw = os.environ.get("VORAG_BOOK_RAG_DISPATCHER_HEARTBEAT_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_HEARTBEAT_PATH


def configured_stale_after_s() -> float:
    """Return the positive heartbeat age ceiling, falling back safely."""
    raw = os.environ.get("VORAG_BOOK_RAG_DISPATCHER_HEARTBEAT_STALE_S", "").strip()
    if not raw:
        return DEFAULT_STALE_AFTER_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_STALE_AFTER_S
    return value if value > 0 else DEFAULT_STALE_AFTER_S


def write_dispatcher_heartbeat(
    state: DispatcherState,
    *,
    path: Path | None = None,
    observed_at: datetime | None = None,
) -> None:
    """Atomically publish one bounded heartbeat without envelope content or ids."""
    if state not in _STATES:
        raise ValueError("dispatcher heartbeat state is invalid")
    target = path or configured_heartbeat_path()
    if target.is_symlink():
        raise RuntimeError("dispatcher heartbeat path must not be a symlink")
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    timestamp = observed_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("dispatcher heartbeat timestamp must be timezone-aware")
    payload = {
        "schema_version": 1,
        "observed_at": timestamp.astimezone(UTC).isoformat(),
        "state": state,
    }
    encoded = (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode()
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(encoded)
        temporary.chmod(0o640)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def dispatcher_health(
    *,
    path: Path | None = None,
    now: datetime | None = None,
    stale_after_s: float | None = None,
) -> dict[str, object]:
    """Read a heartbeat into a bounded fail-closed operator status."""
    target = path or configured_heartbeat_path()
    current = now or datetime.now(UTC)
    ceiling = stale_after_s if stale_after_s is not None else configured_stale_after_s()
    if current.tzinfo is None or ceiling <= 0:
        raise ValueError("dispatcher health requires an aware clock and positive age ceiling")
    try:
        if target.is_symlink():
            return {"ok": False, "error": "unsafe_heartbeat_path"}
        stat = target.stat()
        if not target.is_file() or stat.st_size > MAX_HEARTBEAT_BYTES:
            return {"ok": False, "error": "invalid_heartbeat"}
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"ok": False, "error": "not_configured"}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"ok": False, "error": "invalid_heartbeat"}
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "observed_at", "state"}:
        return {"ok": False, "error": "invalid_heartbeat"}
    if raw.get("schema_version") != 1 or raw.get("state") not in _STATES:
        return {"ok": False, "error": "invalid_heartbeat"}
    try:
        observed = datetime.fromisoformat(str(raw["observed_at"]))
    except ValueError:
        return {"ok": False, "error": "invalid_heartbeat"}
    if observed.tzinfo is None:
        return {"ok": False, "error": "invalid_heartbeat"}
    age = (current.astimezone(UTC) - observed.astimezone(UTC)).total_seconds()
    if age < 0 or age > ceiling:
        return {
            "ok": False,
            "error": "stale_heartbeat",
            "state": raw["state"],
            "age_seconds": round(max(age, 0.0), 3),
        }
    return {
        "ok": raw["state"] != "stopping",
        "state": raw["state"],
        "age_seconds": round(age, 3),
    }


__all__ = [
    "DEFAULT_HEARTBEAT_PATH",
    "DEFAULT_STALE_AFTER_S",
    "DispatcherState",
    "configured_heartbeat_path",
    "configured_stale_after_s",
    "dispatcher_health",
    "write_dispatcher_heartbeat",
]
