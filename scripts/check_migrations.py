#!/usr/bin/env python3
"""Read migration state and fail closed on deployment-blocking drift."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from vo_rag.core.config import get_settings
from vo_rag.core.db.postgres import close_pool, get_pool, migration_status


async def check_migrations() -> dict[str, Any]:
    """Return read-only migration status for the configured database."""
    settings = get_settings()
    pool = await get_pool(
        settings.database.url,
        min_size=1,
        max_size=1,
    )
    try:
        return await migration_status(pool, Path(settings.database.migrations_dir))
    finally:
        await close_pool()


def _write_status(status: dict[str, Any], output: Path | None) -> None:
    """Write status to stdout or atomically replace the requested file."""
    rendered = json.dumps(status, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
        return

    parent = output.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {parent}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main(argv: list[str] | None = None) -> int:
    """Print sanitized JSON status and return nonzero when rollout must stop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="atomically write sanitized JSON to PATH instead of stdout",
    )
    args = parser.parse_args(argv)

    try:
        status = asyncio.run(check_migrations())
    except Exception as exc:  # noqa: BLE001
        status = {"ok": False, "error_type": type(exc).__name__}
        _write_status(status, args.output)
        return 2

    _write_status(status, args.output)
    return 0 if status.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
