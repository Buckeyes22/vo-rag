"""Unit contracts for migration discovery and checksums."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vo_rag.core.db.postgres import _migration_files


def test_migration_files_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _migration_files(tmp_path / "missing")


def test_migration_files_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contains no .sql files"):
        _migration_files(tmp_path)


def test_migration_files_are_sorted_and_sha256_hashed(tmp_path: Path) -> None:
    second = tmp_path / "002_second.sql"
    first = tmp_path / "001_first.sql"
    second.write_text("SELECT 2;\n", encoding="utf-8")
    first.write_text("SELECT 1;\n", encoding="utf-8")

    migrations = _migration_files(tmp_path)

    assert [migration.version for migration in migrations] == ["001_first", "002_second"]
    assert migrations[0].checksum == hashlib.sha256(first.read_bytes()).hexdigest()
