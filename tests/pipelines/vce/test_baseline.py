"""Tests for the baseline rows loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vo_rag.pipelines.vce.baseline import BaselineIndex, load_baseline_rows


@pytest.fixture
def baseline_path(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "AAAAAAAAAAAAAAAAAAAAAAAA",
                    "video_id": "yt:abc",
                    "start_ts": "10.0",
                    "end_ts": "20.0",
                    "observation": "test",
                    "evidence": "explicit_numeric",
                    "anchors": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


class TestLoadBaselineRows:
    def test_loads_index(self, baseline_path: Path) -> None:
        index = load_baseline_rows(baseline_path)
        assert isinstance(index, BaselineIndex)
        assert len(index.rows) == 1

    def test_normalises_string_timestamps_to_floats(self, baseline_path: Path) -> None:
        index = load_baseline_rows(baseline_path)
        for row in index.rows:
            assert isinstance(row["start_ts"], float)
            assert isinstance(row["end_ts"], float)

    def test_indexes_by_interval(self, baseline_path: Path) -> None:
        index = load_baseline_rows(baseline_path)
        row = index.by_interval.get(("yt:abc", 10.0, 20.0))
        assert row is not None

    def test_indexes_by_legacy_id(self, baseline_path: Path) -> None:
        index = load_baseline_rows(baseline_path)
        row = index.by_id.get("AAAAAAAAAAAAAAAAAAAAAAAA")
        assert row is not None

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_baseline_rows(tmp_path / "missing.json")

    def test_invalid_root_type_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a JSON array"):
            load_baseline_rows(path)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"id": "X", "video_id": "yt:abc"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="missing/invalid"):
            load_baseline_rows(path)
