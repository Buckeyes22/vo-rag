"""Shared fixtures for the VCE pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def baseline_rows_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "baseline_rows.json"


@pytest.fixture
def transcript_fixture_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "transcripts"


def make_row_dict(
    *,
    row_id: str,
    video_id: str = "yt:abc",
    start_ts: float = 10.0,
    end_ts: float = 20.0,
    observation: str = "driver carried too much speed into the corner",
    evidence: str = "explicit_numeric",
    excerpt: str = "see how late he turns in here",
    source: str = "whisperx",
) -> dict[str, object]:
    """Build a raw row dict for tests that want to bypass the Pydantic layer."""
    return {
        "id": row_id,
        "video_id": video_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "observation": observation,
        "evidence": evidence,
        "transcript_excerpt": excerpt,
        "transcript_source": source,
        "anchors": [],
    }


@pytest.fixture
def fake_garage():
    from tests.pipelines.vce.fakes import FakeGarage

    return FakeGarage()


__all__ = ["FIXTURES_DIR", "make_row_dict"]
