"""Tests for the export writer and the round-trip into the contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.pipelines.vce._contract import VceRowSchema
from tests.pipelines.vce.conftest import make_row_dict
from vo_rag.pipelines.vce.export import (
    VCE_TRANSCRIPT_SUFFIX,
    VideoTranscriptExport,
    load_export_rows,
    write_export,
)
from vo_rag.pipelines.vce.types import (
    TranscriptArtifact,
    TranscriptSegment,
    VceExportRow,
    VceSupersededEntry,
)


def _row(
    *, row_id: str, video_id: str = "yt:abc", start_ts: float = 10.0, end_ts: float = 20.0
) -> VceExportRow:
    return VceExportRow.model_validate(
        make_row_dict(row_id=row_id, video_id=video_id, start_ts=start_ts, end_ts=end_ts)
    )


def _transcript(video_id: str = "yt:abc") -> TranscriptArtifact:
    return TranscriptArtifact(
        segments=[
            TranscriptSegment(start=8.0, end=12.0, text="see how late he turns in here"),
            TranscriptSegment(start=12.0, end=22.0, text="the apex is much later than reference"),
        ],
        language="en",
        blob_key=f"transcripts/{video_id}.transcript.json",
    )


class TestWriteExport:
    def test_writes_expected_files(self, tmp_path: Path) -> None:
        per_video = [
            VideoTranscriptExport(
                video_id="yt:abc",
                transcript=_transcript(),
                rows=[_row(row_id="AAAAAAAAAAAAAAAAAAAAAAAA")],
                superseded=[],
            )
        ]
        paths = write_export(export_dir=tmp_path, per_video=per_video, run_id="abc123")
        assert Path(str(paths["export"])).exists()
        assert Path(str(paths["superseded"])).exists()
        assert Path(str(paths["manifest"])).exists()
        transcripts = paths["transcripts"]
        assert isinstance(transcripts, list)
        assert any(p.endswith(f"yt:abc{VCE_TRANSCRIPT_SUFFIX}") for p in transcripts)

    def test_export_is_sorted_for_determinism(self, tmp_path: Path) -> None:
        rows = [
            _row(row_id="CCCCCCCCCCCCCCCCCCCCCCCC", start_ts=30.0, end_ts=40.0),
            _row(row_id="AAAAAAAAAAAAAAAAAAAAAAAA", start_ts=10.0, end_ts=20.0),
            _row(row_id="BBBBBBBBBBBBBBBBBBBBBBBB", start_ts=20.0, end_ts=30.0),
        ]
        per_video = [
            VideoTranscriptExport(
                video_id="yt:abc", transcript=_transcript(), rows=rows, superseded=[]
            )
        ]
        paths = write_export(export_dir=tmp_path, per_video=per_video, run_id="abc123")
        payload = json.loads(Path(str(paths["export"])).read_text(encoding="utf-8"))
        ids = [row["id"] for row in payload]
        assert ids == sorted(ids)

    def test_two_writes_are_byte_equal(self, tmp_path: Path) -> None:
        """Re-running on the same inputs is byte-equal."""
        per_video = [
            VideoTranscriptExport(
                video_id="yt:abc",
                transcript=_transcript(),
                rows=[
                    _row(row_id="AAAAAAAAAAAAAAAAAAAAAAAA", start_ts=10.0, end_ts=20.0),
                    _row(row_id="BBBBBBBBBBBBBBBBBBBBBBBB", start_ts=20.0, end_ts=30.0),
                ],
                superseded=[],
            )
        ]
        first = write_export(export_dir=tmp_path / "a", per_video=per_video, run_id="run1")
        second = write_export(export_dir=tmp_path / "b", per_video=per_video, run_id="run1")
        assert Path(str(first["export"])).read_bytes() == Path(str(second["export"])).read_bytes()

    def test_export_parses_under_contract_mirror(self, tmp_path: Path) -> None:
        per_video = [
            VideoTranscriptExport(
                video_id="yt:abc",
                transcript=_transcript(),
                rows=[_row(row_id="AAAAAAAAAAAAAAAAAAAAAAAA")],
                superseded=[],
            )
        ]
        paths = write_export(export_dir=tmp_path, per_video=per_video)
        payload = json.loads(Path(str(paths["export"])).read_text(encoding="utf-8"))
        for row in payload:
            VceRowSchema.model_validate(row)

    def test_superseded_writes_to_file(self, tmp_path: Path) -> None:
        per_video = [
            VideoTranscriptExport(
                video_id="yt:abc",
                transcript=_transcript(),
                rows=[],
                superseded=[
                    VceSupersededEntry(
                        old_id="OLD",
                        new_id="NEW",
                        video_id="yt:abc",
                        old_start_ts=10.0,
                        old_end_ts=20.0,
                        new_start_ts=10.5,
                        new_end_ts=20.5,
                    )
                ],
            )
        ]
        paths = write_export(export_dir=tmp_path, per_video=per_video)
        payload = json.loads(Path(str(paths["superseded"])).read_text(encoding="utf-8"))
        assert len(payload) == 1
        assert payload[0]["old_id"] == "OLD"

    def test_manifest_records_counts(self, tmp_path: Path) -> None:
        per_video = [
            VideoTranscriptExport(
                video_id="yt:abc",
                transcript=_transcript(),
                rows=[_row(row_id="AAAAAAAAAAAAAAAAAAAAAAAA")],
                superseded=[],
            )
        ]
        paths = write_export(export_dir=tmp_path, per_video=per_video, run_id="abc123")
        manifest = json.loads(Path(str(paths["manifest"])).read_text(encoding="utf-8"))
        assert manifest["run_id"] == "abc123"
        assert manifest["row_count"] == 1
        assert manifest["video_count"] == 1


class TestLoadExportRows:
    def test_round_trips(self, tmp_path: Path) -> None:
        rows = [_row(row_id="AAAAAAAAAAAAAAAAAAAAAAAA")]
        per_video = [
            VideoTranscriptExport(
                video_id="yt:abc", transcript=_transcript(), rows=rows, superseded=[]
            )
        ]
        write_export(export_dir=tmp_path, per_video=per_video)
        loaded = load_export_rows(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].id == "AAAAAAAAAAAAAAAAAAAAAAAA"
        assert loaded[0].transcript_excerpt == "see how late he turns in here"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_export_rows(tmp_path)
