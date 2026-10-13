"""Tests for the validator — runs the five contract checks against an export."""

from __future__ import annotations

import json
from pathlib import Path

from tests.pipelines.vce.conftest import make_row_dict
from vo_rag.pipelines.vce.export import (
    VCE_EXPORT_FILENAME,
    VCE_SUPERSEDED_FILENAME,
    write_export,
)
from vo_rag.pipelines.vce.types import (
    TranscriptArtifact,
    TranscriptSegment,
    VceExportRow,
)
from vo_rag.pipelines.vce.validator import (
    DEFAULT_COVERAGE_THRESHOLD,
    ValidationCheck,
    VceExportValidator,
    validate_export,
)


def _row(
    *,
    row_id: str,
    video_id: str = "yt:abc",
    start_ts: float = 10.0,
    end_ts: float = 20.0,
    excerpt: str = "see how late he turns in here",
) -> VceExportRow:
    return VceExportRow.model_validate(
        make_row_dict(
            row_id=row_id, video_id=video_id, start_ts=start_ts, end_ts=end_ts, excerpt=excerpt
        )
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


def _build_export(tmp_path: Path, rows: list[VceExportRow] | None = None) -> Path:
    """Helper — write a valid export directory and return its path."""
    rows = rows or [_row(row_id="AAAAAAAAAAAAAAAAAAAAAAAA")]
    from vo_rag.pipelines.vce.export import VideoTranscriptExport

    write_export(
        export_dir=tmp_path,
        per_video=[
            VideoTranscriptExport(
                video_id="yt:abc",
                transcript=_transcript(),
                rows=rows,
                superseded=[],
            )
        ],
        run_id="test",
    )
    return tmp_path


class TestCheckSchema:
    def test_passes_on_valid_export(self, tmp_path: Path) -> None:
        _build_export(tmp_path)
        result = VceExportValidator(tmp_path).check_schema()
        assert result.passed
        assert result.name == ValidationCheck.SCHEMA

    def test_fails_on_missing_file(self, tmp_path: Path) -> None:
        result = VceExportValidator(tmp_path).check_schema()
        assert not result.passed
        assert "missing export file" in result.reasons[0]


class TestCheckIntervalValidity:
    def test_passes_when_all_intervals_valid(self, tmp_path: Path) -> None:
        _build_export(tmp_path)
        result = VceExportValidator(tmp_path).check_interval_validity()
        assert result.passed

    def test_fails_on_inverted_intervals(self, tmp_path: Path) -> None:
        """The schema permits positive floats for start_ts and end_ts but does
        not enforce start_ts < end_ts at the field level. The interval check
        catches it (per row-level inspection)."""
        # Schema passes (both fields are positive floats); interval check fails.
        (tmp_path / VCE_EXPORT_FILENAME).write_text(
            json.dumps(
                [
                    {
                        "id": "AAAAAAAAAAAAAAAAAAAAAAAA",
                        "video_id": "yt:abc",
                        "start_ts": 20.0,
                        "end_ts": 10.0,
                        "observation": "x",
                        "evidence": "inferred_from_audio",
                        "transcript_excerpt": "",
                        "transcript_source": None,
                        "anchors": [],
                    }
                ]
            ),
            encoding="utf-8",
        )
        validator = VceExportValidator(tmp_path)
        # Schema passes (both fields satisfy gt=0.0).
        assert validator.check_schema().passed
        # The interval check catches the inverted row.
        assert not validator.check_interval_validity().passed


class TestCheckTranscriptCoverage:
    def test_passes_when_all_rows_covered(self, tmp_path: Path) -> None:
        _build_export(tmp_path)
        result = VceExportValidator(tmp_path).check_transcript_coverage()
        assert result.passed
        assert result.details["covered"] == 1
        assert result.details["total"] == 1

    def test_fails_below_threshold(self, tmp_path: Path) -> None:
        rows = [
            _row(row_id="AAAAAAAAAAAAAAAAAAAAAAAA", excerpt="see how late he turns in here"),
            _row(row_id="BBBBBBBBBBBBBBBBBBBBBBBB", start_ts=20.0, end_ts=30.0, excerpt=""),
        ]
        _build_export(tmp_path, rows)
        result = VceExportValidator(tmp_path).check_transcript_coverage()
        assert not result.passed
        assert result.details["ratio"] == 0.5

    def test_threshold_is_configurable(self, tmp_path: Path) -> None:
        """A threshold of 1.0 with one row covered passes; a threshold above 1.0 fails."""
        _build_export(tmp_path)
        # 1.0 threshold with one covered row → ratio=1.0 → passes.
        passing = VceExportValidator(tmp_path, coverage_threshold=1.0).check_transcript_coverage()
        assert passing.passed
        # A threshold above 1.0 is unsatisfiable by definition; the check fails.
        impossible = VceExportValidator(
            tmp_path, coverage_threshold=1.01
        ).check_transcript_coverage()
        assert not impossible.passed


class TestCheckTranscriptSubstring:
    def test_passes_when_excerpt_in_joined_text(self, tmp_path: Path) -> None:
        _build_export(tmp_path)
        result = VceExportValidator(tmp_path).check_transcript_substring()
        assert result.passed

    def test_fails_when_excerpt_missing(self, tmp_path: Path) -> None:
        # Override the row with an excerpt not in the transcript.
        bad_row = _row(row_id="AAAAAAAAAAAAAAAAAAAAAAAA", excerpt="text not in any transcript")
        _build_export(tmp_path, rows=[bad_row])
        result = VceExportValidator(tmp_path).check_transcript_substring()
        assert not result.passed
        assert "excerpts not found" in result.reasons[0]

    def test_missing_transcript_file(self, tmp_path: Path) -> None:
        _build_export(tmp_path)
        (tmp_path / "yt:abc.transcript.json").unlink()
        result = VceExportValidator(tmp_path).check_transcript_substring()
        assert not result.passed
        assert "missing transcript file" in result.reasons[0]


class TestCheckSupersededCompleteness:
    def test_passes_when_no_superseded_file(self, tmp_path: Path) -> None:
        _build_export(tmp_path)
        result = VceExportValidator(tmp_path).check_superseded_completeness()
        assert result.passed

    def test_fails_on_duplicate_old_id(self, tmp_path: Path) -> None:
        (tmp_path / VCE_SUPERSEDED_FILENAME).write_text(
            json.dumps(
                [
                    {
                        "old_id": "DUPLICATED",
                        "new_id": "NEW1",
                        "video_id": "yt:abc",
                        "old_start_ts": 1.0,
                        "old_end_ts": 2.0,
                        "new_start_ts": 1.0,
                        "new_end_ts": 2.0,
                    },
                    {
                        "old_id": "DUPLICATED",
                        "new_id": "NEW2",
                        "video_id": "yt:abc",
                        "old_start_ts": 3.0,
                        "old_end_ts": 4.0,
                        "new_start_ts": 3.0,
                        "new_end_ts": 4.0,
                    },
                ]
            ),
            encoding="utf-8",
        )
        result = VceExportValidator(tmp_path).check_superseded_completeness()
        assert not result.passed


class TestCheckBaselinePreservation:
    def test_passes_with_no_baseline(self, tmp_path: Path) -> None:
        _build_export(tmp_path)
        result = VceExportValidator(tmp_path).check_baseline_preservation()
        assert result.passed  # no baseline loaded — check is a no-op

    def test_passes_when_baseline_ids_match(self, tmp_path: Path, baseline_rows_path: Path) -> None:
        # The fixture baseline has yt:abc / 10.000 / 20.000 with legacy_id
        # BBBB...; write a row with that legacy id and verify preservation.
        from vo_rag.pipelines.vce.baseline import load_baseline_rows

        index = load_baseline_rows(baseline_rows_path)
        baseline_row = index.by_interval.get(("yt:abc", 10.0, 20.0))
        assert baseline_row is not None
        legacy_id = str(baseline_row["legacy_id"])
        row = _row(row_id=legacy_id)
        _build_export(tmp_path, rows=[row])
        result = VceExportValidator(
            tmp_path, baseline_path=baseline_rows_path
        ).check_baseline_preservation()
        assert result.passed, result.reasons


class TestValidateExport:
    def test_full_run_all_passing(self, tmp_path: Path) -> None:
        _build_export(tmp_path)
        report = validate_export(tmp_path)
        assert report.passed, [(r.name, r.reasons) for r in report.results if not r.passed]
        assert len(report.results) == 7

    def test_default_threshold_is_90_percent(self) -> None:
        assert DEFAULT_COVERAGE_THRESHOLD == 0.90
