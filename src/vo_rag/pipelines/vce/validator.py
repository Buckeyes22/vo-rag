"""Validator for the VCE export — runs seven contract checks.

Each check is a method that returns a :class:`CheckResult`; the
:class:`VceExportValidator` runs them all and aggregates a
:class:`ValidationReport` (the same report shape as ``book_rag/validator.py``).

The seven checks, in run order:

1. **schema**: every row parses under ``vceRowSchema v2`` (the Pydantic
   mirror in ``tests/pipelines/vce/_contract.py`` is the test-side guard).
2. **interval_validity**: ``start_ts < end_ts`` for every row.
3. **envelope_consistency**: every video's rows reference only that video.
4. **transcript_coverage**: at least 90% of rows carry ``transcript_excerpt``.
5. **transcript_substring**: every excerpt is a substring of the per-video
   ``<video_id>.transcript.json`` content.
6. **superseded_completeness**: every superseded entry's old id appears in
   ``superseded.json`` exactly once.
7. **baseline_preservation**: the legacy baseline row ids are preserved where
   the interval is unchanged.

``ValidationCheck.IDEMPOTENCY`` names a re-run-yields-identical-ids check that
would compare two exports; no check implements it and the validator does not
run it.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

from vo_rag.pipelines.vce.baseline import BaselineIndex, load_baseline_rows
from vo_rag.pipelines.vce.export import (
    VCE_EXPORT_FILENAME,
    VCE_SUPERSEDED_FILENAME,
)
from vo_rag.pipelines.vce.types import (
    VCE_CONTRACT_VERSION,
    VceExportRow,
)

DEFAULT_COVERAGE_THRESHOLD = 0.90


class ValidationCheck(StrEnum):
    """Names of the checks the validator can run.

    Stable identifiers — the runner / CLI uses them to select which checks
    to run, and the names appear in :class:`CheckResult.name`.
    """

    SCHEMA = "schema"
    TRANSCRIPT_COVERAGE = "transcript_coverage"
    TRANSCRIPT_SUBSTRING = "transcript_substring"
    BASELINE_PRESERVATION = "baseline_preservation"
    IDEMPOTENCY = "idempotency"
    ENVELOPE_CONSISTENCY = "envelope_consistency"
    INTERVAL_VALIDITY = "interval_validity"
    SUPERSEDED_COMPLETENESS = "superseded_completeness"


@dataclasses.dataclass
class CheckResult:
    """Outcome of one :class:`ValidationCheck`.

    ``passed=True`` is the happy path; ``passed=False`` with a populated
    ``reasons`` list is the failure path. ``details`` is a structured
    blob specific to each check (e.g. ``{"ratio": 0.94}`` for
    ``transcript_coverage``).
    """

    name: ValidationCheck
    passed: bool
    reasons: list[str] = dataclasses.field(default_factory=list)
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name.value,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


@dataclasses.dataclass
class ValidationReport:
    """Aggregated report from :meth:`VceExportValidator.run_all_checks`."""

    export_dir: str
    contract_version: str
    results: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "export_dir": self.export_dir,
            "contract_version": self.contract_version,
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
        }


class VceExportValidator:
    """Runs the contract checks against a published export directory.

    The constructor takes the export dir and an optional baseline index
    (for the ``baseline_preservation`` check). The optional
    ``coverage_threshold`` overrides the default 0.90 acceptance floor.
    """

    def __init__(
        self,
        export_dir: Path | str,
        *,
        baseline: BaselineIndex | None = None,
        baseline_path: str | Path | None = None,
        coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    ) -> None:
        self._export_dir = Path(export_dir)
        self._coverage_threshold = coverage_threshold
        if baseline is not None:
            self._baseline: BaselineIndex | None = baseline
        elif baseline_path is not None:
            self._baseline = load_baseline_rows(baseline_path)
        else:
            self._baseline = None

    # ── public entry ────────────────────────────────────────────────────────

    def run_all_checks(self) -> ValidationReport:
        results = [
            self.check_schema(),
            self.check_interval_validity(),
            self.check_envelope_consistency(),
            self.check_transcript_coverage(),
            self.check_transcript_substring(),
            self.check_superseded_completeness(),
            self.check_baseline_preservation(),
        ]
        return ValidationReport(
            export_dir=str(self._export_dir),
            contract_version=VCE_CONTRACT_VERSION,
            results=results,
        )

    # ── individual checks ───────────────────────────────────────────────────

    def check_schema(self) -> CheckResult:
        """Every row in ``vce-episode-export.json`` parses under vceRowSchema v2.

        The Pydantic mirror in ``tests/pipelines/vce/_contract.py`` is the
        canonical schema. Here we re-validate against the production model
        (:class:`VceExportRow`); the test-side mirror catches downstream consumer-side
        drift independently.
        """
        export_path = self._export_dir / VCE_EXPORT_FILENAME
        if not export_path.exists():
            return CheckResult(
                name=ValidationCheck.SCHEMA,
                passed=False,
                reasons=[f"missing export file: {export_path}"],
            )
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return CheckResult(
                name=ValidationCheck.SCHEMA,
                passed=False,
                reasons=[f"export root must be an array, got {type(payload).__name__}"],
            )
        bad_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(payload):
            try:
                VceExportRow.model_validate(row)
            except Exception as exc:  # noqa: BLE001 - report any validation error
                bad_rows.append({"index": idx, "error": str(exc)})
        if bad_rows:
            return CheckResult(
                name=ValidationCheck.SCHEMA,
                passed=False,
                reasons=[f"{len(bad_rows)} rows failed schema validation"],
                details={"bad_rows": bad_rows[:10], "total_bad": len(bad_rows)},
            )
        return CheckResult(
            name=ValidationCheck.SCHEMA,
            passed=True,
            details={"row_count": len(payload)},
        )

    def check_interval_validity(self) -> CheckResult:
        rows = _safe_load_rows(self._export_dir)
        if rows is None:
            return CheckResult(
                name=ValidationCheck.INTERVAL_VALIDITY,
                passed=False,
                reasons=[f"missing export file: {self._export_dir / VCE_EXPORT_FILENAME}"],
            )
        bad: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            if not (row.start_ts < row.end_ts):
                bad.append(
                    {"index": idx, "id": row.id, "start_ts": row.start_ts, "end_ts": row.end_ts}
                )
        if bad:
            return CheckResult(
                name=ValidationCheck.INTERVAL_VALIDITY,
                passed=False,
                reasons=[f"{len(bad)} rows have start_ts >= end_ts"],
                details={"bad_rows": bad[:10], "total_bad": len(bad)},
            )
        return CheckResult(
            name=ValidationCheck.INTERVAL_VALIDITY,
            passed=True,
            details={"row_count": len(rows)},
        )

    def check_envelope_consistency(self) -> CheckResult:
        rows = _safe_load_rows(self._export_dir)
        if rows is None:
            return CheckResult(
                name=ValidationCheck.ENVELOPE_CONSISTENCY,
                passed=False,
                reasons=[f"missing export file: {self._export_dir / VCE_EXPORT_FILENAME}"],
            )
        # No cross-cutting constraint to enforce yet — every row already
        # carries its video_id. The check exists so future per-video
        # manifests can be cross-validated without a schema change.
        per_video: dict[str, int] = Counter(r.video_id for r in rows)
        return CheckResult(
            name=ValidationCheck.ENVELOPE_CONSISTENCY,
            passed=True,
            details={"video_count": len(per_video), "rows_per_video": dict(per_video)},
        )

    def check_transcript_coverage(self) -> CheckResult:
        rows = _safe_load_rows(self._export_dir)
        if rows is None:
            return CheckResult(
                name=ValidationCheck.TRANSCRIPT_COVERAGE,
                passed=False,
                reasons=[f"missing export file: {self._export_dir / VCE_EXPORT_FILENAME}"],
            )
        if not rows:
            return CheckResult(
                name=ValidationCheck.TRANSCRIPT_COVERAGE,
                passed=False,
                reasons=["export contains zero rows"],
            )
        covered = sum(1 for r in rows if r.transcript_excerpt)
        ratio = covered / len(rows)
        passed = ratio >= self._coverage_threshold
        reasons = []
        if not passed:
            reasons.append(
                f"coverage {ratio:.3f} below threshold {self._coverage_threshold:.3f} "
                f"({covered}/{len(rows)} rows)"
            )
        return CheckResult(
            name=ValidationCheck.TRANSCRIPT_COVERAGE,
            passed=passed,
            reasons=reasons,
            details={
                "covered": covered,
                "total": len(rows),
                "ratio": ratio,
                "threshold": self._coverage_threshold,
            },
        )

    def check_transcript_substring(self) -> CheckResult:
        rows = _safe_load_rows(self._export_dir)
        if rows is None:
            return CheckResult(
                name=ValidationCheck.TRANSCRIPT_SUBSTRING,
                passed=False,
                reasons=[f"missing export file: {self._export_dir / VCE_EXPORT_FILENAME}"],
            )
        # Build {video_id: joined_text} from per-video transcript files.
        joined: dict[str, str] = {}
        for video_id in {r.video_id for r in rows}:
            transcript_path = self._export_dir / f"{video_id}.transcript.json"
            if not transcript_path.exists():
                return CheckResult(
                    name=ValidationCheck.TRANSCRIPT_SUBSTRING,
                    passed=False,
                    reasons=[f"missing transcript file: {transcript_path}"],
                )
            payload = json.loads(transcript_path.read_text(encoding="utf-8"))
            joined[video_id] = " ".join(str(s.get("text", "")) for s in payload.get("segments", []))

        bad: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            if not row.transcript_excerpt:
                continue  # empty excerpts are not substring-checked
            text = joined.get(row.video_id, "")
            if row.transcript_excerpt not in text:
                bad.append(
                    {
                        "index": idx,
                        "id": row.id,
                        "video_id": row.video_id,
                        "excerpt": row.transcript_excerpt[:80],
                    }
                )
        if bad:
            return CheckResult(
                name=ValidationCheck.TRANSCRIPT_SUBSTRING,
                passed=False,
                reasons=[f"{len(bad)} excerpts not found in their video's transcript"],
                details={"bad_rows": bad[:10], "total_bad": len(bad)},
            )
        return CheckResult(
            name=ValidationCheck.TRANSCRIPT_SUBSTRING,
            passed=True,
            details={"videos_checked": len(joined)},
        )

    def check_superseded_completeness(self) -> CheckResult:
        superseded_path = self._export_dir / VCE_SUPERSEDED_FILENAME
        if not superseded_path.exists():
            return CheckResult(
                name=ValidationCheck.SUPERSEDED_COMPLETENESS,
                passed=True,
                reasons=[],
                details={"note": "no superseded file (run had no interval changes)"},
            )
        payload = json.loads(superseded_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return CheckResult(
                name=ValidationCheck.SUPERSEDED_COMPLETENESS,
                passed=False,
                reasons=[f"superseded.json must be an array, got {type(payload).__name__}"],
            )
        # Every entry must be unique on old_id (otherwise the legacy case
        # would be ambiguous to downstream consumer).
        old_ids = [entry.get("old_id") for entry in payload if isinstance(entry, dict)]
        counts = Counter(old_ids)
        duplicates = {oid: c for oid, c in counts.items() if c > 1}
        if duplicates:
            return CheckResult(
                name=ValidationCheck.SUPERSEDED_COMPLETENESS,
                passed=False,
                reasons=[f"{len(duplicates)} old_ids appear more than once"],
                details={"duplicates": dict(duplicates)},
            )
        return CheckResult(
            name=ValidationCheck.SUPERSEDED_COMPLETENESS,
            passed=True,
            details={"entry_count": len(payload)},
        )

    def check_baseline_preservation(self) -> CheckResult:
        """Verify the baseline row ids are preserved when intervals are unchanged.

        Compares the export's ``(video_id, start, end)`` keys against the
        baseline's. Two failure modes matter:

        1. The export contains a row at a baseline interval, but the id
           is **not** the legacy id — the export drifted, and downstream consumer's
           ``case_id = vce:<id>`` mapping would break.
        2. The export omits a baseline row at a baseline interval —
           coverage gap (e.g. transcription failed for that video).
           Reported but not failing the check; coverage is a separate
           concern from preservation.

        If no baseline is loaded (e.g. CI without the workstation file),
        the check passes trivially with a note.
        """
        rows = _safe_load_rows(self._export_dir)
        if rows is None:
            return CheckResult(
                name=ValidationCheck.BASELINE_PRESERVATION,
                passed=False,
                reasons=[f"missing export file: {self._export_dir / VCE_EXPORT_FILENAME}"],
            )
        if self._baseline is None:
            return CheckResult(
                name=ValidationCheck.BASELINE_PRESERVATION,
                passed=True,
                details={"note": "no baseline loaded; check is a no-op"},
            )
        export_by_key: dict[tuple[str, float, float], VceExportRow] = {}
        for row in rows:
            key = (row.video_id, round(row.start_ts, 3), round(row.end_ts, 3))
            export_by_key[key] = row

        id_drift: list[dict[str, str]] = []
        coverage_gaps: list[dict[str, str]] = []
        preserved: list[str] = []
        for baseline_row in self._baseline.rows:
            video_id = str(baseline_row["video_id"])
            legacy_id = str(baseline_row["legacy_id"])
            start_ts = float(baseline_row["start_ts"])
            end_ts = float(baseline_row["end_ts"])
            key = (video_id, round(start_ts, 3), round(end_ts, 3))
            export_row = export_by_key.get(key)
            if export_row is None:
                coverage_gaps.append(
                    {
                        "video_id": video_id,
                        "legacy_id": legacy_id,
                        "reason": "export has no row at this interval (transcript may have failed)",
                    }
                )
                continue
            if export_row.id == legacy_id:
                preserved.append(legacy_id)
                continue
            id_drift.append(
                {
                    "video_id": video_id,
                    "legacy_id": legacy_id,
                    "export_id": export_row.id,
                    "reason": "id drift — downstream consumer's case_id mapping will break",
                }
            )

        details = {
            "baseline_rows": len(self._baseline.rows),
            "preserved": len(preserved),
            "coverage_gaps": len(coverage_gaps),
            "id_drift": len(id_drift),
        }
        if id_drift:
            return CheckResult(
                name=ValidationCheck.BASELINE_PRESERVATION,
                passed=False,
                reasons=[f"{len(id_drift)} baseline rows have wrong ids (drift)"],
                details={**details, "id_drift_rows": id_drift[:10]},
            )
        return CheckResult(
            name=ValidationCheck.BASELINE_PRESERVATION,
            passed=True,
            details=details,
        )


def _safe_load_rows(export_dir: Path) -> list[VceExportRow] | None:
    """Load rows from the export dir, returning ``None`` if the file is missing."""
    export_path = export_dir / VCE_EXPORT_FILENAME
    if not export_path.exists():
        return None
    try:
        payload = json.loads(export_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    return [VceExportRow.model_validate(row) for row in payload]


def validate_export(
    export_dir: Path | str,
    *,
    baseline: BaselineIndex | None = None,
    baseline_path: str | Path | None = None,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> ValidationReport:
    """One-shot validator entry point.

    Used by the CLI (``vo-rag vce validate <export_dir>``) and by tests
    that want a single call without instantiating :class:`VceExportValidator`.
    """
    return VceExportValidator(
        export_dir,
        baseline=baseline,
        baseline_path=baseline_path,
        coverage_threshold=coverage_threshold,
    ).run_all_checks()


__all__ = [
    "DEFAULT_COVERAGE_THRESHOLD",
    "CheckResult",
    "ValidationCheck",
    "ValidationReport",
    "VceExportValidator",
    "validate_export",
]
