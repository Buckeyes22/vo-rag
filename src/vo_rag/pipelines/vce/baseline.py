"""Load caller-supplied VCE baseline rows; the default example is synthetic."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vo_rag.pipelines.vce.types import VceExportEvidence, record_id_for_interval

DEFAULT_BASELINE_PATH = os.environ.get(
    "VORAG_VCE_BASELINE_ROWS_PATH",
    "./examples/vce-baseline.json",
)


@dataclass(frozen=True)
class BaselineIndex:
    """An indexed view of the baseline rows, keyed by ``(video_id, start, end)``.

    The key is ``(video_id, round(start, 3), round(end, 3))`` so float drift
    in the baseline's string fields doesn't break lookups.
    """

    rows: tuple[dict[str, Any], ...]
    by_interval: dict[tuple[str, float, float], dict[str, Any]]
    by_id: dict[str, dict[str, Any]]


def _round_for_key(value: float) -> float:
    return round(value, 3)


def load_baseline_rows(path: str | Path | None = None) -> BaselineIndex:
    """Read and index the baseline rows JSON file.

    Args:
        path: Path to the baseline rows JSON. Defaults to
            ``$VORAG_VCE_BASELINE_ROWS_PATH`` or
            ``./examples/vce-baseline.json``.

    Returns:
        A :class:`BaselineIndex` with the rows, an ``(video_id, start, end)``
        lookup, and an ``id`` lookup.

    Raises:
        FileNotFoundError: When the path does not exist.
        ValueError: When a row is missing required fields.
    """
    resolved = Path(path or DEFAULT_BASELINE_PATH).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"baseline rows file not found: {resolved}")

    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"baseline rows file must contain a JSON array; got {type(raw).__name__}")

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw):
        try:
            video_id = str(row["video_id"])
            start_ts = float(row["start_ts"])
            end_ts = float(row["end_ts"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"baseline row {idx} missing/invalid video_id/start_ts/end_ts: {exc}"
            ) from exc
        # Normalise the row in place: keep all original fields, but ensure
        # start_ts / end_ts are floats and observation / evidence / anchors
        # are present (legacy rows may lack anchors).
        normalised = {
            **row,
            "video_id": video_id,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "observation": str(row.get("observation", "")),
            "evidence": str(row.get("evidence", VceExportEvidence.INFERRED_FROM_AUDIO.value)),
            "anchors": list(row.get("anchors") or []),
            "legacy_id": str(row["id"]),
        }
        rows.append(normalised)

    by_interval: dict[tuple[str, float, float], dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = str(row["video_id"])
        start_ts = float(row["start_ts"])
        end_ts = float(row["end_ts"])
        legacy_id = str(row["legacy_id"])
        key = (video_id, _round_for_key(start_ts), _round_for_key(end_ts))
        # Last write wins on duplicate (video, interval) — the baseline should
        # not contain duplicates, but the loader must not crash if it does.
        by_interval[key] = row
        by_id[legacy_id] = row

    return BaselineIndex(rows=tuple(rows), by_interval=by_interval, by_id=by_id)


def expected_legacy_id(video_id: str, start_ts: float, end_ts: float) -> str:
    """Compute the id an export must produce to preserve legacy identity.

    Wraps :func:`record_id_for_interval` so callers don't need to remember
    the exact key format. Used by tests to assert that the export's id
    derivation matches what the legacy baseline carried.
    """
    return record_id_for_interval(
        video_id=video_id,
        start_ts=start_ts,
        end_ts=end_ts,
        length=26,
    )


__all__ = [
    "DEFAULT_BASELINE_PATH",
    "BaselineIndex",
    "expected_legacy_id",
    "load_baseline_rows",
]
