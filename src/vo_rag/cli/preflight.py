"""`vo-rag preflight` — route or reject book files before ingest.

CPU-only per-file checks (format, encryption, text layer, duplicates) over a
directory, a manifest CSV/text file, or a single file. Prints one schema-2
JSON envelope on stdout and exits 1 when any file is rejected.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

log = logging.getLogger("vo_rag.cli.preflight")

SCHEMA_VERSION = "2"


def envelope(
    command: str, *, data: dict[str, Any] | None = None, error: dict[str, Any] | None = None
) -> dict[str, Any]:
    """``{ok, schema_version, command, ...data}`` or an ``ok:false`` error envelope."""
    if error is not None:
        return {
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "command": command,
            "error": {
                "code": str(error.get("code", "unknown_error")),
                "message": str(error.get("message", "")),
                "recoverable": bool(error.get("recoverable", True)),
                "suggestions": list(error.get("suggestions", ())),
            },
        }
    payload: dict[str, Any] = {"ok": True, "schema_version": SCHEMA_VERSION, "command": command}
    if data:
        payload.update(data)
    return payload


def run_command(command: str, handler: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Invoke ``handler``; map an interrupt or unexpected error to an error envelope."""
    try:
        return handler()
    except KeyboardInterrupt:
        return envelope(
            command,
            error={"code": "interrupted", "message": "interrupted", "recoverable": True},
        )
    except Exception as exc:
        log.exception("unexpected error in %r", command)
        return envelope(
            command,
            error={
                "code": "internal_error",
                "message": f"unexpected {type(exc).__name__}",
                "recoverable": False,
            },
        )


def _emit(env: dict[str, Any]) -> None:
    """Print one envelope; exit 1 when it carries ok:false."""
    click.echo(json.dumps(env, indent=2, default=str))
    if not env.get("ok"):
        raise click.exceptions.Exit(1)


def _load_preflight_catalog(catalog_path: Path | None) -> dict[str, Any] | None:
    """The parsed catalog JSON for duplicate lookup, or ``None`` when no path
    is given or the file is missing -- a missing catalog skips the check."""
    if catalog_path is None or not catalog_path.exists():
        return None
    return dict(json.loads(catalog_path.read_text(encoding="utf-8")))


def _collect_preflight_paths(target: Path) -> list[Path]:
    """Every supported file under ``target`` (a directory, walked
    recursively) or listed in ``target`` (a manifest CSV/text file: a
    ``path`` column if the first line names one, else one path per
    non-empty, non-``#`` line) -- or ``[target]`` itself for a single file.
    """
    from vo_rag.core.parsing.preflight import SUPPORTED_EXTENSIONS

    if target.is_dir():
        return sorted(
            p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    if target.suffix.lower() == ".csv" or target.suffix.lower() == ".txt":
        text = target.read_text(encoding="utf-8")
        lines = text.splitlines()
        # A header row exists only when some column is EXACTLY "path"
        # (case-insensitive); a substring match would swallow a header-less
        # first row such as "epath.pdf".
        first_row = next(csv.reader([lines[0]])) if lines else []
        has_header = any(col.strip().lower() == "path" for col in first_row)
        paths: list[Path] = []
        if has_header:
            for row in csv.DictReader(lines):
                path_key = next((k for k in row if k and k.strip().lower() == "path"), None)
                value = (row.get(path_key) or "").strip() if path_key else ""
                if value:
                    paths.append(Path(value))
        else:
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    paths.append(Path(stripped))
        return paths
    return [target]


def _preflight_result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "path": result.path,
        "sha256": result.sha256,
        "fingerprint": result.fingerprint,
        "kind": result.kind,
        "pages": result.pages,
        "route": result.route,
        "verdict": result.verdict,
        "reasons": list(result.reasons),
    }


def _preflight_report_to_dict(report: Any) -> dict[str, Any]:
    return {
        "counts": report.counts,
        "duplicate_groups": report.duplicate_groups,
        "results": [_preflight_result_to_dict(r) for r in report.results],
    }


def _write_preflight_csv(path: Path, report: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256", "kind", "pages", "route", "verdict", "reasons"])
        for result in report.results:
            writer.writerow(
                [
                    result.path,
                    result.sha256 or "",
                    result.kind or "",
                    result.pages if result.pages is not None else "",
                    result.route or "",
                    result.verdict,
                    "; ".join(result.reasons),
                ]
            )


#: One-line caveats written under a bucket header whose name sounds like a
#: count preflight never takes (see ``select_canary``).
_CANARY_BUCKET_NOTES = {
    "image_heavy_epubs": "# note: image-only route flag (is_image_only), not an image count; "
    "picked in report order",
    "most_figures": "# note: longest native PDFs; page count stands in for figure count "
    "(no figure count exists before ingest)",
}


def _write_canary_txt(path: Path, sample: Any) -> None:
    """Write ``select_canary``'s pick as a flat, grouped text file: a
    ``# <bucket>`` header per stratification bucket (plus a ``# note:`` line
    for the two proxy buckets), then one
    ``path\\tsha256\\troute`` line per file in that bucket (pick order). Plain
    text, not JSON/CSV, as ``canary.txt`` --
    envelope ids don't exist yet at preflight time (they're assigned when
    each file is actually ingested), so this is keyed by path/sha256 for a
    later canary-report run to correlate against.
    """
    by_path = {result.path: result for result in sample.results}
    lines: list[str] = []
    for bucket, paths in sample.buckets.items():
        if not paths:
            continue
        lines.append(f"# {bucket}")
        if bucket in _CANARY_BUCKET_NOTES:
            lines.append(_CANARY_BUCKET_NOTES[bucket])
        for file_path in paths:
            result = by_path[file_path]
            lines.append(f"{result.path}\t{result.sha256 or ''}\t{result.route or ''}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _apply_cache_annotations(report: Any) -> Any:
    """Best-effort: annotate each PDF result with ``cache: native``/``cache:
    ocr`` when the OCR farm's Garage cache already has it. Preflight's core
    module stays network-free; this lives here, in the CLI,
    and degrades to a no-op (with a stderr note) whenever Garage isn't
    configured or reachable -- ``--check-cache`` must never fail a run that
    would otherwise succeed.
    """
    import asyncio
    from dataclasses import replace

    from vo_rag.core.config import get_settings
    from vo_rag.core.parsing.factory import _artifacts_bucket, _garage_from_settings
    from vo_rag.core.parsing.router import farm_ocr_meta_key

    settings = get_settings()
    garage = _garage_from_settings(settings)
    if garage is None:
        click.echo("preflight --check-cache: Garage is not configured; skipping", err=True)
        return report
    bucket = _artifacts_bucket(settings)

    async def _pipeline_for(sha256: str) -> str | None:
        meta_key = farm_ocr_meta_key(sha256)
        try:
            if await garage.object_metadata(bucket, meta_key) is None:
                return None
            meta = json.loads(await garage.get_bytes(bucket, meta_key))
        except Exception:  # noqa: BLE001 - a cache lookup failure must not fail the run
            return None
        return str(meta.get("pipeline") or "unknown") if isinstance(meta, dict) else None

    async def _annotate_all() -> list[Any]:
        annotated = []
        for result in report.results:
            if result.sha256 and result.kind == "pdf":
                pipeline = await _pipeline_for(result.sha256)
                if pipeline:
                    result = replace(result, reasons=(*result.reasons, f"cache: {pipeline}"))
            annotated.append(result)
        return annotated

    try:
        new_results = asyncio.run(_annotate_all())
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the run
        click.echo(f"preflight --check-cache: lookup failed, skipping ({exc})", err=True)
        return report
    return replace(report, results=tuple(new_results))


def register_preflight_commands(root: click.Group) -> None:
    """Attach ``vo-rag preflight`` (see vo_rag.cli.main)."""

    @root.command("preflight")
    @click.argument("target", type=click.Path(exists=True, path_type=Path))
    @click.option(
        "--out",
        "out_path",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="write the full report (counts + every result) as JSON here",
    )
    @click.option(
        "--csv",
        "csv_path",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="write a flat CSV summary (one row per file) here",
    )
    @click.option(
        "--check-cache",
        is_flag=True,
        default=False,
        help="best-effort: annotate PDFs with whether the OCR farm's Garage cache "
        "already has them (native/ocr) — needs Garage config; skipped, not failed, "
        "when it isn't reachable",
    )
    @click.option(
        "--jobs",
        type=int,
        default=1,
        help="process pool size for the CPU-bound per-file checks (pandoc, fitz)",
    )
    @click.option(
        "--timeout",
        "timeout_s",
        type=float,
        default=180.0,
        help="hard per-file timeout in seconds under --jobs>1 (a file over this "
        "is rejected as preflight_timeout, never left hanging)",
    )
    @click.option(
        "--catalog",
        "catalog_path",
        type=click.Path(path_type=Path),
        default=None,
        help="catalog JSON for duplicate lookup; omitted or missing skips the check",
    )
    @click.option(
        "--canary-out",
        "canary_out_path",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="write a stratified canary sample here, e.g. "
        "canary.txt -- a stratified pick, not the full batch",
    )
    @click.option(
        "--canary-n",
        type=int,
        default=20,
        help="canary sample size, used together with --canary-out",
    )
    def preflight(
        target: Path,
        out_path: Path | None,
        csv_path: Path | None,
        check_cache: bool,
        jobs: int,
        timeout_s: float,
        catalog_path: Path | None,
        canary_out_path: Path | None,
        canary_n: int,
    ) -> None:
        """Route or reject every book in TARGET (a directory or a manifest CSV)
        before ingest -- CPU-only, no network, no Qdrant, no Postgres.

        Prints one JSON envelope with ``counts`` by route and verdict; exits 1
        when any file is ``reject``.
        """
        from vo_rag.core.parsing import preflight as pf

        def handler() -> dict[str, Any]:
            paths = _collect_preflight_paths(target)
            catalog = _load_preflight_catalog(catalog_path)
            report = pf.inspect_batch(paths, catalog=catalog, jobs=jobs, timeout_s=timeout_s)
            if check_cache:
                report = _apply_cache_annotations(report)

            if out_path:
                out_path.write_text(
                    json.dumps(_preflight_report_to_dict(report), indent=2), encoding="utf-8"
                )
            if csv_path:
                _write_preflight_csv(csv_path, report)
            if canary_out_path:
                sample = pf.select_canary(report, canary_n)
                _write_canary_txt(canary_out_path, sample)

            rejects = report.counts.get("verdict", {}).get("reject", 0)
            env = envelope(
                "preflight",
                data={
                    "counts": report.counts,
                    "duplicate_groups": report.duplicate_groups,
                    "results": [_preflight_result_to_dict(r) for r in report.results],
                },
            )
            env["ok"] = rejects == 0
            return env

        _emit(run_command("preflight", handler))
