"""``vo-rag vce`` Click group — run / validate / status.

Three subcommands:

- ``vo-rag vce run`` — execute the stage DAG over every (or a subset
  of) manifest videos; writes the export directory.
- ``vo-rag vce validate <export_dir>`` — run the seven contract checks
  against a published export directory; exit 0 on pass, 1 on fail.
- ``vo-rag vce status`` — operator smoke: counts per-stage state in
  ``stage_run_log`` and per-status manifest rows.

Heavy imports (the runner, the validators) are deferred to the
subcommand body so ``vo-rag --help`` stays fast.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click


@click.group("vce")
def vce_group() -> None:
    """Video coaching extraction commands.

    Run the VCE pipeline against the manifest; validate a published
    export against the export validator's seven checks; report per-stage
    status counts.
    """


@vce_group.command("run")
@click.option(
    "--export-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Target directory for vce-episode-export.json + per-video transcripts.",
)
@click.option(
    "--video-ids",
    multiple=True,
    help="Subset of video_manifest.video_id rows to process (default: every manifest row).",
)
@click.option(
    "--executor",
    type=click.Choice(["stub", "local", "remote", "speaches"]),
    default="stub",
    show_default=True,
    help=(
        "Transcription executor. 'remote' and 'speaches' require "
        "VORAG_VCE_TRANSCRIPTION_ENDPOINT_URL (or VORAG_VCE_SPEACHES_URL)."
    ),
)
@click.option(
    "--baseline",
    "baseline_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Baseline JSON path (default: $VORAG_VCE_BASELINE_ROWS_PATH).",
)
@click.option(
    "--publish-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional destination for vce-episode-export.json.",
)
@click.option("--run-id", default=None, help="Run identifier (default: random 16-char hex).")
@click.option("--json", "output_json", is_flag=True, default=False, help="Emit JSON summary.")
def vce_run(
    export_dir: Path,
    video_ids: tuple[str, ...],
    executor: str,
    baseline_path: Path | None,
    publish_path: Path | None,
    run_id: str | None,
    output_json: bool,
) -> None:
    """Run the VCE stage DAG and write the export."""
    from vo_rag.cli.vce_impl import run_vce_cli

    summary = asyncio.run(
        run_vce_cli(
            export_dir=export_dir,
            video_ids=list(video_ids) or None,
            executor_name=executor,
            baseline_path=baseline_path,
            publish_path=publish_path,
            run_id=run_id,
        )
    )
    if output_json:
        click.echo(json.dumps(summary, indent=2, default=str))
        return
    click.echo(f"run_id: {summary['run_id']}")
    click.echo(f"videos_processed: {summary['videos_processed']}")
    click.echo(f"videos_failed: {summary['videos_failed']}")
    click.echo(f"rows_written: {summary['rows_written']}")
    click.echo(f"cost_usd: {summary['cost_usd']:.2f}")
    click.echo(f"export_path: {summary['paths']['export']}")
    if summary.get("publish_path"):
        click.echo(f"publish_path: {summary['publish_path']}")


@vce_group.command("validate")
@click.argument(
    "export_dir",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--baseline",
    "baseline_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to the caller-supplied downstream consumer baseline JSON.",
)
@click.option(
    "--coverage-threshold",
    type=float,
    default=0.90,
    show_default=True,
    help="Minimum fraction of rows that must carry transcript_excerpt (transcript_coverage).",
)
@click.option("--json", "output_json", is_flag=True, default=False, help="Emit JSON report.")
def vce_validate(
    export_dir: Path,
    baseline_path: Path | None,
    coverage_threshold: float,
    output_json: bool,
) -> None:
    """Run the seven export validator checks against a published export directory."""
    from vo_rag.pipelines.vce.validator import validate_export

    report = validate_export(
        export_dir=export_dir,
        baseline_path=baseline_path,
        coverage_threshold=coverage_threshold,
    )
    if output_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        for result in report.results:
            marker = "PASS" if result.passed else "FAIL"
            click.echo(f"[{marker}] {result.name.value}: {result.details}")
            for reason in result.reasons:
                click.echo(f"    {reason}")
    if not report.passed:
        sys.exit(1)


@vce_group.command("status")
@click.option("--video-id", default=None, help="Filter to one video.")
@click.option("--json", "output_json", is_flag=True, default=False, help="Emit JSON.")
def vce_status(video_id: str | None, output_json: bool) -> None:
    """Operator smoke — per-stage state counts from stage_run_log."""
    from vo_rag.cli.vce_impl import status_vce_cli

    payload = asyncio.run(status_vce_cli(video_id=video_id))
    if output_json:
        click.echo(json.dumps(payload, indent=2, default=str))
        return
    click.echo(f"videos: {payload['video_count']}")
    click.echo(f"coaching_records: {payload['coaching_records']}")
    for state, count in sorted(payload["stage_states"].items()):
        click.echo(f"  stage_state.{state}: {count}")


def register_vce_commands(root: click.Group) -> None:
    """Attach the ``vo-rag vce`` group to the root CLI."""
    root.add_command(vce_group)


__all__ = ["register_vce_commands", "vce_group"]
