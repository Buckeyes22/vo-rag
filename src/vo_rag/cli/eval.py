"""Eval-gate CLI command."""

from __future__ import annotations

import click

from vo_rag.core.config.pipelines import load_pipeline_registry
from vo_rag.core.eval.registered import run_registered_eval_gate


def register_eval_commands(root: click.Group) -> None:
    @root.command("eval-gate")
    def eval_gate() -> None:
        """Run the retrieval eval gate against the active pipeline roster."""
        registry = load_pipeline_registry()
        result = run_registered_eval_gate(registry)
        if result.blocked_preconditions:
            click.echo("eval gate BLOCKED - retrieval gate preconditions are not satisfied:")
            for pipeline, mode in result.blocked_preconditions.items():
                click.echo(f"  - {pipeline}: mode == {mode}")
            raise click.ClickException("eval gate blocked")
        if result.skipped:
            click.echo(f"skipped inactive pipelines: {result.skipped}")
        if result.checked:
            click.echo(f"checked pipelines (matched baseline): {result.checked}")
        if result.missing_fixtures:
            click.echo(f"failed: active pipelines missing fixture set: {result.missing_fixtures}")
        if result.missing_evaluators:
            click.echo(
                "failed: active pipelines missing registered evaluator: "
                f"{result.missing_evaluators}"
            )
        if result.regressions:
            click.echo("failed: retrieval regression detected:")
            for line in result.regressions:
                click.echo(f"  - {line}")
        if result.failed:
            click.echo(f"eval gate FAIL — failed pipelines: {result.failed}")
            raise click.ClickException("eval gate failed")
        click.echo(f"eval gate PASS (active pipelines checked: {result.checked or 'none'})")
