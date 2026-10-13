"""`vo-rag` CLI. Thin surface over core; derives the roster from the SOT."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import click
import yaml
from pydantic import BaseModel

from vo_rag import __version__
from vo_rag.api.utils import mask_secrets
from vo_rag.cli.admin import register_admin_commands
from vo_rag.cli.client import VoRagApiClient
from vo_rag.cli.corpus import register_corpus_commands
from vo_rag.cli.eval import register_eval_commands
from vo_rag.cli.ingest import register_ingest_commands
from vo_rag.cli.preflight import register_preflight_commands
from vo_rag.cli.query import register_query_commands
from vo_rag.cli.render import emit_json, emit_table, mask_secret
from vo_rag.cli.vce import register_vce_commands
from vo_rag.core.config.pipelines import PipelineStatus, load_pipeline_registry
from vo_rag.ops.corpus_repair import register as register_corpus_repair_commands


class CliConfig(BaseModel):
    api_url: str = "http://localhost:8000"
    api_key: str | None = None


def _load_file_config(path: Path) -> CliConfig:
    if not path.exists():
        return CliConfig()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise click.ClickException(f"invalid config format in {path}")
    return CliConfig.model_validate(raw)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--url", default=None, help="VO-RAG API URL override.")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logs.")
@click.option("--config", type=click.Path(path_type=Path), default=None, help="Config file path.")
@click.pass_context
def cli(ctx: click.Context, url: str | None, verbose: bool, config: Path | None) -> None:
    """vo-rag — multi-domain RAG system."""

    config_path = (
        config or Path(os.getenv("VORAG_CLI_CONFIG", "~/.config/vo-rag/config.yaml")).expanduser()
    )
    file_config = _load_file_config(config_path)

    resolved_api_url = url or os.getenv("VORAG_API_URL") or file_config.api_url
    api_key = os.getenv("VORAG_API_KEY") or file_config.api_key

    client = VoRagApiClient(base_url=resolved_api_url, api_key=api_key, verbose=verbose)
    ctx.obj = {
        "api_url": resolved_api_url,
        "api_key": api_key,
        "config_path": config_path,
        "client": client,
    }
    ctx.call_on_close(lambda: asyncio.run(client.aclose()))


@cli.command()
def version() -> None:
    """Print the vo-rag version."""
    click.echo(__version__)


@cli.group("pipeline")
def pipeline() -> None:
    """Pipeline roster commands."""


def _emit_pipeline_roster() -> None:
    rows: list[list[str]] = []
    for spec in load_pipeline_registry().values():
        rows.append([spec.name, spec.status.value, spec.summary])
    emit_table("Pipelines", ["name", "status", "summary"], rows)


@pipeline.command("list")
def pipeline_list() -> None:
    """List the pipeline roster (from the single source of truth)."""
    _emit_pipeline_roster()


@cli.command("roster")
def roster() -> None:
    """List the pipeline roster (legacy alias for pipeline list)."""
    _emit_pipeline_roster()


@pipeline.command("status")
@click.option("--json", "output_json", is_flag=True, default=False)
def pipeline_status(output_json: bool) -> None:
    """Show pipeline roster summary counts."""
    registry = load_pipeline_registry()
    planned = sum(1 for s in registry.values() if s.status is PipelineStatus.PLANNED)
    nxt = sum(1 for s in registry.values() if s.status is PipelineStatus.NEXT)
    active = sum(1 for s in registry.values() if s.status is PipelineStatus.ACTIVE)

    payload: dict[str, Any] = {
        "planned": planned,
        "next": nxt,
        "active": active,
    }
    if output_json:
        emit_json(payload)
        return
    emit_table(
        "Pipeline status",
        ["status", "count"],
        [
            ["planned", str(planned)],
            ["next", str(nxt)],
            ["active", str(active)],
        ],
    )


@cli.group("config")
def config_group() -> None:
    """CLI configuration commands."""


@config_group.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Show resolved CLI configuration with masked key."""
    api_url = str(ctx.find_root().obj.get("api_url", ""))
    key = str(ctx.find_root().obj.get("api_key") or "")
    config_path = Path(
        str(ctx.find_root().obj.get("config_path", "~/.config/vo-rag/config.yaml"))
    ).expanduser()
    file_config = _read_config_file(config_path)

    payload = {
        "config_path": str(config_path),
        "api_url": api_url,
        "api_key": mask_secret(key),
        "file": mask_secrets(file_config),
    }
    emit_json(payload)


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    return raw


class _LazyBuildGroup(click.Group):
    """Import the heavy build command module only when `vo-rag build` runs."""

    _loaded_group: click.Group | None = None

    def _load_group(self) -> click.Group:
        if self._loaded_group is None:
            root = click.Group()
            from vo_rag.cli.build import register_build_commands

            register_build_commands(root)
            loaded = root.commands["build"]
            if not isinstance(loaded, click.Group):  # pragma: no cover - defensive
                raise click.ClickException("build command group failed to register")
            self._loaded_group = loaded
        return self._loaded_group

    def list_commands(self, ctx: click.Context) -> list[str]:
        return ["dsi", "graph", "raptor", "summary"]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        return self._load_group().get_command(ctx, cmd_name)


@cli.group("build", cls=_LazyBuildGroup)
def build_group() -> None:
    """Direct builder commands for DSI, graph, summary, and RAPTOR corpora."""


register_query_commands(cli)
register_ingest_commands(cli)
register_preflight_commands(cli)
register_corpus_commands(cli)
register_eval_commands(cli)
register_admin_commands(cli)
register_vce_commands(cli)
register_corpus_repair_commands(cli)


def main() -> int:
    cli.main(prog_name="vo-rag", standalone_mode=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
