"""Admin and system CLI commands."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import click

from vo_rag.cli.client import VoRagApiClient, _client_from_ctx, _client_from_ctx_allow_unauth
from vo_rag.cli.render import emit_json

TERMINAL_STATES = {"completed", "failed", "discarded"}


def register_admin_commands(root: click.Group) -> None:
    @root.group("admin")
    def admin() -> None:
        """Administration commands."""

    @admin.command("status")
    @click.pass_context
    def admin_status(ctx: click.Context) -> None:
        """Show admin status."""
        client = _client_from_ctx(ctx)
        response = asyncio.run(client.get_json("/api/v1/admin/status"))
        emit_json(response)

    @root.command("health")
    @click.pass_context
    def health(ctx: click.Context) -> None:
        """Check API health endpoint (no auth required)."""
        client = _client_from_ctx_allow_unauth(ctx)
        response = asyncio.run(client.get_json("/health"))
        emit_json(response)
        ok = bool(response.get("ok", False)) or response.get("status") == "ok"
        if not ok:
            raise click.ClickException("health checks reported failures")

    @root.command("status")
    @click.option("--watch", is_flag=True, default=False)
    @click.pass_context
    def status(ctx: click.Context, watch: bool) -> None:
        """Show system status and optionally watch."""
        client = _client_from_ctx(ctx)
        if not watch:
            response = asyncio.run(client.get_json("/api/v1/admin/status"))
            emit_json(response)
            return
        asyncio.run(_watch_status(client))


async def _watch_status(client: VoRagApiClient) -> None:
    previous: str | None = None
    while True:
        try:
            payload = await client.get_json("/api/v1/admin/status")
        except Exception as exc:
            current = f"error: {exc}"
        else:
            current = "ok" if payload.get("ok") else "degraded"
        if current != previous:
            timestamp = datetime.now(UTC).isoformat()
            click.echo(f"{timestamp}  {current}")
            previous = current
        await asyncio.sleep(2)
