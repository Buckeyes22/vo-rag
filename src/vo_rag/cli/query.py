"""Query and answer CLI commands."""

from __future__ import annotations

import asyncio
from typing import Any

import click

from vo_rag.cli.client import _client_from_ctx
from vo_rag.cli.render import emit_json, emit_table

_QUERY_MODES = ["internal", "publication", "partner"]
_ANSWER_PROFILES = ["quality", "fast", "kimi_benchmark", "local"]
_CONTEXT_SCOPES = ["sterile", "raw"]


def register_query_commands(root: click.Group) -> None:
    @root.command("query")
    @click.argument("query_text", type=str)
    @click.option("--collection", "collections", default=None, help="Comma-separated collections.")
    @click.option("--top-k", default=20, type=click.IntRange(1, 200))
    @click.option("--rerank/--no-rerank", default=True)
    @click.option(
        "--mode",
        type=click.Choice(_QUERY_MODES),
        default="internal",
    )
    @click.option(
        "--visual-mode",
        type=click.Choice(["intent_aware", "baseline", "visual_only", "visual_track"]),
        default=None,
        help=(
            "Override the server-side visual-mode quota strategy. "
            "Server default is intent_aware (auto-detect visual intent in the query)."
        ),
    )
    @click.option(
        "--format", "output_format", type=click.Choice(["json", "table"]), default="table"
    )
    @click.pass_context
    def query_cmd(
        ctx: click.Context,
        query_text: str,
        collections: str | None,
        top_k: int,
        rerank: bool,
        mode: str,
        visual_mode: str | None,
        output_format: str,
    ) -> None:
        """Query indexed knowledge artifacts."""
        client = _client_from_ctx(ctx)
        payload: dict[str, Any] = {
            "query": query_text,
            "collections": _explicit_collections(collections),
            "top_k": top_k,
            "rerank": rerank,
            "mode": mode,
        }
        if visual_mode is not None:
            payload["visual_mode"] = visual_mode
        response = asyncio.run(client.post_json("/api/v1/query", payload))
        if output_format == "json":
            emit_json(response)
            return

        rows: list[list[str]] = []
        for item in response.get("results", []):
            payload_data = item.get("payload", {})
            preview = _preview_text(payload_data)
            rows.append(
                [
                    str(item.get("id", "")),
                    f"{float(item.get('score', 0.0)):.4f}",
                    str(item.get("collection", "")),
                    preview,
                ]
            )
        emit_table("Query results", ["id", "score", "collection", "preview"], rows)

    @root.command("answer")
    @click.argument("query_text", type=str)
    @click.option(
        "--profile",
        type=click.Choice(_ANSWER_PROFILES),
        default="quality",
    )
    @click.option(
        "--context-scope",
        type=click.Choice(_CONTEXT_SCOPES),
        default="sterile",
    )
    @click.option("--include-retrieval/--no-include-retrieval", default=True)
    @click.option(
        "--mode",
        type=click.Choice(_QUERY_MODES),
        default="internal",
    )
    @click.option("--collection", "collections", default=None, help="Comma-separated collections.")
    @click.option("--top-k", default=8, type=click.IntRange(1, 50))
    @click.option("--rerank/--no-rerank", default=True)
    @click.option(
        "--visual-mode",
        type=click.Choice(["intent_aware", "baseline", "visual_only", "visual_track"]),
        default=None,
    )
    @click.option(
        "--format", "output_format", type=click.Choice(["json", "table"]), default="table"
    )
    @click.pass_context
    def answer(
        ctx: click.Context,
        query_text: str,
        profile: str,
        context_scope: str,
        include_retrieval: bool,
        mode: str,
        collections: str | None,
        top_k: int,
        rerank: bool,
        visual_mode: str | None,
        output_format: str,
    ) -> None:
        """Generate an answer for the query."""
        client = _client_from_ctx(ctx)
        payload: dict[str, Any] = {
            "query": query_text,
            "profile": profile,
            "context_scope": context_scope,
            "include_retrieval": include_retrieval,
            "mode": mode,
            "collections": _explicit_collections(collections),
            "top_k": top_k,
            "rerank": rerank,
        }
        if visual_mode is not None:
            payload["visual_mode"] = visual_mode
        response = asyncio.run(client.post_json("/api/v1/answer", payload))
        if output_format == "json":
            emit_json(response)
            return

        answer_text = response.get("answer") or "(no answer)"
        click.echo(answer_text)
        warnings = response.get("warnings", [])
        if warnings:
            click.echo("Warnings:")
            for warning in warnings:
                click.echo(f"  - {warning}")
        citations = response.get("citations", [])
        if citations:
            click.echo("Citations:")
            for citation in citations:
                click.echo(f"  - {citation}")


def _preview_text(payload: dict[str, Any]) -> str:
    for key in ("text", "content", "title", "source"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            if payload.get("stage") == "brain" and (
                payload.get("governing_context_version") != 1
                or payload.get("governing_context_complete") is not True
            ):
                return (
                    "governing evidence unavailable or unattested; "
                    "inspect metadata with --format json"
                )
            compact = " ".join(value.split())
            if (
                payload.get("stage") == "brain"
                and payload.get("governing_context_version") == 1
                and len(compact) > 100
            ):
                return "incomplete preview; full scoped evidence: --format json: " + compact[:100]
            return compact[:100]
    return ""


def _explicit_collections(value: str | None) -> list[str] | None:
    if value is None:
        return None
    selected = _split_csv(value)
    if not selected:
        raise click.BadParameter("requires nonempty collection names", param_hint="--collection")
    return selected


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
