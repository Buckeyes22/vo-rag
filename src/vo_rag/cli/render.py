"""Rendering helpers for the vo-rag CLI."""

from __future__ import annotations

import json
from typing import Any

import click
from rich.console import Console
from rich.table import Table

console = Console()


def emit_json(payload: Any) -> None:
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def emit_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def mask_secret(value: str | None) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 4:
        return "***"
    return "***" + value[-4:]
