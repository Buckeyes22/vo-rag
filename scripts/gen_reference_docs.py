"""Generate the reference tables that live between GENERATED markers in ``docs/``.

Each table is rebuilt from the code, never typed by hand:

====== =============================================== ==========================
name   source                                          page / block
====== =============================================== ==========================
cli    Click tree of ``vo_rag.cli.main:cli``           cli.md / cli-commands
api    routes of ``vo_rag.api.main.create_app()``      api.md / api-routes
mcp    ``build_mcp_server(inspection_only=True)``      mcp.md / mcp-tools
env    environment-variable reads (AST) + settings.yaml configuration.md / env-vars
config ``Settings.model_fields``, recursively          configuration.md / config-keys
====== =============================================== ==========================

A fifth job, ``numbers``, writes ``docs/_measured.json`` (hermetic pytest
selection, portal test count, routes, CLI leaves, MCP tools, Qdrant
collections, Postgres tables) which ``check_docs.py numbers`` compares with the
numbers quoted in prose.

Usage::

    uv run --no-sync python scripts/gen_reference_docs.py            # rewrite blocks
    uv run --no-sync python scripts/gen_reference_docs.py --check    # exit 1 on drift
    uv run --no-sync python scripts/gen_reference_docs.py --only cli,env

The run is idempotent: a second run changes nothing.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import typing
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]

#: Trees scanned for environment-variable reads.
ENV_SCAN_DIRS = ("src", "services", "workers", "scripts")
SETTINGS_YAML = "config/settings.yaml"
MIGRATIONS_DIR = "src/vo_rag/core/db/migrations"
PYTEST_SELECTION = "not integration and not slow and not live"
#: Seconds allowed for ``vitest list`` before falling back to the static count.
PORTAL_LIST_TIMEOUT = 300

#: ``name`` -> (page under docs/, block name). ``numbers`` has no block.
BLOCKS: dict[str, tuple[str, str]] = {
    "cli": ("cli.md", "cli-commands"),
    "api": ("api.md", "api-routes"),
    "mcp": ("mcp.md", "mcp-tools"),
    "env": ("configuration.md", "env-vars"),
    "config": ("configuration.md", "config-keys"),
}

#: Receivers that hold a copy of the environment (``source = os.environ if ...``).
_ENV_MAPPING_NAMES = frozenset({"env", "environ", "environment", "source"})
_YAML_INTERP = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
_TICKET_PAREN = re.compile(r"\s*\([^()]*\bVOR-\d+\b[^()]*\)")
_TICKET_BARE = re.compile(r"\s*\bVOR-\d+\b[,:]?")


def _clean(text: str) -> str:
    """One-line, table-safe text with ticket references removed."""
    text = _TICKET_PAREN.sub("", text)
    text = _TICKET_BARE.sub("", text)
    text = " ".join(text.split())
    return text.replace("|", "\\|")


def _cell(text: object) -> str:
    return _clean(str(text)) or "-"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _type_name(annotation: object) -> str:
    if annotation is None or annotation is type(None):
        return "None"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "").replace("collections.abc.", "")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CliLeaf:
    path: tuple[str, ...]
    command: Any


def cli_leaves() -> list[CliLeaf]:
    """Every leaf command of the ``vo-rag`` Click tree, in tree order."""
    import click

    from vo_rag.cli.main import cli

    def walk(cmd: Any, path: tuple[str, ...]) -> Iterator[CliLeaf]:
        if isinstance(cmd, click.Group):
            ctx = click.Context(cmd, info_name=path[-1] if path else "vo-rag")
            for name in cmd.list_commands(ctx):
                child = cmd.get_command(ctx, name)
                if child is not None:
                    yield from walk(child, (*path, name))
        else:
            yield CliLeaf(path, cmd)

    return list(walk(cli, ()))


def _option_cell(param: Any) -> str:
    import click

    if getattr(param, "hidden", False):
        return ""
    if isinstance(param, click.Argument):
        name = f"`{param.human_readable_name.upper()}`"
    else:
        name = "`" + "/".join([*param.opts, *param.secondary_opts]) + "`"
    ptype = param.type
    if isinstance(ptype, click.Choice):
        type_text = "choice: " + "|".join(str(c) for c in ptype.choices)
    elif getattr(param, "is_flag", False):
        type_text = "flag"
    else:
        type_text = ptype.name
    bits = [type_text]
    default = param.default
    unset = default is None or type(default).__name__ == "Sentinel" or default == ()
    if param.required:
        bits.append("required")
    elif not unset and not (getattr(param, "is_flag", False) and default is False):
        bits.append(f"default {default!r}" if isinstance(default, str) else f"default {default}")
    if getattr(param, "multiple", False):
        bits.append("repeatable")
    envvar = getattr(param, "envvar", None)
    if envvar:
        bits.append("env " + ", ".join(envvar if isinstance(envvar, list) else [envvar]))
    return f"{name} ({', '.join(bits)})"


def cli_rows() -> list[list[str]]:
    rows = []
    for leaf in cli_leaves():
        options = [c for c in (_option_cell(p) for p in leaf.command.params) if c]
        rows.append(
            [
                "`" + " ".join(("vo-rag", *leaf.path)) + "`",
                _cell(leaf.command.get_short_help_str(limit=300)),
                "<br>".join(_clean(o) for o in options) or "-",
            ]
        )
    return rows


def render_cli() -> str:
    return _table(["Command", "Summary", "Options (type, default)"], cli_rows())


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def build_app() -> Any:
    """Construct the ASGI app without entering its lifespan (no sockets open).

    Routes are registered at construction time, so ``create_app()`` with no
    services is enough to enumerate them; the lifespan, which is what opens
    Postgres/Qdrant/Garage connections, only runs under an ASGI server.
    """
    from vo_rag.api.main import create_app

    saved = os.environ.pop("VORAG_OTEL_ENABLED", None)
    try:
        return create_app()
    finally:
        if saved is not None:
            os.environ["VORAG_OTEL_ENABLED"] = saved


def api_rows() -> list[list[str]]:
    from fastapi.routing import APIRoute

    from vo_rag.api.middleware import EXEMPT_PATHS

    app = build_app()
    rows = []
    for route in app.routes:
        methods = sorted(m for m in (getattr(route, "methods", None) or ()) if m != "HEAD")
        request: list[str] = []
        response = "-"
        if isinstance(route, APIRoute):
            request = [_type_name(p.field_info.annotation) for p in route.dependant.body_params]
            if route.response_model is not None:
                response = _type_name(route.response_model)
        rows.append(
            [
                ", ".join(methods) or "-",
                f"`{route.path}`",
                f"`{route.name}`",
                "exempt" if route.path in EXEMPT_PATHS else "API key",
                ", ".join(f"`{r}`" for r in request) or "-",
                f"`{response}`" if response != "-" else "-",
            ]
        )
    rows.sort(key=lambda r: (r[1], r[0]))
    return rows


def api_route_paths() -> list[str]:
    return [row[1].strip("`") for row in api_rows()]


def render_api() -> str:
    header = ["Method", "Path", "Endpoint", "Auth", "Request model", "Response model"]
    return _table(header, api_rows())


# --------------------------------------------------------------------------- #
# MCP
# --------------------------------------------------------------------------- #
def mcp_tools() -> list[Any]:
    from vo_rag.mcp.server import build_mcp_server

    return list(asyncio.run(build_mcp_server(inspection_only=True).list_tools()))


def mcp_rows() -> list[list[str]]:
    rows = []
    for tool in mcp_tools():
        props = list(tool.inputSchema.get("properties", {}))
        required = set(tool.inputSchema.get("required") or [])
        fields = [f"`{name}`" + (" (required)" if name in required else "") for name in props]
        rows.append([f"`{tool.name}`", _cell(tool.description or ""), ", ".join(fields) or "-"])
    return rows


def render_mcp() -> str:
    return _table(["Tool", "Description", "Input fields"], mcp_rows())


# --------------------------------------------------------------------------- #
# Environment variables (audit Appendix B method)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EnvRead:
    name: str
    default: str
    file: str
    line: int


def _is_environ(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "environ"
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            out[node.targets[0].id] = node.value.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            out[node.target.id] = node.value.value
    return out


def _fmt_default(node: ast.AST | None, *, required: bool = False) -> str:
    if required:
        return "required"
    if node is None:
        return "unset"
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return "(computed)"


def env_reads_in_source(source: str, rel: str) -> list[EnvRead]:
    """Environment reads in one Python file, found through the AST.

    Recognised: ``os.environ.get/pop/setdefault``, ``os.getenv``,
    ``os.environ[...]``, ``"X" in os.environ`` (also with ``from os import
    environ, getenv``), ``env.get(...)`` on a mapping named like the
    environment, helper calls whose name contains ``env`` (``_env_float("X", 1.0)``),
    pydantic ``env=``/``validation_alias=``/``AliasChoices`` and Click
    ``envvar=``. A name may be a string literal or a module-level string
    constant of the same file.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    consts = _module_string_constants(tree)

    def name_of(node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return consts.get(node.id)
        return None

    reads: list[EnvRead] = []

    def add(node: ast.expr, name: str | None, default: str) -> None:
        if name and _ENV_NAME.match(name):
            reads.append(EnvRead(name, default, rel, node.lineno))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_getenv = (
                isinstance(func, ast.Attribute)
                and func.attr == "getenv"
                and isinstance(func.value, ast.Name)
                and func.value.id == "os"
            ) or (isinstance(func, ast.Name) and func.id == "getenv")
            is_environ_call = (
                isinstance(func, ast.Attribute)
                and func.attr in {"get", "pop", "setdefault"}
                and _is_environ(func.value)
            )
            func_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            is_env_helper = "env" in func_name.lower() and not is_environ_call and not is_getenv
            is_mapping_get = (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Name)
                and func.value.id in _ENV_MAPPING_NAMES
            )
            if (is_getenv or is_environ_call or is_env_helper or is_mapping_get) and node.args:
                default_node = node.args[1] if len(node.args) > 1 else None
                for kw in node.keywords:
                    if kw.arg == "default":
                        default_node = kw.value
                add(node, name_of(node.args[0]), _fmt_default(default_node))
            for kw in node.keywords:
                if kw.arg in {"env", "envvar", "validation_alias"}:
                    values = kw.value.elts if isinstance(kw.value, ast.List) else [kw.value]
                    for value in values:
                        add(node, name_of(value), "unset")
            if isinstance(func, ast.Name) and func.id == "AliasChoices":
                for arg in node.args:
                    add(node, name_of(arg), "unset")
        elif isinstance(node, ast.Subscript) and _is_environ(node.value):
            add(node, name_of(node.slice), "required")
        elif isinstance(node, ast.Compare) and any(_is_environ(c) for c in node.comparators):
            add(node, name_of(node.left), "unset")
    return reads


def env_reads_in_yaml(text: str, rel: str) -> list[EnvRead]:
    reads = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in _YAML_INTERP.finditer(line):
            default = match.group(2)
            reads.append(
                EnvRead(match.group(1), "unset" if default is None else repr(default), rel, lineno)
            )
    return reads


def _python_files(root: Path, dirs: tuple[str, ...]) -> Iterator[Path]:
    for name in dirs:
        base = root / name
        if base.is_dir():
            for path in sorted(base.rglob("*.py")):
                if "__pycache__" not in path.parts and ".venv" not in path.parts:
                    yield path


def collect_env(root: Path = ROOT) -> dict[str, EnvRead]:
    """Every env var read in code or interpolated by ``config/settings.yaml``.

    The returned read for a name is the first one by (path, line); its default
    is the first concrete default found in that same ordering.
    """
    by_name: dict[str, list[EnvRead]] = {}
    sources: list[EnvRead] = []
    yaml_path = root / SETTINGS_YAML
    if yaml_path.is_file():
        sources.extend(env_reads_in_yaml(yaml_path.read_text(encoding="utf-8"), SETTINGS_YAML))
    for path in _python_files(root, ENV_SCAN_DIRS):
        rel = path.relative_to(root).as_posix()
        sources.extend(env_reads_in_source(path.read_text(encoding="utf-8"), rel))
    for read in sources:
        by_name.setdefault(read.name, []).append(read)
    merged = {}
    for name in sorted(by_name):
        reads = sorted(by_name[name], key=lambda r: (r.file, r.line))
        concrete = next((r.default for r in reads if r.default != "unset"), "unset")
        merged[name] = EnvRead(name, concrete, reads[0].file, reads[0].line)
    return merged


_PLAIN_DEFAULTS = {"unset", "required", "(computed)"}


def _default_cell(default: str) -> str:
    return default if default in _PLAIN_DEFAULTS else f"`{_cell(default)}`"


def render_env(root: Path = ROOT) -> str:
    rows = [
        [f"`{r.name}`", _default_cell(r.default), f"`{r.file}`"] for r in collect_env(root).values()
    ]
    return _table(["Variable", "Default", "First read"], rows)


# --------------------------------------------------------------------------- #
# Config keys
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConfigLeaf:
    path: str
    type: str
    default: str
    description: str


def _field_comments(model: type[BaseModel]) -> dict[str, str]:
    """Description per field from the source: comment above, inline comment, or docstring."""
    module = inspect.getmodule(model)
    if module is None:
        return {}
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return {}
    lines = source.splitlines()
    tree = ast.parse(source)
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == model.__name__):
            continue
        body = node.body
        for index, stmt in enumerate(body):
            if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                continue
            parts: list[str] = []
            row = stmt.lineno - 2
            while row >= 0 and lines[row].strip().startswith("#"):
                text = lines[row].strip().lstrip("#").strip()
                if text and not set(text) <= {"-", "=", " "}:
                    parts.insert(0, text)
                row -= 1
            code, _, trailing = lines[(stmt.end_lineno or stmt.lineno) - 1].partition("#")
            if trailing.strip() and code.count('"') % 2 == 0 and code.count("'") % 2 == 0:
                parts.append(trailing.strip())
            nxt = body[index + 1] if index + 1 < len(body) else None
            if (
                isinstance(nxt, ast.Expr)
                and isinstance(nxt.value, ast.Constant)
                and isinstance(nxt.value.value, str)
            ):
                parts.append(inspect.cleandoc(nxt.value.value))
            if parts:
                out[stmt.target.id] = " ".join(parts)
    return out


def _unwrap_model(annotation: object) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in typing.get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def config_leaves(model: type[BaseModel] | None = None, prefix: str = "") -> list[ConfigLeaf]:
    """Leaf settings (non-model fields) of ``Settings`` as dotted key paths."""
    from pydantic_core import PydanticUndefined

    if model is None:
        from vo_rag.core.config.settings import Settings

        model = Settings
    comments = _field_comments(model)
    leaves: list[ConfigLeaf] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        sub = _unwrap_model(field.annotation)
        if sub is not None:
            leaves.extend(config_leaves(sub, path + "."))
            continue
        if field.default is not PydanticUndefined:
            default = repr(field.default)
        elif field.default_factory is not None:
            try:
                value = field.get_default(call_default_factory=True, validated_data={})
            except (TypeError, ValueError):
                value = "(computed)"
            default = "(computed)" if isinstance(value, BaseModel) else repr(value)
            if value == "(computed)":
                default = "(computed)"
        else:
            default = "required"
        leaves.append(
            ConfigLeaf(
                path,
                _type_name(field.annotation),
                default,
                field.description or comments.get(name, ""),
            )
        )
    return leaves


def render_config() -> str:
    rows = [
        [
            f"`{leaf.path}`",
            f"`{_cell(leaf.type)}`",
            _default_cell(leaf.default),
            _cell(leaf.description),
        ]
        for leaf in config_leaves()
    ]
    return _table(["Key", "Type", "Default", "Description"], rows)


# --------------------------------------------------------------------------- #
# Measured numbers
# --------------------------------------------------------------------------- #
#: Static count: ``it(`` / ``test(`` call sites in the portal's test files. It
#: does not run vitest, so skipped or generated cases are counted as written.
PORTAL_PACKAGE = "@vo-rag/portal"
PORTAL_TEST_CALL = re.compile(r"\b(?:it|test)\(")
PORTAL_EACH_CALL = re.compile(r"\b(?:it|test)\.each\(\s*\[")


def _each_rows(text: str, start: int) -> int:
    """Top-level elements of the array literal whose ``[`` ends just before ``start``."""
    depth, rows, has_item, quote = 1, 0, False, ""
    for ch in text[start:]:
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'`":
            quote, has_item = ch, True
        elif ch in "[({":
            depth += 1
            has_item = True
        elif ch in "])}":
            depth -= 1
            if depth == 0:
                return rows + (1 if has_item else 0)
        elif ch == "," and depth == 1:
            rows += 1 if has_item else 0
            has_item = False
        elif not ch.isspace():
            has_item = True
    return rows


def portal_tests_static(root: Path) -> tuple[int, int]:
    """(tests, files) by regex over the files vitest's ``include`` globs select.

    Counts ``it(``/``test(`` call sites plus the rows of ``it.each([...])``
    arrays; used only when ``vitest list`` cannot run.
    """
    base = root / "packages/portal"
    files = sorted(
        p
        for folder in ("tests", "src")
        for pattern in ("*.test.ts", "*.test.tsx")
        for p in (base / folder).rglob(pattern)
        if "node_modules" not in p.parts
    )
    count = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        count += len(PORTAL_TEST_CALL.findall(text))
        count += sum(_each_rows(text, m.end()) for m in PORTAL_EACH_CALL.finditer(text))
    return count, len(files)


def portal_tests(root: Path) -> tuple[int, int, str]:
    """(tests, files, method) for the portal's vitest suite.

    Prefers ``pnpm --filter @vo-rag/portal exec vitest list`` (each output line is
    ``<file> > <test name>``), which honours the real include globs and
    ``.each`` expansion. Falls back to a static regex count when pnpm or the
    portal's ``node_modules`` is unavailable, or when ``vitest list`` fails or
    runs past ``PORTAL_LIST_TIMEOUT`` seconds.
    """
    if (root / "packages/portal/node_modules").is_dir() and shutil.which("pnpm"):
        try:
            proc = subprocess.run(
                ["pnpm", "--filter", PORTAL_PACKAGE, "exec", "vitest", "list"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=PORTAL_LIST_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            proc = None
        if proc is not None:
            cases = [ln for ln in proc.stdout.splitlines() if " > " in ln]
            if proc.returncode == 0 and cases:
                return len(cases), len({ln.split(" > ", 1)[0] for ln in cases}), "vitest list"
    count, files = portal_tests_static(root)
    return count, files, "static regex"


def pytest_test_files(root: Path) -> int:
    base = root / "tests"
    return sum(1 for p in base.rglob("test_*.py") if "__pycache__" not in p.parts)


_SQL_COMMENT = re.compile(r"--[^\n]*")
_CREATE_TABLE = re.compile(r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.\"]+)", re.I)
_DROP_TABLE = re.compile(r"\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?([\w.\"]+)", re.I)


def postgres_tables(root: Path) -> list[str]:
    """Tables the migrations create (and do not later drop), schema-qualified."""
    created: set[str] = set()
    for path in sorted((root / MIGRATIONS_DIR).glob("*.sql")):
        sql = _SQL_COMMENT.sub("", path.read_text(encoding="utf-8"))
        created.update(m.group(1).strip('"').lower() for m in _CREATE_TABLE.finditer(sql))
        created.difference_update(m.group(1).strip('"').lower() for m in _DROP_TABLE.finditer(sql))
    return sorted(created)


def qdrant_collections() -> list[str]:
    from vo_rag.core.storage.qdrant import COLLECTION_NAMES

    return list(COLLECTION_NAMES)


_COLLECTED = re.compile(r"(?:(\d+)/)?(\d+) tests? collected")


def pytest_command(root: Path) -> list[str]:
    """The Makefile's ``test:`` pytest invocation, switched to ``--collect-only``.

    Reading it from the Makefile keeps the count equal to what ``make test`` runs
    (same extras, same marker selection).
    """
    makefile = root / "Makefile"
    if makefile.is_file():
        recipe = re.search(r"^test:[^\n]*\n((?:\t[^\n]*\n?)+)", makefile.read_text(), re.M)
        for line in recipe.group(1).splitlines() if recipe else []:
            words = shlex.split(line.strip())
            if "pytest" in words:
                at = words.index("pytest") + 1
                return [*words[:at], "--collect-only", "-p", "no:cacheprovider", *words[at:]]
    return pytest_fallback_command()


def pytest_fallback_command() -> list[str]:
    """Collect-only run through the current interpreter, with the Makefile's markers."""
    return [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "-m",
        PYTEST_SELECTION,
    ]


def pytest_selected(root: Path) -> tuple[int, str, str]:
    """(selected test count, command used, method) from a collect-only run.

    The Makefile's command is preferred (method ``makefile``). When its launcher
    (normally ``uv``) is not installed, the same marker expression runs through
    ``sys.executable -m pytest`` (method ``python -m pytest``).
    """
    command = pytest_command(root)
    fallback = pytest_fallback_command()
    try:
        proc = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        command = fallback
        proc = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    method = "python -m pytest" if command == fallback else "makefile"
    match = _COLLECTED.search(proc.stdout)
    if not match:
        raise RuntimeError("could not read the pytest collection summary:\n" + proc.stdout[-500:])
    return int(match.group(1) or match.group(2)), shlex.join(command), method


def measure(root: Path = ROOT, *, count_pytest: bool = True) -> dict[str, Any]:
    portal_count, portal_files, portal_method = portal_tests(root)
    measured: dict[str, Any] = {
        "cli_leaves": len(cli_leaves()),
        "mcp_tools": len(mcp_tools()),
        "portal_count_method": portal_method,
        "portal_test_files": portal_files,
        "portal_tests": portal_count,
        "postgres_tables": len(postgres_tables(root)),
        "pytest_selection": PYTEST_SELECTION,
        "pytest_test_files": pytest_test_files(root),
        "qdrant_collections": len(qdrant_collections()),
        "routes": len(api_rows()),
    }
    if count_pytest:
        count, command, method = pytest_selected(root)
        measured.update(pytest_tests=count, pytest_command=command, pytest_count_method=method)
    return measured


# --------------------------------------------------------------------------- #
# Block rewriting
# --------------------------------------------------------------------------- #
def _markers(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN GENERATED: {name} -->", f"<!-- END GENERATED: {name} -->"


def replace_block(text: str, name: str, body: str) -> str:
    """Return ``text`` with the body between the ``name`` markers replaced."""
    begin, end = _markers(name)
    pattern = re.compile(re.escape(begin) + r"\n(?:.*?\n)?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise ValueError(f"markers for generated block {name!r} not found")
    return pattern.sub(lambda _m: f"{begin}\n{body}\n{end}", text, count=1)


RENDERERS: dict[str, Callable[[], str]] = {
    "cli": render_cli,
    "api": render_api,
    "mcp": render_mcp,
    "env": render_env,
    "config": render_config,
}


def generate(
    root: Path = ROOT,
    *,
    docs_dir: Path | None = None,
    only: list[str] | None = None,
    check: bool = False,
    measure_numbers: bool = True,
) -> list[str]:
    """Rewrite (or, with ``check``, only compare) the generated blocks.

    Returns the names of jobs whose output differs from what is on disk.
    ``root`` is the code tree; ``docs_dir`` defaults to ``root/docs``.
    """
    docs = docs_dir or root / "docs"
    wanted = only or [*BLOCKS, "numbers"]
    changed: list[str] = []
    pages: dict[str, str] = {}
    for job in wanted:
        if job == "numbers" or job not in BLOCKS:
            continue
        page, block = BLOCKS[job]
        path = docs / page
        text = pages.get(page) or path.read_text(encoding="utf-8")
        body = render_env(root) if job == "env" else RENDERERS[job]()
        updated = replace_block(text, block, body)
        if updated != text:
            changed.append(job)
        pages[page] = updated
    if not check:
        for page, text in pages.items():
            path = docs / page
            if path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")
    if "numbers" in wanted and not check and measure_numbers:
        out = docs / "_measured.json"
        new = json.dumps(measure(root), indent=2, sort_keys=True) + "\n"
        if not out.exists() or out.read_text(encoding="utf-8") != new:
            changed.append("numbers")
            out.write_text(new, encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a block would change")
    parser.add_argument("--only", help=f"comma list of: {', '.join([*BLOCKS, 'numbers'])}")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    unknown = set(only or ()) - {*BLOCKS, "numbers"}
    if unknown:
        parser.error(f"unknown job(s): {', '.join(sorted(unknown))}")
    changed = generate(args.root, only=only, check=args.check)
    if args.check:
        for job in changed:
            print(f"DRIFT {job}: regenerate with scripts/gen_reference_docs.py")
        return 1 if changed else 0
    print("updated: " + (", ".join(changed) if changed else "nothing (already current)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
