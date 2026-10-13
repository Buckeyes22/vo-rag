"""Documentation coverage checker.

Each check prints ``PASS|FAIL <name>: <count> <details>`` and the process exits
1 if any check fails. ``--only a,b`` selects checks, ``--report`` lists every
gap under its check.

====== ===============================================================
check  what it proves
====== ===============================================================
modules       every package dir and every .py over 300 lines is named in a docs page
entrypoints   every CLI leaf / route / MCP tool / ``__main__`` module is documented,
              and every ``vo-rag <cmd>`` or route a doc mentions exists
env           every env var is in docs/configuration.md, no documented one is made up
config        every config leaf is in the generated table and the table is current
links         relative links and #anchors resolve
numbers       counts quoted in prose match docs/_measured.json
states        ``%% enum: X`` Mermaid state diagrams list exactly the enum's values
residue       no process/provenance wording; every F-0NN has a docs/security.md entry
capabilities  every row of docs/capabilities.md names a real page and real module paths
====== ===============================================================

The Mermaid *render* check runs in CI, not here.

    uv run --no-sync python scripts/check_docs.py [--only links,config] [--report]
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Trees whose packages and large files the ``modules`` check covers.
LARGE_FILE_LINES = 300
#: Pages that must carry the entry-point inventories.
ENTRYPOINT_PAGES = ("cli.md", "api.md", "mcp.md", "operations.md")

#: ``%% enum: <name>`` in a Mermaid block resolves through this fixed map.
STATE_ENUMS: dict[str, tuple[str, str]] = {
    "EnvelopeStatus": ("vo_rag.core.models.envelope", "EnvelopeStatus"),
    "IngestGenerationState": ("vo_rag.core.models.ingest_generation", "IngestGenerationState"),
    "PublicationState": ("vo_rag.core.ingest.source_publication", "PublicationState"),
    "GateDecision": ("vo_rag.pipelines.book_rag.retrieval_smoke", "GateDecision"),
}


def _generator() -> ModuleType:
    """The reference generator module (loaded lazily; it imports the app)."""
    return importlib.import_module("gen_reference_docs")


@dataclass
class Result:
    name: str
    gaps: list[str] = field(default_factory=list)
    details: str = ""

    @property
    def ok(self) -> bool:
        return not self.gaps

    @property
    def count(self) -> int:
        return len(self.gaps)


# --------------------------------------------------------------------------- #
# Shared text helpers
# --------------------------------------------------------------------------- #
GENERATED_BLOCK = re.compile(
    r"<!-- BEGIN GENERATED: ([\w-]+) -->\n.*?<!-- END GENERATED: \1 -->\n?", re.S
)
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(`+)(.+?)\1", re.S)


def strip_generated(text: str) -> str:
    return GENERATED_BLOCK.sub("", text)


def block_body(text: str, name: str) -> str | None:
    marker = re.escape(name)
    match = re.search(
        rf"<!-- BEGIN GENERATED: {marker} -->\n(.*?)\n?<!-- END GENERATED: {marker} -->", text, re.S
    )
    return match.group(1).strip() if match else None


def _blank(text: str) -> str:
    return re.sub(r"[^\n]", " ", text)


def split_code(text: str, *, inline: bool = True) -> tuple[str, list[str]]:
    """(text with code blanked out but line numbers kept, list of code snippets).

    ``inline=False`` blanks only fenced blocks (headings keep their code spans).
    """
    prose: list[str] = []
    snippets: list[str] = []
    fence: str | None = None
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if fence is None:
            opening = _FENCE_OPEN.match(line)
            if opening:
                fence = opening.group(1)
                current = []
                prose.append(_blank(line))
            else:
                prose.append(line)
        else:
            closing = line.strip()
            if closing.startswith(fence[0] * len(fence)) and set(closing) == {fence[0]}:
                snippets.append("".join(current))
                fence = None
            else:
                current.append(line)
            prose.append(_blank(line))
    if fence is not None:
        snippets.append("".join(current))
    joined = "".join(prose)

    def span(match: re.Match[str]) -> str:
        snippets.append(match.group(2))
        return _blank(match.group(0))

    return (_INLINE_CODE.sub(span, joined) if inline else joined), snippets


def doc_files(root: Path) -> list[Path]:
    """README.md plus every Markdown file under docs/."""
    files = [root / "README.md"] if (root / "README.md").is_file() else []
    docs = root / "docs"
    if docs.is_dir():
        files.extend(sorted(docs.rglob("*.md")))
    return files


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _mentions(text: str, names: Iterable[str], *, top_level: bool = False) -> bool:
    """True if any of ``names`` (a dotted or slash path) appears as a whole token.

    ``vo_rag.core`` is not found inside ``vo_rag.core.raptor`` and
    ``src/vo_rag/core`` is not found inside ``src/vo_rag/core/raptor/tree.py``;
    ``top_level`` stops ``services/x`` matching ``src/vo_rag/services/x``.
    """
    pre = r"(?<![\w])" + (r"(?<!vo_rag/)(?<!vo_rag\.)" if top_level else "")
    return any(re.search(pre + re.escape(n) + r"(?![\w])(?![./]\w)", text) for n in names)


# --------------------------------------------------------------------------- #
# Source tree helpers
# --------------------------------------------------------------------------- #
def _py_files(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


@dataclass(frozen=True)
class Unit:
    label: str  # shown in reports
    names: tuple[str, ...]  # how a doc may name it
    top_level: bool


def _unit_for(root: Path, path: Path, *, package: bool) -> Unit:
    """How a doc may name a package dir (``package``) or a single module file."""
    rel = _rel(root, path)
    top_level = not rel.startswith("src/")
    inner = rel if top_level else rel[len("src/") :]
    inner = inner.removesuffix(".py")
    parts = inner.split("/")
    if not package and parts[-1] in {"__init__", "__main__"}:
        parts = parts[:-1]
    names = [".".join(parts), "/".join(parts)]
    if package:
        label = "/".join(parts) if top_level else ".".join(parts)
    else:
        names.append(inner + ".py")
        label = rel
    return Unit(label, tuple(dict.fromkeys(names)), top_level)


def package_units(root: Path) -> list[Unit]:
    units = []
    src = root / "src" / "vo_rag"
    if src.is_dir():
        for directory in sorted(p for p in src.rglob("*") if p.is_dir()):
            if "__pycache__" in directory.parts:
                continue
            if any(directory.glob("*.py")):
                units.append(_unit_for(root, directory, package=True))
    for top in ("services", "workers"):
        base = root / top
        if base.is_dir():
            for directory in sorted(p for p in base.iterdir() if p.is_dir()):
                if directory.name.startswith(("_", ".")) or not _py_files(directory):
                    continue
                units.append(_unit_for(root, directory, package=True))
    return units


def large_file_units(root: Path) -> list[Unit]:
    units = []
    for base in (root / "src" / "vo_rag", root / "services", root / "workers"):
        for path in _py_files(base):
            if len(path.read_text(encoding="utf-8").splitlines()) > LARGE_FILE_LINES:
                units.append(_unit_for(root, path, package=False))
    return units


def main_module_units(root: Path) -> list[Unit]:
    """Python files under src/services/workers that have a ``__main__`` guard."""
    guard = re.compile(r"""__name__\s*==\s*["']__main__["']""")
    units = []
    for top in ("src", "services", "workers"):
        for path in _py_files(root / top):
            if guard.search(path.read_text(encoding="utf-8")):
                units.append(_unit_for(root, path, package=False))
    return units


# --------------------------------------------------------------------------- #
# 1. modules
# --------------------------------------------------------------------------- #
def check_modules(root: Path = ROOT) -> Result:
    result = Result("modules", details="packages/files over 300 lines not named in a docs page")
    pages = [
        p
        for p in sorted((root / "docs").glob("*.md"))
        if p.name != "README.md" and not p.name.startswith("_")
    ]
    text = "\n".join(strip_generated(_read(p)) for p in pages)
    for unit in [*package_units(root), *large_file_units(root)]:
        if not _mentions(text, unit.names, top_level=unit.top_level):
            result.gaps.append(unit.label)
    return result


# --------------------------------------------------------------------------- #
# 2. entrypoints
# --------------------------------------------------------------------------- #
def live_inventory(root: Path = ROOT) -> dict[str, Any]:
    gen = _generator()
    return {
        "cli": [leaf.path for leaf in gen.cli_leaves()],
        "routes": gen.api_route_paths(),
        "mcp": [tool.name for tool in gen.mcp_tools()],
        "main_modules": [(u.names[0], u.label) for u in main_module_units(root)],
    }


_URL_HOST = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/`'\")\]>]+")
_SLASH_TOKEN = re.compile(r"(?:/[\w{}.\-]+)+")
_METHOD_PATH = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+`?((?:/[\w{}.\-]+)+)")


def _norm_route(path: str) -> str:
    path = re.sub(r"\{[^}]*\}", "{}", path.rstrip(".,;:"))
    return path.rstrip("/") or "/"


def _route_tokens(text: str) -> set[str]:
    text = _URL_HOST.sub("", text)
    return {_norm_route(m.group(0)) for m in _SLASH_TOKEN.finditer(text)}


def _route_regex(route: str) -> re.Pattern[str]:
    return re.compile(re.escape(route).replace(re.escape("{}"), "[^/]+"))


def _cli_tree(leaves: Iterable[tuple[str, ...]]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for path in leaves:
        node = tree
        for part in path:
            node = node.setdefault(part, {})
    return tree


_CLI_MENTION = re.compile(
    r"(?<![\w/.-])vo-rag(?![\w.-])((?:[ \t]+[a-z][a-z0-9-]*(?![\w./=:\[\]<>\"'-]))*)"
)


def _stale_cli(snippets: Iterable[str], tree: dict[str, Any]) -> set[str]:
    stale = set()
    for snippet in snippets:
        for match in _CLI_MENTION.finditer(snippet):
            node, seen = tree, []
            for token in match.group(1).split():
                if not node:  # at a leaf: the rest are arguments
                    break
                seen.append(token)
                if token not in node:
                    stale.add("vo-rag " + " ".join(seen))
                    break
                node = node[token]
    return stale


def check_entrypoints(root: Path = ROOT, inventory: dict[str, Any] | None = None) -> Result:
    """Entry points must be documented, and documented ones must exist.

    Stale CLI detection scans only code spans and fenced blocks for ``vo-rag <cmd>``;
    a command written in plain prose is not checked. Stale routes are found as
    ``/api/...`` or ``METHOD /path`` tokens anywhere in the text.
    """
    result = Result(
        "entrypoints", details="entry points undocumented, or documented but not in the code"
    )
    inv = inventory if inventory is not None else live_inventory(root)
    docs = root / "docs"
    corpus = "\n".join(_read(docs / name) for name in ENTRYPOINT_PAGES)
    tokens = _route_tokens(corpus)

    for path in inv["cli"]:
        words = r"[ \t]+".join(re.escape(w) for w in ("vo-rag", *path))
        if not re.search(rf"(?<![\w-]){words}(?![\w-])", corpus):
            result.gaps.append("undocumented CLI command: vo-rag " + " ".join(path))
    for route in inv["routes"]:
        if _norm_route(route) not in tokens:
            result.gaps.append(f"undocumented route: {route}")
    for tool in inv["mcp"]:
        if not re.search(rf"(?<![\w]){re.escape(tool)}(?![\w])", corpus):
            result.gaps.append(f"undocumented MCP tool: {tool}")
    for dotted, rel in inv["main_modules"]:
        names = [dotted, rel, rel.removeprefix("src/")]
        if not _mentions(corpus, names, top_level=not rel.startswith("src/")):
            result.gaps.append(f"undocumented __main__ module: {dotted} ({rel})")

    tree = _cli_tree(inv["cli"])
    templates = [(_norm_route(r), _route_regex(_norm_route(r))) for r in inv["routes"]]
    for path in doc_files(root):
        rel = _rel(root, path)
        prose, snippets = split_code(_read(path))
        for command in sorted(_stale_cli(snippets, tree)):
            result.gaps.append(f"stale CLI command in {rel}: {command}")
        text = _URL_HOST.sub("", _read(path))
        candidates = {
            _norm_route(t) for t in _SLASH_TOKEN.findall(text) if re.match(r"/api(/|$)", t)
        }
        candidates.update(_norm_route(m.group(1)) for m in _METHOD_PATH.finditer(text))
        for token in sorted(candidates):
            exists = any(
                pattern.fullmatch(token) or route.startswith(token + "/")
                for route, pattern in templates
            )
            if not exists:
                result.gaps.append(f"stale route in {rel}: {token}")
    return result


# --------------------------------------------------------------------------- #
# 3. env
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[A-Z][A-Z0-9_]{2,}")
_ENV_DOC_TOKEN = re.compile(r"`([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)`")
_TOKEN_SUFFIXES = {".py", ".sh", ".ts", ".tsx", ".js", ".mjs", ".yaml", ".yml", ".toml", ".json"}
_SKIP_PARTS = {"node_modules", "dist", "__pycache__", ".venv", ".git"}


def known_tokens(root: Path) -> set[str]:
    """Upper-case identifiers appearing anywhere env vars can legitimately be named."""
    files: list[Path] = []
    for name in ("src", "services", "workers", "scripts", "config", "packages", "tools"):
        base = root / name
        if base.is_dir():
            files.extend(
                p
                for p in base.rglob("*")
                if p.is_file() and p.suffix in _TOKEN_SUFFIXES and not _SKIP_PARTS & set(p.parts)
            )
    for pattern in (".env.example", "docker-compose*.yml", "Makefile", "pyproject.toml"):
        files.extend(p for p in root.glob(pattern) if p.is_file())
    tokens: set[str] = set()
    for path in files:
        try:
            tokens.update(_TOKEN.findall(path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    return tokens


def check_env(root: Path = ROOT) -> Result:
    result = Result(
        "env", details="env vars undocumented in docs/configuration.md, or documented but unknown"
    )
    page = _read(root / "docs" / "configuration.md")
    for name in _generator().collect_env(root):
        if not re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", page):
            result.gaps.append(f"undocumented env var: {name}")
    known = known_tokens(root)
    for token in sorted(set(_ENV_DOC_TOKEN.findall(page))):
        if token not in known:
            result.gaps.append(f"stale env var in docs/configuration.md: {token}")
    return result


# --------------------------------------------------------------------------- #
# 4. config
# --------------------------------------------------------------------------- #
def check_config(
    root: Path = ROOT,
    leaves: list[str] | None = None,
    render: Callable[[], str] | None = None,
) -> Result:
    result = Result(
        "config", details="config keys missing from, or drifted in, the generated table"
    )
    if leaves is None:
        gen = _generator()
        leaves = [leaf.path for leaf in gen.config_leaves()]
        render = render or gen.render_config
    body = block_body(_read(root / "docs" / "configuration.md"), "config-keys")
    if body is None:
        result.gaps.append("docs/configuration.md has no config-keys generated block")
        return result
    result.gaps.extend(leaf for leaf in leaves if f"`{leaf}`" not in body)
    if render is not None and body != render().strip():
        result.gaps.append(
            "drift: config-keys block differs from the code; run scripts/gen_reference_docs.py"
        )
    return result


# --------------------------------------------------------------------------- #
# 5. links
# --------------------------------------------------------------------------- #
_INLINE_LINK = re.compile(r"\]\(\s*(<[^>\n]+>|[^\s)]+)")
_REF_DEF = re.compile(r"^ {0,3}\[[^\]\n]+\]:[ \t]*(<[^>\n]+>|\S+)", re.M)
_HTML_REF = re.compile(r"""\b(?:href|src)\s*=\s*["']([^"']+)["']""")
_HTML_ID = re.compile(r"""\b(?:id|name)\s*=\s*["']([^"']+)["']""")
_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$", re.M)
_SCHEME = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//)")


def slugify(heading: str) -> str:
    """GitHub heading anchor: lower-case, drop punctuation, spaces to hyphens."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"<[^>]+>", "", text).replace("`", "").strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def heading_anchors(text: str) -> set[str]:
    prose, _ = split_code(text, inline=False)
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    for match in _HEADING.finditer(prose):
        slug = slugify(match.group(1))
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        anchors.add(slug if n == 0 else f"{slug}-{n}")
    anchors.update(m.group(1).lower() for m in _HTML_ID.finditer(prose))
    return anchors


def link_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if (root / "README.md").is_file():
        files.append(root / "README.md")
    if (root / "packages" / "README.md").is_file():
        files.append(root / "packages" / "README.md")
    for top in ("docs", "services", "workers", "tools"):
        base = root / top
        if base.is_dir():
            files.extend(p for p in sorted(base.rglob("*.md")) if "node_modules" not in p.parts)
    return files


def check_links(root: Path = ROOT) -> Result:
    result = Result("links", details="broken relative links or anchors")
    anchor_cache: dict[Path, set[str]] = {}

    def anchors_of(path: Path) -> set[str]:
        if path not in anchor_cache:
            anchor_cache[path] = heading_anchors(path.read_text(encoding="utf-8"))
        return anchor_cache[path]

    for path in link_files(root):
        rel = _rel(root, path)
        prose, _ = split_code(path.read_text(encoding="utf-8"))
        found = [
            (m.start(1), m.group(1))
            for rx in (_INLINE_LINK, _REF_DEF, _HTML_REF)
            for m in rx.finditer(prose)
        ]
        for offset, raw in sorted(found):
            target = raw.strip("<>")
            if not target or _SCHEME.match(target):
                continue
            line = prose.count("\n", 0, offset) + 1
            file_part, _, fragment = target.partition("#")
            file_part = unquote(file_part.split("?", 1)[0])
            if file_part:
                dest = (
                    (root / file_part.lstrip("/"))
                    if file_part.startswith("/")
                    else path.parent / file_part
                )
                if not dest.exists():
                    result.gaps.append(f"{rel}:{line}: {target} (file not found)")
                    continue
            else:
                dest = path
            checkable = fragment and dest.is_file() and dest.suffix == ".md"
            if checkable and unquote(fragment).lower() not in anchors_of(dest):
                result.gaps.append(f"{rel}:{line}: {target} (anchor not found)")
    return result


# --------------------------------------------------------------------------- #
# 6. numbers
# --------------------------------------------------------------------------- #
_NUMBER = re.compile(
    r"(?P<hedge>\b(?:about|around|roughly|approximately|nearly)\s+|~\s*)?"
    r"\b(?P<num>[0-9][0-9,]*)\s+(?:[A-Za-z][\w-]*\s+)?"
    r"(?P<unit>test files|tests|routes|commands|tools|collections|tables)\b",
    re.I,
)
_PORTAL = re.compile(r"\b(?:portal|vitest|frontend)\b", re.I)
_MEASURE_KEYS = {
    "routes": "routes",
    "commands": "cli_leaves",
    "tools": "mcp_tools",
    "collections": "qdrant_collections",
    "tables": "postgres_tables",
}
TOLERANCE = 0.05
_SENTENCE_BREAKS = (". ", "; ", "? ", "! ")
_CLAUSE_END = re.compile(r"[,;)]|\.(?:\s|$)")


def _about_the_portal(line: str, match: re.Match[str]) -> bool:
    """True if the portal is named in the match's sentence before it, or its clause after it."""
    starts = [line.rfind(sep, 0, match.start()) for sep in _SENTENCE_BREAKS]
    begin = max((i + 2 for i in starts if i >= 0), default=0)
    after = _CLAUSE_END.split(line[match.end() :], maxsplit=1)[0]
    return bool(_PORTAL.search(line[begin : match.end()] + after))


def check_numbers(root: Path = ROOT, measured: dict[str, Any] | None = None) -> Result:
    """Compare counts quoted in prose with ``docs/_measured.json``.

    The check trusts that file: rerun ``scripts/gen_reference_docs.py`` after adding
    tests, routes, commands, tools, collections or tables.
    """
    result = Result("numbers", details="prose counts that disagree with docs/_measured.json")
    measured_path = root / "docs" / "_measured.json"
    pending: list[tuple[str, int, re.Match[str], bool]] = []
    for path in doc_files(root):
        prose, _ = split_code(strip_generated(path.read_text(encoding="utf-8")))
        for lineno, line in enumerate(prose.splitlines(), start=1):
            for match in _NUMBER.finditer(line):
                pending.append((_rel(root, path), lineno, match, _about_the_portal(line, match)))
    if not pending:
        return result
    if measured is None:
        if not measured_path.is_file():
            result.gaps.append("docs/_measured.json missing; run scripts/gen_reference_docs.py")
            return result
        import json

        measured = json.loads(measured_path.read_text(encoding="utf-8"))
    for rel, lineno, match, portal in pending:
        unit = match.group("unit").lower()
        if unit in {"tests", "test files"}:
            stem = "portal" if portal else "pytest"
            key = f"{stem}_{'tests' if unit == 'tests' else 'test_files'}"
        elif portal:
            continue  # the portal's own routes/tools are not measured
        else:
            key = _MEASURE_KEYS[unit]
        if key not in measured:
            result.gaps.append(f"{rel}:{lineno}: no measurement for {key} (re-run the generator)")
            continue
        expected = int(measured[key])
        stated = int(match.group("num").replace(",", ""))
        within = (
            abs(stated - expected) <= TOLERANCE * expected
            if match.group("hedge")
            else stated == expected
        )
        if not within:
            result.gaps.append(
                f'{rel}:{lineno}: "{match.group("num")} {unit}" but measured {expected} ({key})'
            )
    return result


# --------------------------------------------------------------------------- #
# 7. states
# --------------------------------------------------------------------------- #
_MERMAID = re.compile(r"^```mermaid[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)
_ENUM_TAG = re.compile(r"^%%\s*enum:\s*(\w+)\s*$")
_STATE_ID = r"[\w.-]+"
_TRANSITION = re.compile(rf"^\s*(\[\*\]|{_STATE_ID})\s*-->\s*(\[\*\]|{_STATE_ID})")
_STATE_AS = re.compile(rf'^\s*state\s+"[^"]*"\s+as\s+({_STATE_ID})')
_STATE_DECL = re.compile(rf"^\s*state\s+({_STATE_ID})\b")
_STATE_DESC = re.compile(rf"^\s*({_STATE_ID})\s*:")
_NOT_STATES = {
    "note",
    "direction",
    "end",
    "classDef",
    "class",
    "style",
    "stateDiagram",
    "stateDiagram-v2",
}


def load_state_enums() -> dict[str, set[str]]:
    enums = {}
    for name, (module, attr) in STATE_ENUMS.items():
        enums[name] = {member.value for member in getattr(importlib.import_module(module), attr)}
    return enums


def diagram_states(body: str) -> set[str]:
    states: set[str] = set()
    for line in body.splitlines():
        if line.strip().startswith("%%"):
            continue
        for rx in (_TRANSITION,):
            m = rx.match(line)
            if m:
                states.update(g for g in m.groups() if g != "[*]")
                break
        else:
            m = _STATE_AS.match(line) or _STATE_DECL.match(line) or _STATE_DESC.match(line)
            if m and m.group(1) not in _NOT_STATES:
                states.add(m.group(1))
    return states - _NOT_STATES


def check_states(root: Path = ROOT, enums: dict[str, set[str]] | None = None) -> Result:
    result = Result("states", details="state diagrams whose states differ from their enum")
    for path in doc_files(root):
        rel = _rel(root, path)
        for block in _MERMAID.finditer(path.read_text(encoding="utf-8")):
            body = block.group(1)
            first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
            tag = _ENUM_TAG.match(first)
            if not tag:
                continue
            if enums is None:
                enums = load_state_enums()
            name = tag.group(1)
            if name not in enums:
                result.gaps.append(
                    f"{rel}: unknown enum {name} (known: {', '.join(sorted(enums))})"
                )
                continue
            drawn = diagram_states(body)
            missing = sorted(enums[name] - drawn)
            extra = sorted(drawn - enums[name])
            if missing:
                result.gaps.append(
                    f"{rel}: {name} states missing from diagram: {', '.join(missing)}"
                )
            if extra:
                result.gaps.append(f"{rel}: {name} extra states not in enum: {', '.join(extra)}")
    return result


# --------------------------------------------------------------------------- #
# 8. residue
# --------------------------------------------------------------------------- #
RESIDUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "Lifted from": re.compile(r"Lifted from", re.I),
    "P2a-c": re.compile(r"\bP2[a-c]\b"),
    "Task N": re.compile(r"\bTask [0-9]+\b"),
    "spec §": re.compile(r"spec §", re.I),
    "sanitiz": re.compile(r"sanitiz", re.I),
    # Whole word only: corpus_snapshot, snapshot_download and env names stay legal.
    "snapshot": re.compile(r"\bsnapshots?\b", re.I),
}
_FINDING_ID = re.compile(r"\bF-0\d\d\b")


def check_residue(root: Path = ROOT) -> Result:
    result = Result(
        "residue", details="process wording hits and F-0NN ids missing from docs/security.md"
    )
    for path in doc_files(root):
        rel = _rel(root, path)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for label, pattern in RESIDUE_PATTERNS.items():
                if pattern.search(line):
                    result.gaps.append(f"{rel}:{lineno}: {label}: {line.strip()[:80]}")
    ids: dict[str, str] = {}
    for path in _py_files(root / "src"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for found in _FINDING_ID.findall(line):
                ids.setdefault(found, f"{_rel(root, path)}:{lineno}")
    security = _read(root / "docs" / "security.md")
    documented = set(_FINDING_ID.findall(security))
    for finding, where in sorted(ids.items()):
        if finding not in documented:
            result.gaps.append(f"{finding} (used at {where}) has no entry in docs/security.md")
    return result


# --------------------------------------------------------------------------- #
# 9. capabilities
# --------------------------------------------------------------------------- #
_ROW_SEPARATOR = re.compile(r"^\|[\s:|-]+\|\s*$")
_PAGE_REF = re.compile(r"(?<![\w/])(?:(?:\.\./)?docs/|\./)?([\w-]+\.md)\b")
_PATH_REF = re.compile(
    r"(?<![\w])((?:src|services|workers|scripts|tools|packages)/[\w./{}-]*[\w}])"
)
_DOTTED_REF = re.compile(r"(?<![\w/.])(vo_rag(?:\.\w+)+)")


def _dotted_exists(root: Path, dotted: str) -> bool:
    """``vo_rag.a.b.Name``: walk packages and modules; trailing names need a module file."""
    node = root / "src"
    parts = dotted.split(".")
    for part in parts:
        if (node / part).is_dir():
            node = node / part
        else:
            # a module file: any remaining parts are names inside it
            return (node / f"{part}.py").is_file()
    return True


def check_capabilities(root: Path = ROOT) -> Result:
    result = Result("capabilities", details="capability rows naming a missing page or module path")
    page = root / "docs" / "capabilities.md"
    if not page.is_file():
        result.gaps.append("docs/capabilities.md missing")
        return result
    lines = page.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("|") or _ROW_SEPARATOR.match(line):
            continue
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if _ROW_SEPARATOR.match(following):
            continue  # header row
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        name = cells[0] if cells else line
        pages = _PAGE_REF.findall(line)
        if not pages:
            result.gaps.append(f"row '{name}' names no docs page")
        for ref in pages:
            if not (root / "docs" / ref).is_file():
                result.gaps.append(f"row '{name}': docs page {ref} does not exist")
        for ref in _PATH_REF.findall(line):
            if not (root / ref).exists():
                result.gaps.append(f"row '{name}': path {ref} does not exist")
        for ref in _DOTTED_REF.findall(line):
            if not _dotted_exists(root, ref):
                result.gaps.append(f"row '{name}': module {ref} does not exist")
    return result


# --------------------------------------------------------------------------- #
# Registry and command line
# --------------------------------------------------------------------------- #
CHECKS: dict[str, Callable[..., Result]] = {
    "modules": check_modules,
    "entrypoints": check_entrypoints,
    "env": check_env,
    "config": check_config,
    "links": check_links,
    "numbers": check_numbers,
    "states": check_states,
    "residue": check_residue,
    "capabilities": check_capabilities,
}


def _summary(result: Result, *, first: bool = True) -> str:
    status = "PASS" if result.ok else "FAIL"
    line = f"{status} {result.name}: {result.count} {result.details}"
    if first and not result.ok:
        line += f" (first: {result.gaps[0]})"
    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Documentation coverage checks.")
    parser.add_argument("--only", help=f"comma list of checks: {', '.join(CHECKS)}")
    parser.add_argument("--report", action="store_true", help="list every gap")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    names = [n.strip() for n in args.only.split(",") if n.strip()] if args.only else list(CHECKS)
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        parser.error(f"unknown check(s): {', '.join(unknown)} (choose from {', '.join(CHECKS)})")
    results = []
    for name in names:
        result = CHECKS[name](args.root)
        results.append(result)
        print(_summary(result, first=not args.report))
        if args.report:
            for gap in result.gaps:
                print(f"    {gap}")
    if args.report:
        print("--- summary ---")
        for result in results:
            print(_summary(result, first=False))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
