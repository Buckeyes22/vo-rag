"""Contracts for ``scripts/gen_reference_docs.py``.

The inventory tests run the generator in-process against the real repository
(writing only into a temp copy of the four reference pages); the helper tests
use tiny synthetic inputs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import gen_reference_docs as gen

REPO = gen.ROOT
PAGES = ("cli.md", "api.md", "mcp.md", "configuration.md")


@pytest.fixture(scope="module")
def generated_docs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    docs = tmp_path_factory.mktemp("docs")
    for name in PAGES:
        shutil.copy(REPO / "docs" / name, docs / name)
    gen.generate(REPO, docs_dir=docs, measure_numbers=False)
    return docs


def block(docs: Path, page: str, name: str) -> str:
    text = (docs / page).read_text(encoding="utf-8")
    start = text.index(f"<!-- BEGIN GENERATED: {name} -->")
    end = text.index(f"<!-- END GENERATED: {name} -->")
    return text[start:end]


def test_cli_block_lists_preflight_with_its_options(generated_docs: Path) -> None:
    cli = block(generated_docs, "cli.md", "cli-commands")
    row = next(line for line in cli.splitlines() if line.startswith("| `vo-rag preflight`"))
    assert "`--" in row


def test_cli_block_has_one_row_per_leaf(generated_docs: Path) -> None:
    cli = block(generated_docs, "cli.md", "cli-commands")
    assert cli.count("\n| `vo-rag ") == len(gen.cli_leaves())


def test_api_block_marks_health_exempt_and_query_authenticated(generated_docs: Path) -> None:
    api = block(generated_docs, "api.md", "api-routes")
    health = next(line for line in api.splitlines() if "| `/health` |" in line)
    assert health.startswith("| GET |")
    assert "exempt" in health
    query = next(line for line in api.splitlines() if "| `/api/v1/query` |" in line)
    assert query.startswith("| POST |")
    assert "API key" in query
    assert "`QueryRequest`" in query
    assert "`QueryResponse`" in query


def test_api_block_does_not_list_head_as_a_method(generated_docs: Path) -> None:
    api = block(generated_docs, "api.md", "api-routes")
    assert "HEAD" not in api


def test_mcp_block_lists_tools_with_input_fields(generated_docs: Path) -> None:
    mcp = block(generated_docs, "mcp.md", "mcp-tools")
    row = next(line for line in mcp.splitlines() if line.startswith("| `vo_rag_query`"))
    assert "`query` (required)" in row
    assert "`top_k`" in row
    for name in ("vo_rag_ingest", "vo_rag_ingest_stream", "vo_rag_status"):
        assert f"| `{name}` |" in mcp


def test_env_block_lists_variable_default_and_first_read(generated_docs: Path) -> None:
    env = block(generated_docs, "configuration.md", "env-vars")
    row = next(line for line in env.splitlines() if line.startswith("| `VORAG_API_HOST`"))
    assert "127.0.0.1" in row
    assert row.endswith("| `config/settings.yaml` |")


def test_config_block_lists_leaf_keys_with_defaults(generated_docs: Path) -> None:
    config = block(generated_docs, "configuration.md", "config-keys")
    row = next(line for line in config.splitlines() if line.startswith("| `api.host`"))
    assert "`'127.0.0.1'`" in row
    assert "| `api.api_key` |" in config
    assert "required" in next(
        line for line in config.splitlines() if line.startswith("| `api.api_key`")
    )


def test_generated_tables_carry_no_line_numbers(generated_docs: Path) -> None:
    env = block(generated_docs, "configuration.md", "env-vars")
    assert not re.search(r"\.(?:py|yaml):\d+", env)


def test_second_run_is_a_no_op_and_check_mode_agrees(generated_docs: Path) -> None:
    before = {name: (generated_docs / name).read_text(encoding="utf-8") for name in PAGES}
    assert gen.generate(REPO, docs_dir=generated_docs, measure_numbers=False) == []
    assert gen.generate(REPO, docs_dir=generated_docs, check=True) == []
    after = {name: (generated_docs / name).read_text(encoding="utf-8") for name in PAGES}
    assert before == after


def test_check_mode_reports_drift_without_writing(generated_docs: Path, tmp_path: Path) -> None:
    for name in PAGES:
        shutil.copy(generated_docs / name, tmp_path / name)
    page = tmp_path / "cli.md"
    page.write_text(
        page.read_text(encoding="utf-8").replace("vo-rag preflight", "vo-rag preflite"),
        encoding="utf-8",
    )
    stale = page.read_text(encoding="utf-8")
    assert gen.generate(REPO, docs_dir=tmp_path, only=["cli"], check=True) == ["cli"]
    assert page.read_text(encoding="utf-8") == stale


def test_missing_page_fails_loudly(generated_docs: Path) -> None:
    with pytest.raises(FileNotFoundError):
        gen.generate(REPO, docs_dir=generated_docs / "missing-dir", only=["cli"], check=True)


def test_unknown_job_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        gen.main(["--only", "nonsense"])
    assert exc.value.code == 2


def test_replace_block_is_idempotent_and_requires_markers() -> None:
    text = "a\n<!-- BEGIN GENERATED: x -->\nold\n<!-- END GENERATED: x -->\nb\n"
    once = gen.replace_block(text, "x", "new")
    assert once == "a\n<!-- BEGIN GENERATED: x -->\nnew\n<!-- END GENERATED: x -->\nb\n"
    assert gen.replace_block(once, "x", "new") == once
    with pytest.raises(ValueError):
        gen.replace_block("no markers", "x", "new")


def test_env_reads_cover_each_access_pattern() -> None:
    source = """
import os
from os import environ, getenv

PREFIX_NAME = "VORAG_VIA_CONST"
a = os.environ.get("VORAG_GET", "dflt")
b = os.environ["VORAG_INDEX"]
c = os.getenv("VORAG_GETENV")
d = "VORAG_IN" in os.environ
e = environ.get("VORAG_BARE")
f = os.environ.get(PREFIX_NAME)
g = _env_float("VORAG_HELPER", 2.5)
h = env.get("VORAG_MAPPING", "")
i = getenv("VORAG_BARE_GETENV", default="x")
j = os.environ.get(some_variable)
k = os.environ.get("lowercase_name")
"""
    reads = {r.name: r for r in gen.env_reads_in_source(source, "m.py")}
    assert set(reads) == {
        "VORAG_GET",
        "VORAG_INDEX",
        "VORAG_GETENV",
        "VORAG_IN",
        "VORAG_BARE",
        "VORAG_VIA_CONST",
        "VORAG_HELPER",
        "VORAG_MAPPING",
        "VORAG_BARE_GETENV",
    }
    assert reads["VORAG_GET"].default == "'dflt'"
    assert reads["VORAG_INDEX"].default == "required"
    assert reads["VORAG_GETENV"].default == "unset"
    assert reads["VORAG_HELPER"].default == "2.5"
    assert reads["VORAG_BARE_GETENV"].default == "'x'"


def test_env_reads_pick_up_click_and_pydantic_aliases() -> None:
    source = (
        "import click\n"
        "@click.option('--x', envvar='VORAG_CLICK_VAR')\n"
        "def f(): ...\n"
        "class S:\n"
        "    a: str = Field('', validation_alias='VORAG_ALIAS')\n"
        "    b: str = Field(\n"
        "        '', validation_alias=AliasChoices('VORAG_CHOICE_A', 'VORAG_CHOICE_B'))\n"
    )
    names = {r.name for r in gen.env_reads_in_source(source, "m.py")}
    assert names == {"VORAG_CLICK_VAR", "VORAG_ALIAS", "VORAG_CHOICE_A", "VORAG_CHOICE_B"}


def test_collect_env_merges_yaml_and_code(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config/settings.yaml").write_text(
        'a:\n  b: "${VORAG_X:fallback}"\n', encoding="utf-8"
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src/m.py").write_text(
        'import os\nos.environ.get("VORAG_X")\nos.environ.get("VORAG_Y")\n'
    )
    merged = gen.collect_env(tmp_path)
    assert list(merged) == ["VORAG_X", "VORAG_Y"]
    assert merged["VORAG_X"].default == "'fallback'"
    assert merged["VORAG_X"].file == "config/settings.yaml"
    assert merged["VORAG_Y"].file == "src/m.py"


def test_portal_static_count_covers_tests_and_src_and_each(tmp_path: Path) -> None:
    portal = tmp_path / "packages/portal"
    (portal / "tests").mkdir(parents=True)
    (portal / "src/api").mkdir(parents=True)
    (portal / "tests/a.test.tsx").write_text(
        "it('x', () => {});\ntest('y', () => {});\ndescribe('d', () => {});\n"
    )
    (portal / "src/api/b.test.ts").write_text(
        "it('z', () => {});\nit.each([[1, 'a,b'], [2, 'c'],\n  [3, 'd']])('row %s', () => {});\n"
    )
    (portal / "tests/setup.ts").write_text("it('not a test file', () => {});\n")
    # 2 call sites in a; in b 1 call site plus the 3 rows of the it.each array
    assert gen.portal_tests_static(tmp_path) == (6, 2)
    assert gen.portal_tests(tmp_path) == (6, 2, "static regex")  # no node_modules here


def test_portal_vitest_list_count_matches_the_real_suite() -> None:
    if not (REPO / "packages/portal/node_modules").is_dir():
        pytest.skip("portal dependencies not installed")
    count, files, method = gen.portal_tests(REPO)
    assert method == "vitest list"
    assert count >= files > 0


def test_pytest_command_is_read_from_the_makefile(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text(
        'test:\n\tuv run --extra dev pytest -q -m "not slow and not live"\n\nlint:\n\truff\n'
    )
    assert gen.pytest_command(tmp_path) == [
        "uv",
        "run",
        "--extra",
        "dev",
        "pytest",
        "--collect-only",
        "-p",
        "no:cacheprovider",
        "-q",
        "-m",
        "not slow and not live",
    ]
    real = gen.pytest_command(REPO)
    assert "llamaindex" in real and "--collect-only" in real


def test_postgres_tables_ignores_comments_and_dropped_tables(tmp_path: Path) -> None:
    migrations = tmp_path / gen.MIGRATIONS_DIR
    migrations.mkdir(parents=True)
    (migrations / "001.sql").write_text(
        "-- CREATE TABLE commented_out (id int);\n"
        "CREATE TABLE IF NOT EXISTS r2_log (id int);\n"
        "CREATE TABLE vce.stage_run (id int);\n"
        "CREATE TABLE temp_gone (id int);\n"
    )
    (migrations / "002.sql").write_text("DROP TABLE IF EXISTS temp_gone;\n")
    assert gen.postgres_tables(tmp_path) == ["r2_log", "vce.stage_run"]


def test_real_measurements_have_every_key_the_numbers_check_reads() -> None:
    measured = gen.measure(REPO, count_pytest=False)
    assert set(measured) >= {
        "cli_leaves",
        "mcp_tools",
        "portal_count_method",
        "portal_test_files",
        "portal_tests",
        "postgres_tables",
        "pytest_test_files",
        "qdrant_collections",
        "routes",
    }
    assert measured["cli_leaves"] == len(gen.cli_leaves())
    assert measured["routes"] == len(gen.build_app().routes)
    assert measured["qdrant_collections"] == 19
    assert measured["mcp_tools"] == 4


def test_pytest_summary_line_formats_are_parsed() -> None:
    with_deselected = gen._COLLECTED.search("4941/5066 tests collected (125 deselected) in 3.2s")
    plain = gen._COLLECTED.search("5066 tests collected in 3.2s")
    assert with_deselected is not None and plain is not None
    assert with_deselected.groups() == ("4941", "5066")
    assert plain.groups() == (None, "5066")


def test_ticket_references_are_scrubbed_from_generated_text() -> None:
    assert gen._clean("Gate report for a canary (VOR-1434 task 3).") == "Gate report for a canary."
    assert gen._clean("Rekey tags (pre-VOR-1450 keys) now") == "Rekey tags now"
    assert gen._clean("a | b") == "a \\| b"


def _fake_proc(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_pytest_selected_uses_the_makefile_command_when_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Makefile").write_text('test:\n\tuv run pytest -q -m "not slow"\n')
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return _fake_proc("12/40 tests collected (28 deselected)\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    count, command, method = gen.pytest_selected(tmp_path)
    assert (count, method) == (12, "makefile")
    assert command.startswith("uv run pytest --collect-only")
    assert [c[0] for c in calls] == ["uv"]


def test_pytest_selected_falls_back_to_the_interpreter_when_uv_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Makefile").write_text('test:\n\tuv run pytest -q -m "not slow"\n')
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[0] == "uv":
            raise FileNotFoundError("uv")
        return _fake_proc("7 tests collected in 0.1s\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    count, command, method = gen.pytest_selected(tmp_path)
    assert (count, method) == (7, "python -m pytest")
    assert calls[-1][:3] == [sys.executable, "-m", "pytest"]
    assert f"-m '{gen.PYTEST_SELECTION}'" in command


def test_measure_records_the_pytest_count_method(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen, "pytest_selected", lambda _root: (5, "cmd", "python -m pytest"))
    measured = gen.measure(REPO)
    assert measured["pytest_count_method"] == "python -m pytest"
    assert (measured["pytest_tests"], measured["pytest_command"]) == (5, "cmd")


def _portal_with_a_test(tmp_path: Path) -> None:
    (tmp_path / "packages/portal/node_modules").mkdir(parents=True)
    (tmp_path / "packages/portal/tests").mkdir()
    (tmp_path / "packages/portal/tests/a.test.ts").write_text("it('x', () => {});\n")


def test_portal_vitest_list_has_a_timeout_and_falls_back_when_it_expires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _portal_with_a_test(tmp_path)
    seen: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(command, gen.PORTAL_LIST_TIMEOUT)

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/pnpm")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert gen.portal_tests(tmp_path) == (1, 1, "static regex")
    assert seen["timeout"] == gen.PORTAL_LIST_TIMEOUT == 300


def test_portal_vitest_list_output_is_used_when_it_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _portal_with_a_test(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/pnpm")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: _fake_proc(
            "tests/a.test.ts > x\ntests/a.test.ts > y\ntests/b.test.ts > z\n"
        ),
    )
    assert gen.portal_tests(tmp_path) == (3, 2, "vitest list")
