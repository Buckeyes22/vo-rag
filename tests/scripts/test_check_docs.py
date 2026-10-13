"""Unit contracts for ``scripts/check_docs.py``.

Every check is exercised on a tiny temp-dir tree (a fake docs tree beside a
fake code tree) so the tests never depend on the real repository's docs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_docs

BEGIN = "<!-- BEGIN GENERATED: {} -->"
END = "<!-- END GENERATED: {} -->"


def make_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def generated(name: str, body: str) -> str:
    return f"{BEGIN.format(name)}\n{body}\n{END.format(name)}\n"


def big_module(lines: int) -> str:
    return "".join(f"x{i} = {i}\n" for i in range(lines))


# --------------------------------------------------------------------------- #
# modules
# --------------------------------------------------------------------------- #
ALL_NAMES = "vo_rag.core vo_rag.core.raptor services/exporter workers/embedding"
CODE_TREE = {
    "src/vo_rag/__init__.py": "",
    "src/vo_rag/core/__init__.py": "",
    "src/vo_rag/core/raptor/__init__.py": "",
    "src/vo_rag/core/raptor/tree.py": "x = 1\n",
    "services/exporter/app.py": "x = 1\n",
    "workers/embedding/server.py": "x = 1\n",
}


def test_modules_pass_when_every_package_is_named(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            **CODE_TREE,
            "docs/arch.md": (
                "The `vo_rag.core` package holds `vo_rag.core.raptor`.\n"
                "See services/exporter and workers/embedding.\n"
            ),
        },
    )
    result = check_docs.check_modules(tmp_path)
    assert result.ok, result.gaps
    assert result.count == 0


def test_modules_fail_lists_unnamed_packages(tmp_path: Path) -> None:
    make_tree(tmp_path, {**CODE_TREE, "docs/arch.md": "Only `vo_rag.core` is named.\n"})
    result = check_docs.check_modules(tmp_path)
    assert not result.ok
    joined = "\n".join(result.gaps)
    assert "vo_rag.core.raptor" in joined
    assert "services/exporter" in joined
    assert "workers/embedding" in joined


def test_modules_child_mention_does_not_cover_parent(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            **CODE_TREE,
            "docs/arch.md": "Only `vo_rag.core.raptor` services/exporter workers/embedding\n",
        },
    )
    result = check_docs.check_modules(tmp_path)
    assert [g for g in result.gaps if "vo_rag.core" in g and "raptor" not in g]


def test_modules_accepts_relative_path_with_prefix_and_link(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            **CODE_TREE,
            "docs/arch.md": (
                "[core](../src/vo_rag/core/) [raptor](../src/vo_rag/core/raptor/)\n"
                "[exp](../services/exporter/) [emb](../workers/embedding/)\n"
            ),
        },
    )
    assert check_docs.check_modules(tmp_path).ok


def test_modules_ignores_readme_index_and_generated_blocks(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            **CODE_TREE,
            "docs/README.md": ALL_NAMES + "\n",
            "docs/cli.md": generated("x", ALL_NAMES),
        },
    )
    result = check_docs.check_modules(tmp_path)
    assert not result.ok
    assert result.count == 4


def test_modules_large_file_threshold(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            **CODE_TREE,
            "src/vo_rag/core/raptor/big.py": big_module(301),
            "src/vo_rag/core/raptor/edge.py": big_module(300),
            "docs/arch.md": (
                "vo_rag.core vo_rag.core.raptor services/exporter workers/embedding\n"
            ),
        },
    )
    result = check_docs.check_modules(tmp_path)
    assert not result.ok
    assert result.gaps == ["src/vo_rag/core/raptor/big.py"]
    make_tree(tmp_path, {"docs/more.md": "see `vo_rag.core.raptor.big`\n"})
    assert check_docs.check_modules(tmp_path).ok


def test_modules_directory_with_only_bytecode_is_not_a_package(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "src/vo_rag/__init__.py": "",
            "src/vo_rag/gone/__pycache__/a.cpython-312.pyc": "",
            "docs/arch.md": "nothing\n",
        },
    )
    assert check_docs.check_modules(tmp_path).ok


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #
INVENTORY = {
    "cli": [("corpus", "health"), ("preflight",)],
    "routes": ["/health", "/api/v1/query", "/api/v1/books/{source_id}"],
    "mcp": ["vo_rag_query"],
    "main_modules": [("vo_rag.ops.brain_sync", "src/vo_rag/ops/brain_sync.py")],
}

FULL_DOCS = {
    "docs/cli.md": "`vo-rag corpus health`\n\n```bash\nvo-rag preflight ./books\n```\n",
    "docs/api.md": "GET /health\nPOST /api/v1/query\nGET /api/v1/books/{source_id}\n",
    "docs/mcp.md": "Tool `vo_rag_query`.\n",
    "docs/operations.md": "Run `python -m vo_rag.ops.brain_sync`.\n",
}


def test_entrypoints_pass_when_everything_documented(tmp_path: Path) -> None:
    make_tree(tmp_path, FULL_DOCS)
    result = check_docs.check_entrypoints(tmp_path, inventory=INVENTORY)
    assert result.ok, result.gaps


def test_entrypoints_fail_lists_each_missing_kind(tmp_path: Path) -> None:
    make_tree(tmp_path, dict.fromkeys(FULL_DOCS, "empty\n"))
    result = check_docs.check_entrypoints(tmp_path, inventory=INVENTORY)
    assert not result.ok
    joined = "\n".join(result.gaps)
    for expected in (
        "vo-rag corpus health",
        "vo-rag preflight",
        "/api/v1/query",
        "vo_rag_query",
        "vo_rag.ops.brain_sync",
    ):
        assert expected in joined
    assert result.count == 7


def test_entrypoints_path_form_covers_main_module(tmp_path: Path) -> None:
    docs = {**FULL_DOCS, "docs/operations.md": "File `src/vo_rag/ops/brain_sync.py`.\n"}
    make_tree(tmp_path, docs)
    assert check_docs.check_entrypoints(tmp_path, inventory=INVENTORY).ok


def test_entrypoints_stale_cli_command_detected(tmp_path: Path) -> None:
    docs = {**FULL_DOCS, "docs/setup.md": "Run `vo-rag corpus vanished` first.\n"}
    make_tree(tmp_path, docs)
    result = check_docs.check_entrypoints(tmp_path, inventory=INVENTORY)
    assert not result.ok
    assert any("stale" in g and "vo-rag corpus vanished" in g for g in result.gaps)


def test_entrypoints_stale_cli_in_fenced_block_detected(tmp_path: Path) -> None:
    docs = {**FULL_DOCS, "docs/setup.md": "```bash\n$ vo-rag books search\n```\n"}
    make_tree(tmp_path, docs)
    result = check_docs.check_entrypoints(tmp_path, inventory=INVENTORY)
    assert any("vo-rag books" in g for g in result.gaps)


def test_entrypoints_prose_and_flags_are_not_commands(tmp_path: Path) -> None:
    docs = {
        **FULL_DOCS,
        "docs/setup.md": (
            "vo-rag is a retrieval system. Try `vo-rag --help` or `vo-rag corpus --help`,\n"
            "`pip install vo-rag[dev]`, and `vo-rag preflight --strict`.\n"
        ),
    }
    make_tree(tmp_path, docs)
    result = check_docs.check_entrypoints(tmp_path, inventory=INVENTORY)
    assert result.ok, result.gaps


def test_entrypoints_stale_route_detected(tmp_path: Path) -> None:
    docs = {**FULL_DOCS, "docs/setup.md": "POST `/api/v1/vanished` and `/api/v1/books/abc`.\n"}
    make_tree(tmp_path, docs)
    result = check_docs.check_entrypoints(tmp_path, inventory=INVENTORY)
    stale = [g for g in result.gaps if "stale" in g]
    assert len(stale) == 1
    assert "/api/v1/vanished" in stale[0]


# --------------------------------------------------------------------------- #
# env
# --------------------------------------------------------------------------- #
ENV_CODE = {
    "src/vo_rag/a.py": (
        'import os\nA = os.environ.get("VORAG_ALPHA", "1")\nB = os.environ["VORAG_BETA"]\n'
    ),
    "config/settings.yaml": 'x:\n  y: "${VORAG_GAMMA:zz}"\n',
}


def test_env_pass_when_all_documented(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {**ENV_CODE, "docs/configuration.md": "`VORAG_ALPHA` `VORAG_BETA` `VORAG_GAMMA`\n"},
    )
    result = check_docs.check_env(tmp_path)
    assert result.ok, result.gaps


def test_env_fail_for_undocumented_var(tmp_path: Path) -> None:
    make_tree(tmp_path, {**ENV_CODE, "docs/configuration.md": "`VORAG_ALPHA` `VORAG_BETA`\n"})
    result = check_docs.check_env(tmp_path)
    assert not result.ok
    assert any("VORAG_GAMMA" in g for g in result.gaps)


def test_env_fail_when_page_missing(tmp_path: Path) -> None:
    make_tree(tmp_path, ENV_CODE)
    assert not check_docs.check_env(tmp_path).ok


def test_env_stale_documented_token(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            **ENV_CODE,
            ".env.example": "VORAG_FROM_EXAMPLE=1\n",
            "docs/configuration.md": (
                "`VORAG_ALPHA` `VORAG_BETA` `VORAG_GAMMA` `VORAG_FROM_EXAMPLE` `POSTGRES_URL`\n"
            ),
        },
    )
    result = check_docs.check_env(tmp_path)
    assert not result.ok
    stale = [g for g in result.gaps if "stale" in g]
    assert len(stale) == 1
    assert "POSTGRES_URL" in stale[0]


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
LEAVES = ["api.host", "api.port"]


def test_config_pass_when_block_lists_every_leaf(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {"docs/configuration.md": generated("config-keys", "| `api.host` |\n| `api.port` |")},
    )
    assert check_docs.check_config(tmp_path, leaves=LEAVES).ok


def test_config_fail_when_leaf_missing(tmp_path: Path) -> None:
    make_tree(tmp_path, {"docs/configuration.md": generated("config-keys", "| `api.host` |")})
    result = check_docs.check_config(tmp_path, leaves=LEAVES)
    assert not result.ok
    assert result.gaps == ["api.port"]


def test_config_leaf_outside_block_does_not_count(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "docs/configuration.md": "api.port\n" + generated("config-keys", "| `api.host` |"),
        },
    )
    assert not check_docs.check_config(tmp_path, leaves=LEAVES).ok


def test_config_fail_when_block_drifted(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {"docs/configuration.md": generated("config-keys", "| `api.host` |\n| `api.port` |")},
    )
    result = check_docs.check_config(tmp_path, leaves=LEAVES, render=lambda: "different")
    assert not result.ok
    assert any("drift" in g for g in result.gaps)


def test_config_fail_when_block_absent(tmp_path: Path) -> None:
    make_tree(tmp_path, {"docs/configuration.md": "no block\n"})
    assert not check_docs.check_config(tmp_path, leaves=LEAVES).ok


# --------------------------------------------------------------------------- #
# links
# --------------------------------------------------------------------------- #
def test_links_good_relative_links_and_anchors(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "README.md": "[docs](docs/a.md) [sec](docs/a.md#second-section) [ext](https://x.io/y)\n",
            "docs/a.md": (
                "# A\n\n## Second section\n\n[up](../README.md) [self](#a) "
                "![img](images/p.svg) [src](../src/mod.py#L10)\n"
            ),
            "docs/images/p.svg": "<svg/>",
            "src/mod.py": "x = 1\n",
        },
    )
    result = check_docs.check_links(tmp_path)
    assert result.ok, result.gaps


def test_links_broken_file_reported(tmp_path: Path) -> None:
    make_tree(tmp_path, {"README.md": "[gone](docs/missing.md)\n"})
    result = check_docs.check_links(tmp_path)
    assert not result.ok
    assert "docs/missing.md" in result.gaps[0]
    assert "README.md:1" in result.gaps[0]


def test_links_broken_anchor_reported(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {"README.md": "[x](docs/a.md#nope) [y](#absent)\n", "docs/a.md": "# Real heading\n"},
    )
    result = check_docs.check_links(tmp_path)
    assert result.count == 2


def test_links_heading_slug_rules(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "README.md": (
                "[a](docs/a.md#the-api--cli-v2) [b](docs/a.md#code-and-link) "
                "[c](docs/a.md#dup) [d](docs/a.md#dup-1) [e](docs/a.md#under_score)\n"
            ),
            "docs/a.md": (
                "# The API & CLI (v2)\n\n## `code` and [link](x.md)\n\n## Dup\n\n## Dup\n\n"
                "## under_score\n\n```\n# not a heading\n```\n"
            ),
            "docs/x.md": "x\n",
        },
    )
    result = check_docs.check_links(tmp_path)
    assert result.ok, result.gaps


def test_links_headings_in_code_fence_are_not_anchors(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "README.md": "[a](docs/a.md#not-a-heading)\n",
            "docs/a.md": "# Real\n\n```\n# not a heading\n```\n",
        },
    )
    assert not check_docs.check_links(tmp_path).ok


def test_links_in_code_are_ignored(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {"README.md": "`[a](gone.md)`\n\n```\n[b](gone2.md)\n```\n"},
    )
    assert check_docs.check_links(tmp_path).ok


def test_links_scans_services_workers_packages_tools_not_node_modules(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "services/README.md": "[x](nope.md)\n",
            "workers/embedding/README.md": "[x](nope.md)\n",
            "packages/README.md": "[x](nope.md)\n",
            "tools/guards/README.md": "[x](nope.md)\n",
            "packages/portal/node_modules/pkg/README.md": "[x](nope.md)\n",
        },
    )
    assert check_docs.check_links(tmp_path).count == 4


def test_links_html_and_reference_definitions(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "README.md": '<img src="docs/nope.png">\n\n[ref]: docs/nope2.md\n',
        },
    )
    assert check_docs.check_links(tmp_path).count == 2


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
MEASURED = {
    "pytest_tests": 5000,
    "pytest_test_files": 400,
    "portal_tests": 100,
    "portal_test_files": 10,
    "routes": 28,
    "cli_leaves": 53,
    "mcp_tools": 4,
    "qdrant_collections": 19,
    "postgres_tables": 47,
}


def numbers_tree(root: Path, text: str, measured: dict[str, int] | None = MEASURED) -> Path:
    files = {"docs/testing.md": text}
    if measured is not None:
        files["docs/_measured.json"] = json.dumps(measured)
    return make_tree(root, files)


def test_numbers_exact_matches_pass(tmp_path: Path) -> None:
    numbers_tree(
        tmp_path,
        "5,000 tests, 400 test files, 28 routes, 53 commands, 4 tools, 19 collections, "
        "47 tables.\nThe portal runs 100 tests in 10 test files.\n",
    )
    result = check_docs.check_numbers(tmp_path)
    assert result.ok, result.gaps


def test_numbers_mismatch_fails_with_expected_value(tmp_path: Path) -> None:
    numbers_tree(tmp_path, "There are 4,200 tests and 12 routes.\n")
    result = check_docs.check_numbers(tmp_path)
    assert not result.ok
    assert result.count == 2
    assert "5000" in "\n".join(result.gaps)


def test_numbers_about_allows_five_percent(tmp_path: Path) -> None:
    numbers_tree(tmp_path, "About 4,800 tests ran.\nabout 4,000 tests ran.\n")
    result = check_docs.check_numbers(tmp_path)
    assert result.count == 1
    assert "4,000" in result.gaps[0]


def test_numbers_without_about_is_exact(tmp_path: Path) -> None:
    numbers_tree(tmp_path, "4,800 tests ran.\n")
    assert not check_docs.check_numbers(tmp_path).ok


def test_numbers_portal_context_uses_portal_counts(tmp_path: Path) -> None:
    numbers_tree(tmp_path, "The portal has 100 tests.\nThe web portal suite: 5,000 tests.\n")
    result = check_docs.check_numbers(tmp_path)
    assert result.count == 1
    assert "5,000" in result.gaps[0]


def test_numbers_portal_is_judged_per_clause_not_per_line(tmp_path: Path) -> None:
    numbers_tree(
        tmp_path,
        "Run `make test` (about 5,000 tests, no services) and `pnpm test` for the portal.\n",
    )
    assert check_docs.check_numbers(tmp_path).ok


def test_numbers_allows_one_modifier_word_before_the_unit(tmp_path: Path) -> None:
    numbers_tree(
        tmp_path,
        "The portal has 91 unit tests in 18 test files. The API has 12 HTTP routes.\n"
        "It stores 40 Postgres tables, 4 MCP tools and 53 CLI commands.\n",
    )
    result = check_docs.check_numbers(tmp_path)
    # 91 tests, 18 test files, 12 routes and 40 tables are wrong; 4 tools and 53 commands are right
    assert result.count == 4, result.gaps
    assert not any("tools" in g or "commands" in g for g in result.gaps)


def test_numbers_ignores_code_blocks_and_other_words(tmp_path: Path) -> None:
    numbers_tree(tmp_path, "```\n999 tests\n```\n\nThere are 3 things and 9 test cases.\n")
    assert check_docs.check_numbers(tmp_path).ok


def test_numbers_fails_when_measurements_missing(tmp_path: Path) -> None:
    numbers_tree(tmp_path, "5,000 tests\n", measured=None)
    result = check_docs.check_numbers(tmp_path)
    assert not result.ok
    assert "_measured.json" in result.gaps[0]


def test_numbers_skips_generated_blocks_free_docs_with_no_numbers(tmp_path: Path) -> None:
    numbers_tree(tmp_path, "nothing numeric\n", measured=None)
    assert check_docs.check_numbers(tmp_path).ok


# --------------------------------------------------------------------------- #
# states
# --------------------------------------------------------------------------- #
ENUMS = {"EnvelopeStatus": {"pending", "processing", "done"}}


def mermaid(first: str, body: str) -> str:
    return f"```mermaid\n{first}\nstateDiagram-v2\n{body}\n```\n"


def test_states_match_passes(tmp_path: Path) -> None:
    body = (
        "    [*] --> pending\n    pending --> processing\n"
        "    processing --> done : ok\n    done --> [*]"
    )
    make_tree(tmp_path, {"docs/a.md": mermaid("%% enum: EnvelopeStatus", body)})
    result = check_docs.check_states(tmp_path, enums=ENUMS)
    assert result.ok, result.gaps


def test_states_state_declarations_and_descriptions_count(tmp_path: Path) -> None:
    body = (
        '    state "Waiting" as pending\n    pending : waits\n    processing --> done\n'
        "    pending --> processing"
    )
    make_tree(tmp_path, {"docs/a.md": mermaid("%% enum: EnvelopeStatus", body)})
    assert check_docs.check_states(tmp_path, enums=ENUMS).ok


def test_states_missing_and_extra_reported(tmp_path: Path) -> None:
    body = "    pending --> processing\n    processing --> failed"
    make_tree(tmp_path, {"docs/a.md": mermaid("%% enum: EnvelopeStatus", body)})
    result = check_docs.check_states(tmp_path, enums=ENUMS)
    assert not result.ok
    joined = "\n".join(result.gaps)
    assert "missing" in joined and "done" in joined
    assert "extra" in joined and "failed" in joined


def test_states_unknown_enum_fails(tmp_path: Path) -> None:
    make_tree(tmp_path, {"docs/a.md": mermaid("%% enum: Nope", "a --> b")})
    result = check_docs.check_states(tmp_path, enums=ENUMS)
    assert not result.ok
    assert "Nope" in result.gaps[0]


def test_states_untagged_blocks_ignored(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "docs/a.md": mermaid("%% just a note", "a --> b")
            + "```mermaid\nflowchart LR\n a --> b\n```\n"
        },
    )
    assert check_docs.check_states(tmp_path, enums=ENUMS).ok


def test_states_real_enum_map_resolves_the_named_enums() -> None:
    enums = check_docs.load_state_enums()
    assert set(enums) >= {
        "EnvelopeStatus",
        "IngestGenerationState",
        "PublicationState",
        "GateDecision",
    }
    assert "received" in enums["EnvelopeStatus"]


# --------------------------------------------------------------------------- #
# residue
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "Lifted from elsewhere",
        "phase P2b work",
        "per Task 12 notes",
        "spec § 4",
        "sanitized data",
        "a snapshot of the repo",
    ],
)
def test_residue_flags_each_pattern(tmp_path: Path, text: str) -> None:
    make_tree(tmp_path, {"docs/a.md": f"{text}\n"})
    result = check_docs.check_residue(tmp_path)
    assert not result.ok, text
    assert result.count == 1


def test_residue_clean_text_passes(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {"docs/a.md": "Use corpus_snapshot and snapshot_download. Task-less prose.\n"},
    )
    result = check_docs.check_residue(tmp_path)
    assert result.ok, result.gaps


def test_residue_scans_readme_and_docs(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {"README.md": "Lifted from x\n", "docs/notes.md": "Lifted from y\n"},
    )
    result = check_docs.check_residue(tmp_path)
    assert result.count == 2
    assert any("README.md" in g for g in result.gaps)
    assert any("docs/notes.md" in g for g in result.gaps)


def test_residue_finding_ids_need_security_legend(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "src/vo_rag/a.py": '"""F-014 and F-099 guard."""\n',
            "docs/security.md": "| F-014 | thing |\n",
        },
    )
    result = check_docs.check_residue(tmp_path)
    assert not result.ok
    assert result.count == 1
    assert "F-099" in result.gaps[0]


def test_residue_finding_ids_all_present_passes(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "src/vo_rag/a.py": '"""F-014 guard."""\n',
            "docs/security.md": "| F-014 | thing |\n",
        },
    )
    assert check_docs.check_residue(tmp_path).ok


def test_residue_finding_ids_without_security_page_fail(tmp_path: Path) -> None:
    make_tree(tmp_path, {"src/vo_rag/a.py": "# F-014\n"})
    assert not check_docs.check_residue(tmp_path).ok


# --------------------------------------------------------------------------- #
# capabilities
# --------------------------------------------------------------------------- #
CAP_HEADER = "| Capability | Page | Code |\n| --- | --- | --- |\n"


def test_capabilities_pass(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "docs/capabilities.md": CAP_HEADER
            + "| Query | [api](api.md) | `src/vo_rag/api/routes/query.py`, `vo_rag.core.raptor` |\n"
            + "| Ingest | `docs/cli.md` | [ingest](../src/vo_rag/core/raptor/) |\n",
            "docs/api.md": "x\n",
            "docs/cli.md": "x\n",
            "src/vo_rag/api/routes/query.py": "x = 1\n",
            "src/vo_rag/core/raptor/__init__.py": "",
        },
    )
    result = check_docs.check_capabilities(tmp_path)
    assert result.ok, result.gaps


def test_capabilities_missing_page_and_module(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "docs/capabilities.md": CAP_HEADER
            + "| Query | [api](api.md) | `src/vo_rag/api/routes/nope.py` |\n"
            + "| Ingest | [x](ghost.md) | `vo_rag.core.raptr` |\n"
            + "| Orphan | none | `src/vo_rag/api/routes/query.py` |\n",
            "docs/api.md": "x\n",
            "src/vo_rag/api/routes/query.py": "x = 1\n",
            "src/vo_rag/core/raptor/__init__.py": "",
        },
    )
    result = check_docs.check_capabilities(tmp_path)
    assert not result.ok
    joined = "\n".join(result.gaps)
    assert "nope.py" in joined
    assert "ghost.md" in joined
    assert "raptr" in joined
    assert "Orphan" in joined


def test_capabilities_dotted_class_path_resolves_through_module(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "docs/capabilities.md": CAP_HEADER
            + "| A | [api](api.md) | `vo_rag.core.config.settings.Settings` |\n",
            "docs/api.md": "x\n",
            "src/vo_rag/core/config/settings.py": "class Settings: ...\n",
        },
    )
    assert check_docs.check_capabilities(tmp_path).ok


def test_capabilities_page_missing_fails(tmp_path: Path) -> None:
    make_tree(tmp_path, {"docs/api.md": "x\n"})
    result = check_docs.check_capabilities(tmp_path)
    assert not result.ok
    assert "capabilities.md" in result.gaps[0]


# --------------------------------------------------------------------------- #
# registry and command line
# --------------------------------------------------------------------------- #
def test_registry_has_the_nine_checks() -> None:
    assert list(check_docs.CHECKS) == [
        "modules",
        "entrypoints",
        "env",
        "config",
        "links",
        "numbers",
        "states",
        "residue",
        "capabilities",
    ]


def test_main_only_selects_checks_and_exit_code(tmp_path: Path, capsys) -> None:
    make_tree(tmp_path, {"README.md": "[ok](docs/a.md)\n", "docs/a.md": "# A\n"})
    code = check_docs.main(["--root", str(tmp_path), "--only", "links,residue"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("PASS ") == 2
    assert "PASS links: 0" in out
    assert "modules" not in out


def test_main_exit_one_on_fail_and_report_lists_gaps(tmp_path: Path, capsys) -> None:
    make_tree(tmp_path, {"README.md": "[bad](docs/gone.md)\n"})
    code = check_docs.main(["--root", str(tmp_path), "--only", "links", "--report"])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL links: 1" in out
    assert "docs/gone.md" in out


def test_main_without_report_truncates_gap_details(tmp_path: Path, capsys) -> None:
    make_tree(tmp_path, {"README.md": "".join(f"[b](gone{i}.md)\n" for i in range(12))})
    check_docs.main(["--root", str(tmp_path), "--only", "links"])
    out = capsys.readouterr().out
    assert "FAIL links: 12" in out
    assert "gone11.md" not in out


def test_main_rejects_unknown_check(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        check_docs.main(["--root", str(tmp_path), "--only", "bogus"])
    assert exc.value.code == 2
