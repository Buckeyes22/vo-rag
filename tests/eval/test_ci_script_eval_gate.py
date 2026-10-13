"""Behavioral contract tests for scripts/ci.sh.

ci.sh has no retrieval-quality step; the only eval step is the replay-smoke gate.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ci_script_runs_eval_gate_and_passes(tmp_path: Path) -> None:
    result, uv_log = _run_ci_with_fake_bins(
        tmp_path,
        final_status="0",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "== locked bootstrap (dev + llamaindex) ==" in result.stdout
    assert "== eval gate ==" in result.stdout
    assert "eval gate PASS" in result.stdout
    assert result.stdout.index("== docs ==") < result.stdout.index("== eval gate ==")
    log = uv_log.read_text(encoding="utf-8")
    assert "sync --locked --extra dev --extra llamaindex" in log
    assert "run --no-sync mypy" in log
    assert "run --no-sync python scripts/check_docs.py" in log
    assert "run --no-sync python scripts/gen_reference_docs.py --check" in log


def test_ci_script_fails_when_final_eval_fails(tmp_path: Path) -> None:
    result, _uv_log = _run_ci_with_fake_bins(
        tmp_path,
        final_status="7",
    )

    assert result.returncode == 7
    assert "== eval gate ==" in result.stdout


def test_ci_script_aborts_when_detect_secrets_hook_fails(tmp_path: Path) -> None:
    result, uv_log = _run_ci_with_fake_bins(
        tmp_path,
        final_status="0",
        detect_secrets_status="5",
    )

    assert result.returncode != 0
    assert "== detect-secrets ==" in result.stdout
    assert "== eval gate ==" not in result.stdout
    assert "detect-secrets-hook --baseline .secrets.baseline" in uv_log.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "failing_call",
    ["python scripts/check_docs.py", "python scripts/gen_reference_docs.py --check"],
)
def test_ci_script_aborts_when_a_docs_check_fails(tmp_path: Path, failing_call: str) -> None:
    result, uv_log = _run_ci_with_fake_bins(
        tmp_path,
        final_status="0",
        failing_uv_call=failing_call,
    )

    assert result.returncode != 0
    assert "== docs ==" in result.stdout
    assert "== eval gate ==" not in result.stdout
    assert failing_call in uv_log.read_text(encoding="utf-8")


def _run_ci_with_fake_bins(
    tmp_path: Path,
    *,
    final_status: str,
    detect_secrets_status: str = "0",
    failing_uv_call: str = "",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log = tmp_path / "uv.log"

    _write_executable(
        bin_dir / "git",
        """\
        #!/usr/bin/env bash
        if [[ "$1" == "ls-files" ]]; then
            exit 0
        fi
        exit 0
        """,
    )
    _write_executable(
        bin_dir / "uv",
        """\
        #!/usr/bin/env bash
        printf '%s\n' "$*" >> "$FAKE_UV_LOG"
        case "$*" in
            *"detect-secrets-hook --baseline .secrets.baseline"*)
                exit "$FAKE_DETECT_SECRETS_STATUS"
                ;;
        esac
        if [[ -n "$FAKE_FAILING_UV_CALL" && "$*" == *"$FAKE_FAILING_UV_CALL"* ]]; then
            exit 9
        fi
        case "$*" in
            *"python scripts/run_eval_gate.py"*)
                printf 'eval gate PASS\n'
                exit "$FAKE_FINAL_STATUS"
                ;;
        esac
        exit 0
        """,
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "FAKE_UV_LOG": str(uv_log),
            "FAKE_FINAL_STATUS": final_status,
            "FAKE_DETECT_SECRETS_STATUS": detect_secrets_status,
            "FAKE_FAILING_UV_CALL": failing_uv_call,
        }
    )
    result = subprocess.run(
        ["bash", "scripts/ci.sh"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, uv_log


def _write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(0o755)
