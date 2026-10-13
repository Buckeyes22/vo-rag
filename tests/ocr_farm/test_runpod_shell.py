"""Pod-side scripts, run under bash with stubbed commands (VOR-1434).

No pod is involved: ``VENV_PY`` / ``PYTHON`` / ``pkill`` are small stub
scripts that log their argv, and the cgroup / meminfo files are temp files.
"""

from __future__ import annotations

import os
import re
import subprocess
import tarfile
from pathlib import Path

import pytest

RUNPOD_DIR = Path(__file__).resolve().parents[2] / "src" / "vo_rag" / "ocr_farm" / "runpod"
BOOTSTRAP = RUNPOD_DIR / "bootstrap.sh"
RUN_UNITS = RUNPOD_DIR / "run_units.sh"
GIB = 1024**3


def _stub(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)
    return path


def _bash(script: Path, snippet: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "{script}"; {snippet}'],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


# ---------------------------------------------------------------------------
# static rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", [BOOTSTRAP, RUN_UNITS])
def test_scripts_parse(script: Path) -> None:
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_every_remote_kill_uses_a_bracketed_pattern() -> None:
    kills = [
        line.strip()
        for script in (BOOTSTRAP, RUN_UNITS)
        for line in script.read_text().splitlines()
        if re.search(r"\bpkill\b|\bpgrep\b", line) and not line.strip().startswith("#")
    ]
    assert kills, "run_units.sh must stop workers with pkill"
    for line in kills:
        assert re.search(r"-f '\[[^\]]\]", line), line


def test_bootstrap_pins_torch_cu128() -> None:
    text = BOOTSTRAP.read_text()
    assert '"torch==2.8.0+cu128"' in text
    assert "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128" in text


# ---------------------------------------------------------------------------
# bootstrap.sh
# ---------------------------------------------------------------------------


def _bootstrap_env(
    tmp_path: Path, cuda_rc: int, *, warm_writes_hf: bool = True, models_rc: int = 0
) -> tuple[dict[str, str], Path]:
    """``venv_py`` stub: logs argv; the warm-up run logs the HF env it saw and
    (``warm_writes_hf``) drops a model file under ``$HF_HOME``, like docling's
    first-use download; the RapidOCR model check exits ``models_rc``."""
    log = tmp_path / "calls.log"
    hf_write = (
        'mkdir -p "$HF_HOME/hub" && echo m > "$HF_HOME/hub/model.bin"' if warm_writes_hf else ":"
    )
    venv_py = _stub(
        tmp_path / "venv_py",
        f'echo "$*" >> "{log}"\n'
        f'case "$*" in\n'
        f"  *cuda.is_available*) exit {cuda_rc};;\n"
        f'  *warm.pdf*) echo "WARM_ENV HF_HOME=$HF_HOME HF_TOKEN=${{HF_TOKEN-unset}}" '
        f'>> "{log}"; {hf_write};;\n'
        f"  *torch.load*) exit {models_rc};;\n"
        f"esac\nexit 0\n",
    )
    python = _stub(tmp_path / "python", f'echo "PYTHON $*" >> "{log}"\n')
    env = {"WORKDIR": str(tmp_path), "VENV_PY": str(venv_py), "PYTHON": str(python),
           "HF_TOKEN": "stale-token"}  # fmt: skip
    return env, log


def test_bootstrap_aborts_without_cuda_before_any_worker_runs(tmp_path: Path) -> None:
    env, log = _bootstrap_env(tmp_path, cuda_rc=3)

    result = _bash(BOOTSTRAP, "main", env)

    assert result.returncode == 3
    assert "refusing to OCR on the CPU" in result.stderr
    assert (tmp_path / "setup.failed").read_text().strip() == "NO_CUDA"
    assert not (tmp_path / "setup.done").exists()
    assert "ocr_worker.py" not in log.read_text()


def test_bootstrap_warms_models_in_one_process_then_marks_ready(tmp_path: Path) -> None:
    env, log = _bootstrap_env(tmp_path, cuda_rc=0)

    result = _bash(BOOTSTRAP, "main", env)

    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    torch_install = next(c for c in calls if "torch==2.8.0+cu128" in c)
    assert "--index-url https://download.pytorch.org/whl/cu128" in torch_install
    order = [
        next(i for i, c in enumerate(calls) if key in c)
        for key in ("torch==2.8.0+cu128", "cuda.is_available", "ocr_worker.py --check",
                    "ocr_worker.py warm.pdf", "torch.load")
    ]  # fmt: skip
    assert order == sorted(order)
    assert sum("ocr_worker.py warm.pdf" in c for c in calls) == 1
    assert (tmp_path / "setup.done").read_text().strip() == "READY"
    # warmed into the SAME cache the pool's workers use, without the stale token
    assert f"WARM_ENV HF_HOME={tmp_path}/hf HF_TOKEN=unset" in calls


@pytest.mark.parametrize(("hf", "models_rc"), [(False, 0), (True, 4)])
def test_bootstrap_fails_unless_hf_cache_and_every_rapidocr_model_load(
    tmp_path: Path, hf: bool, models_rc: int
) -> None:
    env, _ = _bootstrap_env(tmp_path, cuda_rc=0, warm_writes_hf=hf, models_rc=models_rc)

    result = _bash(BOOTSTRAP, "main", env)

    assert result.returncode == 4
    assert (tmp_path / "setup.failed").read_text().strip() == "NO_MODELS"
    assert not (tmp_path / "setup.done").exists()


def test_bootstrap_and_pool_share_one_hf_home() -> None:
    for script in (BOOTSTRAP, RUN_UNITS):
        text = script.read_text()
        assert "HF_HOME=${HF_HOME:-$WORKDIR/hf}" in text and "export HF_HOME" in text, script
        assert "unset HF_TOKEN" in text, script
    assert 'HF_HOME="$WORKDIR/hf"' not in RUN_UNITS.read_text()


# ---------------------------------------------------------------------------
# run_units.sh
# ---------------------------------------------------------------------------


def _mem_env(tmp_path: Path, **files: str) -> dict[str, str]:
    root = tmp_path / "cg"
    for rel, value in files.items():
        path = root / rel.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n")
    meminfo = tmp_path / "meminfo"
    if not meminfo.exists():
        meminfo.write_text(f"MemTotal:       {100 * GIB // 1024} kB\n")
    return {"CGROUP_ROOT": str(root), "MEMINFO": str(meminfo), "WORKDIR": str(tmp_path)}


@pytest.mark.parametrize(
    ("files", "max_workers", "expected"),
    [
        ({"memory.max": str(40 * GIB)}, "8", "4"),  # cgroup v2 limit, not host RAM
        ({"memory.max": str(17 * GIB)}, "8", "1"),  # 8.9 GB peaks: 17 GB fits one
        ({"memory.max": str(5 * GIB)}, "8", "1"),  # never zero workers
        ({"memory.max": "max", "memory__memory.limit_in_bytes": str(20 * GIB)}, "8", "2"),
        ({"memory__memory.limit_in_bytes": "9223372036854771712"}, "20", "11"),  # meminfo
        ({"memory.max": str(200 * GIB)}, "8", "8"),  # capped
    ],
)
def test_pool_size_comes_from_the_cgroup_limit_at_9gb_per_worker(
    tmp_path: Path, files: dict[str, str], max_workers: str, expected: str
) -> None:
    env = {**_mem_env(tmp_path, **files), "MAX_WORKERS": max_workers}

    result = _bash(RUN_UNITS, "pool_size", env)

    assert result.stdout.strip() == expected, result.stderr


def _units_env(tmp_path: Path) -> dict[str, str]:
    units = tmp_path / "units"
    units.mkdir()
    (units / "big.pdf").write_bytes(b"x" * 300)
    (units / "bad.pdf").write_bytes(b"x" * 200)
    (units / "done.pdf").write_bytes(b"x" * 100)
    (units / "done.json").write_text("{}")
    (units / "done.note").write_text("ocr_worker: ok pages=1\n")
    venv_py = _stub(
        tmp_path / "venv_py",
        f'echo "$*" >> "{tmp_path}/worker.log"\n'
        'case "$2" in *bad.pdf) echo Killed; exit 137;; esac\n'
        'echo "{}" > "$3"; echo "noise"; echo "ocr_worker: ok pages=1"\n',
    )
    return {**_mem_env(tmp_path, **{"memory.max": str(20 * GIB)}), "VENV_PY": str(venv_py)}


def test_run_ocrs_pending_units_and_skips_finished_ones(tmp_path: Path) -> None:
    env = _units_env(tmp_path)

    result = subprocess.run(
        ["bash", str(RUN_UNITS), "run"], env={**os.environ, **env}, capture_output=True,
        text=True, timeout=60, check=False,
    )  # fmt: skip

    assert result.returncode == 0, result.stderr
    log = (tmp_path / "run.log").read_text().splitlines()
    assert log[0].startswith("POOL 2 ")
    assert sorted(log[1:3]) == ["UNITFAIL bad", "UNITOK big"]
    assert log[-1] == "ALLDONE"
    assert "done.pdf" not in (tmp_path / "worker.log").read_text()
    assert (tmp_path / "units" / "bad.note").read_text().strip() == "Killed"
    assert not (tmp_path / "run.pid").exists()


def test_pack_writes_a_tar_file_with_ok_json_and_every_note(tmp_path: Path) -> None:
    env = _units_env(tmp_path)
    (tmp_path / "units" / "bad.note").write_text("Killed\n")
    (tmp_path / "units" / "bad.json").write_text("")

    result = subprocess.run(
        ["bash", str(RUN_UNITS), "pack"], env={**os.environ, **env}, capture_output=True,
        text=True, timeout=60, check=False,
    )  # fmt: skip

    assert result.returncode == 0, result.stderr
    packed, status = result.stdout.strip().splitlines()
    assert packed == "PACKED ok=1"
    assert status.startswith("STATUS pool=") and "ok=1 failed=1 pending=1 running=no" in status
    with tarfile.open(tmp_path / "results.tar") as tar:
        assert sorted(tar.getnames()) == ["units/bad.note", "units/done.json", "units/done.note"]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("shrink", ["-USR2 -f [x]args -r -P"]),
        ("stop", ["-TERM -f [x]args -r -P", "-TERM -f [o]cr_worker\\.py"]),
    ],
)
def test_shrink_and_stop_signal_by_bracketed_pattern(
    tmp_path: Path, command: str, expected: list[str]
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir / "pkill", f'echo "$*" >> "{tmp_path}/pkill.log"\n')
    env = {**os.environ, "WORKDIR": str(tmp_path), "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    result = subprocess.run(
        ["bash", str(RUN_UNITS), command], env=env, capture_output=True, text=True, timeout=60,
        check=False,
    )  # fmt: skip

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "pkill.log").read_text().splitlines() == expected
