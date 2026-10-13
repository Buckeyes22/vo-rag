"""Example OCR worker hosts; configure your own SSH aliases before use."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Literal

#: Aggregate throughput of the default farm (both hosts together), s/page.
#: Example throughput estimate; calibrate for your own workers before scheduling.
#: This estimate is not a public performance benchmark.
FARM_SECONDS_PER_PAGE = 0.48

CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu130"
PYTHON_VERSION = "3.12"
JOBS_DIR = "jobs"
WORKER_SCRIPT = "ocr_worker.py"
REQUIREMENTS = "requirements.txt"
RESOLVE_TEMP_COMMAND = "echo %TEMP%"
#: Env var ``ocr_worker.py`` reads for ``FarmHost.gpu_mem_fraction``.
GPU_MEM_FRACTION_ENV = "VORAG_OCR_GPU_MEM_FRACTION"

#: Windows wrapper that runs one long step with a remote time limit
#: (``worker/farm_step.ps1``; a script file, because inline multi-line code
#: doesn't survive Windows ssh).
STEP_WRAPPER = "farm_step.ps1"
#: coreutils ``timeout --kill-after``: SIGKILL this long after the SIGTERM.
REMOTE_KILL_AFTER_S = 30
#: Remote time limits must fit ``Wait-Process -Timeout`` (1..32767 s), which
#: ``farm_step.ps1`` uses; the same range is enforced for Linux for one rule.
MAX_REMOTE_TIMEOUT_S = 32767


def validate_remote_timeout(seconds: float, what: str = "remote timeout") -> None:
    """Raise ``ValueError`` unless ``0 < seconds <= MAX_REMOTE_TIMEOUT_S``."""
    if not 0 < seconds <= MAX_REMOTE_TIMEOUT_S:
        raise ValueError(
            f"{what} must be > 0 and <= {MAX_REMOTE_TIMEOUT_S} s "
            f"(farm_step.ps1 / Wait-Process limit), got {seconds}"
        )


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FarmHost:
    """One GPU worker host.

    ``workdir`` is the path as the remote shell sees it (``%TEMP%\\vorag-ocr``
    on Windows, expanded by ``cmd``); ``scp_workdir`` is the resolved
    forward-slash path scp needs (``C:/Users/.../Temp/vorag-ocr``), filled in
    by ``coordinator.resolve_hosts``.
    """

    alias: str
    os: Literal["linux", "windows"]
    workdir: str
    slots: int
    python: str
    uv: str = "uv"
    scp_workdir: str | None = None
    #: Per-process cap on the GPU's memory (``torch.cuda.set_per_process_memory_fraction``),
    #: for hosts whose GPU has another tenant; ``None`` means no cap.
    gpu_mem_fraction: float | None = None

    @property
    def is_windows(self) -> bool:
        return self.os == "windows"

    # -- paths ---------------------------------------------------------------

    def remote(self, *parts: str) -> str:
        """A path under the workdir, as the remote shell sees it."""
        sep = "\\" if self.is_windows else "/"
        return sep.join((self.workdir, *parts))

    def relative(self, *parts: str) -> str:
        """A workdir-relative path, as the remote shell sees it."""
        return ("\\" if self.is_windows else "/").join(parts)

    def scp_target(self, *parts: str) -> str:
        """``<alias>:<forward-slash path>`` for scp."""
        if self.scp_workdir is None:
            raise ValueError(f"{self.alias}: scp_workdir not resolved (run resolve_hosts first)")
        return f"{self.alias}:" + "/".join((self.scp_workdir, *parts))

    def with_scp_workdir(self, scp_workdir: str) -> FarmHost:
        return replace(self, scp_workdir=scp_workdir.rstrip("/"))

    # -- commands ------------------------------------------------------------

    def _in_workdir(self, command: str) -> str:
        if self.is_windows:
            return f'cd /d "{self.workdir}" && {command}'
        return f"cd {self.workdir} && {command}"

    def mkdir_command(self) -> str:
        jobs = self.remote(JOBS_DIR)
        if self.is_windows:
            return f'if not exist "{jobs}" mkdir "{jobs}"'
        return f"mkdir -p {jobs}"

    def worker_command(self, pdf_name: str, out_name: str) -> str:
        pdf = self.relative(JOBS_DIR, pdf_name)
        out = self.relative(JOBS_DIR, out_name)
        hf_home = self.remote("hf")
        cap = self.gpu_mem_fraction
        if self.is_windows:
            env = f"set HF_HOME={hf_home}&& set HF_TOKEN=&& "
            if cap is not None:
                env += f"set {GPU_MEM_FRACTION_ENV}={cap}&& "
        else:
            env = f"env -u HF_TOKEN HF_HOME={hf_home} "
            if cap is not None:
                env += f"{GPU_MEM_FRACTION_ENV}={cap} "
        return self._in_workdir(f"{env}{self.python} {WORKER_SCRIPT} {pdf} {out}")

    def cleanup_command(self, *names: str) -> str:
        paths = [self.remote(JOBS_DIR, name) for name in names]
        if self.is_windows:
            return "del /f /q " + " ".join(f'"{p}"' for p in paths)
        return "rm -f " + " ".join(paths)

    def read_pid_command(self, pid_name: str) -> str:
        """Print the worker PID the worker wrote to ``jobs/<pid_name>``."""
        path = self.remote(JOBS_DIR, pid_name)
        if self.is_windows:
            return f'type "{path}"'
        return f"cat {path}"

    def kill_worker_command(self, pid: int, sha: str) -> str:
        """Kill a timed-out worker by PID, only if that PID is still our worker.

        The command line must contain ``ocr_worker.py`` and the book's sha, so a
        PID recycled by an unrelated program is never touched. Exits non-zero
        (and kills nothing) when it doesn't match.
        """
        if not _SHA256_RE.fullmatch(sha):
            raise ValueError(f"not a sha256: {sha!r}")
        pid = int(pid)
        if self.is_windows:
            return (
                'powershell -NoProfile -Command "'
                f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
                f"if ($p -and $p.CommandLine -like '*{WORKER_SCRIPT}*' "
                f"-and $p.CommandLine -like '*{sha}*') "
                f'{{ Stop-Process -Id {pid} -Force }} else {{ exit 1 }}"'
            )
        return (
            f"grep -qa {WORKER_SCRIPT} /proc/{pid}/cmdline && grep -qa {sha} /proc/{pid}/cmdline "
            f"&& kill {pid}"
        )

    def _bounded(self, program: str, arguments: str, timeout_s: float) -> str:
        """Run ``program arguments`` in the workdir with a *remote* time limit.

        The remote side kills its own command when the limit passes, so
        nothing has to be found and killed by name afterwards: coreutils
        ``timeout`` on Linux (SIGTERM, then SIGKILL after
        ``REMOTE_KILL_AFTER_S``), and on Windows the shipped ``farm_step.ps1``
        wrapper, which stops exactly the PID it started and exits 124.
        """
        validate_remote_timeout(timeout_s)
        secs = max(1, math.ceil(timeout_s))
        if self.is_windows:
            return self._in_workdir(
                f"powershell -NoProfile -ExecutionPolicy Bypass -File {STEP_WRAPPER} "
                f'-TimeoutSeconds {secs} -FilePath {program} -Arguments "{arguments}"'
            )
        return self._in_workdir(
            f"timeout --kill-after={REMOTE_KILL_AFTER_S} {secs} {program} {arguments}"
        )

    def venv_command(self, timeout_s: float) -> str:
        return self._bounded(
            self.uv, f"venv --allow-existing --python {PYTHON_VERSION} .venv", timeout_s
        )

    def install_command(self, timeout_s: float) -> str:
        return self._bounded(
            self.uv,
            f"pip install --python {self.python} --extra-index-url {CUDA_INDEX_URL} "
            f"--index-strategy unsafe-best-match -r {REQUIREMENTS}",
            timeout_s,
        )

    def check_command(self, timeout_s: float) -> str:
        return self._bounded(self.python, f"{WORKER_SCRIPT} --check", timeout_s)

    def support_files(self) -> tuple[str, ...]:
        """Files (in ``worker/``) the host needs next to ``ocr_worker.py``."""
        return (STEP_WRAPPER,) if self.is_windows else ()


win = FarmHost("win", "windows", "%TEMP%\\vorag-ocr", 2, r".venv\Scripts\python.exe", uv="uv")
# A directory on the remote lin worker (ruled path), never a local temp file on
# the coordinator. One-line statement so bandit's nosec covers exactly it.
Z2_WORKDIR = "/tmp/vorag-ocr"  # nosec B108 - remote worker path, not a local temp file
Z2 = FarmHost(
    "lin",
    "linux",
    Z2_WORKDIR,
    2,
    ".venv/bin/python",
    uv="~/.local/bin/uv",
    gpu_mem_fraction=0.25,
)

#: Preference order: the largest books go to the first host with a free slot,
#: so the faster GPU (win) comes first.
DEFAULT_HOSTS: tuple[FarmHost, ...] = (win, Z2)
