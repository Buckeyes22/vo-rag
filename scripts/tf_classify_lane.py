#!/usr/bin/env python3
"""Classify track-and-field tag jobs with a model (VOR-1443, VOR-1451).

The model is called directly through an OpenAI-compatible chat-completions API
(--api-url, or $VORAG_TF_TAGGER_URL; key in the env var named by --api-key-env,
default LLAMA_SWAP_API_KEY): no agent, no tools, thinking off. Without a URL the
lane falls back to an agent shim script given with --shim, which needs
--model provider/model.

One lane = one process over one shard file. Each validated result is appended
immediately, so a restarted lane resumes. A chunk batch is accepted chunk by
chunk: the ids that validate are written as soon as they land, and the next
attempt is asked for the ids that did not, and only for those. A job that still
has ids missing after `max_attempts` goes to RESULTS.failed for a later pass,
with every attempt's error; a book job is all or nothing. A shim that crashes,
times out or exits non-zero is an attempt failure like any other, so one broken
lane worker cannot end the run. Usage:
  uv run python scripts/tf_classify_lane.py JOBS.jsonl RESULTS.jsonl \\
    --api-url http://127.0.0.1:9292/v1/chat/completions
  uv run python scripts/tf_classify_lane.py JOBS.jsonl RESULTS.jsonl \\
    --shim /path/to/agent-shim.sh --model provider/model   # shim fallback
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vo_rag.core.corpus.tf_schema import (
    TF_TAGGER_MODEL,
    TagValidationError,
    validate_batch,
    validate_book,
)

_SENTINEL = re.compile(r"SHIM-DONE exit=(\d+)")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
# A part record is one pass over part of a job's chunk ids: the job's own id with
# the part number after it, so a job answered in several attempts is still one job.
_PART = "#p"
_SHIM_TIMEOUT = 900
_API_MAX_TOKENS = 6000
_REAP_TIMEOUT = 10
# The classifier is text in, text out: no tool may touch the machine (VOR-1447).
# Zen's free tier refuses a session whose tool list lacks `bash` or `read` ("free
# tier can only be used from within OpenCode"), so those two stay offered with
# every call denied except one inert pattern each, and every other tool that
# reads, writes, runs or fetches is denied outright (VOR-1449).
# `--auto` approves anything not denied, so "*" denies every tool this list does
# not name, including ones a later opencode adds. `list`, `codesearch` and `lsp`
# are not tools in opencode 1.18.33; they guard against versions that add them.
TOOL_POLICY: dict[str, Any] = {
    "permission": {
        "*": "deny",
        "bash": {"*": "deny", "true": "allow"},
        "read": {"*": "deny", "/nonexistent/tf-lane-sentinel": "allow"},
        "grep": "deny",
        "glob": "deny",
        "list": "deny",
        "edit": "deny",
        "external_directory": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "codesearch": "deny",
        "task": "deny",
        "skill": "deny",
        "lsp": "deny",
    }
}
_TOOL_POLICY_JSON = json.dumps(TOOL_POLICY)
_STDERR_TAIL = 500
_ERROR_LIMIT = 1500
# Retries are appended to the job prompt: every call is a fresh model session, and
# the job prompt is the only place the passages are (VOR-1448).
_BOOK_RETRY = "\n\nYour previous answer was invalid: {error}\nReturn corrected JSON only."
_ONLY = "\n\nReturn JSON ONLY for these chunk ids: {ids}"
_RETRY = (
    "\n\nYour previous answer was invalid for these chunk ids only: {ids}.\n"
    "Return corrected JSON ONLY for these ids: {ids}\n{errors}"
)


class ShimError(RuntimeError):
    """The shim itself failed: it could not run, timed out, or exited non-zero."""


def _load_json(text: str, order: tuple[str, str]) -> object | None:
    """The first JSON value in ``text`` starting with one of ``order``, or None.

    Every start position is tried rather than slicing from the first opener to the
    last closer: a fence of prose, or a sentence holding a stray brace around the
    real answer, would make the whole span unparseable.
    """
    decoder = json.JSONDecoder()
    for opener in order:
        for index, char in enumerate(text):
            if char != opener:
                continue
            try:
                value, _ = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                continue
            parsed: object = value
            return parsed
    return None


def extract_json(stdout: str) -> object:
    """The model's answer: the last fenced block before the sentinel, else the
    first JSON value in the text before it.

    The un-fenced body is always a candidate -- a model that answers inside prose,
    or fences something that is not JSON at all, still gets read -- and an array
    wins over an object only when the text opens with one, so a bracketed aside
    cannot outrank the answer.
    """
    body = stdout.split("SHIM-DONE", 1)[0]
    order = ("[", "{") if next((c for c in body if c in "[{"), "") == "[" else ("{", "[")
    for text in (*reversed(_FENCE.findall(body)), body):
        value = _load_json(text, order)
        if value is not None:
            return value
    raise ValueError("no JSON object or array in model output")


def _stderr_tail(stderr: str) -> str:
    text = stderr.strip()
    return f": {text[-_STDERR_TAIL:]}" if text else ""


def _checked_stdout(stdout: str, stderr: str = "") -> str:
    """``stdout`` when the shim reported success, else a :class:`ShimError`.

    The shim's last line is ``SHIM-DONE exit=N``. A crash, a timeout or a missing
    sentinel is a shim failure and not a model answer, so it is reported as one:
    the retry prompt says what actually went wrong, and ``.failed`` records it.
    """
    codes = _SENTINEL.findall(stdout)
    if not codes:
        raise ShimError(f"shim produced no SHIM-DONE sentinel{_stderr_tail(stderr)}")
    code = int(codes[-1])
    if code != 0:
        raise ShimError(f"shim exit {code}{_stderr_tail(stderr)}")
    return stdout


def _entry(result: object, chunk_id: str) -> object | None:
    """The model's tags for one chunk id, or None when the answer carries none.

    A keyed object and a list of entries carrying their own ``chunk_id`` are the
    two shapes ``validate_batch`` takes, and the whole-batch path accepted both.
    """
    if isinstance(result, dict):
        return result.get(chunk_id)
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and item.get("chunk_id") == chunk_id:
                return item
    return None


def _split(result: object, chunk_ids: list[str]) -> tuple[dict[str, object], list[str]]:
    """The ids of ``chunk_ids`` that validate on their own, and an error for each id
    that does not.

    Each id is validated with ``validate_batch`` on a one-entry answer: the same
    validator and the same rules as the whole batch, so an id accepted here is one
    the batch would have accepted too. An id the answer does not carry is invalid,
    and an id outside ``chunk_ids`` is never looked at and never written.
    """
    accepted: dict[str, object] = {}
    errors: list[str] = []
    for chunk_id in chunk_ids:
        tags = _entry(result, chunk_id)
        if tags is None:
            errors.append(f"{chunk_id}: no tags in the answer")
            continue
        try:
            validate_batch({chunk_id: tags}, [chunk_id])
        except TagValidationError as exc:
            errors.append(str(exc)[:_ERROR_LIMIT])
        else:
            # The model's own object, as the whole-batch path wrote it.
            accepted[chunk_id] = tags
    return accepted, errors


def _retry_prompt(base: str, chunk_ids: list[str], errors: list[str]) -> str:
    """The job prompt plus what was wrong, naming only the ids still missing, so a
    retry re-asks for those instead of the batch the model already half-answered."""
    ids = json.dumps(list(chunk_ids), ensure_ascii=False)
    # The base prompt is the only place the chunks are: every call is a fresh model
    # session, and a retry without them left the model nothing to classify
    # (VOR-1448).
    return base + _RETRY.format(ids=ids, errors="\n".join(errors))


def _resume_prompt(base: str, chunk_ids: list[str]) -> str:
    """The job prompt, limited to the ids no record covers yet. No error is named:
    the model has not answered this job in this session."""
    return base + _ONLY.format(ids=json.dumps(list(chunk_ids), ensure_ascii=False))


def _split_part(job_id: str) -> tuple[str, int]:
    """``(base job id, part number)``; the part number is 0 for a whole-job record."""
    base, marker, part = job_id.partition(_PART)
    if marker and part.isdigit():
        return base, int(part)
    return job_id, 0


def _read_results(path: Path) -> tuple[dict[str, set[str]], dict[str, int], int]:
    """What a results file already holds: the chunk ids covered per job, the highest
    part number written per job, and how many lines could not be read.

    Coverage is a union over the job's own record and its part records, so a job
    finished in several attempts counts as finished. Read as bytes and decoded line
    by line: a lane killed mid-write can leave a line that is not valid UTF-8 at
    all (not just truncated JSON), and `.read_text()` would raise for the whole
    file on one bad line (VOR-1443). Such a line is counted and its job runs
    again; it does not end the resume.
    """
    covered: dict[str, set[str]] = {}
    parts: dict[str, int] = {}
    unreadable = 0
    if not path.exists():
        return covered, parts, unreadable
    for raw_line in path.read_bytes().splitlines():
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line.decode("utf-8"))
            job_id = str(record["job_id"])
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            unreadable += 1
            continue
        base, part = _split_part(job_id)
        chunk_ids = record.get("chunk_ids")
        covered.setdefault(base, set())
        if isinstance(chunk_ids, list):
            covered[base].update(str(chunk_id) for chunk_id in chunk_ids)
        parts[base] = max(parts.get(base, 0), part)
    return covered, parts, unreadable


def _append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSONL record, starting a fresh line if the file was torn.

    A lane killed mid-write leaves a file that does not end in a newline. Writing
    straight on would glue the next record onto that fragment, and the pair would
    read back as one unparseable line -- losing the result just written.
    """
    with path.open("a+b") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell():
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
        fh.write(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")


def _record(
    job: dict[str, Any],
    job_id: str,
    result: object,
    attempt: int,
    model: str,
    chunk_ids: list[str] | None = None,
    successful_prompt: str | None = None,
) -> dict[str, Any]:
    """One results record: the job's identity, the ids it covers, the answer, and
    the attempt and model that produced it. ``tf-chunks load`` validates each record
    against the ``chunk_ids`` it carries, so a part record loads like any other."""
    record: dict[str, Any] = {"job_id": job_id}
    for key in ("kind", "source_id"):
        if key in job:
            record[key] = job[key]
    if chunk_ids is not None:
        record["chunk_ids"] = chunk_ids
    if "tag_version" in job:
        # The prompt version the answer came from; load refuses any other (VOR-1451).
        record["tag_version"] = job["tag_version"]
    # Carry the whole exported job binding even when a retry writes only a
    # validated subset. The loader compares each returned chunk against this
    # exact export identity and then rechecks the live content before stamping.
    for key in (
        "job_chunk_ids",
        "hashes",
        "input_sha256",
        "recipe_sha256",
        "export_prompt_sha256",
        "job_sha256",
    ):
        if key in job:
            record[key] = job[key]
    record.update(result=result, attempts=attempt, model=model)
    if successful_prompt is not None:
        record["successful_prompt_sha256"] = hashlib.sha256(
            successful_prompt.encode("utf-8")
        ).hexdigest()
        record["successful_prompt_identity_status"] = "computed_actual_attempt"
    return record


def _run_book_job(
    job: dict[str, Any],
    results_path: Path,
    failed_path: Path,
    *,
    call: Callable[[str], str],
    max_attempts: int,
    model: str,
) -> str:
    """Classify one book job: the whole record validates or no record is written,
    and a book still invalid after the last attempt goes to ``.failed``."""
    prompt, errors = job["prompt"], []
    for attempt in range(1, max_attempts + 1):
        try:
            result = extract_json(_checked_stdout(call(prompt)))
            validate_book(result)
        except (ValueError, TagValidationError, ShimError) as exc:
            errors.append(str(exc)[:_ERROR_LIMIT])
            prompt = job["prompt"] + _BOOK_RETRY.format(error=errors[-1])
            continue
        _append_record(
            results_path,
            _record(job, job["job_id"], result, attempt, model, successful_prompt=prompt),
        )
        return "done"
    _append_record(failed_path, {"job_id": job["job_id"], "errors": errors})
    return "failed"


def _run_chunks_job(
    job: dict[str, Any],
    results_path: Path,
    failed_path: Path,
    missing: list[str],
    *,
    call: Callable[[str], str],
    max_attempts: int,
    model: str,
    part: int,
) -> str:
    """Classify one chunk-batch job, accepting the chunk ids that validate.

    A batch is a single answer, so one bad chunk of twenty must not cost the whole
    batch (VOR-1444). Every id is validated on its own and the ids that pass are
    written straight away, so nothing that did validate waits on a retry; the next
    attempt is asked for the ids that are still missing and for no others. A record
    that carries the whole job keeps the job's own id; a record that carries part of
    it is a part record (``<job_id>#p<n>``), numbered after the parts already in the
    results file. Ids that never validate go to ``.failed`` with the errors that
    rejected them.
    """
    pending = list(missing)
    landed: set[str] = set()
    # The model answers under short labels (p1..pN) when the job carries a map
    # (VOR-1451): it garbled long ids under load. Prompts name labels; answers are
    # translated back to chunk ids before validation, so records keep real ids.
    keys: dict[str, str] = dict(job.get("keys") or {})
    labels = {cid: label for label, cid in keys.items()}

    def as_labels(ids: list[str]) -> list[str]:
        return [labels.get(i, i) for i in ids]

    def labelled(text: str) -> str:
        for cid, label in labels.items():
            text = text.replace(cid, label)
        return text

    # A resumed job starts from the ids no record covers: its first ask is the job
    # prompt limited to those ids, with no error it never made (VOR-1448).
    errors: list[str] = []
    # The caller skips a fully covered job, so any id already covered means resumed.
    resumed = len(pending) != len(job["chunk_ids"])
    prompt = _resume_prompt(job["prompt"], as_labels(pending)) if resumed else job["prompt"]
    for attempt in range(1, max_attempts + 1):
        try:
            result = extract_json(_checked_stdout(call(prompt)))
        except (ValueError, ShimError) as exc:
            errors.append(str(exc)[:_ERROR_LIMIT])
        else:
            if keys and isinstance(result, dict):
                result = {keys.get(k, k): v for k, v in result.items()}
            elif keys and isinstance(result, list):
                result = [
                    {**item, "chunk_id": keys.get(str(item.get("chunk_id")), item.get("chunk_id"))}
                    if isinstance(item, dict)
                    else item
                    for item in result
                ]
            accepted, rejected = _split(result, pending)
            if accepted:
                ids = list(accepted)
                whole = set(ids) >= set(job["chunk_ids"])
                job_id = job["job_id"] if whole else f"{job['job_id']}{_PART}{part}"
                _append_record(
                    results_path,
                    _record(
                        job,
                        job_id,
                        accepted,
                        attempt,
                        model,
                        chunk_ids=ids,
                        successful_prompt=prompt,
                    ),
                )
                part += 1
                landed.update(ids)
                pending = [chunk_id for chunk_id in pending if chunk_id not in accepted]
                if not pending:
                    return "done"
            if rejected:
                errors.append("; ".join(rejected)[:_ERROR_LIMIT])
        if pending:
            prompt = _retry_prompt(job["prompt"], as_labels(pending), [labelled(e) for e in errors])
    failure: dict[str, Any] = {"job_id": job["job_id"]}
    if "source_id" in job:
        # The keys the job has, like _record: a job built without a source_id is not
        # given an empty one, so a later pass re-exports it by the same identity.
        failure["source_id"] = job["source_id"]
    failure["chunk_ids"] = pending
    failure["errors"] = errors
    _append_record(failed_path, failure)
    return "partial" if landed else "failed"


def run_lane(
    jobs_path: Path,
    results_path: Path,
    *,
    call: Callable[[str], str],
    max_attempts: int = 2,
    model: str = TF_TAGGER_MODEL,
) -> dict[str, int]:
    """Classify every job in ``jobs_path`` with ``call``, appending each validated
    result to ``results_path`` as it lands.

    ``model`` is stamped into every result so the store can record the tagger
    that actually produced the tags rather than the one configured today. A chunk
    job is finished when its chunk ids are all covered, across its own record and
    its part records; a job that keeps ids after ``max_attempts`` is ``partial``
    when some landed and ``failed`` when none did.
    """
    stats = {"done": 0, "partial": 0, "failed": 0, "skipped": 0}
    covered, parts, unreadable = _read_results(results_path)
    stats["skipped"] = unreadable
    handled: set[str] = set()
    failed_path = Path(str(results_path) + ".failed")
    for line in Path(jobs_path).read_text().splitlines():
        if not line.strip():
            continue
        job = json.loads(line)
        job_id = job["job_id"]
        if job["kind"] == "book":
            missing: list[str] = []
            answered = job_id in covered
        else:
            missing = [cid for cid in job["chunk_ids"] if cid not in covered.get(job_id, set())]
            answered = not missing
        if job_id in handled or answered:
            # A job id repeated in the jobs file is one job, not two.
            stats["skipped"] += 1
            continue
        handled.add(job_id)
        if job["kind"] == "book":
            outcome = _run_book_job(
                job, results_path, failed_path, call=call, max_attempts=max_attempts, model=model
            )
        else:
            outcome = _run_chunks_job(
                job,
                results_path,
                failed_path,
                missing,
                call=call,
                max_attempts=max_attempts,
                model=model,
                part=parts.get(job_id, 0) + 1,
            )
        stats[outcome] += 1
    return stats


def _shim_env() -> dict[str, str]:
    """The environment for the shim, with the repo venv stripped out.

    The shim runs ``python3`` off PATH and needs Python >= 3.14; the repo venv is
    3.12, so both its bin directory (under ``sys.prefix`` or ``VIRTUAL_ENV``) and
    the variables that point back at it are removed. Any inherited opencode config
    override is dropped too, and TOOL_POLICY is passed inline as well as
    in the workdir, so the classifier has no tools whichever source opencode reads
    (VOR-1447). Everything else is inherited.
    """
    roots = [root for root in (sys.prefix, os.environ.get("VIRTUAL_ENV", "")) if root]
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "VIRTUAL_ENV"
        and not key.startswith("UV_")
        and not key.startswith("OPENCODE_CONFIG")
    }
    env["OPENCODE_CONFIG_CONTENT"] = _TOOL_POLICY_JSON
    env["PATH"] = os.pathsep.join(
        entry
        for entry in env.get("PATH", "").split(os.pathsep)
        if entry and not any(entry == root or entry.startswith(root + os.sep) for root in roots)
    )
    return env


def _session_members(sid: int) -> list[int]:
    """Pids whose session id is ``sid``, read from ``/proc/<pid>/stat``."""
    members = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            # Fields after the parenthesised command: state ppid pgrp session ...
            # Bytes, not text: a command name need not be valid UTF-8.
            fields = stat.read_bytes().rpartition(b")")[2].split()
        except OSError:
            continue  # the process exited while we looked
        if len(fields) > 3 and fields[3] == str(sid).encode():
            members.append(int(stat.parent.name))
    return members


def _kill_session(sid: int) -> None:
    """SIGKILL every process in session ``sid``.

    The shim runs `opencode run` in a process group of its own, so killing the
    shim's group misses it; the session is the one id they all share.
    """
    # The leader itself too, so a scan that finds nothing still kills the shim.
    for pid in {sid, *_session_members(sid)}:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


def _run_shim(
    cmd: list[str], *, cwd: Path, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run the shim in its own session; an overrun kills the whole session.

    ``subprocess.run``'s timeout kills only the shim itself, and the model process
    the shim spawned kept running (and holding a model slot) for 1000s+ in the live
    run (VOR-1446). The shim leads a new session, and every process it starts
    stays in that session whatever group it moves to, so the overrun kills the
    session.
    """
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_session(proc.pid)
            # Bounded: a process that escaped the session must not turn the reap
            # into the hang it exists to prevent.
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.communicate(timeout=_REAP_TIMEOUT)
            raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def shim_call(model: str, workdir: Path, shim: Path) -> Callable[[str], str]:
    def call(prompt: str) -> str:
        # The temp workdir or the prompt file can vanish while the lane runs (live,
        # VOR-1445: the classifier itself ran `rm` on its prompt file before tools
        # were denied, VOR-1447): recreate it per call, and never let the prompt
        # file's cleanup end the lane; a removal that lands between the mkdir and
        # the write is a retryable attempt failure like any other shim failure.
        try:
            workdir.mkdir(parents=True, exist_ok=True)
            # `opencode run --auto` approves every tool not explicitly denied, and
            # the classifier used them: it grepped the home directory, queried
            # Postgres and Qdrant, and deleted its own prompt file (VOR-1447). A
            # TOOL_POLICY project config in the workdir leaves it text in, text out.
            # Rewritten every call: the workdir may have been recreated since.
            (workdir / "opencode.json").write_text(_TOOL_POLICY_JSON)
            with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=workdir, delete=False) as fh:
                fh.write(prompt)
        except OSError as exc:
            raise ShimError(f"shim workdir not writable: {exc}") from exc
        try:
            proc = _run_shim(
                [str(shim), model, fh.name],
                cwd=workdir,
                env=_shim_env(),
                timeout=_SHIM_TIMEOUT,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            # A shim that cannot run or overruns is a retryable attempt failure,
            # never an exception that ends the lane.
            raise ShimError(f"shim did not run: {exc}") from exc
        finally:
            Path(fh.name).unlink(missing_ok=True)
        return _checked_stdout(proc.stdout, proc.stderr)

    return call


#: No proxy handler: the bearer key and the chunk text go only to the URL given,
#: never to a proxy an environment happens to set (VOR-1451).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _open(req: urllib.request.Request, timeout: float) -> Any:
    return _OPENER.open(req, timeout=timeout)


def api_call(model: str, url: str, key_env: str) -> Callable[[str], str]:
    """Call an OpenAI-compatible chat-completions endpoint directly (VOR-1451).

    The classifier is text in, text out, so it talks to a model API, never an agent
    CLI: routing it through `opencode run --auto` is what gave it a shell (VOR-1447).
    No tools are offered, temperature is 0 and thinking is off (the thinking pass was
    3.6x slower for a few points of accuracy). The key is read from the environment
    variable ``key_env``; an unset variable sends no Authorization header.
    """

    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise ValueError(f"--api-url must be http or https, got {url!r}")

    def call(prompt: str) -> str:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": _API_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(key_env, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, json.dumps(body).encode(), headers)
        try:
            with _open(req, timeout=_SHIM_TIMEOUT) as resp:
                choice = json.loads(resp.read())["choices"][0]
                content = choice["message"]["content"]
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ShimError(f"api call failed: {exc}") from exc
        if choice.get("finish_reason") == "length":
            # A cut-off answer is unparseable; say why instead of "no JSON".
            raise ShimError(f"completion truncated at max_tokens={_API_MAX_TOKENS}")
        # The same completion sentinel the shim prints, so one parser reads both.
        return f"{content}\nSHIM-DONE exit=0\n"

    return call


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path)
    ap.add_argument("results", type=Path)
    ap.add_argument("--model", default=TF_TAGGER_MODEL)
    ap.add_argument(
        "--api-url",
        default=os.environ.get("VORAG_TF_TAGGER_URL", ""),
        help="OpenAI-compatible chat-completions URL (default $VORAG_TF_TAGGER_URL); "
        "without one the lane falls back to the agent shim given with --shim",
    )
    ap.add_argument(
        "--shim",
        type=Path,
        default=None,
        help="path to an agent shim script (called as SHIM MODEL PROMPT_FILE); "
        "used only when no API URL is given, and needs --model provider/model",
    )
    ap.add_argument("--api-key-env", default="LLAMA_SWAP_API_KEY")
    args = ap.parse_args()
    if not args.api_url and args.shim is None:
        ap.error(
            "no --api-url / $VORAG_TF_TAGGER_URL and no --shim: pass --api-url, or "
            "--shim PATH with --model provider/model for the shim fallback"
        )
    if not args.api_url and "/" not in args.model:
        ap.error("the shim fallback needs --model provider/model")
    if args.api_url:
        call = api_call(args.model, args.api_url, args.api_key_env)
        stats = run_lane(args.jobs, args.results, call=call, model=args.model)
    else:
        # outside the repo: opencode watches its cwd
        with tempfile.TemporaryDirectory(prefix="tf-lane-", ignore_cleanup_errors=True) as workdir:
            stats = run_lane(
                args.jobs,
                args.results,
                call=shim_call(args.model, Path(workdir), args.shim),
                model=args.model,
            )
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
