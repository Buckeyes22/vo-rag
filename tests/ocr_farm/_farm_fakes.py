"""Fakes for the OCR-farm coordinator: Garage, asyncpg pool, ssh/scp recorder,
plus the book/host helpers every farm test shares.

Nothing here touches a real host, Garage or Postgres.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vo_rag.ocr_farm.coordinator import CommandResult, FarmItem, resolve_hosts
from vo_rag.ocr_farm.hosts import DEFAULT_HOSTS, FarmHost

BUCKET = "vorag-artifacts"


def farm_items(pages: list[int], prefix: str = "book") -> tuple[list[FarmItem], dict[str, bytes]]:
    """One item per page count, with stand-in (not real PDF) source bytes.

    Returns ``(items, garage_objects)``. For a book the coordinator must
    split, use ``pdf_item`` (a real PDF).
    """
    items: list[FarmItem] = []
    objects: dict[str, bytes] = {}
    for n, page_count in enumerate(pages):
        data = b"%PDF-1.4\n% " + f"{prefix}-{n}".encode() + b"\n"
        key = f"uploads/{prefix}-{n}.pdf"
        objects[key] = data
        items.append(
            FarmItem(
                f"ENV{prefix}{n:03d}",
                key,
                hashlib.sha256(data).hexdigest(),
                f"{prefix} {n}",
                page_count,
            )
        )
    return items, objects


def pdf_bytes(pages: int, *, text: bool = True) -> bytes:
    """A real PDF (pymupdf); page ``n`` carries the text ``page n`` if ``text``."""
    import fitz  # type: ignore[import-untyped]

    document = fitz.open()
    for n in range(1, pages + 1):
        page = document.new_page(width=200, height=300)
        if text:
            page.insert_text((20, 40), f"page {n}")
    data: bytes = document.tobytes()
    document.close()
    return data


def pdf_item(pages: int, *, n: int = 0, text: bool = True) -> tuple[FarmItem, dict[str, bytes]]:
    """A book whose source is a real ``pages``-page PDF: ``(item, garage_objects)``."""
    data = pdf_bytes(pages, text=text)
    key = f"uploads/split-{n}.pdf"
    item = FarmItem(f"ENVSPLIT{n}", key, hashlib.sha256(data).hexdigest(), f"Split {n}", pages)
    return item, {key: data}


async def resolved_hosts(
    recorder: Recorder, hosts: tuple[FarmHost, ...] = DEFAULT_HOSTS
) -> tuple[FarmHost, ...]:
    return await resolve_hosts(hosts, recorder, state_path=None)


class FakeClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeGarage:
    def __init__(self, objects: dict[str, bytes] | None = None, bucket: str = "vorag-artifacts"):
        self.bucket = bucket
        self.objects: dict[tuple[str, str], bytes] = {
            (bucket, key): value for key, value in (objects or {}).items()
        }
        self.puts: list[tuple[str, str, str | None]] = []
        self.gets: list[tuple[str, str]] = []
        self.heads: list[tuple[str, str]] = []

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        self.gets.append((bucket, key))
        try:
            return self.objects[(bucket, key)]
        except KeyError:
            raise FakeClientError("NoSuchKey") from None

    async def object_metadata(self, bucket: str, key: str) -> Any:
        self.heads.append((bucket, key))
        data = self.objects.get((bucket, key))
        if data is None:
            return None
        return type("Meta", (), {"size": len(data), "content_type": None, "etag": None})()

    async def put_bytes(
        self, bucket: str, key: str, data: bytes, content_type: str | None = None
    ) -> None:
        self.puts.append((bucket, key, content_type))
        self.objects[(bucket, key)] = data


class FakeQdrant:
    """``AsyncQdrantClient.facet`` stand-in: ``source_book``/``source_id`` values."""

    def __init__(
        self,
        source_books: list[str] | None = None,
        source_ids: list[str] | None = None,
    ) -> None:
        self.source_books = list(source_books or [])
        self.source_ids = list(source_ids or [])
        self.facet_calls: list[dict[str, Any]] = []

    async def facet(
        self, collection_name: str, key: str, *, limit: int = 10, exact: bool = False
    ) -> Any:
        self.facet_calls.append(
            {"collection_name": collection_name, "key": key, "limit": limit, "exact": exact}
        )
        by_key = {"source_book": self.source_books, "source_id": self.source_ids}
        values = by_key.get(key, [])[:limit]
        return SimpleNamespace(hits=[SimpleNamespace(value=v, count=1) for v in values])


class FakePool:
    def __init__(
        self,
        failed_rows: list[dict[str, Any]] | None = None,
        completed_titles: list[str] | None = None,
    ) -> None:
        self.failed_rows = failed_rows or []
        self.completed_titles = completed_titles or []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((sql, args))
        if "status = 'completed'" in sql:
            return [{"title": t} for t in self.completed_titles]
        return list(self.failed_rows)

    async def execute(self, sql: str, *args: Any) -> str:
        self.execute_calls.append((sql, args))
        return "UPDATE 1"


_JOB_RE = re.compile(r"jobs[\\/]([0-9a-f]{64})\.pdf")
_JSON_RE = re.compile(r"jobs/([0-9a-f]{64})\.json$")
_PID_READ_RE = re.compile(r"^(?:cat |type ).*jobs[\\/]([0-9a-f]{64})\.pid\"?$")


@dataclass
class Recorder:
    """Records every ssh/scp argv and simulates the remote side.

    - ``ssh <alias> "... ocr_worker.py jobs/<sha>.pdf ..."`` sleeps in
      proportion to the book's pages and tracks per-host concurrency;
      ``exit_codes[sha]`` makes that book's worker fail, ``host_exit_codes``
      makes every worker on that host fail, and ``hang_shas`` makes that
      book's worker never return (until cancelled).
    - ``ssh <alias> "... ocr_worker.py --check"`` answers ``check_by_host``
      (default ``check_output``).
    - ``ssh <alias> "cat|type ...<sha>.pid"`` answers ``pid_output``; the
      first ``pid_missing_reads[sha]`` reads of that book's file find no file,
      and ``pid_reads[sha]`` counts the reads, each book on its own.
    - ``expire_shas`` / ``expire_hosts``: that worker's OCR budget runs out on
      cue -- it raises the ``TimeoutError`` that ``asyncio.wait_for`` raises
      in the coordinator's ``_call``, so a test's timeouts don't race the
      wall clock and the jobs that should succeed keep a budget of several
      seconds (order it against other events with ``hook``).
    - ``scp <local> <alias>:<...>/jobs/<job>.pdf`` reads the shipped file:
      when it is a real PDF, ``job_pages[job]`` gets its page count and
      ``job_texts[job]`` each page's text (a split book's parts are real
      sub-PDFs, so a test can see exactly which pages each part carried).
    - ``scp <alias>:<...>/jobs/<job>.json <local>`` writes a DoclingDocument-shaped
      JSON file to ``<local>`` with ``pages`` 1..n (n from ``job_pages``, else
      ``pages_by_sha``) and one text per page (``job_texts``), each with its
      ``prov`` page -- enough for ``fake_merge`` to merge like docling-core.
    - ``ssh win "echo %TEMP%"`` answers with a Windows temp path.
    - ``fail_commands`` (needle -> exit code), ``hang_commands`` (needles)
      and ``raise_commands`` (needle -> exception) apply to any command.
    - ``hook(argv)``, when set, is awaited first for every command (after
      it is recorded): tests use it to order events without timers.
    """

    pages_by_sha: dict[str, int] = field(default_factory=dict)
    exit_codes: dict[str, int] = field(default_factory=dict)
    host_exit_codes: dict[str, int] = field(default_factory=dict)
    hang_shas: set[str] = field(default_factory=set)
    hang_hosts: set[str] = field(default_factory=set)
    delay_by_sha: dict[str, float] = field(default_factory=dict)
    expire_shas: set[str] = field(default_factory=set)
    expire_hosts: set[str] = field(default_factory=set)
    #: (host, job) of every worker whose budget expired on cue.
    expired: list[tuple[str, str]] = field(default_factory=list)
    #: sha -> the first N reads of that book's PID file find no file (the
    #: worker hasn't written it yet). Books not listed answer on read 1.
    pid_missing_reads: dict[str, int] = field(default_factory=dict)
    #: sha -> PID-file reads so far, scoped per book so that concurrently
    #: timing-out books don't consume each other's reads.
    pid_reads: dict[str, int] = field(default_factory=dict)
    fail_commands: dict[str, int] = field(default_factory=dict)
    hang_commands: list[str] = field(default_factory=list)
    raise_commands: dict[str, BaseException] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)
    worker_starts: list[tuple[str, str]] = field(default_factory=list)
    cancelled: list[list[str]] = field(default_factory=list)
    running: dict[str, int] = field(default_factory=dict)
    peak: dict[str, int] = field(default_factory=dict)
    #: Workers running at once across every host, and the most seen.
    running_total: int = 0
    peak_total: int = 0
    #: job (remote file stem) -> pages / per-page text of the PDF shipped for it.
    job_pages: dict[str, int] = field(default_factory=dict)
    job_texts: dict[str, list[str]] = field(default_factory=dict)
    check_output: str = "torch 2.12.0+cu130 cuda True NVIDIA GeForce RTX 5070"
    check_by_host: dict[str, str] = field(default_factory=dict)
    pid_output: str = "4242\r\n"
    hook: Callable[[list[str]], Awaitable[None]] | None = None

    async def _hang(self, argv: list[str]) -> CommandResult:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.append(argv)
            raise
        raise AssertionError("unreachable")

    async def __call__(self, argv: Sequence[str]) -> CommandResult:
        argv = list(argv)
        self.calls.append(argv)
        if self.hook is not None:
            await self.hook(argv)
        joined = " ".join(argv)
        for needle, exc in self.raise_commands.items():
            if needle in joined:
                raise exc
        for needle, code in self.fail_commands.items():
            if needle in joined:
                return CommandResult(code, "", f"forced failure for {needle}")
        if any(needle in joined for needle in self.hang_commands):
            return await self._hang(argv)
        if argv[0] == "ssh":
            alias, command = argv[-2], argv[-1]
            if command == "echo %TEMP%":
                return CommandResult(0, "C:\\Users\\Demo\\AppData\\Local\\Temp\r\n", "")
            if "ocr_worker.py" in command and "--check" in command:
                output = self.check_by_host.get(alias, self.check_output)
                return CommandResult(0 if " cuda True" in output else 3, output + "\n", "")
            pid_match = _PID_READ_RE.match(command)
            if pid_match:
                sha = pid_match.group(1)
                self.pid_reads[sha] = self.pid_reads.get(sha, 0) + 1
                if self.pid_reads[sha] <= self.pid_missing_reads.get(sha, 0):
                    return CommandResult(1, "", "No such file or directory")
                return CommandResult(0, self.pid_output, "")
            match = _JOB_RE.search(command)
            if "ocr_worker.py" in command and match:
                sha = match.group(1)
                self.worker_starts.append((alias, sha))
                self.running[alias] = self.running.get(alias, 0) + 1
                self.peak[alias] = max(self.peak.get(alias, 0), self.running[alias])
                self.running_total += 1
                self.peak_total = max(self.peak_total, self.running_total)
                try:
                    if sha in self.expire_shas or alias in self.expire_hosts:
                        self.expired.append((alias, sha))
                        raise TimeoutError
                    if sha in self.hang_shas or alias in self.hang_hosts:
                        return await self._hang(argv)
                    pages = self.pages_by_sha.get(sha) or self.job_pages.get(sha, 1)
                    delay = self.delay_by_sha.get(sha, 0.002 * pages / 10)
                    await asyncio.sleep(delay)
                finally:
                    self.running[alias] -= 1
                    self.running_total -= 1
                code = self.host_exit_codes.get(alias, self.exit_codes.get(sha, 0))
                out = "" if code else "ocr_worker: ok status=success gpu_peak_bytes=4096\n"
                return CommandResult(code, out, "")
            return CommandResult(0, "", "")
        if argv[0] == "scp":
            src, dst = argv[-2], argv[-1]
            if ":" in src.split("/", 1)[0] and src.endswith(".json"):
                match = _JSON_RE.search(src)
                job = match.group(1) if match else ""
                Path(dst).write_text(json.dumps(self._docling_json(job, src)), encoding="utf-8")
            elif ":" in dst.split("/", 1)[0] and dst.endswith(".pdf"):
                job = dst.rsplit("/", 1)[-1].removesuffix(".pdf")
                self._read_shipped_pdf(job, Path(src))
            return CommandResult(0, "", "")
        raise AssertionError(f"unexpected command {argv!r}")

    def _read_shipped_pdf(self, job: str, path: Path) -> None:
        data = path.read_bytes()
        if not data.startswith(b"%PDF-"):
            return
        import fitz  # type: ignore[import-untyped, unused-ignore]

        try:
            with fitz.open(stream=data, filetype="pdf") as pdf:
                self.job_pages[job] = int(pdf.page_count)
                self.job_texts[job] = [page.get_text().strip() for page in pdf]
        except Exception:  # noqa: BLE001 - a fake "%PDF-" header without a body
            return

    def _docling_json(self, job: str, src: str) -> dict[str, Any]:
        count = self.job_pages.get(job) or self.pages_by_sha.get(job, 0)
        texts = self.job_texts.get(job, [])
        return {
            "schema_name": "DoclingDocument",
            "name": job,
            "src": src,
            "pages": {
                str(n): {"page_no": n, "size": {"width": 612.0, "height": 792.0}}
                for n in range(1, count + 1)
            },
            "texts": [
                {"text": text, "prov": [{"page_no": n}]} for n, text in enumerate(texts, start=1)
            ],
        }

    def commands_for(self, alias: str) -> list[str]:
        return [argv[-1] for argv in self.calls if argv[0] == "ssh" and argv[-2] == alias]

    def scp_calls(self) -> list[tuple[str, str]]:
        return [(argv[-2], argv[-1]) for argv in self.calls if argv[0] == "scp"]


def fake_merge(part_paths: Sequence[Path], out_path: Path, name: str) -> list[int]:
    """``coordinator.merge_docling_parts`` for the recorder's part JSON.

    Shifts pages and text provenance exactly as docling-core's
    ``DoclingDocument.concatenate`` does: each part moves by
    ``max page so far - its first page + 1``. Returns the merged page numbers.
    """
    merged: dict[str, Any] = {
        "schema_name": "DoclingDocument",
        "name": name,
        "pages": {},
        "texts": [],
    }
    max_page = 0
    for path in part_paths:
        part = json.loads(path.read_text(encoding="utf-8"))
        numbers = [int(n) for n in part.get("pages", {})]
        delta = max_page - min(numbers) + 1 if numbers else 0
        for n, page in part["pages"].items():
            new = int(n) + delta
            merged["pages"][str(new)] = {**page, "page_no": new}
        for text in part.get("texts", []):
            prov = [{**p, "page_no": p["page_no"] + delta} for p in text["prov"]]
            merged["texts"].append({**text, "prov": prov})
        if numbers:
            max_page = max(int(n) for n in merged["pages"])
    out_path.write_text(json.dumps(merged), encoding="utf-8")
    return sorted(int(n) for n in merged["pages"])
