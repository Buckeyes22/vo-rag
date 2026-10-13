"""OCR-farm coordinator: runs on localhost, does all the I/O (VOR-1427).

Workers are thin (a pinned docling/torch venv plus ``ocr_worker.py``) and hold
no credentials. This module:

1. ``plan_backlog`` -- picks the failed scanned envelopes that still need OCR
   (minus titles that already have a completed version), reads each source
   from Garage, hashes it, counts pages and flags books whose
   ``ocr/<sha256>/docling.json`` already exists.
2. ``bootstrap_worker`` -- creates a host's venv from ``worker/requirements.txt``
   with the CUDA wheel index and verifies ``torch.cuda.is_available()``.
3. ``run_farm`` -- splits every book over ``SPLIT_PAGES`` pages into
   contiguous page-range sub-PDFs ("parts"), schedules the jobs (whole books
   and parts) longest-first across every free worker slot (the first host in
   ``hosts`` -- the fastest GPU -- is preferred), ships each PDF over scp,
   runs the worker over ssh, copies the JSON back, merges a split book's part
   JSONs (``merge_docling_parts``) and stores the book's JSON in Garage at
   ``ocr/<sha256>/docling.json``. The book_rag parse
   step then maps that JSON instead of OCR-ing again
   (``ParserRouter._cache_entry`` / ``_from_cache``); the router tells farm
   output apart from off-box native pre-parses under the same keys by the
   ``"pipeline": "ocr"`` in ``meta.json``. Optionally re-queues each envelope.

ssh/scp go through an injected ``Runner`` so every path is testable with a
recorder; nothing here imports torch, and docling-core is imported lazily,
only to merge a split book's parts.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from vo_rag.core.corpus.titles import is_already_replaced, normalize_title
from vo_rag.ocr_farm.hosts import (
    JOBS_DIR,
    REQUIREMENTS,
    RESOLVE_TEMP_COMMAND,
    WORKER_SCRIPT,
    FarmHost,
    validate_remote_timeout,
)

log = logging.getLogger(__name__)

WORKER_DIR = Path(__file__).resolve().parent / "worker"
DEFAULT_BUCKET = "vorag-artifacts"

#: The 3 image-only EPUB envelopes found in planning (VOR-1427); they failed
#: before image-only EPUBs were routed to OCR, with other error texts.
IMAGE_ONLY_EPUB_ENVELOPES: tuple[str, ...] = (
    "01M3AZDRK89N2AEX26QQJCNWK2",
    "01M3AYTR6J5R48HAKG6QH0Y5Y9",
    "01M3AZAWMR0YD10D39CG3QWBE8",
)

FAILED_SCANNED_SQL = """
SELECT id,
       metadata->>'title' AS title,
       raw_content->>'garage_key' AS garage_key,
       updated_at
FROM intake_envelopes
WHERE status = 'failed'
  AND (
        -- Legacy error text written by the removed GLM-OCR forward path
        -- (VOR-1427); kept so old failed envelopes are still found and
        -- retried through the OCR farm.
        error::text LIKE '%GLM-OCR forward path%'
     OR error::text LIKE '%covered content on only%'
        -- A GPU-free dispatcher lane (VORAG_BOOK_RAG_DISPATCHER_ONLY_EXTENSIONS)
        -- that meets an image-only ebook with no farm OCR yet fails it with
        -- OcrRequiresGpu; the farm OCRs it and re-queues it (VOR-1427).
     OR error::text LIKE '%OCR requires a CUDA GPU%'
     OR id = ANY($1::text[])
  )
ORDER BY updated_at DESC, id DESC
"""

#: Completed titles from Postgres. The other half of "completed" is the
#: books collection's ``source_book`` values (``completed_source_books``).
#: ``sources`` is NOT used: it's legacy migration data nothing writes.
COMPLETED_TITLES_SQL = """
SELECT metadata->>'title' AS title FROM intake_envelopes WHERE status = 'completed'
"""

BOOKS_RAW_COLLECTION = "vo_rag_books_raw_v1"
FACET_INITIAL_LIMIT = 10_000
FACET_MAX_LIMIT = 1_000_000

#: Prefix Qdrant's ``source_id`` payload uses for corpus-ingested book bytes
#: (VOR-1427): ``f"{INGESTED_SOURCE_ID_PREFIX}{sha256}"``.
INGESTED_SOURCE_ID_PREFIX = "vo-rag-corpus-sha256:"
ALREADY_INGESTED_PROBLEM = "already_ingested"

#: The ``status='failed'`` guard must stay: never resurrect an envelope some
#: other path already moved on. VOR-1434: a requeued envelope must not keep a
#: streaming ingest's ``metadata.metrics.active_stream`` flag (the dispatcher's
#: claim clears it the same way); metadata without a ``metrics`` object is left alone.
REQUEUE_SQL = (
    "UPDATE intake_envelopes SET status='received', error=NULL, updated_at=NOW(), "
    "metadata = CASE WHEN jsonb_typeof(metadata->'metrics') = 'object' "
    "THEN jsonb_set(metadata, '{metrics,active_stream}', 'false'::jsonb) "
    "ELSE metadata END "
    "WHERE id=$1 AND status='failed'"
)


def ocr_key(sha256: str) -> str:
    return f"ocr/{sha256}/docling.json"


def ocr_meta_key(sha256: str) -> str:
    return f"ocr/{sha256}/meta.json"


# ---------------------------------------------------------------------------
# seams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class Runner(Protocol):
    def __call__(self, argv: Sequence[str], /) -> Awaitable[CommandResult]: ...


class GarageLike(Protocol):
    async def get_bytes(self, bucket: str, key: str) -> bytes: ...

    async def object_metadata(self, bucket: str, key: str) -> Any: ...

    async def put_bytes(
        self, bucket: str, key: str, data: bytes, content_type: str | None = None
    ) -> None: ...


class PoolLike(Protocol):
    async def fetch(self, query: str, *args: Any) -> Any: ...

    async def execute(self, query: str, *args: Any) -> Any: ...


class FacetClient(Protocol):
    """The slice of ``AsyncQdrantClient`` ``completed_source_books`` needs."""

    async def facet(
        self, collection_name: str, key: str, *, limit: int = ..., exact: bool = ...
    ) -> Any: ...


async def subprocess_runner(argv: Sequence[str], /) -> CommandResult:
    """Run a local ssh/scp command; stdin is closed so nothing can prompt.

    The child is killed if the call is cancelled (including by a timeout).
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await proc.communicate()
    except BaseException:
        # Timeout (asyncio.wait_for) or cancellation: never leave the local
        # ssh/scp child running. The remote side is the caller's job.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        raise
    return CommandResult(
        int(proc.returncode if proc.returncode is not None else -1),
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )


def ssh_argv(host: FarmHost, command: str) -> list[str]:
    return ["ssh", host.alias, command]


def scp_argv(src: str, dst: str) -> list[str]:
    return ["scp", "-q", src, dst]


class FarmError(RuntimeError):
    """The farm cannot start (no usable host, unresolvable host path)."""


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FarmItem:
    envelope_id: str
    garage_key: str
    sha256: str
    title: str
    pages: int
    fmt: str = "pdf"
    cached: bool = False
    #: Set by ``plan_backlog`` when the book can't be shipped at all
    #: (``missing_garage_key``, ``missing_source_object``, ``unsupported_format``,
    #: ``unreadable_source``, ``no_pages``) -- ``run_farm`` reports it as
    #: ``failed(<problem>)`` -- or to ``duplicate_sha256`` for an envelope with
    #: the same bytes as ``duplicate_of``, which is re-queued with its twin, or
    #: to ``already_ingested`` when ``duplicate_of`` is the matching
    #: ``source_id`` already present in the books collection (never OCR'd,
    #: never re-queued: those bytes are already in the corpus).
    problem: str | None = None
    duplicate_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FarmResult:
    envelope_id: str
    title: str
    sha256: str
    #: The host that ran it; for a split book, every host that ran a part
    #: (comma-separated, in part order).
    host: str | None
    pages: int
    #: Worker-slot time: the sum of the book's job durations (every part of a
    #: split book), plus the merge and upload.
    seconds: float
    #: ``ok`` | ``cached`` | ``duplicate`` | ``failed(<reason>)``
    status: str
    requeued: bool = False

    @property
    def failed(self) -> bool:
        return self.status.startswith("failed(")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FarmReport:
    results: list[FarmResult]
    host_errors: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "books": len(self.results),
            "ok": sum(1 for r in self.results if r.status == "ok"),
            "cached": sum(1 for r in self.results if r.status == "cached"),
            "duplicate": sum(1 for r in self.results if r.status == "duplicate"),
            "failed": sum(1 for r in self.results if r.failed),
            "requeued": sum(1 for r in self.results if r.requeued),
            "pages": sum(r.pages for r in self.results if r.status == "ok"),
            "seconds": round(sum(r.seconds for r in self.results), 1),
            "host_errors": dict(self.host_errors),
        }


@dataclass(frozen=True)
class BootstrapResult:
    host: str
    ok: bool
    check: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# page counting / EPUB -> PDF
# ---------------------------------------------------------------------------


def count_epub_spine_images(data: bytes) -> int:
    """Pages of an image-only EPUB: its spine images, in reading order, found
    by the same helper the parser uses to synthesize the OCR PDF."""
    from vo_rag.core.parsing.parsers.epub import _spine_ordered_image_bytes

    return len(_spine_ordered_image_bytes(_read_epub(data)))


def epub_to_pdf_bytes(data: bytes) -> bytes:
    """Synthesize the page-image PDF the worker OCRs, exactly as the parser's
    image-only path does (``_write_pdf_from_images``)."""
    from vo_rag.core.parsing.parsers.epub import (
        _spine_ordered_image_bytes,
        _write_pdf_from_images,
    )

    images = _spine_ordered_image_bytes(_read_epub(data))
    if not images:
        raise ValueError("image-only ebook has no spine images")
    with tempfile.TemporaryDirectory(prefix="vorag-ocr-farm-epub-") as tmp:
        pdf_path = Path(tmp) / "synthesized.pdf"
        _write_pdf_from_images(images, pdf_path)
        return pdf_path.read_bytes()


def _read_epub(data: bytes) -> Any:
    from ebooklib import epub  # type: ignore[import-not-found, unused-ignore]

    with tempfile.TemporaryDirectory(prefix="vorag-ocr-farm-epub-") as tmp:
        path = Path(tmp) / "book.epub"
        path.write_bytes(data)
        return epub.read_epub(str(path))


def count_pages(data: bytes, name: str = "") -> tuple[str, int]:
    """``(fmt, pages)`` for a source file: ``pdf`` via pymupdf, ``epub`` as its
    spine-image count, ``("unknown", 0)`` otherwise."""
    lowered = name.lower()
    if b"%PDF-" in data[:1024] or lowered.endswith(".pdf"):
        import fitz  # type: ignore[import-untyped]

        with fitz.open(stream=data, filetype="pdf") as pdf:
            return "pdf", int(pdf.page_count)
    if data[:4] == b"PK\x03\x04" and (
        lowered.endswith(".epub") or b"application/epub+zip" in data[:256]
    ):
        return "epub", count_epub_spine_images(data)
    return "unknown", 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


async def _facet_values(qdrant: FacetClient, collection: str, key: str) -> set[str]:
    """Every value of ``key`` in ``collection`` (a keyword facet).

    Qdrant's facet API has no cursor, so this pages by growing the limit
    until a response comes back short of it; a response still full at
    ``FACET_MAX_LIMIT`` raises rather than silently truncating.
    """
    limit = FACET_INITIAL_LIMIT
    while True:
        response = await qdrant.facet(collection_name=collection, key=key, limit=limit, exact=False)
        hits = list(getattr(response, "hits", None) or [])
        if len(hits) < limit:
            return {str(hit.value) for hit in hits if str(hit.value or "").strip()}
        if limit >= FACET_MAX_LIMIT:
            raise FarmError(f"{collection} {key} facet truncated at {limit} values")
        limit = min(limit * 4, FACET_MAX_LIMIT)


async def completed_source_books(
    qdrant: FacetClient, collection: str = BOOKS_RAW_COLLECTION
) -> set[str]:
    """Every ``source_book`` value in the books collection (a keyword facet).

    See ``_facet_values`` for the growing-limit paging and truncation guard.
    """
    return await _facet_values(qdrant, collection, "source_book")


async def ingested_source_ids(
    qdrant: FacetClient, collection: str = BOOKS_RAW_COLLECTION
) -> set[str]:
    """Every ``source_id`` value in the books collection (a keyword facet).

    Used by ``plan_backlog`` to exclude a failed envelope whose source bytes
    (``f"{INGESTED_SOURCE_ID_PREFIX}{sha256}"``) are already ingested under
    some *other* envelope -- exact hash matching, independent of the title
    match (an untitled envelope can't be matched by title at all). See
    ``_facet_values`` for the growing-limit paging and truncation guard.
    """
    return await _facet_values(qdrant, collection, "source_id")


async def plan_backlog(
    pool: PoolLike,
    garage: GarageLike,
    qdrant: FacetClient,
    *,
    bucket: str = DEFAULT_BUCKET,
    collection: str = BOOKS_RAW_COLLECTION,
) -> list[FarmItem]:
    """The failed scanned books that still need OCR, one item per book.

    Read-only: two SELECTs, one Qdrant facet, Garage HEAD/GETs.

    - "Completed" titles are ``intake_envelopes`` (``status='completed'``)
      plus the books collection's ``source_book`` values.
    - Repeated failures of the same book (same normalized title) keep only
      the most recent envelope; titles that already have a completed version
      are dropped (``is_already_replaced``); untitled envelopes are kept.
    - An envelope with the same bytes (sha256) as one already kept is
      reported as ``problem="duplicate_sha256"`` with ``duplicate_of`` set; it
      isn't OCR'd again, and ``run_farm`` re-queues it with its twin.
    - An envelope whose bytes (sha256) are already ingested under some other
      envelope -- ``f"{INGESTED_SOURCE_ID_PREFIX}{sha256}"`` is one of the
      books collection's ``source_id`` values (``ingested_source_ids``) -- is
      reported as ``problem="already_ingested"`` with ``duplicate_of`` set to
      that ``source_id``. This hash match is exact and runs in addition to
      the title match above (it also catches the untitled envelopes the title
      match can't); such a book is never OCR'd and never re-queued.
    - Books whose farm OCR already exists are kept with ``cached=True`` so
      ``run_farm`` can still re-queue them.
    """
    rows = await pool.fetch(FAILED_SCANNED_SQL, list(IMAGE_ONLY_EPUB_ENVELOPES))
    completed = {
        normalize_title(str(row["title"] or "")) for row in await pool.fetch(COMPLETED_TITLES_SQL)
    }
    completed |= {normalize_title(v) for v in await completed_source_books(qdrant, collection)}
    completed.discard("")
    ingested_ids = await ingested_source_ids(qdrant, collection)

    seen_titles: set[str] = set()
    kept_by_sha: dict[str, str] = {}
    items: list[FarmItem] = []
    for row in rows:
        title = str(row["title"] or "").strip()
        normalized = normalize_title(title)
        if normalized:
            if normalized in seen_titles:
                continue
            seen_titles.add(normalized)
            if is_already_replaced(title, completed):
                continue
        item = await _plan_item(str(row["id"]), title, row["garage_key"], garage, bucket)
        if item.sha256:
            source_id = f"{INGESTED_SOURCE_ID_PREFIX}{item.sha256}"
            if source_id in ingested_ids:
                item = replace(item, problem=ALREADY_INGESTED_PROBLEM, duplicate_of=source_id)
            else:
                twin = kept_by_sha.get(item.sha256)
                if twin is None:
                    kept_by_sha[item.sha256] = item.envelope_id
                else:
                    item = replace(item, problem=DUPLICATE_PROBLEM, duplicate_of=twin)
        items.append(item)
    return items


async def _plan_item(
    envelope_id: str, title: str, garage_key: Any, garage: GarageLike, bucket: str
) -> FarmItem:
    key = str(garage_key or "")
    if not key:
        return FarmItem(envelope_id, "", "", title, 0, problem="missing_garage_key")
    if await garage.object_metadata(bucket, key) is None:
        return FarmItem(envelope_id, key, "", title, 0, problem="missing_source_object")
    data = await garage.get_bytes(bucket, key)
    sha = await asyncio.to_thread(lambda: hashlib.sha256(data).hexdigest())
    try:
        fmt, pages = await asyncio.to_thread(count_pages, data, key)
    except Exception:  # noqa: BLE001 - one unreadable book must not abort the plan
        log.exception("ocr_farm.unreadable_source envelope_id=%s key=%s", envelope_id, key)
        return FarmItem(envelope_id, key, sha, title, 0, problem="unreadable_source")
    if fmt == "unknown":
        return FarmItem(envelope_id, key, sha, title, 0, fmt=fmt, problem="unsupported_format")
    if pages <= 0:
        return FarmItem(envelope_id, key, sha, title, 0, fmt=fmt, problem="no_pages")
    cached = await garage.object_metadata(bucket, ocr_key(sha)) is not None
    return FarmItem(envelope_id, key, sha, title, pages, fmt=fmt, cached=cached)


# ---------------------------------------------------------------------------
# hosts
# ---------------------------------------------------------------------------


async def resolve_hosts(
    hosts: Sequence[FarmHost], runner: Runner, *, state_path: Path | None
) -> tuple[FarmHost, ...]:
    """Fill each host's ``scp_workdir``.

    Linux workdirs are used as-is. A Windows ``%TEMP%`` workdir is resolved
    once with ``ssh <alias> "echo %TEMP%"`` and stored (forward slashes) in
    ``state_path``, which later runs read instead of asking again.
    """
    state: dict[str, str] = {}
    if state_path is not None and state_path.is_file():
        state = dict(json.loads(state_path.read_text(encoding="utf-8")))
    changed = False
    resolved: list[FarmHost] = []
    for host in hosts:
        if host.scp_workdir:
            resolved.append(host)
        elif not host.is_windows:
            resolved.append(host.with_scp_workdir(host.workdir))
        elif host.alias in state:
            resolved.append(host.with_scp_workdir(state[host.alias]))
        else:
            result = await runner(ssh_argv(host, RESOLVE_TEMP_COMMAND))
            temp = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
            if result.returncode != 0 or not temp or "%" in temp:
                raise FarmError(
                    f"{host.alias}: cannot resolve %TEMP% (exit {result.returncode}): "
                    f"{(result.stderr or result.stdout).strip()[-300:]}"
                )
            scp_workdir = host.workdir.replace("%TEMP%", temp).replace("\\", "/")
            state[host.alias] = scp_workdir
            changed = True
            resolved.append(host.with_scp_workdir(scp_workdir))
    if changed and state_path is not None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return tuple(resolved)


def _tail(result: CommandResult) -> str:
    return (result.stderr or result.stdout).strip()[-500:]


@dataclass(frozen=True)
class BootstrapTimeouts:
    """Per-command bounds for ``bootstrap_worker``, in seconds.

    ``install_s`` covers ``uv venv`` and ``uv pip install`` each (the first
    CUDA torch download is several GB); ``check_s`` the CUDA check;
    ``command_s`` mkdir and the scp copies. The long steps enforce their bound
    on the remote side (``FarmHost._bounded``); the local ``wait_for`` gets
    ``backstop_s`` more, so it only fires if the remote limit didn't.
    """

    command_s: float = 300.0
    install_s: float = 3600.0
    check_s: float = 300.0
    backstop_s: float = 90.0

    def __post_init__(self) -> None:
        # These three become remote time limits (FarmHost._bounded).
        validate_remote_timeout(self.install_s, "BootstrapTimeouts.install_s")
        validate_remote_timeout(self.check_s, "BootstrapTimeouts.check_s")
        validate_remote_timeout(self.command_s, "BootstrapTimeouts.command_s")


def _copy_steps(host: FarmHost, *names: str) -> list[tuple[str, list[str]]]:
    return [
        (f"scp {name}", scp_argv(str(WORKER_DIR / name), host.scp_target(name)))
        for name in (*names, *host.support_files())
    ]


async def bootstrap_worker(
    host: FarmHost,
    *,
    runner: Runner,
    timeouts: BootstrapTimeouts = BootstrapTimeouts(),  # noqa: B008 - frozen, immutable default
) -> BootstrapResult:
    """Create the worker venv from ``requirements.txt`` (CUDA wheel index) and
    verify ``torch.cuda.is_available()`` with ``ocr_worker.py --check``.

    ``uv venv`` / ``uv pip install`` / the CUDA check carry their own remote
    time limit, so a hung step kills itself on the host (nothing is ever
    killed by name). The local bound (limit + ``backstop_s``) kills the local
    ssh child if the remote limit somehow didn't fire. A timeout, a non-zero
    exit or an exception (e.g. no ``ssh`` binary) reports this host as
    failed; one host never stops the other's bootstrap.
    """
    backstop = timeouts.backstop_s
    steps: list[tuple[str, list[str], float]] = [
        ("mkdir", ssh_argv(host, host.mkdir_command()), timeouts.command_s),
        *(
            (name, argv, timeouts.command_s)
            for name, argv in _copy_steps(host, WORKER_SCRIPT, REQUIREMENTS)
        ),
        (
            "uv venv",
            ssh_argv(host, host.venv_command(timeouts.install_s)),
            timeouts.install_s + backstop,
        ),
        (
            "pip install",
            ssh_argv(host, host.install_command(timeouts.install_s)),
            timeouts.install_s + backstop,
        ),
        (
            "cuda check",
            ssh_argv(host, host.check_command(timeouts.check_s)),
            timeouts.check_s + backstop,
        ),
    ]
    result = CommandResult(0)
    for step, argv, timeout in steps:
        try:
            result = await _call(runner, argv, timeout)
        except TimeoutError:
            log.error(
                "ocr_farm.bootstrap_timeout host=%s step=%s after=%ss", host.alias, step, timeout
            )
            return BootstrapResult(host.alias, False, error=f"{step} timeout after {timeout:g}s")
        except Exception as exc:  # noqa: BLE001 - one host must not stop the other
            return BootstrapResult(
                host.alias, False, error=f"{step} failed: {type(exc).__name__}: {exc}"
            )
        if step != "cuda check" and result.returncode != 0:
            return BootstrapResult(
                host.alias, False, error=f"{step} exit {result.returncode}: {_tail(result)}"
            )
    check, error = _check_line(result)
    if error:
        return BootstrapResult(host.alias, False, check=check, error=error)
    return BootstrapResult(host.alias, True, check=check)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

#: Exit codes of the remote worker command that mean the *host* is broken,
#: not the book. Such a host is dropped for the rest of the run and its book
#: goes back to the queue for a healthy host.
#:
#: - 69: ``ocr_worker.py`` found CUDA unavailable (``torch.cuda.is_available()``
#:   is False), before loading any model. (Exit 3 is a Windows ``abort()``
#:   crash, e.g. ``std::bad_alloc``: a book failure, not a host fault.)
#: - 127: Linux shell "command not found" (the venv interpreter is missing).
#: - 9009: Windows cmd "is not recognized" (the venv interpreter is missing).
#: - 255: ssh itself failed (could not reach or authenticate to the host, or
#:   the connection dropped mid-command). Only this code triggers the
#:   ``probe_s`` reachability probe before cleanup; 69, 127 and 9009 come from a
#:   host that answered, so they don't.
#:
#: Worker exit 4 (the GPU guard: peak GPU memory 0 after the conversion), 5
#: (docling conversion failed) and any other code are *book* failures.
SSH_EXIT = 255
HOST_FAULT_EXITS = frozenset({69, 127, 9009, SSH_EXIT})

#: This many ``failed(timeout)`` books in a row on one host, with no success
#: in between, mean the host (or its link) is stalled: it's dropped and those
#: books go back to the queue.
TIMEOUT_STREAK_LIMIT = 2

DUPLICATE_PROBLEM = "duplicate_sha256"

#: Books over this many pages are OCR'd as contiguous parts of at most this
#: many pages, each its own worker job (VOR-1427).
#:
#: The worker keeps every 100-page batch's document (pictures included) until
#: it concatenates them, so its memory grows with the pages it is given. On
#: 2026-09-26 two win workers on a 4,178- and a 2,036-page book grew
#: from the ~6-6.5 GB a worker commits at start to ~8 GB each within 40
#: minutes (~1,000-1,600 pages each at 1.5-2.5 s/page): roughly 1-2 MB per
#: page kept, while the PC itself holds ~13 GB of a 36.5 GB commit limit.
#: At 400 pages a worker keeps at most ~0.4-0.8 GB (twice that while
#: concatenating and saving), so two workers stay near 29 GB even at the high
#: estimate. Each part reloads the models (40-60 s): 7-10% of a 400-page
#: part of light pages (10 min at 1.5 s/page), 2-4% on lin (28 min at
#: 4.2 s/page), 1-2% for dense pages (>=54 min at >=8.1 s/page, see
#: ``FarmTimeouts``); smaller parts would waste more. A dense 400-page part
#: still fits its 4,120 s budget with ~25% headroom. 400 is also a whole
#: number of the worker's 100-page batches, and the largest backlog book
#: (4,178 pages) becomes 11 parts spread over every slot.
SPLIT_PAGES = 400

#: ``merge(part_json_paths, out_path, name) -> merged page numbers``: writes
#: one DoclingDocument JSON from a split book's part JSONs (in page order).
PartMerger = Callable[[Sequence[Path], Path, str], list[int]]


def part_ranges(pages: int, split_pages: int = SPLIT_PAGES) -> list[tuple[int, int]]:
    """Inclusive 1-based page ranges of at most ``split_pages`` covering ``1..pages``.

    One range (the whole book) for a book of ``split_pages`` pages or fewer.
    """
    if split_pages < 1:
        raise ValueError(f"split_pages must be >= 1, got {split_pages}")
    if pages <= split_pages:
        return [(1, max(pages, 0))]
    return [
        (start, min(start + split_pages - 1, pages)) for start in range(1, pages + 1, split_pages)
    ]


def part_key(sha256: str, first: int, last: int, gen: int = 0) -> str:
    """The worker file stem (``jobs/<key>.pdf|.json|.pid``) of pages
    ``first..last`` of a book in attempt ``gen``.

    A sha256 of its own, so parts of one book -- and two attempts at the same
    part, after a stalled host revives the book while the old attempt is
    still killing its worker and removing its files -- never share worker
    files, and ``FarmHost.kill_worker_command`` (which requires a
    sha256-shaped id and checks it is on the worker's command line) can only
    ever kill that attempt of that part.
    """
    return hashlib.sha256(f"{sha256}:{first}-{last}:g{gen}".encode()).hexdigest()


def merge_docling_parts(part_paths: Sequence[Path], out_path: Path, name: str) -> list[int]:
    """Merge a split book's part JSONs, in order, into ``out_path``.

    Each part document starts at page 1 and has exactly its page count (the
    worker re-adds pages docling dropped), so ``DoclingDocument.concatenate``
    -- which shifts each document by ``max page so far - its first page + 1``
    -- gives absolute page numbers. Parts are read one after another from
    disk (only the parsed documents are kept, never the JSON text) and the
    merged document is written with its pictures embedded, like the worker's.
    Returns the merged document's page numbers, sorted; the caller checks them.

    docling-core is imported here (the ``runtime`` extra on localhost).
    """
    from docling_core.types.doc import (  # type: ignore[import-not-found, unused-ignore]
        DoclingDocument,
        ImageRefMode,
    )

    documents = [DoclingDocument.load_from_json(path) for path in part_paths]
    merged = DoclingDocument.concatenate(documents)
    del documents
    merged.name = name
    merged.save_as_json(out_path, image_mode=ImageRefMode.EMBEDDED)
    return sorted(int(page_no) for page_no in merged.pages)


@dataclass(frozen=True)
class FarmTimeouts:
    """Per-command bounds in seconds.

    ``command_s`` bounds every scp / cleanup / kill command and the CUDA
    check; the OCR command gets ``ocr_s(pages)``; ``probe_s`` bounds the
    ``ssh <alias> exit`` reachability probe after an scp failure. The CUDA
    check also carries a remote limit of ``command_s``, with the local bound
    ``backstop_s`` longer.

    The OCR budget is ``ocr_startup_s + pages * ocr_s_per_page``, at least
    ``ocr_min_s``. Every worker run loads its models before the first page
    (40-60 s on lin), so the load is budgeted once per book instead of eating
    into the per-page allowance: on 2026-09-26 a dense 31-page book ran past
    the old ``max(300, pages * 10)`` = 310 s and was failed as a timeout.
    ``ocr_startup_s`` is 2x the slowest observed load. ``ocr_s_per_page``
    stays 10 s: lin averages ~4.2 s/page per worker (2.10 s/page aggregate
    over 2 workers), but that dense book took at least ~8.1 s/page after its
    model load (>=311 s, less at most 60 s of load, over 31 pages).

    The budget is per job: a book over ``SPLIT_PAGES`` pages runs as parts,
    and each part (which loads the models again) gets ``ocr_s(part pages)``,
    at most ``ocr_s(SPLIT_PAGES)`` = 4,120 s. A 400-page part of such dense
    pages takes >=~54 min (400 x 8.1 s + the load), so the budget leaves
    about 25% headroom for it.

    The OCR command has no remote time limit (``FarmHost.worker_command``
    isn't ``_bounded``), so this budget is not capped by
    ``MAX_REMOTE_TIMEOUT_S``: it is enforced locally by ``asyncio.wait_for``
    in ``_call``, and on expiry the remote worker is killed by its verified
    PID (``_kill_remote``).
    """

    command_s: float = 300.0
    ocr_startup_s: float = 120.0
    ocr_s_per_page: float = 10.0
    ocr_min_s: float = 300.0
    probe_s: float = 30.0
    backstop_s: float = 90.0
    pid_read_attempts: int = 3
    pid_retry_s: float = 5.0

    def __post_init__(self) -> None:
        # command_s is also the CUDA check's remote time limit.
        validate_remote_timeout(self.command_s, "FarmTimeouts.command_s")
        if self.pid_read_attempts < 1:
            raise ValueError("FarmTimeouts.pid_read_attempts must be >= 1")

    def ocr_s(self, pages: int) -> float:
        return max(self.ocr_min_s, self.ocr_startup_s + pages * self.ocr_s_per_page)


class _HostFault(Exception):
    """A failure that says the host, not the book, is broken.

    ``unreachable`` means a reachability probe failed: nothing more (not even
    cleanup) is sent to the host.
    """

    def __init__(self, reason: str, *, unreachable: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.unreachable = unreachable


async def _call(runner: Runner, argv: list[str], timeout: float) -> CommandResult:
    """Run one command with a bound; raises ``TimeoutError`` past it."""
    return await asyncio.wait_for(runner(argv), timeout)


async def run_farm(
    items: Sequence[FarmItem],
    hosts: Sequence[FarmHost],
    garage: GarageLike,
    pool: PoolLike,
    *,
    requeue: bool,
    bucket: str = DEFAULT_BUCKET,
    runner: Runner = subprocess_runner,
    timeouts: FarmTimeouts = FarmTimeouts(),  # noqa: B008 - frozen, immutable default
    split_pages: int = SPLIT_PAGES,
    merge: PartMerger = merge_docling_parts,
) -> FarmReport:
    """OCR ``items`` on ``hosts`` and store each result in Garage.

    A book over ``split_pages`` pages is cut into contiguous page-range parts
    (``part_ranges``); each part is a sub-PDF and its own job. Jobs (whole
    books and parts alike) run longest first, onto the first host (in
    ``hosts`` order) with a free slot; never more than ``host.slots`` jobs per
    host at once. A split book's parts can run on different hosts at the same
    time. Once every part of a book is in, the coordinator merges them
    (``merge``), checks the merged pages are exactly 1..N and stores the
    book; the book is re-queued once.

    - A host that fails its preparation (mkdir, worker copy, CUDA check) gets
      no jobs. A host whose worker exits with a ``HOST_FAULT_EXITS`` code is
      dropped for the rest of the run and that job (only that part of a split
      book) goes back to the queue. Both land in ``FarmReport.host_errors``.
    - An scp failure followed by a failed ``ssh <alias> exit`` probe is a host
      fault too; so are ``TIMEOUT_STREAK_LIMIT`` timed-out books in a row on
      one host (a stalled link), which puts those books back in the queue
      (every part of a split book). A worker command exiting 255 (ssh itself
      failed) is probed the same way; when the probe fails, no cleanup is sent
      and ``host_errors`` notes that the job's worker files may remain.
    - A job failure (any other nonzero exit, bad output, an scp failure on a
      reachable host, an isolated timeout, merged pages that aren't 1..N)
      fails its whole book: nothing is stored, the book's queued parts are
      dropped, its in-flight parts are cancelled (each kills its remote
      worker and removes its files), and it is reported as
      ``failed(<reason>)`` and never retried or re-queued.
    - ``duplicate_sha256`` items are never OCR'd; they're re-queued (when
      ``requeue``) once their twin's OCR exists.
    - Cancelling the run cancels every job: each kills its remote worker and
      removes its files before the cancellation propagates.
    """
    for host in hosts:
        if host.scp_workdir is None:
            raise ValueError(f"{host.alias}: scp_workdir not resolved (call resolve_hosts)")

    host_errors: dict[str, str] = {}
    usable: list[FarmHost] = []
    for host in hosts:
        try:
            error = await _prepare_host(host, runner, timeouts)
        except Exception as exc:  # noqa: BLE001 - one broken host must not stop the farm
            error = f"prepare failed: {type(exc).__name__}: {exc}"
        if error:
            log.error("ocr_farm.host_unusable host=%s error=%s", host.alias, error)
            host_errors[host.alias] = error
        else:
            usable.append(host)

    results: list[FarmResult] = []
    pending: list[FarmItem] = []
    duplicates: list[FarmItem] = []
    for item in items:
        if item.problem == DUPLICATE_PROBLEM:
            duplicates.append(item)
        elif item.problem:
            results.append(_result(item, None, 0.0, f"failed({item.problem})"))
        else:
            pending.append(item)
    pending.sort(key=lambda item: item.pages, reverse=True)

    with tempfile.TemporaryDirectory(
        prefix="vorag-ocr-farm-", ignore_cleanup_errors=True
    ) as scratch:
        farm = _Farm(
            usable,
            host_errors,
            results,
            garage,
            pool,
            bucket=bucket,
            requeue=requeue,
            runner=runner,
            timeouts=timeouts,
            merge=merge,
        )
        for n, item in enumerate(pending):
            farm.add(item, part_ranges(item.pages, split_pages), Path(scratch) / f"book-{n}")
        await farm.run()

    for item in duplicates:
        results.append(
            await _resolve_duplicate(item, results, garage, pool, bucket=bucket, requeue=requeue)
        )
    return FarmReport(results, host_errors)


@dataclass(frozen=True)
class _Job:
    """One worker run: a whole book, or part ``index`` (pages ``first..last``)."""

    item: FarmItem
    index: int
    first: int
    last: int
    parts: int
    #: The book's generation; a book revived after a stalled host gets a new
    #: one, so outcomes of the old generation's jobs are ignored.
    gen: int

    @property
    def pages(self) -> int:
        return self.last - self.first + 1

    @property
    def key(self) -> str:
        """Worker file stem: the book's sha256 for an unsplit book's first
        attempt (as before splitting), else ``part_key`` -- scoped to the
        attempt, so a revived book never reuses files its old attempt may
        still be killing or removing."""
        if self.parts == 1 and self.gen == 0:
            return self.item.sha256
        return part_key(self.item.sha256, self.first, self.last, self.gen)

    @property
    def label(self) -> str:
        """``book <id>`` or ``book <id> part <i>/<n>``, for logs and host errors."""
        book = f"book {self.item.envelope_id}"
        return book if self.parts == 1 else f"{book} part {self.index + 1}/{self.parts}"


@dataclass(frozen=True)
class _JobDone:
    """A job's outcome: ``status`` is ``""`` (its JSON is at ``path``),
    ``"cached"`` (the book's OCR already exists) or ``failed(<reason>)``;
    ``host`` is the host that ran it."""

    status: str
    seconds: float
    host: str
    path: Path | None = None
    note: str = ""


@dataclass
class _Attempt:
    """One attempt at a book: its local directory, its shared prep, and how
    many of its jobs are still running (the directory goes once none are)."""

    dir: Path
    prep: asyncio.Task[list[Path] | str] | None = None
    running: int = 0


@dataclass
class _Book:
    item: FarmItem
    ranges: list[tuple[int, int]]
    root: Path
    #: The current attempt; a stalled host's revival starts a new one.
    gen: int = 0
    #: ``None`` while its jobs run, ``"finishing"`` while it is merged and
    #: stored, then its final status.
    status: str | None = None
    attempts: dict[int, _Attempt] = field(default_factory=dict)
    done: dict[int, _JobDone] = field(default_factory=dict)
    #: Sum of its jobs' durations (worker-slot time) this attempt.
    seconds: float = 0.0

    @property
    def attempt(self) -> _Attempt:
        """The current attempt (created on first use): prep and files in ``g<gen>``."""
        if self.gen not in self.attempts:
            self.attempts[self.gen] = _Attempt(self.root / f"g{self.gen}")
        return self.attempts[self.gen]

    def jobs(self) -> list[_Job]:
        return [
            _Job(self.item, index, first, last, len(self.ranges), self.gen)
            for index, (first, last) in enumerate(self.ranges)
        ]

    def hosts(self, *extra: str) -> str:
        """The hosts that ran its parts, in part order (then ``extra``), comma-separated."""
        ordered = [self.done[index].host for index in sorted(self.done)]
        return ",".join(dict.fromkeys([*ordered, *extra]))


class _Farm:
    """``run_farm``'s scheduling loop and per-book state."""

    def __init__(
        self,
        usable: list[FarmHost],
        host_errors: dict[str, str],
        results: list[FarmResult],
        garage: GarageLike,
        pool: PoolLike,
        *,
        bucket: str,
        requeue: bool,
        runner: Runner,
        timeouts: FarmTimeouts,
        merge: PartMerger,
    ) -> None:
        self.usable = usable
        self.host_errors = host_errors
        self.results = results
        self.garage = garage
        self.pool = pool
        self.bucket = bucket
        self.requeue = requeue
        self.runner = runner
        self.timeouts = timeouts
        self.merge = merge
        self.books: dict[str, _Book] = {}
        self.pending: list[_Job] = []
        self.free = {host.alias: host.slots for host in usable}
        self.running: dict[asyncio.Task[_JobDone], tuple[FarmHost, _Job]] = {}
        self.finishing: dict[asyncio.Task[FarmResult], _Book] = {}
        self.slot_of: dict[str, int] = {}
        self.timeout_streak: dict[str, list[_Book]] = {}
        self.requeued_for_timeouts: set[str] = set()

    def add(self, item: FarmItem, ranges: list[tuple[int, int]], root: Path) -> None:
        """Queue a book's jobs (``add`` books longest first)."""
        book = _Book(item, ranges, root)
        self.books[item.envelope_id] = book
        self.pending.extend(book.jobs())

    async def run(self) -> None:
        # Longest job first. Stable: equal-sized jobs keep book order, then
        # part order.
        self.pending.sort(key=lambda job: job.pages, reverse=True)
        try:
            while self.pending or self.running or self.finishing:
                self._dispatch()
                if not self.running and not self.finishing:
                    break
                waiting: set[asyncio.Task[Any]] = {*self.running, *self.finishing}
                done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task in self.running:
                        self._job_done(task)
                    else:
                        self._book_stored(task)
        except BaseException:
            tasks: list[asyncio.Task[Any]] = [*self.running, *self.finishing]
            tasks += [
                attempt.prep
                for book in self.books.values()
                for attempt in book.attempts.values()
                if attempt.prep is not None
            ]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        # Only reached with jobs left when every host was dropped.
        for envelope_id in dict.fromkeys(job.item.envelope_id for job in self.pending):
            book = self.books[envelope_id]
            unusable = _result(book.item, None, 0.0, "failed(no_usable_host)")
            if envelope_id not in self.slot_of:
                self.slot_of[envelope_id] = len(self.results)
                self.results.append(unusable)
            self._record(unusable)
            book.status = unusable.status
            self._log_done(book)
            self._tidy(book)
        # Timed-out books whose host's streak never resolved are final now.
        for streak in self.timeout_streak.values():
            self._clear_streak(streak)

    # -- dispatch --------------------------------------------------------------

    def _dispatch(self) -> None:
        while self.pending:
            target = next((h for h in self.usable if self.free[h.alias] > 0), None)
            if target is None:
                return
            job = self.pending.pop(0)
            book = self.books[job.item.envelope_id]
            self.free[target.alias] -= 1
            if job.item.envelope_id not in self.slot_of:
                self.slot_of[job.item.envelope_id] = len(self.results)
                self.results.append(_result(job.item, target.alias, 0.0, "failed(not_finished)"))
            attempt = book.attempt
            if attempt.prep is None:
                attempt.prep = asyncio.create_task(
                    _prepare_book(book.item, book.ranges, attempt.dir, self.garage, self.bucket)
                )
            task = asyncio.create_task(
                _run_job(job, attempt.prep, attempt.dir, target, self.runner, self.timeouts)
            )
            attempt.running += 1
            self.running[task] = (target, job)

    # -- outcomes --------------------------------------------------------------

    def _job_done(self, task: asyncio.Task[_JobDone]) -> None:
        host, job = self.running.pop(task)
        self.free[host.alias] += 1
        book = self.books[job.item.envelope_id]
        book.attempts[job.gen].running -= 1
        try:
            self._job_outcome(task, host, job, book)
        finally:
            self._tidy(book)

    def _job_outcome(
        self, task: asyncio.Task[_JobDone], host: FarmHost, job: _Job, book: _Book
    ) -> None:
        if task.cancelled():
            return  # its book failed or was resolved while it ran
        live = job.gen == book.gen and book.status is None
        fault = task.exception()
        if isinstance(fault, _HostFault):
            self._drop(host, f"{fault.reason} ({job.label})")
            if live:
                _requeue_locally(self.pending, job)
            return
        if fault is not None:
            raise fault
        outcome = task.result()
        if not live:
            return  # a sibling already failed (or resolved) the book
        book.seconds += outcome.seconds
        streak = self.timeout_streak.setdefault(host.alias, [])
        if outcome.status == "":
            self._clear_streak(streak)
            book.done[job.index] = outcome
            if job.parts > 1:
                log.info(
                    "ocr_farm.part_done host=%s envelope_id=%s part=%d/%d pages=%d-%d job=%s "
                    "seconds=%.1f",
                    host.alias,
                    job.item.envelope_id,
                    job.index + 1,
                    job.parts,
                    job.first,
                    job.last,
                    job.key,
                    outcome.seconds,
                )
            if len(book.done) == len(book.ranges):
                self._store(book)
        elif outcome.status == "cached":
            self._store(book, cached_on=host.alias)
        elif outcome.status == "failed(timeout)" and host not in self.usable:
            # The host was already dropped (another fault) while this job
            # ran: the timeout is the host's, so the job always goes back,
            # whatever its book's history.
            self.requeued_for_timeouts.add(job.item.envelope_id)
            _requeue_locally(self.pending, job)
        elif outcome.status == "failed(timeout)":
            self._fail(book, outcome.status, host)
            if job.item.envelope_id in self.requeued_for_timeouts:
                # A book already requeued once for a stalled host is final on
                # its next timeout on a healthy host and doesn't count: two
                # books that stall everywhere must not take down every host.
                self._log_done(book)
            else:
                # Not final yet: the streak may still revive it.
                streak.append(book)
                if len(streak) >= TIMEOUT_STREAK_LIMIT:
                    self._drop(host, f"{len(streak)} consecutive timeouts")
                    for stalled in streak:
                        self.requeued_for_timeouts.add(stalled.item.envelope_id)
                        self._revive(stalled)
                    streak.clear()
        else:
            self._fail(book, outcome.status, host)
            self._log_done(book)

    def _book_stored(self, task: asyncio.Task[FarmResult]) -> None:
        book = self.finishing.pop(task)
        result = task.result()
        book.status = result.status
        self._record(result)
        self._log_done(book)
        self._tidy(book)

    # -- book state --------------------------------------------------------------

    def _store(self, book: _Book, *, cached_on: str | None = None) -> None:
        """Every part is in (or ``cached_on`` found the OCR already exists):
        merge and store it."""
        book.status = "finishing"
        self._stop_jobs(book)
        parts = [] if cached_on else [book.done[index] for index in range(len(book.ranges))]
        task = asyncio.create_task(
            _store_book(
                book.item,
                parts,
                book.attempt.dir,
                cached_on or book.hosts(),
                book.seconds,
                self.garage,
                self.pool,
                bucket=self.bucket,
                requeue=self.requeue,
                merge=self.merge,
            )
        )
        self.finishing[task] = book

    def _fail(self, book: _Book, status: str, host: FarmHost) -> None:
        """One job failed: the whole book fails, and nothing of it is stored.

        The caller logs ``book_done`` once the failure is final (a timeout
        may still be revived by its host's streak).
        """
        book.status = status
        self._stop_jobs(book)
        self._record(_result(book.item, book.hosts(host.alias), book.seconds, status))

    def _stop_jobs(self, book: _Book) -> None:
        """Drop the book's queued jobs and cancel its running ones (each kills
        its remote worker and removes its files as it unwinds)."""
        envelope_id = book.item.envelope_id
        self.pending[:] = [job for job in self.pending if job.item.envelope_id != envelope_id]
        for task, (_, job) in self.running.items():
            if job.item.envelope_id == envelope_id:
                task.cancel()
        prep = book.attempt.prep
        if prep is not None and not prep.done():
            prep.cancel()

    def _revive(self, book: _Book) -> None:
        """A book failed by a stalled host goes back to the queue, every part."""
        book.gen += 1
        book.status = None
        book.done.clear()
        book.seconds = 0.0
        for job in book.jobs():
            _requeue_locally(self.pending, job)

    def _record(self, result: FarmResult) -> None:
        self.results[self.slot_of[result.envelope_id]] = result

    def _log_done(self, book: _Book) -> None:
        """``ocr_farm.book_done``, once per book, with its final result."""
        result = self.results[self.slot_of[book.item.envelope_id]]
        log.info(
            "ocr_farm.book_done host=%s envelope_id=%s status=%s pages=%d seconds=%.1f",
            result.host,
            result.envelope_id,
            result.status,
            result.pages,
            result.seconds,
        )

    def _clear_streak(self, streak: list[_Book]) -> None:
        """A success on the host: its timed-out books are final after all."""
        for book in streak:
            if book.status == "failed(timeout)":
                self._log_done(book)
        streak.clear()

    def _drop(self, host: FarmHost, reason: str) -> None:
        if host in self.usable:
            self.usable.remove(host)
            self.host_errors[host.alias] = reason
            log.error("ocr_farm.host_dropped host=%s reason=%s", host.alias, reason)

    def _tidy(self, book: _Book) -> None:
        """Remove an attempt's local files once none of its jobs run and it is
        over: an old attempt of a revived book, or the book is finished."""
        finished = book.status not in (None, "finishing")
        for gen, attempt in list(book.attempts.items()):
            if gen == book.gen and not finished:
                continue
            if attempt.running or (attempt.prep is not None and not attempt.prep.done()):
                continue  # later; the run's scratch directory is the backstop
            shutil.rmtree(attempt.dir, ignore_errors=True)
            del book.attempts[gen]


def _requeue_locally(pending: list[_Job], job: _Job) -> None:
    """Put a job back in the queue, keeping it longest-first."""
    index = next((n for n, other in enumerate(pending) if other.pages < job.pages), len(pending))
    pending.insert(index, job)


def _check_line(result: CommandResult) -> tuple[str, str]:
    """``(check_line, error)`` for an ``ocr_worker.py --check`` result."""
    lines = result.stdout.strip().splitlines()
    check = lines[-1].strip() if lines else ""
    if result.returncode != 0 or " cuda True" not in check:
        return check, f"cuda check exit {result.returncode}: {check or _tail(result)}"
    return check, ""


async def _prepare_host(host: FarmHost, runner: Runner, timeouts: FarmTimeouts) -> str:
    """Make ``jobs/``, refresh the worker files and require ``cuda True``.

    Returns an error string, or "" when the host can take books.
    """
    result = await _call(runner, ssh_argv(host, host.mkdir_command()), timeouts.command_s)
    if result.returncode != 0:
        return f"mkdir exit {result.returncode}: {_tail(result)}"
    for step, argv in _copy_steps(host, WORKER_SCRIPT):
        result = await _call(runner, argv, timeouts.command_s)
        if result.returncode != 0:
            return f"{step} exit {result.returncode}: {_tail(result)}"
    try:
        result = await _call(
            runner,
            ssh_argv(host, host.check_command(timeouts.command_s)),
            timeouts.command_s + timeouts.backstop_s,
        )
    except TimeoutError:
        return f"cuda check timeout after {timeouts.command_s + timeouts.backstop_s:g}s"
    return _check_line(result)[1]


def _result(
    item: FarmItem, host: str | None, seconds: float, status: str, requeued: bool = False
) -> FarmResult:
    return FarmResult(
        envelope_id=item.envelope_id,
        title=item.title,
        sha256=item.sha256,
        host=host,
        pages=item.pages,
        seconds=round(seconds, 1),
        status=status,
        requeued=requeued,
    )


async def _requeue(pool: PoolLike, envelope_id: str) -> bool:
    status = await pool.execute(REQUEUE_SQL, envelope_id)
    return str(status).strip().endswith(" 1")


async def _resolve_duplicate(
    item: FarmItem,
    results: Sequence[FarmResult],
    garage: GarageLike,
    pool: PoolLike,
    *,
    bucket: str,
    requeue: bool,
) -> FarmResult:
    """A same-bytes envelope rides on its twin's OCR: same sha256, same cache."""
    ocr_exists = any(r.sha256 == item.sha256 and r.status in ("ok", "cached") for r in results) or (
        bool(item.sha256) and await garage.object_metadata(bucket, ocr_key(item.sha256)) is not None
    )
    if not ocr_exists:
        return _result(item, None, 0.0, f"failed({DUPLICATE_PROBLEM})")
    requeued = await _requeue(pool, item.envelope_id) if requeue else False
    return _result(item, None, 0.0, "duplicate", requeued)


async def _prepare_book(
    item: FarmItem,
    ranges: Sequence[tuple[int, int]],
    workdir: Path,
    garage: GarageLike,
    bucket: str,
) -> list[Path] | str:
    """Once per book: the PDF(s) its jobs ship, or a final status.

    ``"cached"`` when its OCR already exists; ``failed(<reason>)`` for a
    sha256 mismatch, a split source whose page count isn't the planned one,
    or any error. Otherwise one local PDF per part: the source itself (EPUBs
    converted first) for an unsplit book, else ``_split_pdf``'s sub-PDFs.
    """
    try:
        if await garage.object_metadata(bucket, ocr_key(item.sha256)) is not None:
            return "cached"
        data = await garage.get_bytes(bucket, item.garage_key)
        sha = await asyncio.to_thread(lambda: hashlib.sha256(data).hexdigest())
        if sha != item.sha256:
            return "failed(sha256_mismatch)"
        pdf = data if item.fmt == "pdf" else await asyncio.to_thread(epub_to_pdf_bytes, data)
        await asyncio.to_thread(workdir.mkdir, parents=True, exist_ok=True)
        if len(ranges) == 1:
            path = workdir / "book.pdf"
            await asyncio.to_thread(path.write_bytes, pdf)
            return [path]
        return await asyncio.to_thread(_split_pdf, pdf, ranges, item.pages, workdir)
    except Exception as exc:  # noqa: BLE001 - one book's failure must not stop the farm
        log.exception("ocr_farm.book_error envelope_id=%s", item.envelope_id)
        return f"failed(error_{type(exc).__name__})"


def _split_pdf(
    pdf: bytes, ranges: Sequence[tuple[int, int]], pages: int, workdir: Path
) -> list[Path] | str:
    """Write one sub-PDF per range (pymupdf ``insert_pdf``) into ``workdir``.

    The ranges come from the planned page count, so a source with any other
    count is ``failed(page_count_mismatch)``: its parts would not cover it.
    """
    import fitz  # type: ignore[import-untyped, unused-ignore]

    with fitz.open(stream=pdf, filetype="pdf") as source:
        if int(source.page_count) != pages:
            log.error("ocr_farm.page_count_mismatch planned=%d source=%d", pages, source.page_count)
            return "failed(page_count_mismatch)"
        paths: list[Path] = []
        for index, (first, last) in enumerate(ranges):
            path = workdir / f"part-{index:03d}.pdf"
            with fitz.open() as part:
                part.insert_pdf(source, from_page=first - 1, to_page=last - 1)
                part.save(str(path))
            paths.append(path)
    return paths


async def _run_job(
    job: _Job,
    prep: asyncio.Task[list[Path] | str],
    workdir: Path,
    host: FarmHost,
    runner: Runner,
    timeouts: FarmTimeouts,
) -> _JobDone:
    """Ship, OCR and fetch one job's PDF; the JSON stays in ``workdir``."""
    started = time.monotonic()

    def done(status: str, path: Path | None = None, note: str = "") -> _JobDone:
        return _JobDone(status, time.monotonic() - started, host.alias, path, note)

    try:
        # Shielded: cancelling one job must not cancel the book's shared prep.
        prepared = await asyncio.shield(prep)
        if isinstance(prepared, str):
            return done(prepared)
        pdf = prepared[job.index]
        out = workdir / f"part-{job.index:03d}.json"
        failure, note = await _ocr_on_host(job.key, job.pages, pdf, out, host, runner, timeouts)
        if failure:
            return done(f"failed({failure})")
        await asyncio.to_thread(pdf.unlink, missing_ok=True)
        return done("", out, note)
    except _HostFault:
        raise
    except Exception as exc:  # noqa: BLE001 - one book's failure must not stop the farm
        log.exception("ocr_farm.book_error host=%s %s", host.alias, job.label)
        return done(f"failed(error_{type(exc).__name__})")


async def _store_book(
    item: FarmItem,
    parts: Sequence[_JobDone],
    workdir: Path,
    hosts: str,
    seconds: float,
    garage: GarageLike,
    pool: PoolLike,
    *,
    bucket: str,
    requeue: bool,
    merge: PartMerger,
) -> FarmResult:
    """Store a book whose jobs all succeeded (merging a split book's parts)
    and re-queue it once; with no ``parts`` the OCR already exists (cached).

    A split book's merged pages must be exactly ``1..item.pages``, or the book
    is ``failed(merged_pages)`` and nothing is stored. ``seconds`` is the sum
    of the book's job durations; the merge and upload time is added.
    """
    started = time.monotonic()
    host = hosts or None

    def done(status: str, requeued: bool = False) -> FarmResult:
        return _result(item, host, seconds + time.monotonic() - started, status, requeued)

    try:
        if not parts:
            return done("cached", await _requeue(pool, item.envelope_id) if requeue else False)
        paths = [part.path for part in parts if part.path is not None]
        if len(parts) == 1:
            payload = await asyncio.to_thread(paths[0].read_bytes)
        else:
            merged = workdir / "merged.json"
            pages = await asyncio.to_thread(merge, paths, merged, item.sha256)
            if pages != list(range(1, item.pages + 1)):
                log.error(
                    "ocr_farm.merged_pages envelope_id=%s parts=%d expected=1..%d got=%d pages "
                    "(%s..%s)",
                    item.envelope_id,
                    len(parts),
                    item.pages,
                    len(pages),
                    pages[0] if pages else None,
                    pages[-1] if pages else None,
                )
                return done("failed(merged_pages)")
            payload = await asyncio.to_thread(merged.read_bytes)
        meta = {
            # Tells the router this is OCR output, not an off-box native
            # (layout-only) pre-parse, which is stored under the same keys.
            "pipeline": "ocr",
            "envelope_id": item.envelope_id,
            "title": item.title,
            "host": hosts,
            "pages": item.pages,
            "fmt": item.fmt,
            "parts": len(parts),
            "seconds": round(seconds + time.monotonic() - started, 1),
            # The worker's last stdout line: "ocr_worker: ok ... gpu_peak_bytes=N"
            # (one per part, in page order, for a split book).
            "worker": parts[0].note if len(parts) == 1 else [part.note for part in parts],
        }
        # docling.json first, then meta.json (VOR-1434 cache contract): watchers
        # flip an envelope when its meta.json appears, so the document must
        # already be there. The envelope is still 'failed' between the two
        # writes (requeue runs after both), so no parse can see a half cache.
        await garage.put_bytes(bucket, ocr_key(item.sha256), payload, "application/json")
        await garage.put_bytes(
            bucket, ocr_meta_key(item.sha256), json.dumps(meta).encode(), "application/json"
        )
        return done("ok", await _requeue(pool, item.envelope_id) if requeue else False)
    except Exception as exc:  # noqa: BLE001 - one book's failure must not stop the farm
        log.exception("ocr_farm.book_error envelope_id=%s", item.envelope_id)
        return done(f"failed(error_{type(exc).__name__})")


async def _to_completion(step: Awaitable[None]) -> None:
    """Await a kill / cleanup step to its end even if the caller is cancelled
    meanwhile (e.g. a timed-out part is still killing its worker when a
    sibling's failure cancels it); the cancellation is re-raised after."""
    task = asyncio.ensure_future(step)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Also when the step finished in the same loop iteration: that
            # cancellation was the caller's, and must not be swallowed.
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


async def _ocr_on_host(
    job: str,
    pages: int,
    pdf: Path,
    out: Path,
    host: FarmHost,
    runner: Runner,
    timeouts: FarmTimeouts,
) -> tuple[str, str]:
    """Ship ``pdf`` as ``jobs/<job>.pdf``, OCR it, fetch the JSON to ``out``;
    always remove the job's worker files.

    Returns ``(failure_reason, "")`` or ``("", worker_last_stdout_line)``.
    Raises ``_HostFault`` for a ``HOST_FAULT_EXITS`` worker exit, or for an
    scp failure when the host then fails the reachability probe. A worker
    exit of ``SSH_EXIT`` (255) is probed the same way: an unreachable host
    gets no cleanup either.

    Only a timeout or cancellation that interrupts the *worker command* kills
    the remote worker (by the PID in ``jobs/<job>.pid``, verified to still be
    this job's ``ocr_worker.py``). During the scp steps the worker isn't
    running, so nothing is killed. The kill and the cleanup run to completion
    even if the job is cancelled while they run.
    """
    pdf_name, out_name, pid_name = f"{job}.pdf", f"{job}.json", f"{job}.pid"
    worker_files = (pdf_name, out_name, f"{out_name}.part", pid_name)
    worker_note = ""
    phase = "upload"
    reachable = True
    try:
        result = await _call(
            runner, scp_argv(str(pdf), host.scp_target(JOBS_DIR, pdf_name)), timeouts.command_s
        )
        if result.returncode != 0:
            return await _scp_failure(
                host, job, f"scp_upload_exit_{result.returncode}", runner, timeouts
            )
        phase = "worker"
        result = await _call(
            runner,
            ssh_argv(host, host.worker_command(pdf_name, out_name)),
            timeouts.ocr_s(pages),
        )
        phase = "download"
        if result.returncode != 0:
            log.error(
                "ocr_farm.worker_failed host=%s job=%s exit=%d stderr=%s",
                host.alias,
                job,
                result.returncode,
                _tail(result),
            )
            if result.returncode in HOST_FAULT_EXITS:
                reason = f"worker_exit_{result.returncode}: {_tail(result)}"
                if result.returncode == SSH_EXIT:
                    # ssh itself failed: the connection may have dropped
                    # mid-command. Probe before spending command_s on a
                    # cleanup that can't reach the host.
                    await _probe_reachable(host, job, reason, runner, timeouts)
                raise _HostFault(reason)
            return f"worker_exit_{result.returncode}", ""
        lines = result.stdout.strip().splitlines()
        worker_note = lines[-1].strip() if lines else ""
        result = await _call(
            runner, scp_argv(host.scp_target(JOBS_DIR, out_name), str(out)), timeouts.command_s
        )
        if result.returncode != 0:
            return await _scp_failure(
                host, job, f"scp_download_exit_{result.returncode}", runner, timeouts
            )
    except TimeoutError:
        log.error(
            "ocr_farm.timeout host=%s job=%s pages=%d phase=%s", host.alias, job, pages, phase
        )
        if phase == "worker":
            await _to_completion(_kill_remote(host, job, runner, timeouts))
        return "timeout", ""
    except asyncio.CancelledError:
        if phase == "worker":
            await _to_completion(_kill_remote(host, job, runner, timeouts))
        raise
    except _HostFault as fault:
        # An unreachable host would only burn command_s on cleanup; the
        # leftover files are recorded in the fault reason instead.
        reachable = not fault.unreachable
        raise
    finally:
        if reachable:
            await _to_completion(_cleanup(host, worker_files, runner, timeouts))
    if not await asyncio.to_thread(_is_docling_file, out):
        return "invalid_output", ""
    return "", worker_note


async def _scp_failure(
    host: FarmHost, job: str, reason: str, runner: Runner, timeouts: FarmTimeouts
) -> tuple[str, str]:
    """An scp failure is the job's unless the host fails a reachability probe.

    scp reports a lost connection as exit 1, like any other error, so
    ``_probe_reachable`` decides: an unreachable host raises, otherwise the
    failure is the job's.
    """
    await _probe_reachable(host, job, reason, runner, timeouts)
    return reason, ""


async def _probe_reachable(
    host: FarmHost, sha: str, reason: str, runner: Runner, timeouts: FarmTimeouts
) -> None:
    """``ssh <alias> exit``, bounded by ``probe_s``; raises if the host is gone.

    If the probe fails (non-zero, including 255, or times out), the host is
    unreachable and this raises an ``unreachable`` ``_HostFault``; the caller
    then attempts no cleanup, and the reason records that the book's worker
    files may remain on the host. Returns normally when the host answers.
    """
    leftover = f"worker files for {sha} may remain on {host.alias}"
    try:
        probe = await _call(runner, ssh_argv(host, "exit"), timeouts.probe_s)
    except TimeoutError:
        raise _HostFault(
            f"{reason}; reachability probe timed out; {leftover}", unreachable=True
        ) from None
    except Exception as exc:  # noqa: BLE001 - any probe failure means unreachable
        raise _HostFault(
            f"{reason}; reachability probe failed: {type(exc).__name__}; {leftover}",
            unreachable=True,
        ) from exc
    if probe.returncode != 0:
        raise _HostFault(
            f"{reason}; reachability probe exit {probe.returncode}; {leftover}", unreachable=True
        )


async def _read_worker_pid(host: FarmHost, sha: str, runner: Runner, timeouts: FarmTimeouts) -> str:
    """The PID in ``jobs/<sha>.pid``, retried: a worker whose command timed out
    may still be starting Python and not have written the file yet."""
    for attempt in range(1, timeouts.pid_read_attempts + 1):
        result = await _call(
            runner, ssh_argv(host, host.read_pid_command(f"{sha}.pid")), timeouts.command_s
        )
        pid = result.stdout.strip()
        if result.returncode == 0 and pid.isdigit():
            return pid
        if attempt < timeouts.pid_read_attempts:
            await asyncio.sleep(timeouts.pid_retry_s)
    return ""


async def _kill_remote(host: FarmHost, sha: str, runner: Runner, timeouts: FarmTimeouts) -> None:
    """Best effort: killing the local ssh client leaves the remote worker running.

    Reads ``jobs/<sha>.pid`` (up to ``pid_read_attempts`` times,
    ``pid_retry_s`` apart) and kills that PID only if its command line is
    still ``ocr_worker.py`` for this sha (``FarmHost.kill_worker_command``).
    """
    try:
        pid = await _read_worker_pid(host, sha, runner, timeouts)
        if not pid:
            log.error(
                "ocr_farm.orphan_risk host=%s sha=%s: no PID file after %d reads; the worker "
                "may still be running on the host and was not killed",
                host.alias,
                sha,
                timeouts.pid_read_attempts,
            )
            return
        killed = await _call(
            runner, ssh_argv(host, host.kill_worker_command(int(pid), sha)), timeouts.command_s
        )
        if killed.returncode != 0:
            log.warning(
                "ocr_farm.kill_skipped host=%s pid=%s exit=%d (not our worker, or gone) %s",
                host.alias,
                pid,
                killed.returncode,
                _tail(killed),
            )
    except Exception:  # noqa: BLE001 - best effort
        log.warning("ocr_farm.kill_error host=%s sha=%s", host.alias, sha, exc_info=True)


async def _cleanup(
    host: FarmHost, names: Sequence[str], runner: Runner, timeouts: FarmTimeouts
) -> None:
    try:
        cleanup = await _call(
            runner, ssh_argv(host, host.cleanup_command(*names)), timeouts.command_s
        )
        if cleanup.returncode != 0:
            log.warning(
                "ocr_farm.cleanup_failed host=%s files=%s exit=%d %s",
                host.alias,
                names,
                cleanup.returncode,
                _tail(cleanup),
            )
    except Exception:  # noqa: BLE001 - best effort
        log.warning("ocr_farm.cleanup_error host=%s files=%s", host.alias, names, exc_info=True)


def _is_docling_json(payload: bytes) -> bool:
    try:
        parsed = json.loads(payload)
    except ValueError:
        return False
    return isinstance(parsed, dict) and parsed.get("schema_name") == "DoclingDocument"


def _is_docling_file(path: Path) -> bool:
    return path.is_file() and _is_docling_json(path.read_bytes())
