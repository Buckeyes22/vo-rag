"""``vo-rag preflight`` — per-file CPU-only checks that route or
reject every book before ingest (VOR-1434).

Pure checks: no network, no Qdrant, no Postgres, no Garage. Every detection
rule is imported from the real parser code (``_epub_needs_normalization``,
``_normalize_epub``, ``is_image_only``, ``ParserRouter._pdf_kind``,
``core.parsing.fingerprint``, ``epub._escaping_hrefs``,
``docling.OVERSIZE_PAGE_PT``) so preflight and ingest can never disagree.
Defects the parser repairs itself (an EPUB
manifest href escaping the OPF directory, an oversize PDF page; VOR-1434)
are reported as ``info`` annotations, never rejects.

Per-file checks run in a fixed order; the first hard
failure (an out-of-order dependency, a corrupt container, a duplicate, a
failed pandoc dry run, or ``not_a_book``) stops that file
and sets ``verdict="reject"`` (or ``"duplicate_of=<id>"``) with the reason(s)
collected so far in ``reasons``. A file that clears every check is
``verdict="ok"``; ``route`` records where ingest should send it
(``native``/``ocr``/``giant``/``image_epub_ocr``, or the plain format name
for anything else).

``inspect_file`` never raises for a bad input file — an unexpected exception
during a check is caught and turned into a ``reject: preflight_error: ...``
result, because a single unreadable file must never abort a batch run.
"""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from vo_rag.core.parsing.fingerprint import content_fingerprint, file_sha256
from vo_rag.core.parsing.parsers import epub as epub_module
from vo_rag.core.parsing.parsers.docling import OVERSIZE_PAGE_PT as OVERSIZE_PAGE_PT
from vo_rag.core.parsing.parsers.docling import _longest_side
from vo_rag.core.parsing.parsers.epub import (
    _epub_needs_normalization,
    _normalize_epub,
    is_image_only,
)
from vo_rag.core.parsing.router import ParserRouter

#: Giant routing: the operator's rule, shared with the ingest-time routing.
GIANT_MAX_PAGES = 1000
GIANT_IMAGE_HEAVY_MIN_PAGES = 700

# ``OVERSIZE_PAGE_PT`` (imported from the docling parser, which rescales such
# pages before docling opens the PDF): a page whose longest side exceeds it is
# reported as the ``oversize_pages`` info annotation, not a reject.

#: ``not_a_book``: short PDFs with almost no extractable text.
NOT_A_BOOK_MAX_PAGES = 10
NOT_A_BOOK_MAX_CHARS = 2_000

#: 60s pandoc dry-run timeout.
PANDOC_DRY_RUN_TIMEOUT_S = 60.0

#: Filenames that are almost never books, whatever their content looks like
#: (race-schedule PDFs).
_NOT_A_BOOK_FILENAME = re.compile(
    r"(?<![a-z])(schedule|flyer|flier|brochure|poster|program(me)?|menu|calendar|ticket)s?(?![a-z])",
    re.IGNORECASE,
)
#: The filename rule only rejects short PDFs: race schedules and flyers run a
#: few pages, while a 70-page beginners' training program is a book (VOR-1434).
#: Formats without a page count keep the rule.
NOT_A_BOOK_FILENAME_MAX_PAGES = 40

#: Extensions ``inspect_batch`` walks a directory for.
SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".epub",
    ".mobi",
    ".docx",
    ".pptx",
    ".html",
    ".htm",
    ".xlsx",
    ".txt",
    ".md",
)


@dataclass(frozen=True)
class PreflightResult:
    """One file's preflight verdict.

    ``verdict`` is ``"ok"``, ``"reject"``, or ``"duplicate_of=<id>"`` — an
    in-batch duplicate's ``<id>`` is the earlier file's own sha256; a
    catalog duplicate's ``<id>`` is the matching catalog ``source_id``.
    ``reasons`` accumulates every info/reject annotation encountered before
    the file's checks stopped (informational codes like ``"zip64"`` or
    ``"normalized: text_html"`` alongside the terminal reject code, if any).
    ``kind`` is the detected file format (``"pdf"``, ``"epub"``, ...) or
    ``None`` when it could not be detected at all (``unknown_type``).
    ``route`` is where ingest should send the file (unset — ``None`` — for a
    file that never reached routing because it was rejected or a duplicate).
    """

    path: str
    sha256: str | None
    fingerprint: str | None
    kind: str | None
    pages: int | None
    route: str | None
    verdict: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreflightReport:
    results: tuple[PreflightResult, ...]
    counts: dict[str, dict[str, int]]
    duplicate_groups: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# check_container — public so other callers can reuse it.
# ---------------------------------------------------------------------------


def check_container(path: Path, kind: str | None = None) -> str | None:
    """``None`` when the container opens cleanly; else ``"corrupt_container"``
    or ``"zip_bomb"``.

    ``kind`` (``"epub"``/``"pdf"``) overrides the extension, for a caller
    checking a file before it has its final name (the downloader's
    ``.part`` file). EPUB: the zip-bomb guard on the declared sizes first,
    then ``testzip()`` finds the first bad member; any exception while
    reading the archive (``zlib.error``, ``EOFError``, ``RuntimeError`` for
    an encrypted member, ``NotImplementedError`` for an unknown compression)
    is ``corrupt_container``. PDF: ``fitz.open`` plus ``page_count`` (pymupdf
    can open a truncated/garbage PDF lazily and only fail once a page is
    read); a PDF with no pages is ``corrupt_container`` too. Any other
    format is not this check's job — ``None``.
    """
    kind = (kind or path.suffix.lstrip(".")).lower()
    if kind == "epub":
        try:
            with zipfile.ZipFile(path) as archive:
                try:
                    epub_module.check_zip_bomb(archive, source=str(path))
                except epub_module.ZipBombError:
                    return "zip_bomb"
                return "corrupt_container" if archive.testzip() is not None else None
        except Exception:  # noqa: BLE001 - BadZipFile, zlib.error, EOFError, RuntimeError (encrypted), NotImplementedError (compression), OSError
            return "corrupt_container"
    if kind == "pdf":
        try:
            import fitz  # type: ignore[import-untyped, unused-ignore]  # PyMuPDF

            with fitz.open(path, filetype="pdf") as doc:
                if doc.page_count == 0:
                    return "corrupt_container"
        except Exception:  # noqa: BLE001 - any of pymupdf's many "not a PDF" errors
            return "corrupt_container"
        return None
    return None


# ---------------------------------------------------------------------------
# Format detection — reuses ParserRouter._detect_format (magic -> mime ->
# extension), the exact production dispatch, the same way ``_pdf_kind`` is reused.
# ---------------------------------------------------------------------------


def _router_instance() -> ParserRouter:
    router = ParserRouter.__new__(ParserRouter)
    router.scanned_char_threshold = 50
    return router


def _detect_kind(path: Path) -> str | None:
    """The production format name for ``path``, or ``None`` for ``unknown_type``."""
    try:
        return _router_instance()._detect_format(str(path), None)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# EPUB: escaping hrefs (detected by ``epub._escaping_hrefs``; ``_normalize_epub``
# repairs them, so preflight only reports them).
# ---------------------------------------------------------------------------


def _epub_escaping_hrefs(path: Path) -> list[str]:
    """``_escaping_hrefs`` across every OPF manifest in the EPUB at ``path``."""
    escaping: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(".opf"):
                escaping.extend(epub_module._escaping_hrefs(name, archive.read(name)))
    return escaping


# ---------------------------------------------------------------------------
# EPUB: pandoc dry run + image-only detection (both need ebooklib/pypandoc,
# the ``runtime`` extra).
# ---------------------------------------------------------------------------


def _pandoc_dry_run(parse_path: str, *, timeout_s: float = PANDOC_DRY_RUN_TIMEOUT_S) -> str | None:
    """``None`` on a clean pandoc conversion; else ``"pandoc_failed: <detail>"``.

    A ``ThreadPoolExecutor`` wrapping a blocking ``pypandoc.convert_file``
    call cannot actually stop a hung pandoc
    process — ``future.result(timeout=...)`` gives up waiting, but the
    thread (and the pandoc child process under it) keeps running, and the
    ``with ThreadPoolExecutor(...)`` block's ``__exit__`` calls
    ``shutdown(wait=True)``, which then blocks forever on that same thread.
    Shelling out to the pandoc binary directly with
    ``subprocess.run(timeout=...)`` is the fix: on a timeout, ``subprocess.run``
    kills and reaps the child process itself before re-raising
    ``TimeoutExpired``, so this function is guaranteed to return within
    ``timeout_s`` (plus process-teardown overhead), never longer.
    """
    import subprocess

    import pypandoc  # type: ignore[import-not-found, import-untyped, unused-ignore]

    try:
        pandoc_path = pypandoc.get_pandoc_path()
    except Exception as exc:  # noqa: BLE001 - pandoc not resolvable is a reject, not a crash
        return f"pandoc_failed: {type(exc).__name__}: {exc}"

    try:
        # ``--sandbox``: pandoc reads only the input named here, never other
        # files or the network. The input is absolute and after ``--`` so a
        # name starting with ``-`` is never read as an option.
        subprocess.run(  # noqa: S603 - fixed argv, pandoc_path resolved by pypandoc itself
            [
                pandoc_path,
                "--sandbox",
                "-f",
                "epub",
                "-t",
                "plain",
                "-o",
                os.devnull,
                "--",
                str(Path(parse_path).absolute()),
            ],
            timeout=timeout_s,
            capture_output=True,
            check=True,
        )
    except subprocess.TimeoutExpired:
        return f"pandoc_failed: timed out after {timeout_s:.0f}s"
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace").strip()
        detail = stderr.splitlines()[0] if stderr else f"pandoc exited {exc.returncode}"
        return f"pandoc_failed: {detail}"
    except Exception as exc:  # noqa: BLE001 - pandoc's failure surface is wide
        detail = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return f"pandoc_failed: {detail}"
    return None


def _runtime_extras_available() -> bool:
    """``True`` when ebooklib + pypandoc (the ``runtime`` extra) are
    importable. A separate function so tests can force the "extras missing"
    fallback deterministically, regardless of what's actually installed."""
    try:
        import ebooklib  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: F401
        import pypandoc  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: F401
    except ImportError:
        return False
    return True


def _epub_is_image_only(parse_path: str, source_path: str) -> bool:
    """``is_image_only`` on the real parsed elements — the same conversion
    (``ebooklib.read_epub`` + ``pypandoc.convert_file`` to ``gfm-raw_html`` +
    ``markdown_to_elements``) ``EpubParser`` runs in production, so preflight
    and ingest can never disagree on which EPUBs are page scans.
    """
    from vo_rag.core.parsing.parsers.epub_markdown import markdown_to_elements

    _book, _metadata, toc_tree, markdown = epub_module._read_epub_and_convert(
        parse_path, source_path, source_format=epub_module.EPUB_FORMAT
    )
    elements = markdown_to_elements(markdown, toc_tree)
    return is_image_only(elements)


def _inspect_epub(path: Path, sha256: str, fingerprint: str | None) -> PreflightResult:
    reasons: list[str] = []

    container_reject = check_container(path, "epub")
    if container_reject:
        return PreflightResult(
            str(path), sha256, fingerprint, "epub", None, None, "reject", (container_reject,)
        )

    if epub_module._has_zip64_eocd(path):
        reasons.append("zip64")

    # Info only: ``_normalize_epub`` moves escaping members into the OPF dir.
    escaping = _epub_escaping_hrefs(path)
    if escaping:
        reasons.append(f"escaping_hrefs: {', '.join(escaping)}")

    needs_normalization = _epub_needs_normalization(path)
    with tempfile.TemporaryDirectory(prefix="vorag_preflight_epub_") as tmp:
        parse_path = str(_normalize_epub(path, Path(tmp))) if needs_normalization else str(path)
        if needs_normalization:
            reasons.append("normalized")

        if not _runtime_extras_available():
            reasons.append("skipped: ebooklib/pypandoc not installed (--extra runtime)")
            return PreflightResult(
                str(path), sha256, fingerprint, "epub", None, "epub", "ok", tuple(reasons)
            )

        pandoc_reject = _pandoc_dry_run(parse_path)
        if pandoc_reject:
            reasons.append(pandoc_reject)
            return PreflightResult(
                str(path), sha256, fingerprint, "epub", None, None, "reject", tuple(reasons)
            )

        try:
            image_only = _epub_is_image_only(parse_path, str(path))
        except Exception as exc:  # noqa: BLE001 - a conversion-stage failure is a reject, not a crash
            detail = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            reasons.append(f"pandoc_failed: {detail}")
            return PreflightResult(
                str(path), sha256, fingerprint, "epub", None, None, "reject", tuple(reasons)
            )

    route = "image_epub_ocr" if image_only else "epub"
    return PreflightResult(
        str(path), sha256, fingerprint, "epub", None, route, "ok", tuple(reasons)
    )


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _pdf_page_stats(path: Path) -> tuple[int, float, str]:
    """``(page_count, max_page_side_pt, sampled_text)`` — text sampled only
    across the first ``NOT_A_BOOK_MAX_PAGES`` pages (the only case that needs
    it, and the only case cheap enough to always pay for)."""
    import fitz  # type: ignore[import-untyped, unused-ignore]  # PyMuPDF

    with fitz.open(path) as doc:
        page_count = doc.page_count
        max_side = 0.0
        text_parts: list[str] = []
        for index in range(page_count):
            page = doc[index]
            max_side = max(max_side, _longest_side(page))
            if page_count < NOT_A_BOOK_MAX_PAGES:
                text_parts.append(page.get_text("text"))
    return page_count, max_side, "".join(text_parts)


async def _pdf_kind(path: Path) -> str:
    return await _router_instance()._pdf_kind(str(path))


def _giant_route(pages: int, pdf_kind: str) -> str | None:
    if pages > GIANT_MAX_PAGES:
        return "giant"
    if pages >= GIANT_IMAGE_HEAVY_MIN_PAGES and pdf_kind != "native":
        return "giant"
    return None


def _docling_backend_check(path: Path) -> str | None:
    """Whether docling's PDF backend (pypdfium2) can open ``path``, using the
    parser's own repair step (VOR-1427): ``None`` when it opens as-is,
    ``"unlocked_for_docling"`` (info) when the parser's unencrypted copy opens,
    ``"unreadable_by_docling"`` (reject) otherwise. Skipped (``None``) when
    pypdfium2 isn't installed."""
    try:
        from vo_rag.core.parsing.parsers.docling import pdfium_can_open, unlock_for_pdfium
    except ImportError:
        return None
    try:
        if pdfium_can_open(path):
            return None
    except ImportError:
        return None
    with tempfile.TemporaryDirectory(prefix="vorag_preflight_unlock_") as tmp:
        unlocked = unlock_for_pdfium(path, Path(tmp))
    return "unlocked_for_docling" if unlocked is not None else "unreadable_by_docling"


def _inspect_pdf(path: Path, sha256: str, fingerprint: str | None) -> PreflightResult:
    import asyncio

    reasons: list[str] = []

    container_reject = check_container(path, "pdf")
    if container_reject:
        return PreflightResult(
            str(path), sha256, fingerprint, "pdf", None, None, "reject", (container_reject,)
        )

    pages, max_side, sampled_text = _pdf_page_stats(path)

    if max_side > OVERSIZE_PAGE_PT:
        # Info only: the docling parser rescales these pages before parsing.
        reasons.append("oversize_pages")

    docling_reject = _docling_backend_check(path)
    if docling_reject == "unreadable_by_docling":
        return PreflightResult(
            str(path), sha256, fingerprint, "pdf", pages, None, "reject", (*reasons, docling_reject)
        )
    if docling_reject:
        reasons.append(docling_reject)

    if pages < NOT_A_BOOK_MAX_PAGES and len(sampled_text.strip()) < NOT_A_BOOK_MAX_CHARS:
        reasons.append("not_a_book")
        return PreflightResult(
            str(path), sha256, fingerprint, "pdf", pages, None, "reject", tuple(reasons)
        )

    kind = asyncio.run(_pdf_kind(path))
    route = _giant_route(pages, kind) or ("native" if kind == "native" else "ocr")
    return PreflightResult(
        str(path), sha256, fingerprint, "pdf", pages, route, "ok", tuple(reasons)
    )


# ---------------------------------------------------------------------------
# Any other supported format (mobi, docx, pptx, html, xlsx, text): the
# incident table has no failures in these formats, so preflight limits
# itself to detection + the filename-pattern not_a_book check.
# ---------------------------------------------------------------------------


def _inspect_other(path: Path, sha256: str, fingerprint: str | None, kind: str) -> PreflightResult:
    return PreflightResult(str(path), sha256, fingerprint, kind, None, kind, "ok", ())


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


def _catalog_duplicate(sha256: str, fingerprint: str | None, catalog: dict[str, Any]) -> str | None:
    """The catalog ``source_id`` this file's sha256/fingerprint already
    matches, or ``None``. Checked strongest-first: an exact sha256 match
    before a fingerprint (same text, different bytes) match."""
    indexes = catalog.get("indexes") or {}
    for index_name, key in (("by_sha256", sha256), ("by_fingerprint", fingerprint)):
        if not key:
            continue
        hits = (indexes.get(index_name) or {}).get(key)
        if hits:
            return str(sorted(hits)[0])
    return None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def inspect_file(path: Path, *, catalog: dict[str, Any] | None = None) -> PreflightResult:
    """Preflight one file: duplicate (catalog only — see ``inspect_batch`` for
    in-batch duplicates) -> format detection -> format-specific checks.

    Never raises: any unexpected exception during a check becomes a
    ``reject: preflight_error: ...`` result rather than aborting the caller.
    """
    path = Path(path)
    try:
        sha256 = file_sha256(path)
    except OSError as exc:
        return PreflightResult(
            str(path), None, None, None, None, None, "reject", (f"unreadable: {exc}",)
        )
    fingerprint = content_fingerprint(path)

    if catalog is not None:
        duplicate_of = _catalog_duplicate(sha256, fingerprint, catalog)
        if duplicate_of:
            return PreflightResult(
                str(path), sha256, fingerprint, None, None, None, f"duplicate_of={duplicate_of}", ()
            )

    try:
        kind = _detect_kind(path)
        if kind is None:
            return PreflightResult(
                str(path), sha256, fingerprint, None, None, None, "reject", ("unknown_type",)
            )

        filename_not_a_book = bool(_NOT_A_BOOK_FILENAME.search(path.name))

        if kind == "epub":
            result = _inspect_epub(path, sha256, fingerprint)
        elif kind == "pdf":
            result = _inspect_pdf(path, sha256, fingerprint)
        else:
            result = _inspect_other(path, sha256, fingerprint, kind)

        if (
            filename_not_a_book
            and result.pages is not None
            and result.pages >= NOT_A_BOOK_FILENAME_MAX_PAGES
        ):
            filename_not_a_book = False
        if filename_not_a_book and result.verdict == "ok":
            result = replace(
                result, route=None, verdict="reject", reasons=(*result.reasons, "not_a_book")
            )
        return result
    except Exception as exc:  # noqa: BLE001 - one bad file must never abort a batch
        detail = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return PreflightResult(
            str(path),
            sha256,
            fingerprint,
            None,
            None,
            None,
            "reject",
            (f"preflight_error: {type(exc).__name__}: {detail}",),
        )


def _verdict_class(verdict: str) -> str:
    return "duplicate" if verdict.startswith("duplicate_of=") else verdict


def _counts(results: list[PreflightResult]) -> dict[str, dict[str, int]]:
    routes: dict[str, int] = {}
    verdicts: dict[str, int] = {}
    for result in results:
        route_key = result.route or "none"
        routes[route_key] = routes.get(route_key, 0) + 1
        verdict_key = _verdict_class(result.verdict)
        verdicts[verdict_key] = verdicts.get(verdict_key, 0) + 1
    return {"route": routes, "verdict": verdicts}


def _in_batch_duplicate_pass(
    paths: list[Path],
) -> tuple[dict[str, PreflightResult], list[Path], dict[str, list[str]]]:
    """Cheap sequential first pass: hash + fingerprint every file (no pandoc,
    no fitz page walk) and flag in-batch duplicates before the expensive
    per-format checks run. Returns ``(results so far, files still needing a
    full inspection, duplicate_groups)``.
    """
    resolved: dict[str, PreflightResult] = {}
    remaining: list[Path] = []
    #: sha256 -> first path seen with that sha256 (also the ``duplicate_of``
    #: identifier for every later occurrence, exact or by fingerprint).
    seen_sha: dict[str, str] = {}
    #: fingerprint -> the sha256 of the first path seen with that fingerprint.
    seen_fingerprint: dict[str, str] = {}
    duplicate_groups: dict[str, list[str]] = {}

    for path in paths:
        key = str(path)
        try:
            sha256 = file_sha256(path)
        except OSError as exc:
            resolved[key] = PreflightResult(
                key, None, None, None, None, None, "reject", (f"unreadable: {exc}",)
            )
            continue
        fingerprint = content_fingerprint(path)

        earlier_sha: str | None = sha256 if sha256 in seen_sha else None
        if earlier_sha is None and fingerprint:
            earlier_sha = seen_fingerprint.get(fingerprint)

        if earlier_sha is not None:
            duplicate_groups.setdefault(earlier_sha, [seen_sha[earlier_sha]]).append(key)
            resolved[key] = PreflightResult(
                key, sha256, fingerprint, None, None, None, f"duplicate_of={earlier_sha}", ()
            )
            continue

        seen_sha[sha256] = key
        if fingerprint:
            seen_fingerprint[fingerprint] = sha256
        remaining.append(path)

    return resolved, remaining, duplicate_groups


def inspect_batch(
    paths: list[Path],
    catalog: dict[str, Any] | None = None,
    *,
    jobs: int = 1,
    timeout_s: float | None = None,
) -> PreflightReport:
    """``inspect_file`` over every path, plus in-batch duplicate grouping:
    two files in the same batch sharing a sha256 or fingerprint are flagged
    ``duplicate_of=<earlier file's sha256>`` on every occurrence after the
    first (the first-seen file is inspected and routed normally).

    ``jobs > 1`` runs the (expensive: pandoc, fitz) per-file inspection of
    every non-duplicate file across a process pool; ``timeout_s`` bounds how
    long any single file may take there -- a file that blows past it becomes
    ``reject: preflight_timeout`` rather than hanging the whole batch.
    ``jobs == 1`` (the default) runs fully in this process, matching
    ``inspect_file``'s own behavior exactly.

    ``concurrent.futures.ProcessPoolExecutor`` is not used here because its
    context manager calls ``shutdown(wait=True)`` on exit -- so even though a per-future
    ``.result(timeout=...)`` correctly gave up waiting on a hung file,
    exiting the ``with`` block still blocked forever on that same
    still-running worker. ``multiprocessing.Pool``'s context manager calls
    ``terminate()`` on exit instead (SIGTERM to every worker,
    finished or not), which is what actually guarantees this function
    returns -- a hung worker is forcibly killed the moment every file's
    result has been collected (or timed out), not waited on.
    """
    ordered = [Path(p) for p in paths]
    resolved, remaining, duplicate_groups = _in_batch_duplicate_pass(ordered)

    if jobs <= 1:
        for path in remaining:
            resolved[str(path)] = inspect_file(path, catalog=catalog)
    else:
        import multiprocessing

        with multiprocessing.Pool(processes=jobs) as pool:
            async_results = {
                str(path): pool.apply_async(inspect_file, (path,), {"catalog": catalog})
                for path in remaining
            }
            for key, async_result in async_results.items():
                try:
                    resolved[key] = async_result.get(timeout=timeout_s)
                except multiprocessing.TimeoutError:
                    resolved[key] = PreflightResult(
                        key, None, None, None, None, None, "reject", ("preflight_timeout",)
                    )
                except Exception as exc:  # noqa: BLE001 - a worker crash rejects that file only
                    resolved[key] = PreflightResult(
                        key,
                        None,
                        None,
                        None,
                        None,
                        None,
                        "reject",
                        (f"preflight_error: {type(exc).__name__}: {exc}",),
                    )
            # __exit__ -> pool.terminate(): forcibly kills any worker still
            # running a timed-out file's check, so this call can never hang
            # past the sum of the per-file timeouts above.

    results = [resolved[str(path)] for path in ordered]
    return PreflightReport(tuple(results), _counts(results), duplicate_groups)


# ---------------------------------------------------------------------------
# Canary sample selection (VOR-1434) -- a stratified pick from an
# ``inspect_batch`` report, run before any file in the batch is ingested.
# ---------------------------------------------------------------------------

#: Category sizes: 3 largest giants, 3 most
#: image-heavy EPUBs, 4 scans, 4 books with the most figures, the rest
#: random native.
CANARY_GIANTS = 3
CANARY_IMAGE_HEAVY_EPUBS = 3
CANARY_SCANS = 4
CANARY_MOST_FIGURES = 4


@dataclass(frozen=True)
class CanarySample:
    """``select_canary``'s output.

    ``results`` is the flat, deduplicated pick in the source report's
    original order -- what ``vo-rag preflight --canary-out`` writes.
    ``buckets`` records which stratification bucket picked each path (in
    pick order within the bucket), for a caller that wants to show its work.
    """

    results: tuple[PreflightResult, ...]
    buckets: dict[str, tuple[str, ...]]


def select_canary(report: PreflightReport, n: int = 20, *, seed: int = 0) -> CanarySample:
    """A stratified sample of ``report``'s ``ok`` files for the pre-batch
    canary run: the ``CANARY_GIANTS`` largest giants,
    the ``CANARY_IMAGE_HEAVY_EPUBS`` most image-heavy EPUBs,
    ``CANARY_SCANS`` scans, ``CANARY_MOST_FIGURES`` books with the most
    figures, and the rest random native -- up to ``n`` total. Only
    ``verdict == "ok"`` files are eligible; a reject or duplicate was never
    going to be ingested.

    Preflight is a lightweight structural check, not a parse, so it never
    computes an embedded-image or figure count for any file (that's a
    post-ingest Qdrant read, ``health.BookStats.figure_chunks``,
    which doesn't exist for a book that hasn't been ingested). "Most
    image-heavy" and "most figures" therefore use the only signals
    ``PreflightResult`` actually carries, not a fabricated score:

    - *image-heavy EPUBs*: ``route == "image_epub_ocr"`` is the real
      ``is_image_only`` classification the production EPUB parser already
      computed (preflight never re-derives a detection rule). It is boolean,
      not a density ranking, so qualifying EPUBs are taken in the report's
      own order -- deterministic, not invented.
    - *most figures*: ``pages`` (a real, already-computed field) stands in
      for figure count -- longer native-text PDFs are the ones most likely
      to carry the most figures. This is a documented proxy; a genuine
      ranking needs a per-book figure/image count persisted by preflight or
      ingest, which canary selection does not add.

    Deterministic except for the final random-native fill, which is seeded
    by ``seed`` so a rerun (or a test) reproduces the same pick.
    """
    import random

    ok = [r for r in report.results if r.verdict == "ok"]
    chosen: set[str] = set()
    buckets: dict[str, tuple[str, ...]] = {}

    def _pick(name: str, pool: list[PreflightResult], count: int, key: Any) -> None:
        candidates = sorted((r for r in pool if r.path not in chosen), key=key)
        picked = candidates[:count]
        chosen.update(r.path for r in picked)
        buckets[name] = tuple(r.path for r in picked)

    _pick(
        "giants",
        [r for r in ok if r.route == "giant"],
        CANARY_GIANTS,
        lambda r: (-(r.pages or 0), r.path),
    )
    _pick(
        "image_heavy_epubs",
        [r for r in ok if r.kind == "epub" and r.route == "image_epub_ocr"],
        CANARY_IMAGE_HEAVY_EPUBS,
        lambda r: r.path,
    )
    _pick(
        "scans",
        [r for r in ok if r.route == "ocr"],
        CANARY_SCANS,
        lambda r: (-(r.pages or 0), r.path),
    )
    _pick(
        "most_figures",
        [r for r in ok if r.kind == "pdf" and r.route == "native"],
        CANARY_MOST_FIGURES,
        lambda r: (-(r.pages or 0), r.path),
    )

    remaining_native = [r for r in ok if r.route == "native" and r.path not in chosen]
    remaining_slots = max(0, n - len(chosen))
    rng = random.Random(seed)
    random_pick = rng.sample(remaining_native, k=min(remaining_slots, len(remaining_native)))
    chosen.update(r.path for r in random_pick)
    buckets["random_native"] = tuple(sorted(r.path for r in random_pick))

    results = tuple(r for r in ok if r.path in chosen)
    return CanarySample(results, buckets)
