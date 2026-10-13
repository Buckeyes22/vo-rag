"""Admit genuine PDF page recovery by replaying the normal parser and splice.

Historical Garage bytes stay immutable. Every admission verifies the baseline
against Garage, replays a finite source in isolated workers of at most five physical pages,
and compares the complete candidate to the normal splice of that baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from vo_rag.core.ingest.repair_evidence import (
    RepairEvidence,
    SourceArtifact,
    SourceArtifactRole,
)
from vo_rag.core.ingest.representation_receipt import canonical_sha256
from vo_rag.core.parsing.contract import ParsedDocument, ParsedElement
from vo_rag.core.parsing.parsers.pdf_formula import verify_equation_crops
from vo_rag.core.parsing.parsers.pdf_page_rebuild import splice_pages
from vo_rag.core.storage.figures import FigureStore
from vo_rag.ops import repair_legacy_parse
from vo_rag.ops.repair_legacy_parse import LegacyParseBlocked

_MAX_REPLAY_BYTES = 64 * 1024 * 1024
_REPLAY_TIMEOUT_SECONDS = 120
_MAX_SOURCE_PAGES = 512
_MAX_WORKER_PAGES = 5
_MAX_REPLAY_CONCURRENCY = 2
_PARSER_ID = "docling.physical_pages.no_ocr.v1"


def page_parser_recipe_sha256() -> str:
    """Bind the offline physical-page parser options and installed runtimes."""
    try:
        versions = {
            name: importlib.metadata.version(name)
            for name in ("docling", "docling-core", "docling-ibm-models", "torch", "PyMuPDF")
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise LegacyParseBlocked("page_recovery_parser_runtime_unavailable") from exc
    return canonical_sha256(
        {
            "parser_id": _PARSER_ID,
            "versions": versions,
            "parser_code": {
                path: hashlib.sha256(
                    (Path(__file__).resolve().parents[2] / path).read_bytes()
                ).hexdigest()
                for path in (
                    "vo_rag/core/parsing/parsers/docling_mapping.py",
                    "vo_rag/core/parsing/parsers/pdf_formula.py",
                    "vo_rag/core/storage/figures.py",
                )
            },
            "do_ocr": False,
            "offline": True,
            "max_source_pages": _MAX_SOURCE_PAGES,
            "max_worker_pages": _MAX_WORKER_PAGES,
            "max_concurrency": _MAX_REPLAY_CONCURRENCY,
            "max_replay_bytes": _MAX_REPLAY_BYTES,
            "output_reservation": "equal_per_batch.v1",
            "crop_output_reservation": "same_worker_share_with_json_across_physical_pages.v1",
            "worker_timeout_seconds": _REPLAY_TIMEOUT_SECONDS,
            "timeout_fallback": "once_single_pages_same_batch_share_and_deadline.v1",
        }
    )


def _read_bound_bytes(path: Path, size: int, digest: str, limit: int) -> bytes:
    """Read at most the approved size plus one byte, even if a file grows."""
    if size > limit or size < 0:
        raise LegacyParseBlocked("page_recovery_artifact_size_mismatch")
    try:
        with path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != size:
                raise LegacyParseBlocked("page_recovery_artifact_size_mismatch")
            raw = stream.read(size + 1)
    except OSError as exc:
        raise LegacyParseBlocked("page_recovery_artifact_unavailable") from exc
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise LegacyParseBlocked("page_recovery_artifact_digest_mismatch")
    return raw


def _read_bound_parse(artifact: SourceArtifact, limit: int) -> ParsedDocument:
    raw = _read_bound_bytes(artifact.path, artifact.size_bytes, artifact.sha256, limit)
    try:
        return ParsedDocument.model_validate_json(raw)
    except ValueError as exc:
        raise LegacyParseBlocked("page_recovery_parse_invalid") from exc


async def _replay_batch(
    original: SourceArtifact,
    pages: tuple[int, ...],
    timeout: float,
    *,
    max_output_bytes: int = _MAX_REPLAY_BYTES,
) -> dict[int, list[ParsedElement]]:
    if not 0 < max_output_bytes <= _MAX_REPLAY_BYTES:
        raise LegacyParseBlocked("page_recovery_replay_output_over_budget")
    with tempfile.TemporaryDirectory(prefix="vo-rag-page-recovery-") as directory:
        root = Path(directory)
        os.chmod(root, 0o700)
        output = root / "replayed-pages.json"
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "vo_rag.ops.repair_page_parse",
            "--original",
            str(original.path),
            "--original-sha256",
            original.sha256,
            "--original-size",
            str(original.size_bytes),
            "--pages",
            ",".join(str(page) for page in pages),
            "--output",
            str(output),
            "--max-output-bytes",
            str(max_output_bytes),
            env=environment,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except BaseException as exc:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    await process.wait()
            if isinstance(exc, TimeoutError):
                raise LegacyParseBlocked("page_recovery_normal_parser_replay_timed_out") from exc
            raise
        if process.returncode != 0 or not output.is_file():
            raise LegacyParseBlocked("page_recovery_normal_parser_replay_failed")
        if output.stat().st_size > max_output_bytes:
            raise LegacyParseBlocked("page_recovery_replay_output_over_budget")
        try:
            with output.open("rb") as stream:
                raw_output = stream.read(max_output_bytes + 1)
            if len(raw_output) > max_output_bytes:
                raise LegacyParseBlocked("page_recovery_replay_output_over_budget")
            value = json.loads(raw_output)
            if not isinstance(value, dict) or set(value) != {str(page) for page in pages}:
                raise ValueError("physical replay page set differs")
            return {
                page: [
                    ParsedElement.model_validate_json(json.dumps(row)) for row in value[str(page)]
                ]
                for page in pages
            }
        except (TypeError, ValueError, KeyError) as exc:
            raise LegacyParseBlocked("page_recovery_replay_output_invalid") from exc


async def _replay_pages(
    original: SourceArtifact, pages: tuple[int, ...], timeout: float
) -> dict[int, list[ParsedElement]]:
    """Replay a finite source with bounded workers inside one aggregate deadline."""
    if (
        not pages
        or len(pages) > _MAX_SOURCE_PAGES
        or pages != tuple(sorted(set(pages)))
        or min(pages) <= 0
    ):
        raise LegacyParseBlocked("page_recovery_physical_page_scope_invalid")
    batch_count = (len(pages) + _MAX_WORKER_PAGES - 1) // _MAX_WORKER_PAGES
    batch_output_limit = _MAX_REPLAY_BYTES // batch_count
    semaphore = asyncio.Semaphore(_MAX_REPLAY_CONCURRENCY)
    stopped = asyncio.Event()
    replayed: dict[int, list[ParsedElement]] = {}
    total_bytes = 0
    deadline = time.monotonic() + timeout

    async def run_batch(batch: tuple[int, ...]) -> None:
        nonlocal total_bytes
        async with semaphore:
            if stopped.is_set():
                raise LegacyParseBlocked("page_recovery_prior_batch_failed")
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LegacyParseBlocked("page_recovery_wall_budget_exhausted")
                try:
                    rows = await _replay_batch(
                        original,
                        batch,
                        min(_REPLAY_TIMEOUT_SECONDS, remaining),
                        max_output_bytes=batch_output_limit,
                    )
                except LegacyParseBlocked as exc:
                    if (
                        str(exc) != "page_recovery_normal_parser_replay_timed_out"
                        or len(batch) == 1
                    ):
                        raise
                    # A slow page can exhaust a multi-page worker's allowance.
                    # Keep this concurrency slot and retry each page only once,
                    # sharing the original output reservation and deadline.
                    rows = {}
                    page_output_limit = batch_output_limit // len(batch)
                    for page in batch:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise LegacyParseBlocked("page_recovery_wall_budget_exhausted") from exc
                        one = await _replay_batch(
                            original,
                            (page,),
                            min(_REPLAY_TIMEOUT_SECONDS, remaining),
                            max_output_bytes=page_output_limit,
                        )
                        if set(one) != {page}:
                            raise LegacyParseBlocked("page_recovery_replay_output_invalid") from exc
                        rows.update(one)
                if set(rows) != set(batch):
                    raise LegacyParseBlocked("page_recovery_replay_output_invalid")
                encoded = json.dumps(
                    {
                        str(page): [row.model_dump(mode="json") for row in rows[page]]
                        for page in batch
                    },
                    sort_keys=True,
                ).encode()
                total_bytes += len(encoded)
                if total_bytes > _MAX_REPLAY_BYTES:
                    raise LegacyParseBlocked("page_recovery_replay_output_over_budget")
                replayed.update(rows)
            except BaseException:
                stopped.set()
                raise

    tasks = [
        asyncio.create_task(run_batch(pages[start : start + _MAX_WORKER_PAGES]))
        for start in range(0, len(pages), _MAX_WORKER_PAGES)
    ]
    try:
        async with asyncio.timeout(timeout):
            await asyncio.gather(*tasks)
    except BaseException as exc:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if isinstance(exc, TimeoutError):
            raise LegacyParseBlocked("page_recovery_wall_budget_exhausted") from exc
        raise
    return {page: replayed[page] for page in pages}


async def verify_repair_parse(
    evidence: RepairEvidence,
    parsed: ParsedDocument,
    settings: Any,
    pool: Any,
    *,
    deadline: float | None = None,
) -> dict[str, object]:
    """Verify an unchanged cache or genuine recovery within the remaining job time."""
    if evidence.recovered_page_parse is None:
        binding = await repair_legacy_parse.verify_legacy_parse(evidence, parsed, settings, pool)
        if _has_equation_crops(parsed):
            used = 2 * (evidence.original.size_bytes + evidence.accepted_parse.size_bytes)
            used += int(getattr(getattr(evidence, "census", None), "evidence_bytes", 0))
            used += int(getattr(evidence, "evidence_snapshot_bytes", 0))
            lineage = getattr(evidence, "reddit_lineage", None)
            if lineage is not None:
                used += lineage.archive.size_bytes + lineage.manifest.size_bytes
            binding["equation_crop_files"] = _verify_crop_files(
                parsed, settings, evidence.budget.max_evidence_bytes - used
            )
        return binding
    limit = (
        deadline if deadline is not None else time.monotonic() + evidence.budget.max_wall_seconds
    )
    if time.monotonic() >= limit:
        raise LegacyParseBlocked("page_recovery_wall_budget_exhausted")
    try:
        async with asyncio.timeout_at(limit):
            return await _verify_recovered_parse(evidence, parsed, settings, pool, limit)
    except TimeoutError as exc:
        raise LegacyParseBlocked("page_recovery_wall_budget_exhausted") from exc


def _has_equation_crops(parsed: ParsedDocument) -> bool:
    return any(
        isinstance(element.metadata.get(key), dict)
        and element.metadata[key].get("status") == "source_bound"
        for element in parsed.elements
        for key in ("equation_visual", "table_visual", "source_page_visual")
    )


def _verify_crop_files(
    parsed: ParsedDocument, settings: Any, remaining_bytes: int
) -> dict[str, int]:
    try:
        return verify_equation_crops(parsed, FigureStore(settings.figures.root), remaining_bytes)
    except (OSError, ValueError) as exc:
        raise LegacyParseBlocked("page_recovery_equation_crop_unavailable_or_changed") from exc


async def _verify_recovered_parse(
    evidence: RepairEvidence, parsed: ParsedDocument, settings: Any, pool: Any, deadline: float
) -> dict[str, object]:
    """Bind all actual evidence bytes before reading the immutable recovery artifacts."""
    recovery = evidence.recovered_page_parse
    if recovery is None:
        raise LegacyParseBlocked("page_recovery_evidence_missing")
    maximum = evidence.budget.max_evidence_bytes
    artifacts = (
        evidence.original,
        evidence.accepted_parse,
        recovery.baseline_parse,
        recovery.recovered_pages_parse,
    )
    try:
        sizes = [artifact.path.stat().st_size for artifact in artifacts]
    except OSError as exc:
        raise LegacyParseBlocked("page_recovery_artifact_unavailable") from exc
    if any(size != artifact.size_bytes for size, artifact in zip(sizes, artifacts, strict=True)):
        raise LegacyParseBlocked("page_recovery_artifact_size_mismatch")
    # Include both the local evidence and original/cache bytes fetched for the
    # baseline check. Never trust a caller's snapshot byte declaration alone.
    actual_bytes = sum(sizes) + evidence.original.size_bytes + recovery.baseline_parse.size_bytes
    actual_bytes += int(getattr(getattr(evidence, "census", None), "evidence_bytes", 0))
    actual_bytes += int(getattr(evidence, "evidence_snapshot_bytes", 0))
    if actual_bytes > maximum:
        raise LegacyParseBlocked("page_recovery_aggregate_evidence_budget_exceeded")
    baseline = _read_bound_parse(recovery.baseline_parse, maximum)
    recovered = _read_bound_parse(recovery.recovered_pages_parse, maximum)
    candidate = _read_bound_parse(evidence.accepted_parse, maximum)
    if candidate != parsed:
        raise LegacyParseBlocked("page_recovery_candidate_object_mismatch")
    baseline_artifact = recovery.baseline_parse.model_copy(
        update={"role": SourceArtifactRole.ACCEPTED_PARSE}
    )
    baseline_evidence = evidence.model_copy(
        update={"accepted_parse": baseline_artifact, "recovered_page_parse": None}
    )
    baseline_binding = await repair_legacy_parse.verify_legacy_parse(
        baseline_evidence, baseline, settings, pool
    )
    expected_id = f"vo-rag-corpus-sha256:{evidence.original.sha256}"
    if any(
        doc.document_id != evidence.original.sha256 or doc.source_id != expected_id
        for doc in (baseline, recovered, candidate)
    ):
        raise LegacyParseBlocked("page_recovery_source_identity_mismatch")
    pages = recovery.physical_pages
    if any(page > baseline.total_pages for page in pages):
        raise LegacyParseBlocked("page_recovery_physical_page_out_of_range")
    if not recovered.elements or any(
        element.page_number not in pages for element in recovered.elements
    ):
        raise LegacyParseBlocked("page_recovery_element_outside_approved_pages")
    if page_parser_recipe_sha256() != recovery.parser_recipe_sha256:
        raise LegacyParseBlocked("page_recovery_parser_recipe_changed")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise LegacyParseBlocked("page_recovery_wall_budget_exhausted")
    replayed = await _replay_pages(evidence.original, pages, remaining)
    if any(not rows or any(e.page_number != page for e in rows) for page, rows in replayed.items()):
        raise LegacyParseBlocked("page_recovery_replayed_page_identity_mismatch")
    actual_elements = [element for page in pages for element in replayed[page]]
    if actual_elements != recovered.elements:
        raise LegacyParseBlocked("page_recovery_differs_from_genuine_parser_output")
    expected = splice_pages(baseline, replayed)
    if expected != candidate:
        raise LegacyParseBlocked("page_recovery_candidate_differs_from_normal_splice")
    crop_receipt: dict[str, int] = {"occurrences": 0, "unique_images": 0, "bytes": 0}
    if _has_equation_crops(candidate):
        crop_receipt = _verify_crop_files(candidate, settings, maximum - actual_bytes)
    return {
        "schema": "source-bound-physical-page-recovery.v1",
        "baseline_current_byte_binding": baseline_binding,
        "original_sha256": evidence.original.sha256,
        "baseline_parse_sha256": recovery.baseline_parse.sha256,
        "recovered_pages_parse_sha256": recovery.recovered_pages_parse.sha256,
        "accepted_parse_sha256": evidence.accepted_parse.sha256,
        "physical_pages": list(pages),
        "parser_recipe_sha256": recovery.parser_recipe_sha256,
        "normal_parser_replay_verified": True,
        "complete_normal_splice_verified": True,
        "equation_crop_files": crop_receipt,
        "historical_cache_replaced": False,
    }


def _worker(
    original: Path,
    digest: str,
    size: int,
    pages: tuple[int, ...],
    output: Path,
    max_output_bytes: int = _MAX_REPLAY_BYTES,
) -> None:
    """Run only the normal offline Docling converter on verified copied bytes."""
    from vo_rag.core.parsing.parsers.docling import DoclingParser
    from vo_rag.core.parsing.parsers.docling_mapping import docling_to_parsed_document
    from vo_rag.core.storage.figures import FigureStore

    if (
        not pages
        or len(pages) > _MAX_WORKER_PAGES
        or pages != tuple(sorted(set(pages)))
        or min(pages) <= 0
    ):
        raise ValueError("invalid physical page scope")
    if not 0 < max_output_bytes <= _MAX_REPLAY_BYTES:
        raise ValueError("invalid finite replay output budget")
    raw = _read_bound_bytes(original, size, digest, size)
    if not raw.startswith(b"%PDF-") or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("exact original PDF bytes differ")
    source = output.parent / "verified-original.pdf"
    with source.open("xb") as stream:
        stream.write(raw)
    os.chmod(source, 0o600)
    parser = DoclingParser(FigureStore(output.parent / "figures"), do_ocr=False)
    converter = parser._get_docling_parse_converter()
    result: dict[str, object] = {}
    remaining_crop_bytes = max_output_bytes
    for page in pages:
        converted = converter.convert(str(source), page_range=(page, page))
        doc = docling_to_parsed_document(
            converted.document,
            parser._figures,
            str(source),
            "pdf",
            src_hash=digest,
            equation_crop_max_bytes=remaining_crop_bytes,
        )
        remaining_crop_bytes -= sum(
            int(visual.get("size_bytes", 0))
            for element in doc.elements
            for key in ("equation_visual", "table_visual", "source_page_visual")
            if isinstance((visual := element.metadata.get(key)), dict)
            and visual.get("status") == "source_bound"
        )
        if remaining_crop_bytes < 0:
            raise ValueError("bounded equation crop output exceeded")
        if not doc.elements or any(element.page_number != page for element in doc.elements):
            raise ValueError("normal parser did not return the requested physical page")
        result[str(page)] = [element.model_dump(mode="json") for element in doc.elements]
    encoded = json.dumps(result, sort_keys=True).encode()
    if len(encoded) > remaining_crop_bytes:
        raise ValueError("bounded parser output exceeded")
    with output.open("xb") as stream:
        stream.write(encoded)
    os.chmod(output, 0o600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--original-sha256", required=True)
    parser.add_argument("--original-size", type=int, required=True)
    parser.add_argument("--pages", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-output-bytes", type=int, default=_MAX_REPLAY_BYTES)
    arguments = parser.parse_args()
    _worker(
        arguments.original,
        arguments.original_sha256,
        arguments.original_size,
        tuple(int(page) for page in arguments.pages.split(",")),
        arguments.output,
        arguments.max_output_bytes,
    )
