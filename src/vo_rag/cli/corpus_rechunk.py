"""`vo-rag corpus rechunk-preview` (VOR-1452) -- read-only chunker verification.

For each source id: load its stored ``ParsedDocument`` (or, with ``--reparse``,
a fresh docling parse of the original upload), rebuild extraction composites
and chunks with the code on ``PYTHONPATH`` -- exactly what a re-ingest would
produce -- and compare against the book's *live* chunks in Qdrant. Strictly
read-only: no embedding, no Qdrant/Postgres/Garage writes, and a re-parse's
figures go to a throwaway directory, not the live figure store. Proves the
chunker fixes on real books before a corpus re-ingest.

The metric logic lives in ``vo_rag.core.corpus.rechunk_preview`` so it is unit-testable
without Garage/Qdrant/Postgres. Resources come from
``vo_rag.cli.corpus_ocr._resources`` (pool, garage, bucket, qdrant client),
the same seam ``corpus ocr-farm`` uses.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any

import click

from vo_rag.cli.corpus_ocr import _resources
from vo_rag.core.chunking.hierarchical import HierarchicalChunker
from vo_rag.core.chunking.models import RetrievalChunk
from vo_rag.core.corpus.health import BOOKS_RAW_COLLECTION, scroll_book_payloads
from vo_rag.core.corpus.rechunk_preview import (
    RawCounts,
    add_counts,
    carry_over,
    chunk_view_from_payload,
    chunk_view_from_retrieval,
    coverage_baseline,
    input_coverage,
    rates,
    raw_counts,
    sentence_coverage,
    word_coverage,
)
from vo_rag.core.corpus.source_format import SHA_ENVELOPE_MATCH, source_sha256
from vo_rag.core.parsing.contract import ParsedDocument
from vo_rag.pipelines.book_rag.boilerplate import filter_boilerplate_elements
from vo_rag.pipelines.book_rag.ocr_quality import assess
from vo_rag.pipelines.book_rag.steps.chunk_embed import do_build_chunks
from vo_rag.pipelines.book_rag.visual_sidecar import build_extraction_composites

#: One completed envelope per book, newest first; its garage_key is only
#: needed for --reparse (the original upload), same lookup as the prototype
#: and vo_rag.core.corpus.source_format's own docstring.
_ENVELOPE_SQL = f"""
SELECT e.id,
       COALESCE(NULLIF(e.raw_content->>'garage_key', ''), e.source_data->>'garage_key')
           AS garage_key
FROM intake_envelopes e, (SELECT $1::text AS sha256) k
WHERE {SHA_ENVELOPE_MATCH} AND e.status = 'completed'
ORDER BY e.updated_at DESC LIMIT 1
"""  # nosec B608 -- SHA_ENVELOPE_MATCH is source_format's own static literal

_LIVE_PAYLOAD_FIELDS = ["content", "section_path", "has_image", "chunk_type"]
_RATE_FIELDS = (
    "chunks",
    "frag",
    "frag_text",
    "tiny_text",
    "figures",
    "junk",
    "back_matter",
    "garb",
    "fig_bleed",
    "path_quality",
)


def _gpu_parse_available() -> bool:
    """True when this host can run docling inference on CUDA.

    the operator's rule: OCR/inference never runs on the CPU, on any machine
    (mirrors ``DoclingParser``'s own ``_require_cuda`` guard). ``--reparse``
    checks this once, up front, rather than mid-run.
    """
    try:
        import torch  # type: ignore[import-not-found]  # runtime-only dep
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


async def _build_gpu_parser_router(garage: Any, bucket: str, figures_root: Path) -> Any:
    """The book_rag ingest's parser router, with docling pinned to CUDA.

    Built by ``vo_rag.core.parsing.factory.build_parser_router`` from the
    same settings the ingest worker uses, so a re-parse
    here is what a re-ingest would parse. Two things differ, both on purpose:

    - the device is pinned to "cuda" -- ``--reparse`` never runs docling on
      the CPU, whatever ``VORAG_DOCLING_DEVICE`` says;
    - its ``FigureStore`` writes under ``figures_root``, a throwaway
      directory: parsing writes every extracted picture at its deterministic
      served key, so a store on ``settings.figures.root`` would rewrite the
      files live Qdrant points' ``image_ref`` resolve to.

    ``garage``/``bucket`` are the command's own clients, the ones the factory
    would build from the same settings (the OCR farm cache is only read).
    """
    from vo_rag.core.config import get_settings
    from vo_rag.core.parsing.factory import build_parser_router
    from vo_rag.core.storage.figures import FigureStore

    return build_parser_router(
        get_settings(),
        FigureStore(figures_root),
        garage=garage,
        bucket=bucket,
        docling_device="cuda",
    )


async def _reparsed_document(
    *,
    garage: Any,
    bucket: str,
    garage_key: str,
    sha: str,
    source_id: str,
    parser_router: Any,
) -> ParsedDocument:
    """Re-parse the original upload with ``parser_router``; never written to Garage."""
    data = await garage.get_bytes(bucket, garage_key)
    with tempfile.TemporaryDirectory(prefix="vorag_rechunk_preview_") as tmp:
        local = Path(tmp) / (Path(garage_key).name or sha)
        local.write_bytes(data)
        parsed: ParsedDocument = await parser_router.parse(str(local), sha)
    parsed.document_id = sha
    parsed.source_id = source_id
    return parsed


def _read_source_ids(sources: tuple[str, ...], sources_file: str | None) -> list[str]:
    """``--source`` values then ``--sources-file`` lines, deduped, order kept."""
    ids: list[str] = list(sources)
    if sources_file:
        with open(sources_file, encoding="utf-8") as handle:
            ids.extend(line.strip() for line in handle if line.strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            ordered.append(sid)
    return ordered


def _emit(payload: dict[str, Any]) -> None:
    click.echo(json.dumps(payload, default=str))


def _print_human_sides(label: str, live: dict[str, Any], new: dict[str, Any]) -> None:
    click.echo(label)
    for field in _RATE_FIELDS:
        click.echo(f"  {field:<10} live={live[field]!s:<8} new={new[field]!s}")


class _SourceError(Exception):
    """One source id couldn't be turned into a record; message is user-facing."""


async def _process_source(
    sid: str,
    *,
    pool: Any,
    garage: Any,
    bucket: str,
    qdrant: Any,
    reparse: bool,
    parser_router: Any | None,
    show_drops: int = 0,
) -> tuple[dict[str, Any], RawCounts, RawCounts]:
    """One book's ``(record, live_counts, new_counts)``; raises ``_SourceError``."""
    sha = source_sha256(sid)
    if sha is None:
        raise _SourceError("source id carries no sha256")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(_ENVELOPE_SQL, sha)
    if row is None:
        raise _SourceError("no completed envelope")

    if reparse:
        garage_key = row["garage_key"]
        if not garage_key:
            raise _SourceError("completed envelope has no garage_key to re-parse")
        parsed = await _reparsed_document(
            garage=garage,
            bucket=bucket,
            garage_key=garage_key,
            sha=sha,
            source_id=sid,
            parser_router=parser_router,
        )
    else:
        raw = await garage.get_bytes(bucket, f"parsed/{row['id']}/parsed.json")
        parsed = ParsedDocument.model_validate_json(raw)

    composites = await build_extraction_composites(parsed, document_metadata={})
    ocr_dropped: list[RetrievalChunk] = []
    try:
        new_chunks = do_build_chunks(
            parsed, composites, book_metadata={"domain_tags": []}, ocr_dropped=ocr_dropped
        )
    except RuntimeError as exc:
        # Zero chunks (the OCR gate may drop everything), degenerate or runaway
        # output: this book's error line; the batch goes on.
        raise _SourceError(str(exc)) from exc
    new_views = [chunk_view_from_retrieval(c) for c in new_chunks]
    # The chunker's own input: the parse after the reviewed-boilerplate filter
    # do_build_chunks applies, as body text the chunker does not skip as page
    # furniture (input_cov).
    chunker_input, _ = filter_boilerplate_elements(parsed, None)
    input_texts = HierarchicalChunker().body_input_texts(chunker_input)

    live_views = [
        chunk_view_from_payload(point.payload or {})
        async for point in scroll_book_payloads(
            qdrant, sid, with_payload=_LIVE_PAYLOAD_FIELDS, collection=BOOKS_RAW_COLLECTION
        )
    ]

    live_counts = raw_counts(live_views)
    new_counts = raw_counts(new_views)
    # Coverage is measured against live text the OCR gate keeps: its drops are intended.
    baseline = coverage_baseline(live_views, lambda text: assess(text) == "drop")
    record = {
        "sid": sid,
        "live": rates(live_counts),
        "new": rates(new_counts),
        "live_ocr_noise": len(live_views) - len(baseline),
        # The gate's drops among the *new* chunks: coverage is
        # measured against live text the gate keeps, so a clean text the gate
        # wrongly drops would never show there -- it shows here.
        "new_ocr_drops": len(ocr_dropped),
        "coverage": word_coverage(baseline, new_views),
        "sent_cov": sentence_coverage(baseline, new_views),
        # Of the sentences the chunker was given, the share its output holds
        # (the OCR gate's drops aside): parse-independent loss.
        "input_cov": input_coverage(
            input_texts, new_views, [chunk.content for chunk in ocr_dropped]
        ),
        "carry_over": carry_over(live_views, new_views),
    }
    if show_drops > 0:
        record["ocr_drop_samples"] = [chunk.content for chunk in ocr_dropped[:show_drops]]
    return record, live_counts, new_counts


def register(group: click.Group) -> None:
    @group.command("rechunk-preview")
    @click.option(
        "--source",
        "sources",
        multiple=True,
        help="Book source id to re-chunk and compare (repeatable).",
    )
    @click.option(
        "--sources-file",
        "sources_file",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="File of source ids, one per line (combined with --source).",
    )
    @click.option(
        "--reparse",
        is_flag=True,
        default=False,
        help="Re-parse the original upload with docling on GPU instead of using the "
        "stored parse. Requires CUDA; never runs on CPU. In-memory only -- "
        "nothing is written back to Garage.",
    )
    @click.option(
        "--show-drops",
        "show_drops",
        type=click.IntRange(min=0),
        default=0,
        show_default=True,
        help="Also print the first N texts the OCR gate drops from each book's new chunks.",
    )
    @click.option(
        "--json", "as_json", is_flag=True, default=False, help="Print one JSON object per line."
    )
    def rechunk_preview(
        sources: tuple[str, ...],
        sources_file: str | None,
        reparse: bool,
        show_drops: int,
        as_json: bool,
    ) -> None:
        """Re-chunk stored parses with the current chunker and diff against live Qdrant chunks.

        Read-only: no embedding, no Qdrant/Postgres/Garage writes, and with
        --reparse the parser's figures go to a throwaway directory, never the
        live figure store (VOR-1452). Each book reports new_ocr_drops,
        the new chunks the OCR gate drops (--show-drops N prints the first N
        texts). Exits 1 if any source id errors (missing sha256, no completed
        envelope, zero chunks, or -- with --reparse -- no CUDA available);
        the other books and the totals are still reported.
        """
        source_ids = _read_source_ids(sources, sources_file)
        if not source_ids:
            raise click.UsageError("at least one --source or a --sources-file entry is required")
        if reparse and not _gpu_parse_available():
            click.echo(
                "error: --reparse requires a CUDA GPU (docling never runs on the CPU); "
                "none is available on this host",
                err=True,
            )
            raise SystemExit(1)

        async def _run() -> tuple[bool, RawCounts, RawCounts, int]:
            had_error = False
            live_total = RawCounts()
            new_total = RawCounts()
            drops_total = 0
            async with contextlib.AsyncExitStack() as stack:
                pool, garage, bucket, qdrant = await stack.enter_async_context(_resources())
                parser_router = None
                if reparse:
                    figures_root = Path(
                        stack.enter_context(
                            tempfile.TemporaryDirectory(prefix="vorag_rechunk_figures_")
                        )
                    )
                    parser_router = await _build_gpu_parser_router(garage, bucket, figures_root)
                for sid in source_ids:
                    try:
                        record, live_counts, new_counts = await _process_source(
                            sid,
                            pool=pool,
                            garage=garage,
                            bucket=bucket,
                            qdrant=qdrant,
                            reparse=reparse,
                            parser_router=parser_router,
                            show_drops=show_drops,
                        )
                    except _SourceError as exc:
                        _emit({"sid": sid, "error": str(exc)})
                        had_error = True
                        continue
                    if as_json:
                        _emit(record)
                    else:
                        _print_human_sides(f"book: {record['sid']}", record["live"], record["new"])
                        click.echo(
                            f"  coverage={record['coverage']}  sent_cov={record['sent_cov']}  "
                            f"input_cov={record['input_cov']}  "
                            f"carry_over={record['carry_over']}  "
                            f"live_ocr_noise={record['live_ocr_noise']}  "
                            f"new_ocr_drops={record['new_ocr_drops']}"
                        )
                        for number, text in enumerate(record.get("ocr_drop_samples", []), 1):
                            click.echo(f"  ocr drop {number}: {text!r}")
                    live_total = add_counts(live_total, live_counts)
                    new_total = add_counts(new_total, new_counts)
                    drops_total += int(record["new_ocr_drops"])
            return had_error, live_total, new_total, drops_total

        had_error, live_total, new_total, drops_total = asyncio.run(_run())

        live_rates, new_rates = rates(live_total), rates(new_total)
        if as_json:
            _emit({"live": live_rates, "new": new_rates, "new_ocr_drops": drops_total})
        else:
            _print_human_sides("TOTAL", live_rates, new_rates)
            click.echo(f"  new_ocr_drops={drops_total}")

        if had_error:
            raise SystemExit(1)
