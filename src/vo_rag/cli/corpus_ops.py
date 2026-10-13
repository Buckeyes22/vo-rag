"""`vo-rag corpus purge`/`health`/`boilerplate` (VOR-1427).

``purge``: dry run by default -- prints exactly what would be deleted.
``--yes`` applies. Opens the Postgres pool and Qdrant store the same way
``cli/corpus_ocr.py`` does.

``health``: read-only report over the books-raw collection (plus one
Postgres read to tell legacy PDFs from ebooks for books with no
``parser_used``) -- degenerate books, a dominant section, unpaginated page-bucket
titles, leaked test points, boilerplate points, duplicate groups
with a planned delete, and (VOR-1434) empty-figure, low-text-density,
image-only-suspect, too-few-chunks and no-chunks books. Exits 1 when the
report isn't clean.

``boilerplate``: thin wrapper over ``vo_rag.core.corpus.boilerplate_ops``.
Exactly one of ``--scan``/``--delete`` is required; ``--scan`` (and
``--delete`` without ``--yes``) report per-book boilerplate counts in the
books-raw collection read-only, ending ``{"points": N}``; ``--delete`` also
reports (and with ``--yes`` deletes) the matching points in the sterile and
contextualized collections, one ``{"collection": ..., ...}`` line each
(``"absent"`` for a collection that doesn't exist). Only ``--delete --yes``
actually deletes, ending ``{"deleted": N}`` (N = raw points, the same N
``--scan`` reports).

``duplicates``: read-only report over the books-raw collection plus one
Postgres query (VOR-1427). Groups books by normalized title + the
payload's author (``plan_duplicates``), dates each copy by its earliest
completed intake envelope (``envelope_ingested_at``), and writes every
planned row to a CSV, printing ``{"groups": G, "delete": D}``. Never
deletes anything itself -- the operator approves the CSV's ``action == "delete"``
rows before ``corpus purge`` is ever run against them.

``canary-report`` (VOR-1434): read-only gate report over a pre-batch
canary sample (``vo_rag.core.parsing.preflight.select_canary``). Reads each
envelope's status/timestamps and ``metadata->'quality'`` findings
from Postgres, checks Qdrant's payload indexes read-only, and prints the six
gates plus an optional wall-time projection for the full batch. Gates
whose input isn't persisted anywhere (per-stage timing, embedder 429 rate,
Qdrant call p95) are ``unmeasured``, which forces INCOMPLETE rather than
PASS -- see ``vo_rag.core.corpus.canary`` for exactly what each gate reads.
Exits 1 unless the verdict is PASS.
"""

from __future__ import annotations

import asyncio
import csv
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, fields
from typing import Any

import click

from vo_rag.core.corpus.boilerplate_ops import (
    boilerplate_payload_matcher as _boilerplate_payload_matcher,
)
from vo_rag.core.corpus.boilerplate_ops import boilerplate_point_ids as _boilerplate_point_ids
from vo_rag.core.corpus.boilerplate_ops import (
    delete_boilerplate_points as _delete_boilerplate_points,
)
from vo_rag.core.corpus.boilerplate_ops import (
    derived_boilerplate_point_ids as _derived_boilerplate_point_ids,
)
from vo_rag.core.corpus.canary import CanaryReport, run_canary_report
from vo_rag.core.corpus.duplicates import (
    DupRow,
    book_authors,
    envelope_ingested_at,
    plan_duplicates,
)
from vo_rag.core.corpus.health import HealthReport, collect_stats, run_health
from vo_rag.core.corpus.purge import purge_source


def _emit(payload: dict[str, Any]) -> None:
    click.echo(json.dumps(payload, default=str))


@asynccontextmanager
async def _resources() -> AsyncIterator[tuple[Any, Any]]:
    """``(pool, store)`` from settings; both closed on exit."""
    from vo_rag.core.config import get_settings
    from vo_rag.core.db.postgres import close_pool, get_pool
    from vo_rag.core.storage.qdrant import QdrantStore

    settings = get_settings()
    store = QdrantStore(
        settings.qdrant.url,
        settings.qdrant.api_key or None,
        prefer_grpc=settings.qdrant.prefer_grpc,
    )
    try:
        pool = await get_pool(settings.database.url, min_size=1, max_size=4)
        try:
            yield pool, store
        finally:
            await close_pool()
    finally:
        await store._client.close()  # noqa: SLF001


@asynccontextmanager
async def _store_only() -> AsyncIterator[Any]:
    """Just the Qdrant store from settings, closed on exit. ``boilerplate``
    has no Postgres dependency, so it doesn't pay for opening a pool."""
    from vo_rag.core.config import get_settings
    from vo_rag.core.storage.qdrant import QdrantStore

    settings = get_settings()
    store = QdrantStore(
        settings.qdrant.url,
        settings.qdrant.api_key or None,
        prefer_grpc=settings.qdrant.prefer_grpc,
    )
    try:
        yield store
    finally:
        await store._client.close()  # noqa: SLF001


@asynccontextmanager
async def _pool_only() -> AsyncIterator[Any]:
    """Just the Postgres pool from settings, closed on exit. Commands with no
    Qdrant dependency (``tf-books sheet``/``load``/``load-drafts``) use this so
    they don't build a store to close again."""
    from vo_rag.core.config import get_settings
    from vo_rag.core.db.postgres import close_pool, get_pool

    settings = get_settings()
    pool = await get_pool(settings.database.url, min_size=1, max_size=4)
    try:
        yield pool
    finally:
        await close_pool()


def register(group: click.Group) -> None:
    @group.command("purge")
    @click.argument("source_ids", metavar="SOURCE_ID...", nargs=-1, required=True)
    @click.option("--yes", is_flag=True, default=False, help="Actually delete (default: dry run).")
    @click.option(
        "--drop-categories",
        is_flag=True,
        default=False,
        help="Also delete book_categories rows (kept by default; re-ingest reuses them).",
    )
    def purge(source_ids: tuple[str, ...], yes: bool, drop_categories: bool) -> None:
        """Remove every trace of each SOURCE_ID from Qdrant and Postgres."""

        # Emit each id's report as soon as it finishes (not batched at the
        # end) so an already-applied purge is never lost if a later id
        # raises.
        async def _run() -> None:
            async with _resources() as (pool, store):
                for source_id in source_ids:
                    report = await purge_source(
                        store,
                        pool,
                        source_id,
                        apply=yes,
                        keep_categories=not drop_categories,
                    )
                    _emit(asdict(report))

        asyncio.run(_run())
        _emit({"applied": yes})

    @group.command("health")
    @click.option(
        "--source-id",
        "source_ids",
        multiple=True,
        help="Restrict the scan to these source ids (repeatable). Default: every book.",
    )
    @click.option("--json", "as_json", is_flag=True, default=False, help="Print one JSON object.")
    def health(source_ids: tuple[str, ...], as_json: bool) -> None:
        """Report degenerate, dominant-section and unpaginated books, test points,
        boilerplate points, duplicate groups with a planned delete, and (VOR-1434)
        empty-figure, low-text-density, image-only-suspect, too-few-chunks
        and no-chunks books; exit 1 unless clean."""

        async def _run() -> HealthReport:
            async with _resources() as (pool, store):
                return await run_health(
                    store,
                    source_ids=list(source_ids) or None,
                    boilerplate_matcher=_boilerplate_payload_matcher,
                    pool=pool,
                )

        report = asyncio.run(_run())

        if as_json:
            _emit(asdict(report))
        else:
            for line in (
                f"books: {report.books}",
                f"degenerate: {report.degenerate}",
                f"dominant_section: {report.dominant_section}",
                f"unpaginated_page_titles: {report.unpaginated_page_titles}",
                f"test_points: {report.test_points}",
                f"boilerplate_points: {report.boilerplate_points}",
                f"duplicate_groups: {report.duplicate_groups}",
                f"empty_figures: {report.empty_figures}",
                f"low_text_density: {report.low_text_density}",
                f"image_only_suspect: {report.image_only_suspect}",
                f"too_few_chunks: {report.too_few_chunks}",
                f"no_chunks: {report.no_chunks}",
            ):
                click.echo(line)

        if not report.ok():
            raise SystemExit(1)

    @group.command("boilerplate")
    @click.option(
        "--scan",
        "do_scan",
        is_flag=True,
        default=False,
        help="Report boilerplate point counts per book, read-only. Exactly one "
        "of --scan / --delete is required.",
    )
    @click.option(
        "--delete",
        "do_delete",
        is_flag=True,
        default=False,
        help="Delete matching points (dry run -- same report as --scan -- unless "
        "--yes). Exactly one of --scan / --delete is required.",
    )
    @click.option(
        "--yes",
        is_flag=True,
        default=False,
        help="Actually delete, together with --delete (default: dry run).",
    )
    @click.option(
        "--source-id",
        "source_ids",
        multiple=True,
        help="Restrict the scan to these source ids (repeatable). Default: every book.",
    )
    def boilerplate(do_scan: bool, do_delete: bool, yes: bool, source_ids: tuple[str, ...]) -> None:
        """Scan for (`--scan`) or remove (`--delete`, needs `--yes` to apply) boilerplate chunks.

        Exactly one of `--scan`/`--delete` is required; both are explicit,
        always-typed flags (``corpus boilerplate --scan``,
        ``corpus boilerplate --delete``).
        """
        if do_scan == do_delete:  # both False, or both True
            raise click.UsageError("exactly one of --scan or --delete is required")

        async def _run() -> tuple[
            dict[str, list[str]], dict[str, int | str], dict[str, int | str], int | None
        ]:
            async with _store_only() as store:
                ids_filter = list(source_ids) or None
                by_book = await _boilerplate_point_ids(store, source_ids=ids_filter)
                derived_found: dict[str, int | str] = {}
                derived_deleted: dict[str, int | str] = {}
                deleted: int | None = None
                if do_delete:
                    derived = await _derived_boilerplate_point_ids(store, source_ids=ids_filter)
                    for collection, derived_by_book in derived.items():
                        if derived_by_book is None:
                            derived_found[collection] = derived_deleted[collection] = "absent"
                            continue
                        derived_ids = [i for ids in derived_by_book.values() for i in ids]
                        derived_found[collection] = len(derived_ids)
                        if yes:
                            derived_deleted[collection] = await _delete_boilerplate_points(
                                store, derived_ids, collection=collection
                            )
                    if yes:
                        all_ids = [point_id for ids in by_book.values() for point_id in ids]
                        deleted = await _delete_boilerplate_points(store, all_ids)
                return by_book, derived_found, derived_deleted, deleted

        by_book, derived_found, derived_deleted, deleted = asyncio.run(_run())

        if deleted is not None:
            for collection, count in derived_deleted.items():
                _emit({"collection": collection, "deleted": count})
            _emit({"deleted": deleted})
            return

        points = 0
        for source_id, ids in by_book.items():
            count = len(ids)
            _emit({"source_id": source_id, "boilerplate": count})
            points += count
        for collection, found in derived_found.items():
            _emit({"collection": collection, "boilerplate": found})
        _emit({"points": points})

    @group.command("duplicates")
    @click.option(
        "--report",
        "report_path",
        required=True,
        type=click.Path(dir_okay=False, writable=True),
        help="CSV path to write the duplicate-title plan to.",
    )
    def duplicates(report_path: str) -> None:
        """Group duplicate-titled books and plan a keep/delete list.

        Read-only and non-destructive: writes every planned `DupRow` to
        `--report` and prints `{"groups": G, "delete": D}`. Nothing is
        deleted here -- the operator approves the CSV first.
        """

        async def _run() -> list[DupRow]:
            async with _resources() as (pool, store):
                stats = await collect_stats(store, boilerplate_matcher=_boilerplate_payload_matcher)
                ingested_at = await envelope_ingested_at(pool, [b.source_id for b in stats])
                return plan_duplicates(stats, book_authors(stats), ingested_at)

        dup_rows = asyncio.run(_run())

        with open(report_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(DupRow)])
            writer.writeheader()
            for row in dup_rows:
                writer.writerow(asdict(row))

        groups = len({row.group for row in dup_rows})
        delete = sum(1 for row in dup_rows if row.action == "delete")
        _emit({"groups": groups, "delete": delete})

    @group.command("canary-report")
    @click.argument("envelope_ids", metavar="ENVELOPE_ID...", nargs=-1, required=True)
    @click.option(
        "--total-pages",
        type=int,
        default=None,
        help="total page count of the full batch, for the wall-time projection; "
        "if omitted, no projection is printed.",
    )
    @click.option(
        "--lanes",
        "lane_count",
        type=int,
        default=1,
        help="lane count the full batch will run with; the projection "
        "divides the measured per-lane pages/min by it.",
    )
    @click.option(
        "--include-queue-wait",
        is_flag=True,
        default=False,
        help="for a book with no recorded claim time, measure its wall time "
        "from intake (created_at), queue wait included; off by default.",
    )
    @click.option("--json", "as_json", is_flag=True, default=False, help="Print one JSON object.")
    def canary_report(
        envelope_ids: tuple[str, ...],
        total_pages: int | None,
        lane_count: int,
        include_queue_wait: bool,
        as_json: bool,
    ) -> None:
        """Read-only gate report for a pre-batch canary sample (VOR-1434).

        Reads ENVELOPE_ID... from Postgres (status, wall time, and the
        quality findings recorded in `metadata->'quality'`) and checks Qdrant's
        payload indexes read-only. Gates whose input isn't persisted anywhere
        (per-stage timing, embedder 429 rate, Qdrant call p95) are reported
        `unmeasured`, which forces the verdict to INCOMPLETE rather than PASS --
        never faked as a pass. Exits 1 unless the verdict is PASS.
        """

        async def _run() -> CanaryReport:
            async with _resources() as (pool, store):
                return await run_canary_report(
                    pool,
                    store._client,  # noqa: SLF001 - same private access as _resources' own close()
                    envelope_ids,
                    total_pages=total_pages,
                    lane_count=lane_count,
                    include_queue_wait=include_queue_wait,
                )

        report = asyncio.run(_run())

        if as_json:
            _emit(asdict(report))
        else:
            for book in report.books:
                _emit(
                    {
                        "envelope_id": book.envelope_id,
                        "status": book.status,
                        "source_id": book.source_id,
                        "title": book.title,
                        "pages": book.pages,
                        "chunks": book.chunks,
                        "pages_per_min": book.pages_per_min,
                        "wall_time_s": book.wall_time_s,
                        "findings": list(book.findings) if book.findings is not None else None,
                        "error": book.error,
                    }
                )
            for gate in report.gates:
                click.echo(f"{gate.name}: {gate.status} -- {gate.detail}")
            if report.projection is not None:
                p = report.projection
                click.echo(
                    f"projection: {p.formula} = {p.projected_wall_time_s:.0f}s "
                    f"(total_pages={p.total_pages}, "
                    f"pages_per_min_per_lane={p.pages_per_min_per_lane:.2f}, "
                    f"lane_count={p.lane_count}); {p.lane_count_note}"
                )
            click.echo(f"verdict: {report.verdict}")

        if report.verdict != "PASS":
            raise SystemExit(1)
