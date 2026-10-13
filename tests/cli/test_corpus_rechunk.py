"""`vo-rag corpus rechunk-preview` (VOR-1452).

Postgres/Garage/Qdrant come from a monkeypatched `_resources` seam; every
write-shaped method on the fakes raises, so any accidental upsert/
set_payload/delete/put fails the test loudly. Nothing real is touched --
no network.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import pytest
from click.testing import CliRunner

from vo_rag.cli import corpus_rechunk
from vo_rag.cli.main import cli
from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement

SHA_A = "a" * 64
SHA_B = "b" * 64
SID_A = f"vo-rag-corpus-sha256:{SHA_A}"
SID_B = f"vo-rag-corpus-sha256:{SHA_B}"
BUCKET = "vorag-artifacts"


def _parsed_doc(source_id: str) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        source_id=source_id,
        title="T",
        author="A",
        total_pages=1,
        elements=[
            ParsedElement(
                element_id="e-1", element_type=ElementType.TITLE, text="Chapter 1", page_number=1
            ),
            ParsedElement(
                element_id="e-2",
                element_type=ElementType.NARRATIVE_TEXT,
                text=(
                    "Narrative text forming a real chunk body for the test, with plenty "
                    "of words in it so it survives any boilerplate filter easily."
                ),
                page_number=1,
            ),
        ],
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        metadata={},
        parser_used="test",
    )


class _Ctx:
    """An async context manager yielding the fake itself (``pool.acquire()``)."""

    def __init__(self, target: Any) -> None:
        self._target = target

    async def __aenter__(self) -> Any:
        return self._target

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeConn:
    def __init__(self, rows_by_sha: dict[str, dict[str, Any]]) -> None:
        self.rows_by_sha = rows_by_sha
        self.queried_shas: list[str] = []

    async def fetchrow(self, sql: str, sha: str) -> dict[str, Any] | None:
        _ = sql
        self.queried_shas.append(sha)
        return self.rows_by_sha.get(sha)


class FakePool:
    def __init__(self, rows_by_sha: dict[str, dict[str, Any]]) -> None:
        self.conn = FakeConn(rows_by_sha)

    def acquire(self) -> _Ctx:
        return _Ctx(self.conn)


class FakeGarage:
    """``get_bytes`` reads from a fixed object map; every write raises."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.gets: list[str] = []

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        assert bucket == BUCKET
        self.gets.append(key)
        try:
            return self.objects[key]
        except KeyError:
            raise AssertionError(f"unexpected garage get_bytes({key!r})") from None

    async def put_bytes(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"garage write attempted: put_bytes{args, kwargs}")


class FakeQdrant:
    """``AsyncQdrantClient.scroll`` stand-in (one page per source_id filter).

    Modeled on tests/core/corpus/test_health.py's ``_FakeScrollClient``.
    Every mutating method raises.
    """

    def __init__(self, books: dict[str, list[dict[str, Any]]]) -> None:
        self.books = books
        self.scrolled_source_ids: list[str] = []

    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Any,
        with_payload: Any,
        with_vectors: bool,
        limit: int,
        offset: Any,
    ) -> tuple[list[Any], Any]:
        _ = (collection_name, with_payload, with_vectors, limit, offset)
        from types import SimpleNamespace

        condition = scroll_filter.must[0]
        source_id = condition.match.value
        self.scrolled_source_ids.append(source_id)
        payloads = self.books.get(source_id, [])
        points = [SimpleNamespace(payload=p) for p in payloads]
        return points, None

    async def upsert(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"qdrant write attempted: upsert{args, kwargs}")

    async def set_payload(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"qdrant write attempted: set_payload{args, kwargs}")

    async def delete(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"qdrant write attempted: delete{args, kwargs}")


def _lines(output: str) -> list[dict[str, Any]]:
    # structlog's INFO logging (book_rag.chunk.drop_summary etc.) can land on
    # the same captured stream as the CLI's own JSON lines; only the lines
    # that actually parse as JSON are the command's output.
    result = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


@pytest.fixture
def seams(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    pool = FakePool(
        {
            SHA_A: {"id": "ENV-A", "garage_key": "uploads/a.pdf"},
        }
    )
    garage = FakeGarage(
        {
            "parsed/ENV-A/parsed.json": _parsed_doc(SID_A).model_dump_json().encode("utf-8"),
        }
    )
    qdrant = FakeQdrant(
        {
            SID_A: [
                {
                    "content": "Old live content for book A, long enough to not be tiny.",
                    "section_path": "[Book > Chapter 1]",
                    "has_image": False,
                    "chunk_type": "text",
                }
            ]
        }
    )

    @asynccontextmanager
    async def fake_resources() -> Any:
        yield pool, garage, BUCKET, qdrant

    monkeypatch.setattr(corpus_rechunk, "_resources", fake_resources)
    return {"pool": pool, "garage": garage, "qdrant": qdrant}


# ---------------------------------------------------------------------------
# --help / registration
# ---------------------------------------------------------------------------


def test_help_works() -> None:
    result = CliRunner().invoke(cli, ["corpus", "rechunk-preview", "--help"])
    assert result.exit_code == 0, result.output
    assert "--reparse" in result.output
    assert "--json" in result.output
    assert "--sources-file" in result.output


def test_registered_under_corpus() -> None:
    result = CliRunner().invoke(cli, ["corpus", "--help"])
    assert result.exit_code == 0, result.output
    assert "rechunk-preview" in result.output


def test_requires_at_least_one_source() -> None:
    result = CliRunner().invoke(cli, ["corpus", "rechunk-preview"])
    assert result.exit_code != 0
    assert "--source" in result.output


# ---------------------------------------------------------------------------
# source id parsing (flags + file)
# ---------------------------------------------------------------------------


def test_source_ids_parsed_from_flags_and_file(
    seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    sources_file = tmp_path / "sources.txt"
    sources_file.write_text(f"{SID_A}\n\n  {SID_A}  \n")  # dupe + blank line, both ignored

    result = CliRunner().invoke(
        cli,
        [
            "corpus",
            "rechunk-preview",
            "--source",
            SID_A,
            "--sources-file",
            str(sources_file),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    # SID_A given via both --source and the file: queried exactly once (deduped).
    assert seams["pool"].conn.queried_shas == [SHA_A]
    lines = _lines(result.output)
    assert lines[0]["sid"] == SID_A
    assert "TOTAL" not in result.output  # human labels only in the non-json path


def test_default_human_output_includes_book_and_total_labels(seams: dict[str, Any]) -> None:
    result = CliRunner().invoke(cli, ["corpus", "rechunk-preview", "--source", SID_A])
    assert result.exit_code == 0, result.output
    assert f"book: {SID_A}" in result.output
    assert "TOTAL" in result.output


# ---------------------------------------------------------------------------
# missing envelope / invalid source id: error line, continue, exit 1
# ---------------------------------------------------------------------------


def test_missing_envelope_prints_error_line_and_continues(seams: dict[str, Any]) -> None:
    result = CliRunner().invoke(
        cli, ["corpus", "rechunk-preview", "--source", SID_B, "--source", SID_A, "--json"]
    )

    assert result.exit_code == 1, result.output
    lines = _lines(result.output)
    errors = [line for line in lines if "error" in line]
    records = [
        line for line in lines if "sid" in line and "error" not in line and line["sid"] == SID_A
    ]
    assert errors == [{"sid": SID_B, "error": "no completed envelope"}]
    assert len(records) == 1  # book A still processed after B's error


def test_invalid_source_id_prints_error_and_continues(seams: dict[str, Any]) -> None:
    result = CliRunner().invoke(
        cli, ["corpus", "rechunk-preview", "--source", "not-a-sha256-id", "--json"]
    )

    assert result.exit_code == 1, result.output
    lines = _lines(result.output)
    assert lines[0] == {"sid": "not-a-sha256-id", "error": "source id carries no sha256"}


# ---------------------------------------------------------------------------
# --reparse with no GPU endpoint: exit non-zero, before touching resources
# ---------------------------------------------------------------------------


def test_reparse_without_gpu_exits_nonzero_before_touching_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpus_rechunk, "_gpu_parse_available", lambda: False)

    @asynccontextmanager
    async def poison_resources() -> Any:
        raise AssertionError("resources must not be opened when the GPU check fails")
        yield  # pragma: no cover - unreachable, satisfies the generator protocol

    monkeypatch.setattr(corpus_rechunk, "_resources", poison_resources)

    result = CliRunner().invoke(cli, ["corpus", "rechunk-preview", "--source", SID_A, "--reparse"])

    assert result.exit_code != 0
    assert "CUDA" in result.output or "GPU" in result.output


# ---------------------------------------------------------------------------
# happy path: real build_extraction_composites + do_build_chunks, no mocking
# of the chunker itself -- only I/O is faked.
# ---------------------------------------------------------------------------


def test_default_path_json_output_matches_pure_metrics(seams: dict[str, Any]) -> None:
    from vo_rag.core.corpus import rechunk_preview

    result = CliRunner().invoke(cli, ["corpus", "rechunk-preview", "--source", SID_A, "--json"])
    assert result.exit_code == 0, result.output

    lines = _lines(result.output)
    assert len(lines) == 2  # one book record + the total
    record, total = lines
    assert record["sid"] == SID_A
    assert record["live"]["chunks"] == 1
    assert record["new"]["chunks"] == 1  # the fixture's one narrative-text element
    assert total["live"] == record["live"]
    assert total["new"] == record["new"]

    # Cross-check against the pure module directly, over the same views the
    # CLI must have built: live has 1 old text chunk, new should carry the
    # freshly parsed narrative text (real do_build_chunks output).
    live_view = rechunk_preview.chunk_view_from_payload(seams["qdrant"].books[SID_A][0])
    assert rechunk_preview.metrics([live_view]) == record["live"]


def test_default_path_never_calls_a_write_method(seams: dict[str, Any]) -> None:
    """Garage/Qdrant fakes raise on every write method; a clean exit proves none fired."""
    result = CliRunner().invoke(cli, ["corpus", "rechunk-preview", "--source", SID_A, "--json"])
    assert result.exit_code == 0, result.output
    assert result.exception is None


# ---------------------------------------------------------------------------
# --reparse wiring (fake GPU + fake parser_router; no real docling/torch)
# ---------------------------------------------------------------------------


def test_reparse_downloads_raw_upload_and_uses_parser_router(
    seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpus_rechunk, "_gpu_parse_available", lambda: True)
    seams["garage"].objects["uploads/a.pdf"] = b"%PDF-1.4 fake bytes"

    calls: list[tuple[str, str]] = []

    class FakeParserRouter:
        async def parse(self, file_path: str, file_hash: str) -> ParsedDocument:
            calls.append((file_path, file_hash))
            return _parsed_doc(SID_A)

    async def fake_build_router(garage: Any, bucket: str, figures_root: Any) -> Any:
        _ = (garage, bucket, figures_root)
        return FakeParserRouter()

    monkeypatch.setattr(corpus_rechunk, "_build_gpu_parser_router", fake_build_router)

    result = CliRunner().invoke(
        cli, ["corpus", "rechunk-preview", "--source", SID_A, "--reparse", "--json"]
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    file_path, file_hash = calls[0]
    assert file_hash == SHA_A
    assert file_path.endswith("a.pdf")
    # parsed.json is never touched in --reparse mode.
    assert "parsed/ENV-A/parsed.json" not in seams["garage"].gets


# ---------------------------------------------------------------------------
# VOR-1452: --reparse writes nothing to the live figure store
# ---------------------------------------------------------------------------


def test_reparse_never_writes_to_the_live_figure_store(
    seams: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Parsing writes every picture through the parser's ``FigureStore``; on
    the live root that rewrites the files live Qdrant ``image_ref``s serve.
    ``--reparse`` must hand the parser a throwaway store."""
    from types import SimpleNamespace

    import vo_rag.core.config as config_module
    from vo_rag.core.parsing import factory as factory_module
    from vo_rag.core.parsing.parsers import docling as docling_module
    from vo_rag.core.parsing.parsers import epub as epub_module

    live_root = tmp_path / "live-figures"
    live_root.mkdir()
    monkeypatch.setattr(
        config_module,
        "get_settings",
        lambda: SimpleNamespace(figures=SimpleNamespace(root=str(live_root))),
    )
    monkeypatch.setattr(corpus_rechunk, "_gpu_parse_available", lambda: True)
    seams["garage"].objects["uploads/a.pdf"] = b"%PDF-1.4 fake bytes"

    class FakeParser:
        def __init__(self, figures: Any, **kwargs: Any) -> None:
            self.figures = figures
            self.kwargs = kwargs

    written: list[Any] = []

    built: dict[str, Any] = {}

    class FakeRouter:
        def __init__(self, *, figures: Any, **kwargs: Any) -> None:
            self.figures = figures
            built.update(kwargs)

        async def parse(self, file_path: str, file_hash: str) -> ParsedDocument:
            _ = file_path
            # What docling_mapping / epub do for every extracted picture.
            self.figures.write(file_hash, 1, 0, b"\x89PNG fake", ext="png")
            written.append(self.figures.root)
            return _parsed_doc(SID_A)

    monkeypatch.setattr(docling_module, "DoclingParser", FakeParser)
    monkeypatch.setattr(epub_module, "EpubParser", FakeParser)
    monkeypatch.setattr(factory_module, "ParserRouter", FakeRouter)

    result = CliRunner().invoke(
        cli, ["corpus", "rechunk-preview", "--source", SID_A, "--reparse", "--json"]
    )

    assert result.exit_code == 0, result.output
    assert written, "the fake parse never ran"
    assert list(live_root.rglob("*")) == []  # nothing written to the live store
    # The router is the ingest's own (the factory), docling pinned to CUDA and
    # layout-only like production.
    docling = built["docling"].unwrap()
    assert docling.kwargs == {"do_ocr": False, "device": "cuda"}
    assert not written[0].is_relative_to(live_root.resolve())
    assert not written[0].exists()  # the throwaway store is removed after the run


# ---------------------------------------------------------------------------
# VOR-1452: the new chunks' OCR-gate drops are reported
# ---------------------------------------------------------------------------

_GATE_DROP = "~#| \\\\ =| °° -//4 &C-5E. ~}{ |=| ¥¥ -/-/ ^^% ir oe"


def _with_noise(source_id: str) -> ParsedDocument:
    """The fixture book plus a chapter of pure OCR noise (its own chunk)."""
    doc = _parsed_doc(source_id)
    doc.elements.extend(
        [
            ParsedElement(
                element_id="e-3", element_type=ElementType.TITLE, text="Chapter 2", page_number=2
            ),
            ParsedElement(
                element_id="e-4",
                element_type=ElementType.NARRATIVE_TEXT,
                text=_GATE_DROP,
                page_number=2,
            ),
        ]
    )
    doc.total_pages = 2
    return doc


def test_ocr_gate_drops_are_counted_per_book_and_in_the_total(seams: dict[str, Any]) -> None:
    from vo_rag.pipelines.book_rag.ocr_quality import assess

    assert assess(_GATE_DROP) == "drop"
    seams["garage"].objects["parsed/ENV-A/parsed.json"] = (
        _with_noise(SID_A).model_dump_json().encode("utf-8")
    )
    result = CliRunner().invoke(cli, ["corpus", "rechunk-preview", "--source", SID_A, "--json"])
    assert result.exit_code == 0, result.output
    record, total = _lines(result.output)
    assert record["new_ocr_drops"] == 1
    assert "ocr_drop_samples" not in record  # texts only with --show-drops
    assert total["new_ocr_drops"] == 1

    human = CliRunner().invoke(cli, ["corpus", "rechunk-preview", "--source", SID_A])
    assert "new_ocr_drops=1" in human.output
    assert "TOTAL" in human.output and human.output.count("new_ocr_drops=1") == 2


def test_show_drops_prints_the_first_dropped_texts(seams: dict[str, Any]) -> None:
    seams["garage"].objects["parsed/ENV-A/parsed.json"] = (
        _with_noise(SID_A).model_dump_json().encode("utf-8")
    )
    result = CliRunner().invoke(
        cli, ["corpus", "rechunk-preview", "--source", SID_A, "--json", "--show-drops", "5"]
    )
    assert result.exit_code == 0, result.output
    record = _lines(result.output)[0]
    assert len(record["ocr_drop_samples"]) == 1
    assert "°°" in record["ocr_drop_samples"][0]

    human = CliRunner().invoke(
        cli, ["corpus", "rechunk-preview", "--source", SID_A, "--show-drops", "5"]
    )
    assert "ocr drop 1:" in human.output and "°°" in human.output


def test_do_build_chunks_reports_gate_drops_into_a_sink() -> None:
    from vo_rag.pipelines.book_rag.steps.chunk_embed import do_build_chunks

    dropped: list[Any] = []
    chunks = do_build_chunks(_with_noise(SID_A), ocr_dropped=dropped)
    assert chunks
    assert len(dropped) == 1
    assert "°°" in dropped[0].content


# ---------------------------------------------------------------------------
# VOR-1452: a zero-chunk book is an error line, batch goes on
# ---------------------------------------------------------------------------


def test_zero_chunk_book_is_an_error_line_and_the_batch_continues(
    seams: dict[str, Any],
) -> None:
    empty = _parsed_doc(SID_B)
    empty.elements = [
        ParsedElement(
            element_id="e-1",
            element_type=ElementType.NARRATIVE_TEXT,
            text=_GATE_DROP,
            page_number=1,
        )
    ]
    seams["pool"].conn.rows_by_sha[SHA_B] = {"id": "ENV-B", "garage_key": "uploads/b.pdf"}
    seams["garage"].objects["parsed/ENV-B/parsed.json"] = empty.model_dump_json().encode("utf-8")
    seams["qdrant"].books[SID_B] = []

    result = CliRunner().invoke(
        cli, ["corpus", "rechunk-preview", "--source", SID_B, "--source", SID_A, "--json"]
    )

    assert result.exit_code == 1, result.output
    lines = _lines(result.output)
    errors = [line for line in lines if "error" in line]
    assert len(errors) == 1 and errors[0]["sid"] == SID_B
    assert "zero chunks" in errors[0]["error"]
    assert any(line.get("sid") == SID_A and "error" not in line for line in lines)
    assert "live" in lines[-1] and "new" in lines[-1]  # the totals still print
