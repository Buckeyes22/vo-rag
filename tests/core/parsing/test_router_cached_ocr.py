"""Pre-parsed Garage cache handoff in ``ParserRouter`` (VOR-1427).

When ``ocr/<sha256>/docling.json`` is already in Garage, the router maps that
JSON instead of parsing the file again. ``ocr/<sha256>/meta.json`` says which
pipeline produced it:

- ``"pipeline": "ocr"`` (the OCR farm): used for any PDF / image-only EPUB
  as ``docling-ocr``. The farm only OCRs books whose live parse failed, so an
  explicit OCR label stands in even for a PDF ``_pdf_kind`` calls native.
- no ``meta.json``, or no pipeline in it (older farm output):
  used for a scanned PDF / image-only EPUB as ``docling-ocr``; ignored for a
  native PDF, which is parsed live (the pipeline is never guessed).
- ``"pipeline": "native"`` (an off-box layout-only docling parse):
  used for any PDF as plain ``docling``.

Two layers:

- the DoclingDocument JSON loader is stubbed with a duck-typed document, so
  the routing / fall-through / error semantics run in the locked CI env
  (no docling there);
- a real ``docling_core`` round trip (2 text items + 1 embedded picture,
  saved with ``ImageRefMode.EMBEDDED``) runs where docling_core is installed.
"""

from __future__ import annotations

import base64
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import vo_rag.core.parsing.router as router_module
from tests.core.parsing._router_fakes import FakeDocling, FakeParser
from vo_rag.core.parsing.contract import ElementType, ParsedDocument
from vo_rag.core.parsing.parsers.epub import ImageOnlyEbook
from vo_rag.core.parsing.router import ParserRouter
from vo_rag.core.storage.figures import FigureStore

BUCKET = "vorag-artifacts"
HASH = "a" * 64


class _Garage:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.heads: list[str] = []
        self.gets: list[str] = []

    async def object_metadata(self, bucket: str, key: str) -> Any:
        assert bucket == BUCKET
        self.heads.append(key)
        return SimpleNamespace(size=len(self.objects[key])) if key in self.objects else None

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        assert bucket == BUCKET
        self.gets.append(key)
        return self.objects[key]


class _BrokenGarage(_Garage):
    """Fails every HEAD, or only the HEAD of ``fail_key`` when one is given."""

    def __init__(
        self, objects: dict[str, bytes] | None = None, fail_key: str | None = None
    ) -> None:
        super().__init__(objects)
        self.fail_key = fail_key

    async def object_metadata(self, bucket: str, key: str) -> Any:
        if self.fail_key is None or key == self.fail_key:
            raise RuntimeError("garage 503 SlowDown")
        return await super().object_metadata(bucket, key)


def _png(size: int = 200) -> bytes:
    # Pillow is runtime-only (via docling); the light CI job skips these tests.
    image_mod = pytest.importorskip("PIL.Image")

    image = image_mod.new("RGB", (size, size), (255, 255, 255))
    for x in range(size):
        image.putpixel((x, x), (0, 0, 0))
        image.putpixel((x, size // 3), (200, 10, 10))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _fake_docling_doc() -> Any:
    uri = "data:image/png;base64," + base64.b64encode(_png()).decode()
    return SimpleNamespace(
        name=HASH,  # the worker OCRs jobs/<sha256>.pdf
        texts=[
            SimpleNamespace(
                text="First OCR paragraph.", label="text", prov=[SimpleNamespace(page_no=1)]
            ),
            SimpleNamespace(
                text="Second OCR paragraph.", label="text", prov=[SimpleNamespace(page_no=2)]
            ),
        ],
        tables=[],
        pictures=[
            SimpleNamespace(image=SimpleNamespace(uri=uri), prov=[SimpleNamespace(page_no=2)])
        ],
    )


def _router(
    garage: Any, figures: FigureStore, docling: Any = None, epub: Any = None
) -> ParserRouter:
    return ParserRouter(
        docling=docling or FakeDocling(),
        epub=epub or FakeParser("epub"),
        garage=garage,
        bucket=BUCKET,
        figures=figures,
    )


@pytest.fixture(autouse=True)
def _stub_magic_and_pdf_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_from_file(path: str, mime: bool = True) -> str:
        _ = (path, mime)
        return "application/octet-stream"

    async def scanned(self: ParserRouter, path: str) -> str:
        _ = (self, path)
        return "scanned"

    monkeypatch.setattr(router_module.magic, "from_file", fake_from_file)  # type: ignore[attr-defined]
    monkeypatch.setattr(ParserRouter, "_pdf_kind", scanned)


@pytest.fixture
def figures(tmp_path: Path) -> FigureStore:
    return FigureStore(tmp_path / "figures")


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    seen: list[bytes] = []

    def fake_loader(data: bytes) -> Any:
        seen.append(data)
        return _fake_docling_doc()

    monkeypatch.setattr(router_module, "_load_docling_json", fake_loader)
    return seen


def _pdf(tmp_path: Path) -> str:
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    return str(path)


def _assert_farm_doc(doc: ParsedDocument) -> None:
    texts = [e.text for e in doc.elements if e.element_type != ElementType.IMAGE]
    assert texts == ["First OCR paragraph.", "Second OCR paragraph."]
    assert doc.parser_used == "docling-ocr"
    assert len(doc.images) == 1
    assert doc.images[0].page_number == 2
    assert doc.metadata["ocr_source"] == "farm"


async def test_cached_farm_ocr_is_used_for_scanned_pdf(
    tmp_path: Path, figures: FigureStore, loaded: list[bytes]
) -> None:
    payload = b'{"schema_name": "DoclingDocument"}'
    garage = _Garage(
        {
            f"ocr/{HASH}/docling.json": payload,
            f"ocr/{HASH}/meta.json": json.dumps({"host": "win"}).encode(),
        }
    )
    docling = FakeDocling()
    router = _router(garage, figures, docling)

    doc = await router.parse(_pdf(tmp_path), file_hash=HASH, mime_type="application/pdf")

    _assert_farm_doc(doc)
    assert doc.title == "scan"  # the source file's stem, not the worker's hash name
    assert doc.metadata["ocr_host"] == "win"
    assert doc.document_id == HASH
    assert docling.scanned_called is False
    assert loaded == [payload]
    assert figures.root.exists() and any(figures.root.rglob("*.png"))


async def test_missing_meta_sidecar_records_unknown_host(
    tmp_path: Path, figures: FigureStore, loaded: list[bytes]
) -> None:
    garage = _Garage({f"ocr/{HASH}/docling.json": b"{}"})
    doc = await _router(garage, figures).parse(_pdf(tmp_path), file_hash=HASH)

    assert doc.metadata["ocr_host"] == "unknown"
    assert doc.metadata["ocr_source"] == "farm"


async def test_no_cache_object_falls_back_to_parse_scanned(
    tmp_path: Path, figures: FigureStore, loaded: list[bytes]
) -> None:
    garage = _Garage()
    docling = FakeDocling()

    doc = await _router(garage, figures, docling).parse(_pdf(tmp_path), file_hash=HASH)

    assert docling.scanned_called is True
    assert docling.scanned_file_hash == HASH
    assert doc.parser_used == "docling-ocr"
    assert "ocr_source" not in doc.metadata
    assert garage.heads == [f"ocr/{HASH}/docling.json"]
    assert garage.gets == []
    assert loaded == []


async def test_router_without_garage_never_looks_for_cache(tmp_path: Path) -> None:
    docling = FakeDocling()
    router = ParserRouter(docling=docling, epub=FakeParser("epub"))

    await router.parse(_pdf(tmp_path), file_hash=HASH)

    assert docling.scanned_called is True


async def test_garage_errors_other_than_not_found_raise(
    tmp_path: Path, figures: FigureStore, loaded: list[bytes]
) -> None:
    docling = FakeDocling()
    router = _router(_BrokenGarage(), figures, docling)

    with pytest.raises(RuntimeError, match="SlowDown"):
        await router.parse(_pdf(tmp_path), file_hash=HASH)

    assert docling.scanned_called is False


class _EpubRaisingImageOnly:
    def __init__(self, image_only: ImageOnlyEbook) -> None:
        self.image_only = image_only

    async def parse(self, file_path: str) -> ParsedDocument:
        _ = file_path
        raise self.image_only


async def test_image_only_epub_uses_cached_farm_ocr(
    tmp_path: Path, figures: FigureStore, loaded: list[bytes]
) -> None:
    tmpdir = tempfile.TemporaryDirectory(dir=str(tmp_path))
    pdf_path = Path(tmpdir.name) / "synthesized.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    image_only = ImageOnlyEbook(
        str(pdf_path),
        tmpdir,
        metadata={
            "document_id": "epub-doc-id",
            "source_id": "epub-source-id",
            "title": "The Scanned Book",
            "author": "Some Author",
            "format": "epub",
        },
    )
    garage = _Garage({f"ocr/{HASH}/docling.json": b"{}"})
    docling = FakeDocling()
    router = _router(garage, figures, docling, epub=_EpubRaisingImageOnly(image_only))

    doc = await router.parse("original.epub", file_hash=HASH, mime_type=None)

    _assert_farm_doc(doc)
    assert docling.scanned_called is False
    assert doc.document_id == "epub-doc-id"
    assert doc.source_id == "epub-source-id"
    assert doc.title == "The Scanned Book"
    assert doc.author == "Some Author"
    assert doc.metadata["source_format"] == "epub_image_only"
    assert not Path(tmpdir.name).exists()


def test_garage_without_figures_is_rejected() -> None:
    with pytest.raises(ValueError, match="figures"):
        ParserRouter(
            docling=FakeDocling(),
            epub=FakeParser("epub"),
            garage=_Garage(),
            bucket=BUCKET,
        )


async def test_real_docling_json_round_trip(tmp_path: Path, figures: FigureStore) -> None:
    """2 text items + 1 embedded picture saved by docling_core, then routed."""
    doc_mod = pytest.importorskip("docling_core.types.doc", reason="docling_core not installed")
    from PIL import Image

    document = doc_mod.DoclingDocument(name=HASH)
    for page_no in (1, 2):
        document.add_page(page_no=page_no, size=doc_mod.Size(width=612, height=792))
    bbox = doc_mod.BoundingBox(l=72, t=720, r=540, b=700, coord_origin="BOTTOMLEFT")
    for page_no, text in ((1, "First OCR paragraph."), (2, "Second OCR paragraph.")):
        document.add_text(
            label=doc_mod.DocItemLabel.TEXT,
            text=text,
            prov=doc_mod.ProvenanceItem(page_no=page_no, bbox=bbox, charspan=(0, len(text))),
        )
    picture = Image.open(io.BytesIO(_png()))
    document.add_picture(
        image=doc_mod.ImageRef.from_pil(picture, dpi=72),
        prov=doc_mod.ProvenanceItem(
            page_no=2,
            bbox=doc_mod.BoundingBox(l=100, t=500, r=300, b=300, coord_origin="BOTTOMLEFT"),
            charspan=(0, 0),
        ),
    )
    out = tmp_path / "docling.json"
    document.save_as_json(out, image_mode=doc_mod.ImageRefMode.EMBEDDED)
    garage = _Garage({f"ocr/{HASH}/docling.json": out.read_bytes()})
    docling = FakeDocling()

    doc = await _router(garage, figures, docling).parse(_pdf(tmp_path), file_hash=HASH)

    _assert_farm_doc(doc)
    assert doc.title == "scan"
    assert docling.scanned_called is False


# ---------------------------------------------------------------------------
# The cache covers native PDFs too (off-box RunPod pre-parse)
# ---------------------------------------------------------------------------

DOCLING_KEY = f"ocr/{HASH}/docling.json"
META_KEY = f"ocr/{HASH}/meta.json"
RUNPOD_META = {"pipeline": "native", "host": "runpod", "docling": "2.97.0", "backend": "pypdfium"}


def _kind(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    async def fixed(self: ParserRouter, path: str) -> str:
        _ = (self, path)
        return kind

    monkeypatch.setattr(ParserRouter, "_pdf_kind", fixed)


def _text_docling_doc() -> Any:
    """A text-only DoclingDocument stand-in (no Pillow, so it runs in light CI)."""
    return SimpleNamespace(
        name=HASH,
        texts=[
            SimpleNamespace(
                text="First native paragraph.", label="text", prov=[SimpleNamespace(page_no=1)]
            ),
            SimpleNamespace(
                text="Second native paragraph.", label="text", prov=[SimpleNamespace(page_no=2)]
            ),
        ],
        tables=[],
        pictures=[],
    )


@pytest.fixture
def loaded_text(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    seen: list[bytes] = []

    def fake_loader(data: bytes) -> Any:
        seen.append(data)
        return _text_docling_doc()

    monkeypatch.setattr(router_module, "_load_docling_json", fake_loader)
    return seen


def _cache(meta: dict[str, Any] | None, payload: bytes = b"{}") -> dict[str, bytes]:
    objects = {DOCLING_KEY: payload}
    if meta is not None:
        objects[META_KEY] = json.dumps(meta).encode()
    return objects


@pytest.mark.parametrize("kind", ["native", "scanned"])
async def test_native_labelled_cache_is_used_as_plain_docling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    figures: FigureStore,
    loaded_text: list[bytes],
    kind: str,
) -> None:
    _kind(monkeypatch, kind)
    payload = b'{"schema_name": "DoclingDocument", "native": true}'
    garage = _Garage(_cache(RUNPOD_META, payload))
    docling = FakeDocling()

    doc = await _router(garage, figures, docling).parse(_pdf(tmp_path), file_hash=HASH)

    assert doc.parser_used == "docling"
    assert {qa.parser_used for qa in doc.page_qa} == {"docling"}
    assert [e.text for e in doc.elements] == ["First native paragraph.", "Second native paragraph."]
    assert doc.metadata["pre_parsed"] == "native"
    assert doc.metadata["ocr_host"] == "runpod"
    assert "ocr_source" not in doc.metadata  # no OCR farm involved
    assert doc.title == "scan"
    assert doc.document_id == HASH
    assert docling.called is False
    assert docling.scanned_called is False
    assert loaded_text == [payload]
    # One HEAD for docling.json, then meta.json (read only because docling.json exists).
    assert garage.heads == [DOCLING_KEY, META_KEY]
    assert garage.gets == [META_KEY, DOCLING_KEY]


async def test_native_pdf_uses_an_ocr_labelled_cache_as_docling_ocr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    figures: FigureStore,
    loaded_text: list[bytes],
) -> None:
    """An explicit ``"pipeline": "ocr"`` label is trusted for a native PDF:
    the farm only OCRs books whose live parse already failed, so parsing
    live again would fail again (a 34-page guide whose live parse
    covered 0/34 pages)."""
    _kind(monkeypatch, "native")
    payload = b'{"schema_name": "DoclingDocument", "ocr": true}'
    garage = _Garage(_cache({"pipeline": "ocr", "host": "win"}, payload))
    docling = FakeDocling()

    doc = await _router(garage, figures, docling).parse(_pdf(tmp_path), file_hash=HASH)

    assert doc.parser_used == "docling-ocr"
    assert {qa.parser_used for qa in doc.page_qa} == {"docling-ocr"}
    assert doc.metadata["ocr_source"] == "farm"
    assert doc.metadata["ocr_host"] == "win"
    assert "pre_parsed" not in doc.metadata
    assert docling.called is False
    assert docling.scanned_called is False
    assert loaded_text == [payload]
    assert garage.heads == [DOCLING_KEY, META_KEY]
    assert garage.gets == [META_KEY, DOCLING_KEY]


@pytest.mark.parametrize(
    "meta",
    [
        {"host": "win"},  # legacy farm meta, no pipeline
        None,  # no meta.json at all
    ],
    ids=["meta-without-pipeline", "no-meta"],
)
async def test_native_pdf_ignores_a_cache_without_a_pipeline_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    figures: FigureStore,
    loaded_text: list[bytes],
    meta: dict[str, Any] | None,
) -> None:
    _kind(monkeypatch, "native")
    garage = _Garage(_cache(meta))
    docling = FakeDocling()

    doc = await _router(garage, figures, docling).parse(_pdf(tmp_path), file_hash=HASH)

    assert docling.called is True
    assert docling.called_file_hash == HASH
    assert docling.scanned_called is False
    assert doc.parser_used == "docling"
    assert "pre_parsed" not in doc.metadata
    assert "ocr_host" not in doc.metadata
    assert loaded_text == []
    assert DOCLING_KEY not in garage.gets  # the (large) cached JSON is never downloaded


@pytest.mark.parametrize(
    "meta",
    [{"pipeline": "ocr", "host": "win"}, {"host": "win"}],
    ids=["with-pipeline", "without-pipeline"],
)
async def test_scanned_pdf_with_farm_meta_is_docling_ocr(
    tmp_path: Path, figures: FigureStore, loaded_text: list[bytes], meta: dict[str, Any]
) -> None:
    garage = _Garage(_cache(meta))
    docling = FakeDocling()

    doc = await _router(garage, figures, docling).parse(_pdf(tmp_path), file_hash=HASH)

    assert doc.parser_used == "docling-ocr"
    assert doc.metadata["ocr_source"] == "farm"
    assert doc.metadata["ocr_host"] == "win"
    assert "pre_parsed" not in doc.metadata
    assert docling.scanned_called is False
    assert docling.called is False


async def test_native_pdf_without_cache_makes_one_head_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, figures: FigureStore
) -> None:
    _kind(monkeypatch, "native")
    garage = _Garage()
    docling = FakeDocling()

    doc = await _router(garage, figures, docling).parse(_pdf(tmp_path), file_hash=HASH)

    assert docling.called is True
    assert doc.parser_used == "docling"
    assert garage.heads == [DOCLING_KEY]
    assert garage.gets == []


@pytest.mark.parametrize("kind", ["native", "scanned"])
@pytest.mark.parametrize("fail_key", [DOCLING_KEY, META_KEY], ids=["docling-head", "meta-head"])
async def test_garage_error_on_cache_check_raises_naming_the_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    figures: FigureStore,
    loaded_text: list[bytes],
    kind: str,
    fail_key: str,
) -> None:
    _kind(monkeypatch, kind)
    garage = _BrokenGarage(_cache(RUNPOD_META), fail_key=fail_key)
    docling = FakeDocling()

    with pytest.raises(RuntimeError, match=HASH) as excinfo:
        await _router(garage, figures, docling).parse(_pdf(tmp_path), file_hash=HASH)

    assert "SlowDown" in str(excinfo.value)
    assert docling.called is False  # no silent fallback to a live parse
    assert docling.scanned_called is False
    assert loaded_text == []


@pytest.mark.parametrize(
    "meta_bytes",
    [
        b"not json",
        b'["a", "list"]',
        json.dumps({"pipeline": "vlm", "host": "runpod"}).encode(),
    ],
    ids=["malformed-json", "not-an-object", "unknown-pipeline"],
)
async def test_unreadable_or_unknown_meta_raises_naming_the_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    figures: FigureStore,
    loaded_text: list[bytes],
    meta_bytes: bytes,
) -> None:
    _kind(monkeypatch, "native")
    garage = _Garage({DOCLING_KEY: b"{}", META_KEY: meta_bytes})
    docling = FakeDocling()

    with pytest.raises(ValueError, match=HASH):
        await _router(garage, figures, docling).parse(_pdf(tmp_path), file_hash=HASH)

    assert docling.called is False
    assert docling.scanned_called is False
    assert loaded_text == []


async def test_image_only_epub_rejects_a_native_labelled_cache(
    tmp_path: Path, figures: FigureStore, loaded_text: list[bytes]
) -> None:
    """An image-only ebook has no text layer: a native pre-parse under its hash
    is contradictory, and a ``docling`` result for a non-PDF upload would skip
    the parse step's PDF gates. Refuse it rather than guess."""
    tmpdir = tempfile.TemporaryDirectory(dir=str(tmp_path))
    pdf_path = Path(tmpdir.name) / "synthesized.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    image_only = ImageOnlyEbook(str(pdf_path), tmpdir, metadata={"format": "epub"})
    garage = _Garage(_cache(RUNPOD_META))
    docling = FakeDocling()
    router = _router(garage, figures, docling, epub=_EpubRaisingImageOnly(image_only))

    with pytest.raises(ValueError, match=HASH):
        await router.parse("original.epub", file_hash=HASH, mime_type=None)

    assert docling.scanned_called is False
    assert loaded_text == []
    assert not Path(tmpdir.name).exists()
