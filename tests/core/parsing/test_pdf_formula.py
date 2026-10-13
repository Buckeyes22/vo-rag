"""Original PDF crops preserve geometry and reject changed or over-budget files."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fitz  # type: ignore[import-untyped, unused-ignore]
import pytest

from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement
from vo_rag.core.parsing.parsers import pdf_formula
from vo_rag.core.parsing.parsers.pdf_formula import PdfEquationCrops, verify_equation_crops
from vo_rag.core.storage.figures import FigureStore


def _source(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "source.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=200, height=300)
        page.insert_text((60, 70), "numerator")
        page.draw_line((50, 75), (115, 75))
        page.insert_text((60, 90), "denominator")
        page.set_cropbox(fitz.Rect(20, 20, 180, 280))
        doc.save(path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _location(origin: str = "TOPLEFT") -> dict[str, Any]:
    box = {"l": 30.0, "t": 35.0, "r": 100.0, "b": 75.0, "coord_origin": origin}
    if origin == "BOTTOMLEFT":
        box.update(t=225.0, b=185.0)
    return {
        "geometry_status": "available",
        "total_occurrence_count": 1,
        "occurrences": [{"page_no": 1, "bbox": box}],
    }


def _parsed(digest: str, visual: dict[str, Any]) -> ParsedDocument:
    return ParsedDocument(
        document_id=digest,
        source_id="vo-rag-corpus-sha256:" + digest,
        title="Book",
        total_pages=1,
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        parser_used="docling",
        elements=[
            ParsedElement(
                element_id="eq",
                element_type=ElementType.EQUATION,
                text="numerator denominator ----",
                page_number=1,
                metadata={"equation_visual": visual},
            )
        ],
    )


def test_cropbox_coordinates_and_both_origins_produce_identical_original_pixels(tmp_path) -> None:
    path, digest = _source(tmp_path)
    store = FigureStore(tmp_path / "figures")
    crops = PdfEquationCrops(path, digest, store)
    try:
        top = crops.crop(_location(), (160.0, 260.0))
        bottom = crops.crop(_location("BOTTOMLEFT"), (160.0, 260.0))
    finally:
        crops.close()
    assert top["status"] == bottom["status"] == "source_bound"
    assert top["image_ref"] == bottom["image_ref"]
    assert top["media_box"] == [0.0, 0.0, 200.0, 300.0]
    assert top["crop_box"] == [20.0, 20.0, 180.0, 280.0]
    with fitz.open(path) as doc:
        expected = (
            doc[0]
            .get_pixmap(
                matrix=fitz.Matrix(1.67, 1.67),
                clip=fitz.Rect(top["clip_bbox_top_left"]),
                alpha=False,
            )
            .tobytes("png")
        )
    assert store.open_bytes(str(top["image_ref"])) == expected
    assert hashlib.sha256(expected).hexdigest() == top["sha256"]
    assert verify_equation_crops(_parsed(digest, top), store, len(expected)) == {
        "occurrences": 1,
        "unique_images": 1,
        "bytes": len(expected),
    }


def test_aggregate_budget_prevents_writes_during_mapping(tmp_path) -> None:
    path, digest = _source(tmp_path)
    store = FigureStore(tmp_path / "figures")
    crops = PdfEquationCrops(path, digest, store, max_bytes=1)
    try:
        value = crops.crop(_location(), (160.0, 260.0))
    finally:
        crops.close()
    assert value["status"] == "unavailable"
    assert not store.root.exists()


def test_original_table_pixels_use_separate_namespace_and_shared_admission_budget(tmp_path) -> None:
    path, digest = _source(tmp_path)
    store = FigureStore(tmp_path / "figures")
    crops = PdfEquationCrops(path, digest, store)
    try:
        equation = crops.crop(_location(), (160.0, 260.0))
        table = crops.crop(_location(), (160.0, 260.0), kind="table")
    finally:
        crops.close()
    assert table["status"] == "source_bound"
    assert table["sha256"] == equation["sha256"]
    assert "/p001_tbl" in str(table["image_ref"])
    assert table["image_ref"] != equation["image_ref"]
    assert store.open_bytes(str(table["image_ref"])) == store.open_bytes(str(equation["image_ref"]))
    parsed = _parsed(digest, equation)
    parsed.elements.append(
        ParsedElement(
            element_id="table",
            element_type=ElementType.TABLE,
            text="| fraction | value |",
            page_number=1,
            metadata={"table_visual": table},
        )
    )
    size = table["size_bytes"]
    assert isinstance(size, int)
    assert verify_equation_crops(parsed, store, size * 2)["occurrences"] == 2
    with pytest.raises(ValueError, match="aggregate_budget"):
        verify_equation_crops(parsed, store, size * 2 - 1)
    store.path_for(str(table["image_ref"])).unlink()
    with pytest.raises(FileNotFoundError):
        verify_equation_crops(parsed, store, size * 2)


def test_table_and_equation_mapping_share_one_source_crop_byte_cap(tmp_path) -> None:
    path, digest = _source(tmp_path)
    store = FigureStore(tmp_path / "figures")
    sizing = PdfEquationCrops(path, digest, store)
    try:
        receipt = sizing.crop(_location(), (160.0, 260.0))
    finally:
        sizing.close()
    size = receipt["size_bytes"]
    assert isinstance(size, int)
    capped = PdfEquationCrops(path, digest, store, max_bytes=size)
    try:
        assert capped.crop(_location(), (160.0, 260.0), kind="table")["status"] == "source_bound"
        assert capped.crop(_location(), (160.0, 260.0))["status"] == "unavailable"
    finally:
        capped.close()


@pytest.mark.parametrize("label", ["table", "document_index"])
def test_native_table_mapping_retains_original_pixels_without_formula_nodes(
    tmp_path, label
) -> None:
    from vo_rag.core.parsing.parsers.docling_mapping import docling_to_parsed_document

    path, digest = _source(tmp_path)
    item = SimpleNamespace(
        label=label,
        text="| Term | Page |\n|---|---|\n| Shearing stress | 141 |",
        prov=[SimpleNamespace(page_no=1, bbox=_location()["occurrences"][0]["bbox"])],
    )
    doc = SimpleNamespace(
        texts=[],
        tables=[item],
        pictures=[],
        pages={1: SimpleNamespace(size=SimpleNamespace(width=160.0, height=260.0))},
    )
    store = FigureStore(tmp_path / "figures")
    parsed = docling_to_parsed_document(doc, store, str(path), "pdf", src_hash=digest)
    assert len(parsed.elements) == 1
    element = parsed.elements[0]
    assert element.element_type is ElementType.TABLE
    assert element.text == item.text
    assert element.metadata["table_visual"]["status"] == "source_bound"
    assert verify_equation_crops(parsed, store, 1_000_000)["occurrences"] == 1


def test_mixed_prose_formulas_retain_one_complete_original_page_with_shared_crop_budget(
    tmp_path,
) -> None:
    from vo_rag.core.parsing.parsers.docling_mapping import docling_to_parsed_document

    path, digest = _source(tmp_path)
    formula = SimpleNamespace(
        label="formula",
        text="",
        orig="numerator denominator --",
        prov=[SimpleNamespace(page_no=1, bbox=_location()["occurrences"][0]["bbox"])],
    )
    prose = SimpleNamespace(
        label="text",
        text="The fraction is embedded in this prose.",
        prov=[SimpleNamespace(page_no=1)],
    )
    doc = SimpleNamespace(
        texts=[prose, formula, formula],
        tables=[],
        pictures=[],
        pages={1: SimpleNamespace(size=SimpleNamespace(width=160.0, height=260.0))},
    )
    store = FigureStore(tmp_path / "figures")
    parsed = docling_to_parsed_document(doc, store, str(path), "pdf", src_hash=digest)
    declarations = [
        e.metadata["source_page_visual"]
        for e in parsed.elements
        if "source_page_visual" in e.metadata
    ]
    assert len(declarations) == 1
    visual = declarations[0]
    assert visual["status"] == "source_bound"
    assert "/p001_pg" in visual["image_ref"]
    assert visual["representation"] == "original_mathematical_page"
    with fitz.open(path) as pdf:
        expected = pdf[0].get_pixmap(matrix=fitz.Matrix(1.67, 1.67), alpha=False).tobytes("png")
    assert store.open_bytes(visual["image_ref"]) == expected
    proof = verify_equation_crops(parsed, store, 1_000_000)
    assert proof["occurrences"] == 3 and proof["unique_images"] == 2
    with pytest.raises(ValueError, match="aggregate_budget"):
        verify_equation_crops(parsed, store, proof["bytes"] - 1)
    store.path_for(visual["image_ref"]).unlink()
    with pytest.raises(FileNotFoundError):
        verify_equation_crops(parsed, store, 1_000_000)


@pytest.mark.parametrize(
    "source_text", ["Mixed fraction 2⁄3", "Volume r²", "Maximum hole 2 + 0.0005 = 2.0005"]
)
def test_original_math_glyphs_preserve_page_without_native_formula(tmp_path, source_text) -> None:
    from vo_rag.core.parsing.parsers.docling_mapping import docling_to_parsed_document

    path = tmp_path / "original-math.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=200, height=300)
        page.insert_font(fontname="math", fontbuffer=fitz.Font("helv").buffer)
        page.insert_text((30, 70), source_text, fontname="math")
        pdf.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    doc = SimpleNamespace(
        texts=[
            SimpleNamespace(
                label="text",
                text="Native flattened source characters.",
                prov=[SimpleNamespace(page_no=1)],
            )
        ],
        tables=[],
        pictures=[],
        pages={1: SimpleNamespace(size=SimpleNamespace(width=200.0, height=300.0))},
    )
    store = FigureStore(tmp_path / "figures")
    parsed = docling_to_parsed_document(doc, store, str(path), "pdf", src_hash=digest)
    assert all(e.element_type is not ElementType.EQUATION for e in parsed.elements)
    visual = parsed.elements[0].metadata["source_page_visual"]
    assert visual["status"] == "source_bound"
    with fitz.open(path) as pdf:
        assert source_text in pdf[0].get_text()
        expected = pdf[0].get_pixmap(matrix=fitz.Matrix(1.67, 1.67), alpha=False).tobytes("png")
    assert store.open_bytes(visual["image_ref"]) == expected
    assert verify_equation_crops(parsed, store, 1_000_000)["occurrences"] == 1


def test_plain_original_prose_without_formula_renders_no_page_image(tmp_path) -> None:
    path, digest = _source(tmp_path)
    store = FigureStore(tmp_path / "figures")
    renderer = PdfEquationCrops(path, digest, store)
    try:
        assert renderer.has_mathematical_notation(1) is False
        assert renderer.has_mathematical_notation(None) is False
    finally:
        renderer.close()
    assert not list((tmp_path / "figures").rglob("*.png"))


def test_empty_native_formula_still_preserves_original_page_for_prose(tmp_path) -> None:
    from vo_rag.core.chunking.hierarchical import HierarchicalChunker
    from vo_rag.core.parsing.parsers.docling_mapping import docling_to_parsed_document

    path, digest = _source(tmp_path)
    prose = SimpleNamespace(
        label="text",
        text="The original fraction is embedded in this explanatory paragraph.",
        prov=[SimpleNamespace(page_no=1)],
    )
    formula = SimpleNamespace(label="formula", text="", orig="", prov=[SimpleNamespace(page_no=1)])
    doc = SimpleNamespace(
        texts=[prose, formula],
        tables=[],
        pictures=[],
        pages={1: SimpleNamespace(size=SimpleNamespace(width=160.0, height=260.0))},
    )
    store = FigureStore(tmp_path / "figures")
    parsed = docling_to_parsed_document(doc, store, str(path), "pdf", src_hash=digest)
    assert len(parsed.elements) == 1
    assert parsed.elements[0].text == prose.text
    visual = parsed.elements[0].metadata["source_page_visual"]
    assert visual["status"] == "source_bound"
    assert verify_equation_crops(parsed, store, 1_000_000)["occurrences"] == 1
    chunks = HierarchicalChunker(counter=lambda text: len(text.split())).chunk(
        parsed, composites=[]
    )
    assert len(chunks) == 1
    assert chunks[0].content == prose.text
    assert chunks[0].has_image and chunks[0].image_ref == visual["image_ref"]
    assert chunks[0].metadata["source_page_visuals"] == [visual]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_mathematical_whole_page_pixels_equal_original_displayed_page(tmp_path, rotation) -> None:
    original, _ = _source(tmp_path)
    path = tmp_path / "mathematical-page.pdf"
    with fitz.open(original) as pdf:
        pdf[0].set_rotation(rotation)
        pdf.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with fitz.open(path) as pdf:
        frame = (pdf[0].rect.width, pdf[0].rect.height)
        expected = pdf[0].get_pixmap(matrix=fitz.Matrix(1.67, 1.67), alpha=False).tobytes("png")
    store = FigureStore(tmp_path / "figures")
    renderer = PdfEquationCrops(path, digest, store)
    try:
        assert renderer.page(1, (1.0, 1.0))["reason"] == "native_page_frame_differs"
        visual = renderer.page(1, frame)
    finally:
        renderer.close()
    assert visual["status"] == "source_bound" and visual["rotation"] == rotation
    assert store.open_bytes(str(visual["image_ref"])) == expected


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_table_display_frame_crop_matches_full_original_page_pixels(
    tmp_path, rotation
) -> None:
    from PIL import Image

    source, _ = _source(tmp_path)
    path = tmp_path / "rotated-table.pdf"
    with fitz.open(source) as pdf:
        pdf[0].set_rotation(rotation)
        pdf.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    store = FigureStore(tmp_path / "figures")
    with fitz.open(path) as pdf:
        native_frame = (pdf[0].rect.width, pdf[0].rect.height)
    crops = PdfEquationCrops(path, digest, store)
    try:
        assert crops.crop(_location(), native_frame)["reason"] == (
            "rotated_parser_geometry_unverified"
        )
        assert crops.crop(_location(), (1.0, 1.0), kind="table")["reason"] == (
            "native_page_frame_differs"
        )
        table = crops.crop(_location(), native_frame, kind="table")
    finally:
        crops.close()
    assert table["status"] == "source_bound" and table["rotation"] == rotation
    with fitz.open(path) as pdf:
        matrix = fitz.Matrix(pdf_formula.CROP_SCALE, pdf_formula.CROP_SCALE)
        full = pdf[0].get_pixmap(matrix=matrix, alpha=False)
        expected_region = (fitz.Rect(table["clip_bbox_top_left"]) * matrix).irect
        expected = Image.frombytes("RGB", (full.width, full.height), full.samples).crop(
            tuple(expected_region)
        )
    actual = Image.open(io.BytesIO(store.open_bytes(str(table["image_ref"]))))
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()


def test_source_count_and_parser_frame_bounds_are_explicit(tmp_path, monkeypatch) -> None:
    path, digest = _source(tmp_path)
    store = FigureStore(tmp_path / "figures")
    monkeypatch.setattr(pdf_formula, "MAX_SOURCE_CROPS", 1)
    crops = PdfEquationCrops(path, digest, store)
    try:
        assert crops.crop(_location(), (200.0, 300.0))["reason"] == "native_page_frame_differs"
        assert (
            crops.crop(_location(), (160.0, 260.0))["reason"]
            == "finite_source_crop_budget_exceeded"
        )
    finally:
        crops.close()
    assert not store.root.exists()


def test_replay_rejects_missing_corrupt_and_unbudgeted_crop_bytes(tmp_path) -> None:
    path, digest = _source(tmp_path)
    store = FigureStore(tmp_path / "figures")
    crops = PdfEquationCrops(path, digest, store)
    try:
        visual = crops.crop(_location(), (160.0, 260.0))
    finally:
        crops.close()
    parsed = _parsed(digest, visual)
    with pytest.raises(ValueError, match="aggregate_budget"):
        verify_equation_crops(parsed, store, 0)
    target = store.path_for(str(visual["image_ref"]))
    original = target.read_bytes()
    target.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError, match="bytes_changed"):
        verify_equation_crops(parsed, store, len(original))
    target.unlink()
    with pytest.raises(FileNotFoundError):
        verify_equation_crops(parsed, store, len(original))


def test_original_digest_mismatch_stops_before_crop_publication(tmp_path) -> None:
    path, _ = _source(tmp_path)
    store = FigureStore(tmp_path / "figures")
    with pytest.raises(ValueError, match="original PDF identity"):
        PdfEquationCrops(path, "ab" * 32, store)
    assert not store.root.exists()
