"""Unit contract for the read-only heading-stream replay script (VOR-1427).

Only the pure fixture builder and the file writer are exercised, on a
synthetic ``ParsedDocument``; no Garage access.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from scripts import replay_headings
from vo_rag.core.parsing.contract import ElementType, ParsedDocument, ParsedElement

BODY_SENTINEL = "zqxbodysentinel"
LONG_BODY = f"{BODY_SENTINEL} " + "the quick brown fox jumps over the lazy dog " * 8
LONG_NUMBERED_BODY = "3 " + LONG_BODY


def _el(
    idx: int,
    element_type: ElementType,
    text: str,
    page: int | None,
    *,
    depth: int = 0,
    metadata: dict[str, object] | None = None,
) -> ParsedElement:
    return ParsedElement(
        element_id=f"e{idx}",
        element_type=element_type,
        text=text,
        page_number=page,
        section_depth=depth,
        metadata=dict(metadata or {}),
    )


def _doc(
    elements: list[ParsedElement], *, total_pages: int, quarantined: list[int] | None = None
) -> ParsedDocument:
    return ParsedDocument(
        document_id="d",
        source_id="s",
        title="Synthetic Book",
        total_pages=total_pages,
        elements=elements,
        images=[],
        page_qa=[],
        markdown="",
        heading_tree=[],
        parser_used="docling",
        quarantined_pages=list(quarantined or []),
    )


def _synthetic_doc() -> ParsedDocument:
    specs: list[tuple[ElementType, str, int | None, int, dict[str, object] | None]] = [
        (ElementType.TITLE, "Preface", 1, 1, {"docling_label": "section_header"}),
        (ElementType.NARRATIVE_TEXT, LONG_BODY, 1, 1, None),
        (ElementType.TITLE, "[7.9 Exercises](#chapter007.xhtml#Sec_7.9)", 2, 2, None),
        (ElementType.NARRATIVE_TEXT, "35", 3, 1, None),
        (ElementType.TITLE, "Interpretability", 3, 1, None),
        (ElementType.NARRATIVE_TEXT, LONG_NUMBERED_BODY, 3, 1, None),
        (ElementType.NARRATIVE_TEXT, "Bibliography", 4, 1, None),
        (ElementType.NARRATIVE_TEXT, "12 Angry Men and other films are covered next", 4, 1, None),
        (ElementType.HEADER, "Chapter 35", 5, 0, None),
        (ElementType.TABLE, f"| {BODY_SENTINEL} | 1 |", 5, 1, None),
        (ElementType.TITLE, "Quarantined Heading", 6, 2, None),
        (ElementType.LIST_ITEM, f"{BODY_SENTINEL} list row", 6, 2, None),
    ]
    elements = [
        _el(i, etype, text, page, depth=depth, metadata=meta)
        for i, (etype, text, page, depth, meta) in enumerate(specs)
    ]
    return _doc(elements, total_pages=6, quarantined=[7])


def test_every_element_is_recorded_in_order_with_type_page_level() -> None:
    doc = _synthetic_doc()
    stream = replay_headings.build_heading_stream(doc, envelope_id="ENV1")
    rows = stream["elements"]
    assert len(rows) == len(doc.elements)
    for row, element in zip(rows, doc.elements, strict=True):
        assert row["type"] == element.element_type.value
        assert row["page"] == element.page_number
        assert row["level"] == element.section_depth
    assert stream["envelope_id"] == "ENV1"
    assert stream["title"] == "Synthetic Book"
    assert stream["total_pages"] == 6
    assert stream["quarantined_pages"] == [7]
    assert stream["parser_used"] == "docling"


def test_non_heading_elements_carry_no_text() -> None:
    stream = replay_headings.build_heading_stream(_synthetic_doc(), envelope_id="ENV1")
    by_index = stream["elements"]
    for idx in (1, 5, 9, 11):  # body paragraph, long numbered body, table, list item
        assert set(by_index[idx]) == {"type", "page", "level"}
    assert BODY_SENTINEL not in json.dumps(stream)


def test_heading_like_elements_carry_text_reject_and_routing() -> None:
    rows = replay_headings.build_heading_stream(_synthetic_doc(), envelope_id="ENV1")["elements"]

    preface = rows[0]
    assert preface["text"] == "Preface"
    assert preface["reject"] is None
    assert preface["route"] == "update_title"
    assert preface["update_title"] is True
    assert preface["label"] == "section_header"

    toc_link = rows[2]
    assert toc_link["reject"] == "markdown_link"
    assert toc_link["route"] == "title_rejected"
    assert toc_link["update_title"] is False

    split_number = rows[3]  # "35" split from "Interpretability"
    assert split_number["text"] == "35"
    assert split_number["reject"] == "numeric_only"
    assert split_number["update_title"] is False

    assert rows[4]["route"] == "update_title"

    bibliography = rows[6]
    assert bibliography["route"] == "pending_heading"
    assert bibliography["update_title"] is False

    numbered_body = rows[7]  # matches chapter numbering; short enough to be a heading
    assert numbered_body["text"].startswith("12 Angry Men")
    assert numbered_body["route"] == "body"

    header = rows[8]
    assert header["text"] == "Chapter 35"
    assert header["route"] == "skip_header_footer"
    assert header["update_title"] is False

    assert rows[10]["route"] == "update_title"


def test_quarantined_title_is_still_routed_like_hierarchical() -> None:
    # hierarchical.py handles TITLE before the quarantine check.
    title = _el(0, ElementType.TITLE, "Kept Heading", 7, depth=1)
    body = _el(1, ElementType.NARRATIVE_TEXT, "Standalone Words", 7, depth=1)
    rows = replay_headings.build_heading_stream(
        _doc([title, body], total_pages=7, quarantined=[7]), envelope_id="E"
    )["elements"]
    assert rows[0]["route"] == "update_title"
    assert rows[1]["route"] == "skip_quarantined"


def test_repeated_running_title_is_skipped_like_hierarchical() -> None:
    elements: list[ParsedElement] = []
    for page in range(1, 7):
        elements.append(_el(len(elements), ElementType.TITLE, "Running Head", page, depth=1))
        elements.append(_el(len(elements), ElementType.NARRATIVE_TEXT, LONG_BODY, page, depth=1))
    rows = replay_headings.build_heading_stream(_doc(elements, total_pages=6), envelope_id="E")[
        "elements"
    ]
    titles = [row for row in rows if row["type"] == "title"]
    # Page 1's TITLE is exempt from the repeated-text skip; the others are dropped.
    assert titles[0]["route"] == "update_title"
    assert {row["route"] for row in titles[1:]} == {"skip_repeated"}
    assert all(row["update_title"] is False for row in titles[1:])


def test_run_writes_fixture_json(tmp_path: Path) -> None:
    doc = _synthetic_doc()
    seen: list[str] = []

    async def fake_fetch(envelope_id: str) -> ParsedDocument:
        seen.append(envelope_id)
        return doc

    full = tmp_path / "nested" / "full.json"
    asyncio.run(replay_headings.run("ENV9", full, full=True, fetch=fake_fetch))
    compact = tmp_path / "nested" / "compact.json"
    asyncio.run(replay_headings.run("ENV9", compact, fetch=fake_fetch))
    assert seen == ["ENV9", "ENV9"]
    stream = replay_headings.build_heading_stream(doc, envelope_id="ENV9")
    assert json.loads(full.read_text(encoding="utf-8")) == stream
    # Compact is the default output.
    assert json.loads(compact.read_text(encoding="utf-8")) == (
        replay_headings.compact_heading_stream(stream)
    )


def test_parsed_key_matches_parse_step() -> None:
    assert replay_headings.parsed_key("ABC") == "parsed/ABC/parsed.json"


# --- compact output -------------------------------------------------------


def _compact_source_doc() -> ParsedDocument:
    specs: list[tuple[ElementType, str, int]] = [
        (ElementType.TITLE, "Preface", 1),
        (ElementType.NARRATIVE_TEXT, LONG_BODY, 1),
        (ElementType.NARRATIVE_TEXT, LONG_BODY, 2),
        (ElementType.NARRATIVE_TEXT, "A", 2),  # short standalone fragment -> run
        (ElementType.IMAGE, "", 2),
        (ElementType.IMAGE, "", 3),
        (ElementType.TITLE, "[7.9 Exercises](#chapter007.xhtml#Sec_7.9)", 3),
        (ElementType.NARRATIVE_TEXT, "35", 3),  # bare number, under 4 chars -> run
        (ElementType.NARRATIVE_TEXT, "3.2 Numbered Candidate Line", 4),  # kept
        (ElementType.NARRATIVE_TEXT, "Bibliography", 4),  # unnumbered standalone -> run
        (ElementType.NARRATIVE_TEXT, "[**1 Introduction**](#toc.xhtml#chap-1)", 5),  # kept
        (ElementType.HEADER, "Running Header", 5),  # no reject -> run
        (ElementType.TABLE, f"| {BODY_SENTINEL} |", 5),
        (ElementType.TITLE, "Interpretability", 6),
    ]
    return _doc(
        [_el(i, etype, text, page, depth=1) for i, (etype, text, page) in enumerate(specs)],
        total_pages=6,
    )


def test_compact_keeps_headings_and_collapses_the_rest_into_runs() -> None:
    stream = replay_headings.build_heading_stream(_compact_source_doc(), envelope_id="E")
    compact = replay_headings.compact_heading_stream(stream)
    entries = replay_headings.expand_compact_entries(compact)

    headings = [entry["text"] for entry in entries if "text" in entry]
    assert headings == [
        "Preface",
        "[7.9 Exercises](#chapter007.xhtml#Sec_7.9)",
        "3.2 Numbered Candidate Line",
        "[**1 Introduction**](#toc.xhtml#chap-1)",
        "Interpretability",
    ]
    runs = [entry for entry in entries if "count" in entry]
    assert all(set(run) == {"type", "page_start", "page_end", "count"} for run in runs)
    assert runs[0] == {"type": "narrative_text", "page_start": 1, "page_end": 2, "count": 3}
    assert runs[1] == {"type": "image", "page_start": 2, "page_end": 3, "count": 2}
    toc_link = entries[[e.get("text") for e in entries].index(headings[1])]
    assert toc_link["reject"] == "markdown_link"
    assert toc_link["route"] == "title_rejected"
    assert toc_link["update_title"] is False
    assert compact["title"] == "Synthetic Book"
    assert BODY_SENTINEL not in replay_headings.dumps_compact(compact)


def test_compact_preserves_order_and_element_count() -> None:
    doc = _compact_source_doc()
    stream = replay_headings.build_heading_stream(doc, envelope_id="E")
    entries = replay_headings.expand_compact_entries(replay_headings.compact_heading_stream(stream))
    expanded_types: list[str] = []
    for entry in entries:
        expanded_types.extend([entry["type"]] * entry.get("count", 1))
    assert expanded_types == [row["type"] for row in stream["elements"]]
    title_texts = [row["text"] for row in stream["elements"] if row["type"] == "title"]
    assert [e["text"] for e in entries if e["type"] == "title"] == title_texts


def test_compact_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    stream = replay_headings.build_heading_stream(_synthetic_doc(), envelope_id="E")
    first = replay_headings.dumps_compact(replay_headings.compact_heading_stream(stream))
    second = replay_headings.dumps_compact(
        replay_headings.compact_heading_stream(json.loads(json.dumps(stream)))
    )
    assert first == second
    assert json.loads(first) == replay_headings.compact_heading_stream(stream)

    full_path = tmp_path / "full.json"
    full_path.write_text(json.dumps(stream), encoding="utf-8")
    out = tmp_path / "compact.json"
    assert replay_headings.main(["--compact-existing", str(full_path), str(out)]) == 0
    assert out.read_text(encoding="utf-8") == first


def test_expand_rejects_non_compact_input() -> None:
    with pytest.raises(ValueError, match="heading-stream-compact"):
        replay_headings.expand_compact_entries({"elements": []})


def test_routing_comes_from_the_chunker_predicate() -> None:
    """The script must not carry its own copy of the chunker's routing."""
    from vo_rag.core.chunking import hierarchical

    assert vars(replay_headings)["route_element"] is hierarchical.route_element
    assert vars(replay_headings)["RoutingContext"] is hierarchical.RoutingContext


# --- compact output --------------------------------------------------------------


def test_compact_drops_text_of_numbered_body_lines() -> None:
    """Footnote citations / sentence fragments that the chunker treats as body."""
    specs: list[tuple[ElementType, str, int]] = [
        (ElementType.TITLE, "1 Introduction", 1),
        (ElementType.NARRATIVE_TEXT, "6 Orrell and Wilmott, The Money Formula of finance.", 1),
        (
            ElementType.NARRATIVE_TEXT,
            "3 heads in a row. The MLE is ![art](images/Art_P595.jpg).",
            2,
        ),
        (ElementType.NARRATIVE_TEXT, "3.2 Numbered Candidate Line", 2),  # pending heading: kept
    ]
    doc = _doc(
        [_el(i, etype, text, page, depth=1) for i, (etype, text, page) in enumerate(specs)],
        total_pages=2,
    )
    stream = replay_headings.build_heading_stream(doc, envelope_id="E")
    assert [row.get("route") for row in stream["elements"][1:3]] == ["body", "body"]
    compact = replay_headings.compact_heading_stream(stream)
    dumped = replay_headings.dumps_compact(compact)
    assert "Orrell" not in dumped and "MLE" not in dumped
    entries = replay_headings.expand_compact_entries(compact)
    assert entries[1] == {"type": "narrative_text", "page_start": 1, "page_end": 2, "count": 2}
    assert entries[2]["text"] == "3.2 Numbered Candidate Line"


def test_recompacting_a_compact_stream_is_idempotent_and_applies_current_rules() -> None:
    stream = replay_headings.build_heading_stream(_compact_source_doc(), envelope_id="E")
    compact = replay_headings.compact_heading_stream(stream)
    assert replay_headings.compact_heading_stream(compact) == compact

    # A compact stream written under an older rule (a body line kept with text)
    # re-compacts to what a fresh capture would give.
    older = json.loads(json.dumps(compact))
    body_row = [
        "H",
        "narrative_text",
        4,
        1,
        "12 Angry Men and other films",
        None,
        "body",
        False,
        None,
    ]
    older["entries"].insert(3, body_row)
    fresh = replay_headings.compact_heading_stream(older)
    assert fresh["entries"] == compact["entries"] or all(
        row[0] == "R" or row[6] != "body" for row in fresh["entries"]
    )
    assert "12 Angry Men" not in replay_headings.dumps_compact(fresh)


def test_compact_existing_accepts_a_compact_file(tmp_path: Path) -> None:
    stream = replay_headings.build_heading_stream(_compact_source_doc(), envelope_id="E")
    source = tmp_path / "compact.json"
    source.write_text(
        replay_headings.dumps_compact(replay_headings.compact_heading_stream(stream)), "utf-8"
    )
    out = tmp_path / "again.json"
    assert replay_headings.main(["--compact-existing", str(source), str(out)]) == 0
    assert out.read_text("utf-8") == source.read_text("utf-8")
