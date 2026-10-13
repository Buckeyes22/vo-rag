from __future__ import annotations

from vo_rag.core.chunking.table_linearizer import TableLinearizer
from vo_rag.core.parsing.contract import ElementType, ParsedElement


def _el(**meta: str) -> ParsedElement:
    return ParsedElement(
        element_id="t",
        element_type=ElementType.TABLE,
        text="",
        page_number=1,
        metadata=meta,
    )


def test_html_two_cols_five_rows() -> None:
    html = """
    <table>
      <tr><th>Car</th><th>HP</th></tr>
      <tr><td>A</td><td>500</td></tr>
      <tr><td>B</td><td>600</td></tr>
      <tr><td>C</td><td>700</td></tr>
      <tr><td>D</td><td>800</td></tr>
      <tr><td>E</td><td>900</td></tr>
    </table>"""
    out = TableLinearizer().linearize(_el(html=html))
    assert "Comparison across columns: Car, HP." in out
    assert "A: HP=500." in out and "E: HP=900." in out


def test_headerless_html_table_keeps_every_row_as_data() -> None:
    html = """
    <table>
      <tr><td>0-10</td><td>baseline</td></tr>
      <tr><td>10-20</td><td>next</td></tr>
    </table>"""
    out = TableLinearizer().linearize(_el(html=html))

    assert "Row 1: Column 1=0-10, Column 2=baseline." in out
    assert "Row 2: Column 1=10-20, Column 2=next." in out
    assert "Comparison across columns" not in out


def test_html_table_retains_cells_beyond_header_width() -> None:
    html = """
    <table>
      <tr><th>Item</th><th>Value</th></tr>
      <tr><td>Brake</td><td>80%</td><td>rear axle</td></tr>
    </table>"""
    out = TableLinearizer().linearize(_el(html=html))

    assert "Brake: Value=80%, Column 3=rear axle." in out


def test_markdown_fallback() -> None:
    md = """
| Car | HP |
| --- | --- |
| A | 500 |
| B | 600 |
"""
    out = TableLinearizer().linearize(_el(markdown=md))
    assert "Car" in out and "A: HP=500." in out


def test_markdown_without_delimiter_row_keeps_all_rows_as_data() -> None:
    md = "| 0-10 | baseline |\n| 10-20 | next |"

    out = TableLinearizer().linearize(_el(markdown=md))

    assert "Row 1: Column 1=0-10, Column 2=baseline." in out
    assert "Row 2: Column 1=10-20, Column 2=next." in out


def test_markdown_headerless_punctuation_only_row_is_data() -> None:
    md = "| label | value |\n| A | ordinary |\n| --- | :--: |\n| B | final |"

    out = TableLinearizer().linearize(_el(markdown=md))

    assert "Row 1: Column 1=label, Column 2=value." in out
    assert "Row 2: Column 1=A, Column 2=ordinary." in out
    assert "Row 3: Column 1=---, Column 2=:--:." in out
    assert "Row 4: Column 1=B, Column 2=final." in out


def test_markdown_header_keeps_cells_beyond_header_width() -> None:
    md = "| Item | Value |\n| --- | --- |\n| Brake | 80% | rear axle |"

    out = TableLinearizer().linearize(_el(markdown=md))

    assert "Brake: Value=80%, Column 3=rear axle." in out


def test_text_fallback_when_no_html_or_markdown() -> None:
    out = TableLinearizer().linearize(
        ParsedElement(
            element_id="t",
            element_type=ElementType.TABLE,
            text="raw table text",
            page_number=1,
        )
    )
    assert out == "raw table text"


def test_over_budget_html_table_keeps_every_row_for_downstream_splitting() -> None:
    rows = "\n".join(f"<tr><td>R{i}</td><td>V{i}</td></tr>" for i in range(100))
    html = f"<table><tr><th>X</th><th>Y</th></tr>{rows}</table>"
    out = TableLinearizer(max_tokens=30).linearize(_el(html=html))
    assert "X, Y" in out
    assert "R0: Y=V0." in out
    assert "R50: Y=V50." in out
    assert "R99: Y=V99." in out
    assert "rows omitted" not in out


def test_single_column_table_keeps_its_original_header_for_html_and_markdown() -> None:
    """The first row is source data even when the table has just one column."""
    for element in (
        _el(html="<table><tr><th>Cutting Speeds</th></tr><tr><td>Feeds</td></tr></table>"),
        _el(markdown="| Cutting Speeds |\n| --- |\n| Feeds |"),
    ):
        text = TableLinearizer().linearize(element)
        assert "Column header: Cutting Speeds." in text
        assert "Feeds." in text


def test_single_column_header_only_table_keeps_header_and_empty_table_stays_empty() -> None:
    """A sole header remains searchable without fabricating missing data rows."""
    for element in (
        _el(html="<table><tr><th>Feed Limits</th></tr></table>"),
        _el(markdown="| Feed Limits |\n| --- |"),
    ):
        assert TableLinearizer().linearize(element) == "Column header: Feed Limits."
    assert TableLinearizer().linearize(_el(html="<table></table>")) == ""
