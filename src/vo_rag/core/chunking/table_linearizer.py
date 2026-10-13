from __future__ import annotations

import re
from collections.abc import Callable

from vo_rag.core.chunking.counters import naive_counter
from vo_rag.core.parsing.contract import ParsedElement


class TableLinearizer:
    def __init__(
        self,
        *,
        max_tokens: int = 600,
        counter: Callable[[str], int] | None = None,
    ) -> None:
        self.max = max_tokens
        self.count = counter or naive_counter

    def linearize(self, table_element: ParsedElement) -> str:
        html = table_element.metadata.get("html")
        if html:
            return self._from_html(html)

        md = table_element.metadata.get("markdown")
        if md:
            return self._from_markdown(md)

        return table_element.text or ""

    def _from_html(self, html: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        if not rows:
            return soup.get_text(" ", strip=True)

        first_cells = rows[0].find_all(["th", "td"], recursive=False)
        has_headers = any(cell.name == "th" for cell in first_cells)
        headers = [cell.get_text(" ", strip=True) for cell in first_cells] if has_headers else []
        lines: list[str] = []
        if len(headers) >= 2:
            lines.append(f"Comparison across columns: {', '.join(headers)}.")
        elif headers:
            lines.append(f"Column header: {headers[0]}.")

        data_rows = rows[1:] if has_headers else rows
        for row_number, tr in enumerate(data_rows, start=2 if has_headers else 1):
            cells = [
                cell.get_text(" ", strip=True) or "[empty]"
                for cell in tr.find_all(["td", "th"], recursive=False)
            ]
            if not cells:
                continue

            if not has_headers:
                values = ", ".join(
                    f"Column {column}={value}" for column, value in enumerate(cells, start=1)
                )
                lines.append(f"Row {row_number}: {values}.")
                continue

            row_label = cells[0]
            pairs = ", ".join(
                f"{headers[column] if column < len(headers) else f'Column {column + 1}'}={value}"
                for column, value in enumerate(cells[1:], start=1)
            )
            lines.append(f"{row_label}: {pairs}." if pairs else f"{row_label}.")
        return "\n".join(lines)

    def _from_markdown(self, md: str) -> str:
        # A GFM delimiter row is explicit evidence that the first row is a header.
        clean = [
            line.strip().strip("|") for line in md.splitlines() if line.strip().startswith("|")
        ]
        cells = [[c.strip() for c in row.split("|")] for row in clean]
        has_headers = len(cells) > 1 and all(
            re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in cells[1]
        )
        if has_headers:
            # Only the recognized GFM delimiter row is syntax. A later row
            # made of punctuation can be real table data (for example a
            # separator used as a value in a table without a header row).
            cells = [cells[0], *cells[2:]]
        if not cells:
            return md

        headers = cells[0] if has_headers else []
        lines: list[str] = []
        if len(headers) >= 2:
            lines.append(f"Comparison across columns: {', '.join(headers)}.")
        elif headers:
            lines.append(f"Column header: {headers[0]}.")

        data_rows = cells[1:] if has_headers else cells
        for row_number, row in enumerate(data_rows, start=2 if has_headers else 1):
            if not row:
                continue

            if not has_headers:
                values = ", ".join(
                    f"Column {column}={value or '[empty]'}"
                    for column, value in enumerate(row, start=1)
                )
                lines.append(f"Row {row_number}: {values}.")
                continue

            row_label = row[0] or "[empty]"
            pairs = ", ".join(
                f"{headers[column] if column < len(headers) else f'Column {column + 1}'}="
                f"{value or '[empty]'}"
                for column, value in enumerate(row[1:], start=1)
            )
            lines.append(f"{row_label}: {pairs}." if pairs else f"{row_label}.")
        return "\n".join(lines)
