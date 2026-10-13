"""Bind actual original and accepted parser bytes without claiming extraction fidelity."""

from __future__ import annotations

import hashlib
import json

from vo_rag.core.parsing.contract import ParsedDocument

__all__ = ["accepted_element_inventory_sha256", "accepted_parse_binding"]


def accepted_element_inventory_sha256(parsed: ParsedDocument) -> str:
    """Hash ordered accepted element occurrences, including repeated content."""
    digest = hashlib.sha256()
    for ordinal, element in enumerate(parsed.elements):
        record = {
            "accepted_element_ordinal": ordinal,
            "element_id": element.element_id,
            "element_type": str(element.element_type),
            "page_number": element.page_number,
            "text_sha256": hashlib.sha256(element.text.encode("utf-8")).hexdigest(),
            "text_chars": len(element.text),
        }
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def accepted_parse_binding(
    parsed: ParsedDocument, original_sha256: str, accepted_bytes: bytes
) -> dict[str, object]:
    """Bind actual stage artifacts; original extraction and chunk spans remain unverified."""
    return {
        "version": 1,
        "original_sha256": original_sha256,
        "accepted_parse_sha256": hashlib.sha256(accepted_bytes).hexdigest(),
        "accepted_element_inventory_sha256": accepted_element_inventory_sha256(parsed),
        "accepted_element_count": len(parsed.elements),
        "parser_used": parsed.parser_used,
        "original_to_parse_fidelity": "unverified",
        "parse_to_chunk_intervals": "unverified",
    }
