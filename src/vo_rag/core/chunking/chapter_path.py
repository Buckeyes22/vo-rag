from __future__ import annotations

import re


def derive_chapter_path(section_path: str, doc_title: str) -> str:
    stripped = section_path.strip()
    if stripped.startswith("["):
        stripped = stripped[1:]
    if stripped.endswith("]"):
        stripped = stripped[:-1]

    segments = stripped.split(" > ")

    if len(segments) <= 1:
        return f"[{doc_title}]"

    heading_segments = segments[1:]
    first_heading = heading_segments[0].strip()

    number_match = re.match(r"^\s*(\d+)(?:\.\d+)*", first_heading)
    if number_match:
        n = number_match.group(1)
        return f"[{doc_title} > Ch {n}]"

    return f"[{doc_title} > {first_heading}]"
