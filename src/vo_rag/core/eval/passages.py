"""Passage-level projection helpers for retrieval evals.

The eval metrics operate on ranked string identifiers. These helpers translate
chunk-shaped records into stable passage identifiers so chunking changes do not
orphan relevance judgments.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from vo_rag.core.eval.qrels import Judgment, Qrels

type RelevanceMode = Literal["chunk_id", "passage"]
type GoldPassageProjectionFn = Callable[[str], set[str]]


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    """Search result with both a ranking id and payload metadata."""

    doc_id: str
    payload: dict[str, Any] = field(default_factory=dict)


type RetrievedItem = str | RetrievedDocument | Mapping[str, Any]

_PASSAGE_KEY_RE = re.compile(r"^.+#p[1-9][0-9]*$")
_PAGE_RE = re.compile(r"\d+")


def passage_key(source_id: str, page: int) -> str:
    """Return the canonical passage relevance key for a source page."""
    source = source_id.strip()
    if not source:
        msg = "source_id must be non-empty"
        raise ValueError(msg)
    if page < 1:
        msg = "page must be >= 1"
        raise ValueError(msg)
    return f"{source}#p{page}"


def is_passage_key(value: str) -> bool:
    """Return True when `value` is a canonical passage key."""
    return _PASSAGE_KEY_RE.match(value) is not None


def passage_keys_from_payload(payload: Mapping[str, Any]) -> set[str]:
    """Project a retrieved chunk payload to all passage keys it covers.

    `page_range` wins when present because a chunk can span multiple pages.
    When no usable range is present, the function falls back to `page_number`.
    """
    source = _string_field(payload, "source_id")
    if source is None:
        return set()

    pages = _pages_from_range(payload.get("page_range"))
    if not pages:
        pages = _pages_from_single(payload.get("page_number"))
    return {passage_key(source, page) for page in pages}


def passage_keys_from_gold_payload(payload: Mapping[str, Any]) -> set[str]:
    """Project a gold chunk payload to its single judged passage page."""
    source = _string_field(payload, "source_id")
    if source is None:
        return set()
    return {passage_key(source, page) for page in _pages_from_single(payload.get("page_number"))}


def project_retrieved_to_doc_ids(retrieved: Sequence[RetrievedItem]) -> list[str]:
    """Normalize legacy search results to ranked document ids."""
    out: list[str] = []
    for item in retrieved:
        doc_id = doc_id_from_retrieved(item)
        if doc_id is not None:
            out.append(doc_id)
    return out


def project_retrieved_to_passage_keys(retrieved: Sequence[RetrievedItem]) -> list[str]:
    """Flatten ranked retrieved chunks into ranked, de-duplicated passage keys."""
    out: list[str] = []
    seen: set[str] = set()
    for item in retrieved:
        for key in _passage_keys_from_retrieved_item(item):
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def project_qrels_to_passage_keys(
    qrels: Qrels,
    gold_doc_projection_fn: GoldPassageProjectionFn | None = None,
) -> Qrels:
    """Project qrels judgments into passage-key space with max-grade de-duping."""
    grades: dict[tuple[str, str], int] = {}
    for judgment in qrels.judgments:
        keys = _passage_keys_from_gold_doc_id(judgment.doc_id, gold_doc_projection_fn)
        for key in sorted(keys, key=_passage_sort_key):
            current = grades.get((judgment.query_id, key))
            if current is None or judgment.grade > current:
                grades[(judgment.query_id, key)] = judgment.grade

    judgments = [
        Judgment(query_id=query_id, doc_id=key, grade=grade)
        for (query_id, key), grade in sorted(grades.items(), key=lambda item: item[0])
    ]
    return Qrels(pipeline=qrels.pipeline, collection=qrels.collection, judgments=judgments)


def doc_id_from_retrieved(item: RetrievedItem) -> str | None:
    """Return the best legacy id available on a retrieved item."""
    if isinstance(item, str):
        return item
    if isinstance(item, RetrievedDocument):
        return item.doc_id

    for field_name in ("doc_id", "id", "chunk_id"):
        value = item.get(field_name)
        if value is not None:
            return str(value)

    payload = payload_from_retrieved(item)
    chunk_id = payload.get("chunk_id")
    return str(chunk_id) if chunk_id is not None else None


def payload_from_retrieved(item: RetrievedItem) -> Mapping[str, Any]:
    """Return payload metadata from a retrieved item, or an empty mapping."""
    if isinstance(item, str):
        return {}
    if isinstance(item, RetrievedDocument):
        return item.payload

    payload = item.get("payload")
    if isinstance(payload, Mapping):
        return payload
    return item


def _passage_keys_from_retrieved_item(item: RetrievedItem) -> list[str]:
    if isinstance(item, str):
        return [item] if is_passage_key(item) else []

    keys = passage_keys_from_payload(payload_from_retrieved(item))
    if keys:
        return sorted(keys, key=_passage_sort_key)

    doc_id = doc_id_from_retrieved(item)
    if doc_id is not None and is_passage_key(doc_id):
        return [doc_id]
    return []


def _passage_keys_from_gold_doc_id(
    doc_id: str,
    gold_doc_projection_fn: GoldPassageProjectionFn | None,
) -> set[str]:
    if gold_doc_projection_fn is not None:
        return {key for key in gold_doc_projection_fn(doc_id) if is_passage_key(key)}
    if is_passage_key(doc_id):
        return {doc_id}
    return set()


def _string_field(payload: Mapping[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pages_from_single(value: Any) -> list[int]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value] if value >= 1 else []

    match = _PAGE_RE.search(str(value).strip())
    if match is None:
        return []
    page = int(match.group(0))
    return [page] if page >= 1 else []


def _pages_from_range(value: Any) -> list[int]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value] if value >= 1 else []

    pages = [int(part) for part in _PAGE_RE.findall(str(value).strip())]
    pages = [page for page in pages if page >= 1]
    if not pages:
        return []
    if len(pages) == 1:
        return [pages[0]]

    start = min(pages[0], pages[-1])
    end = max(pages[0], pages[-1])
    return list(range(start, end + 1))


def _passage_sort_key(key: str) -> tuple[str, int, str]:
    source, separator, page = key.rpartition("#p")
    if separator and page.isdigit():
        return (source, int(page), key)
    return (key, 0, key)


__all__ = [
    "GoldPassageProjectionFn",
    "RetrievedDocument",
    "RetrievedItem",
    "RelevanceMode",
    "doc_id_from_retrieved",
    "is_passage_key",
    "passage_key",
    "passage_keys_from_gold_payload",
    "passage_keys_from_payload",
    "payload_from_retrieved",
    "project_qrels_to_passage_keys",
    "project_retrieved_to_doc_ids",
    "project_retrieved_to_passage_keys",
]
