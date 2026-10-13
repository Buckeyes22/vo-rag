"""Compliance primitives: license filter + annotation seam for retrieval.

The seam is intentionally tiny and side-effect-free: pure Qdrant ``Filter``
construction and a result-list annotation, no DB, no observability, no model writes.

This module is the load-bearing boundary between retrieval and
licensing. The retriever (``vo_rag.core.retrieval.search``) imports it
eagerly — the ``license_class`` payload field is part of the Qdrant
schema, so a missing/optional seam would break every public-mode query.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from qdrant_client.models import FieldCondition, Filter, MatchAny

from vo_rag.core.models.artifact import LicenseClass

if TYPE_CHECKING:
    from vo_rag.core.retrieval.query_mode import QueryMode
    from vo_rag.core.retrieval.search import SearchResult


PUBLISHABLE: tuple[LicenseClass, ...] = (LicenseClass.CLEARED, LicenseClass.PUBLIC_DOMAIN)


def license_filter_for_mode(mode: QueryMode) -> Filter | None:
    """Qdrant filter enforcing license restrictions for the query mode.

    INTERNAL mode returns ``None`` (operators see everything, regardless
    of license class). PUBLICATION and PARTNER return a ``Filter``
    that restricts to ``cleared`` + ``public_domain`` license classes.
    """
    if str(mode) == "internal":
        return None

    return Filter(
        must=[
            FieldCondition(
                key="license_class",
                match=MatchAny(any=[license_class.value for license_class in PUBLISHABLE]),
            )
        ]
    )


def _as_list(value: Any) -> list[Any]:
    """Normalise the qdrant ``Filter.must/must_not/should`` field to a list.

    The qdrant-client types these as ``Optional[Union[List[Condition],
    Condition]]`` — a single condition or a list. The retriever only ever
    puts lists in, so the normaliser is a defensive cast.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def merge_with_user_filter(
    mode_filter: Filter | None,
    user_filter: Filter | None,
) -> Filter | None:
    """AND-combine mode and user filters while preserving all clauses."""
    if mode_filter is None:
        return user_filter
    if user_filter is None:
        return mode_filter

    must = _as_list(mode_filter.must) + _as_list(user_filter.must)
    must_not = _as_list(mode_filter.must_not) + _as_list(user_filter.must_not)
    should = _as_list(mode_filter.should) + _as_list(user_filter.should)

    # ``Filter`` types its clause fields as ``Optional[Union[List[Condition],
    # Condition]]``; we always pass a list, so the cast is a defensive
    # narrowing. The cast happens at the boundary because the union
    # prevents direct assignment.
    return Filter(
        must=cast("list[FieldCondition] | None", must or None),  # type: ignore[arg-type]
        must_not=cast("list[FieldCondition] | None", must_not or None),  # type: ignore[arg-type]
        should=cast("list[FieldCondition] | None", should or None),  # type: ignore[arg-type]
    )


def annotate_unlicensed(results: list[SearchResult], mode: QueryMode) -> list[SearchResult]:
    """In INTERNAL mode, add a warning payload key for non-publishable
    license classes. Public modes are never annotated — the license
    filter already constrains the result set.

    Returns NEW ``SearchResult`` objects (via ``dataclasses.replace``)
    so callers can rely on the input list being unchanged.
    """
    if str(mode) != "internal":
        return results

    output: list[SearchResult] = []
    for result in results:
        payload = result.payload or {}
        license_value = str(payload.get("license_class", LicenseClass.UNKNOWN.value))
        warning_license_value = license_value
        try:
            license_class = LicenseClass(license_value)
        except ValueError:
            license_class = LicenseClass.UNKNOWN

        if license_class not in PUBLISHABLE:
            new_payload = dict(payload)
            new_payload["_vo_rag_license_warning"] = f"source_license_class={warning_license_value}"
            output.append(replace(result, payload=new_payload))
            continue

        output.append(result)

    return output


__all__ = [
    "PUBLISHABLE",
    "annotate_unlicensed",
    "license_filter_for_mode",
    "merge_with_user_filter",
]
