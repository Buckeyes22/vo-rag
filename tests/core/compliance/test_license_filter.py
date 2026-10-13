"""Tests for the ``core.compliance.license_filter`` seam.

The seam is NOT optional at construction time — the
retriever imports it eagerly because the persistence contract for
``license_class`` payload field is part of the Qdrant schema. The
seam IS narrow: three pure functions, no DB, no observability, no
``models.common`` writes.

The compliance log / similarity / citation primitives already exist
in ``core.models.compliance`` and are unrelated to this seam.
"""

from __future__ import annotations

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from vo_rag.core.compliance.license_filter import (
    PUBLISHABLE,
    annotate_unlicensed,
    license_filter_for_mode,
    merge_with_user_filter,
)
from vo_rag.core.models.artifact import LicenseClass
from vo_rag.core.retrieval.query_mode import QueryMode
from vo_rag.core.retrieval.search import SearchResult


def _field_conds(filt: Filter) -> list[FieldCondition]:
    """Return the ``FieldCondition`` entries in ``filt.must``."""
    must = filt.must or []
    return [c for c in must if isinstance(c, FieldCondition)]


def test_publishable_is_cleared_or_public_domain() -> None:
    assert set(PUBLISHABLE) == {LicenseClass.CLEARED, LicenseClass.PUBLIC_DOMAIN}


def test_license_filter_for_internal_is_none() -> None:
    """INTERNAL mode = no license restriction (operators see everything)."""
    assert license_filter_for_mode(QueryMode.INTERNAL) is None


@pytest.mark.parametrize("mode", [QueryMode.PUBLICATION, QueryMode.PARTNER])
def test_license_filter_for_restricted_modes_is_match_any_publishable(mode: QueryMode) -> None:
    filt = license_filter_for_mode(mode)
    assert isinstance(filt, Filter)
    conds = _field_conds(filt)
    assert len(conds) == 1
    cond = conds[0]
    assert cond.key == "license_class"
    assert isinstance(cond.match, MatchAny)
    assert set(cond.match.any) == {"cleared", "public_domain"}


def test_merge_with_user_filter_both_none() -> None:
    assert merge_with_user_filter(None, None) is None


def test_merge_with_user_filter_only_mode() -> None:
    mode = license_filter_for_mode(QueryMode.PUBLICATION)
    assert merge_with_user_filter(mode, None) is mode


def test_merge_with_user_filter_only_user() -> None:
    user = Filter(must=[FieldCondition(key="source_book", match=MatchValue(value="x"))])
    assert merge_with_user_filter(None, user) is user


def test_merge_with_user_filter_concatenates_must_clauses() -> None:
    mode = license_filter_for_mode(QueryMode.PUBLICATION)
    user = Filter(
        must=[FieldCondition(key="source_book", match=MatchValue(value="x"))],
    )
    merged = merge_with_user_filter(mode, user)
    assert isinstance(merged, Filter)
    conds = _field_conds(merged)
    assert len(conds) == 2
    keys = {cond.key for cond in conds}
    assert keys == {"license_class", "source_book"}


# ----------------------------------------------------------------------
# annotate_unlicensed — INTERNAL mode adds warnings to non-publishable
# ----------------------------------------------------------------------


def test_annotate_unlicensed_noop_in_publication_mode() -> None:
    results = [
        SearchResult(id="p", score=0.5, collection="c", payload={"license_class": "pending"}),
    ]
    out = annotate_unlicensed(results, QueryMode.PUBLICATION)
    assert out == results
    assert "_vo_rag_license_warning" not in out[0].payload


def test_annotate_unlicensed_adds_warning_for_non_publishable_in_internal() -> None:
    results = [
        SearchResult(
            id="p",
            score=0.5,
            collection="c",
            payload={"license_class": LicenseClass.PENDING.value},
        ),
    ]
    out = annotate_unlicensed(results, QueryMode.INTERNAL)
    assert out[0].payload.get("_vo_rag_license_warning") == "source_license_class=pending"


def test_annotate_unlicensed_no_warning_for_publishable_in_internal() -> None:
    results = [
        SearchResult(
            id="p",
            score=0.5,
            collection="c",
            payload={"license_class": LicenseClass.CLEARED.value},
        ),
    ]
    out = annotate_unlicensed(results, QueryMode.INTERNAL)
    assert "_vo_rag_license_warning" not in out[0].payload


def test_annotate_unlicensed_handles_missing_license_class() -> None:
    """Backwards-compatible: a payload with no license_class field at all
    should still produce a sensible warning (UNKNOWN)."""
    results = [
        SearchResult(id="p", score=0.5, collection="c", payload={"source_id": "src-1"}),
    ]
    out = annotate_unlicensed(results, QueryMode.INTERNAL)
    assert out[0].payload.get("_vo_rag_license_warning") == "source_license_class=unknown"


def test_annotate_unlicensed_does_not_mutate_input_results() -> None:
    """The seam returns NEW SearchResult objects (via dataclasses.replace)
    so callers can rely on the original list being unchanged."""
    original = SearchResult(
        id="p",
        score=0.5,
        collection="c",
        payload={"license_class": LicenseClass.PENDING.value},
    )
    results = [original]
    annotate_unlicensed(results, QueryMode.INTERNAL)
    assert "_vo_rag_license_warning" not in original.payload
