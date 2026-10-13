"""Every `tf_*` payload field declared in `tf_schema` has a matching Qdrant
payload index on the book collections (VOR-1443).

Without the index a `tf_*` filter degrades to a full payload scan, so the
schema and `qdrant.BOOK_EXTRA_INDEXES` must not drift apart.
"""

from __future__ import annotations

import pytest
from qdrant_client.models import PayloadSchemaType

from vo_rag.core.corpus.tf_schema import TF_INDEXED_FIELDS
from vo_rag.core.storage.qdrant import (
    BOOK_EXTRA_INDEXES,
    _tf_payload_kind,
)


def test_every_tf_field_has_a_matching_index() -> None:
    """Pin one representative field of each ``tf_schema`` index kind to its
    literal ``PayloadSchemaType`` — independently of ``_tf_payload_kind``, the
    very helper under test — so this cannot regress into comparing
    ``BOOK_EXTRA_INDEXES`` to the same mapping that built it (VOR-1443 r3)."""
    assert BOOK_EXTRA_INDEXES["tf_pub_year"] == PayloadSchemaType.INTEGER
    assert BOOK_EXTRA_INDEXES["tf_tag_version"] == PayloadSchemaType.INTEGER
    assert BOOK_EXTRA_INDEXES["tf_event_family"] == PayloadSchemaType.KEYWORD
    assert BOOK_EXTRA_INDEXES["tf_book_reviewed"] == PayloadSchemaType.BOOL
    for field in TF_INDEXED_FIELDS:
        assert field in BOOK_EXTRA_INDEXES, field
    # 15 ALLOWED fields (incl. the 3 book-level population fields, copied onto
    # every chunk) + 6 extras (tf_pub_year, tf_pub_decade,
    # tf_population_source, tf_tag_version, tf_tagger_model, tf_book_reviewed).
    assert len(TF_INDEXED_FIELDS) == 21
    assert BOOK_EXTRA_INDEXES["tf_book_athlete_level"] == PayloadSchemaType.KEYWORD


def test_unknown_tf_index_kind_names_the_field_and_kind() -> None:
    """A kind that ``tf_schema`` adds without ``qdrant`` teaching must fail
    loudly, naming the field — a silent skip drops the payload index and the
    ``tf_*`` filter degrades to a full scan (VOR-1443)."""
    with pytest.raises(KeyError, match="tf_probe.*float"):
        _tf_payload_kind("tf_probe", "float")
