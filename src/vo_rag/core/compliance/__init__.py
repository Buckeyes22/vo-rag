"""Compliance primitives for the retrieval/answer path.

The seam is intentionally narrow: three pure functions for
the retriever to compose with the Qdrant store. The compliance log /
similarity / citation primitives used by answer synthesis live in
``vo_rag.core.models.compliance`` and are unrelated to this seam.
"""

from __future__ import annotations

from vo_rag.core.compliance.license_filter import (
    PUBLISHABLE,
    annotate_unlicensed,
    license_filter_for_mode,
    merge_with_user_filter,
)
from vo_rag.core.compliance.sterilizer import (
    SYSTEM_PROMPT,
    SterilizerParseError,
    sterilize_chunks,
)

__all__ = [
    "PUBLISHABLE",
    "SYSTEM_PROMPT",
    "SterilizerParseError",
    "annotate_unlicensed",
    "license_filter_for_mode",
    "merge_with_user_filter",
    "sterilize_chunks",
]
