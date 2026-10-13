"""Shared Pydantic base + primitive types for vo_rag models."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class VoBaseModel(BaseModel):
    """Base for every vo_rag model.

    - extra='forbid': catch typos in unknown fields during dict→model coercion
    - str_strip_whitespace: trim inputs at model boundary
    - validate_assignment: re-run validators when fields are reassigned
    - use_enum_values=False: keep enum instances so callers can compare by type
    """

    model_config = ConfigDict(
        use_enum_values=False,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )


# Crockford-base32 ULID, 26 chars. Used for all primary keys.
# Pattern excludes I, L, O, U to avoid 0/O and 1/I/L confusion.
ULID = Annotated[
    str,
    Field(
        min_length=26,
        max_length=26,
        pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$",
        description="Crockford-base32 ULID",
    ),
]


def ulid_new() -> str:
    """Generate a new ULID string. Lazily imports python-ulid so importing this
    module stays fast when only types (not generators) are needed."""
    from ulid import ULID as _ULID  # python-ulid package

    return str(_ULID())
