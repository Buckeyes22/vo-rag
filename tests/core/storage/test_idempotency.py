"""Unit tests for `vo_rag.core.idempotency` (VOR-2).

`vo_rag.core.storage.qdrant` depends on this module for its point ids.

The deterministic UUID is **byte-for-byte** the F-024 point-id key (retries
overwrite; purge filters on the payload ``source_id``) — it
must not change between Python versions, processes, or host OS. We assert a
known-good digest below to lock that contract.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from vo_rag.core.idempotency import deterministic_hex, deterministic_uuid

# ----------------------------------------------------------------------
# deterministic_uuid
# ----------------------------------------------------------------------


def test_deterministic_uuid_is_stable_across_calls() -> None:
    a = deterministic_uuid("source-1", "0", "raw")
    b = deterministic_uuid("source-1", "0", "raw")
    assert a == b


def test_deterministic_uuid_matches_sha256_first_16_bytes() -> None:
    """Lock the digest formula: SHA-256 over '|'.join(parts), first 16 bytes -> UUID."""
    parts = ("alpha", "1", "stage-x")
    expected_digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()[:16]
    expected_uuid = str(uuid.UUID(bytes=expected_digest))

    assert deterministic_uuid(*parts) == expected_uuid


def test_deterministic_uuid_is_a_valid_uuid_string() -> None:
    out = deterministic_uuid("x", "y", "z")
    # Round-trip through uuid.UUID — raises if malformed.
    parsed = uuid.UUID(out)
    assert str(parsed) == out


def test_deterministic_uuid_differs_on_any_part_change() -> None:
    base = deterministic_uuid("source-1", "0", "raw")
    assert deterministic_uuid("source-2", "0", "raw") != base
    assert deterministic_uuid("source-1", "1", "raw") != base
    assert deterministic_uuid("source-1", "0", "sterile") != base


def test_deterministic_uuid_order_matters() -> None:
    """The parts are joined positionally; order is part of the contract."""
    a = deterministic_uuid("a", "b", "c")
    b = deterministic_uuid("b", "a", "c")
    assert a != b


def test_deterministic_uuid_rejects_zero_parts() -> None:
    with pytest.raises(ValueError):
        deterministic_uuid()


# ----------------------------------------------------------------------
# deterministic_hex
# ----------------------------------------------------------------------


def test_deterministic_hex_is_stable() -> None:
    a = deterministic_hex("foo", "bar")
    b = deterministic_hex("foo", "bar")
    assert a == b


def test_deterministic_hex_default_length_is_32() -> None:
    assert len(deterministic_hex("foo", "bar")) == 32


def test_deterministic_hex_length_truncates() -> None:
    assert len(deterministic_hex("foo", "bar", length=8)) == 8
    assert len(deterministic_hex("foo", "bar", length=64)) == 64


def test_deterministic_hex_rejects_zero_parts() -> None:
    with pytest.raises(ValueError):
        deterministic_hex()


@pytest.mark.parametrize("bad_len", [0, -1, 65, 1000])
def test_deterministic_hex_rejects_out_of_range_length(bad_len: int) -> None:
    with pytest.raises(ValueError):
        deterministic_hex("foo", length=bad_len)


def test_deterministic_hex_matches_sha256_prefix() -> None:
    parts = ("alpha", "beta")
    expected = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]
    assert deterministic_hex(*parts) == expected
