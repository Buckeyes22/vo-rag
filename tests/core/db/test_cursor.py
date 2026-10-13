"""Unit tests for the URL-safe opaque keyset cursor."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timezone

import pytest

from vo_rag.core.db.cursor import decode_cursor, encode_cursor


class TestCursorRoundTrip:
    """encode_cursor → decode_cursor must be a lossless bijection."""

    def test_round_trip_with_aware_utc_datetime(self) -> None:
        created_at = datetime(2026, 5, 14, 9, 30, 45, 123456, tzinfo=UTC)
        row_id = "01HZP5XQ8R2G7K9F4N1A0B3C7D"

        token = encode_cursor(created_at, row_id)
        decoded_at, decoded_id = decode_cursor(token)

        assert decoded_id == row_id
        assert decoded_at == created_at

    def test_round_trip_with_naive_datetime(self) -> None:
        created_at = datetime(2026, 1, 1, 0, 0, 0)
        row_id = "01HZP5XQ8R2G7K9F4N1A0B3C7D"

        token = encode_cursor(created_at, row_id)
        decoded_at, decoded_id = decode_cursor(token)

        assert decoded_id == row_id
        assert decoded_at == created_at

    def test_round_trip_with_non_utc_timezone(self) -> None:
        # non-UTC offset (UTC-05:00)
        from datetime import timedelta as _td

        tz = timezone(_td(hours=-5))
        created_at = datetime(2026, 5, 14, 9, 30, 45, 123456, tzinfo=tz)
        row_id = "01HZP5XQ8R2G7K9F4N1A0B3C7D"

        token = encode_cursor(created_at, row_id)
        decoded_at, decoded_id = decode_cursor(token)

        assert decoded_id == row_id
        assert decoded_at == created_at

    def test_round_trip_microsecond_precision_preserved(self) -> None:
        created_at = datetime(2026, 5, 14, 9, 30, 45, 999999, tzinfo=UTC)
        row_id = "01HZP5XQ8R2G7K9F4N1A0B3C7D"

        token = encode_cursor(created_at, row_id)
        decoded_at, _ = decode_cursor(token)

        assert decoded_at.microsecond == 999999

    def test_round_trip_ulid_with_all_valid_chars(self) -> None:
        # ULID excludes I, L, O, U — confirm a valid Crockford-base32 ULID
        # round-trips unchanged.
        created_at = datetime(2026, 5, 14, 9, 30, 45, tzinfo=UTC)
        row_id = "0123456789ABCDEFGHJKMNPQRST"

        token = encode_cursor(created_at, row_id)
        _, decoded_id = decode_cursor(token)

        assert decoded_id == row_id


class TestCursorShape:
    """Cursor encoding shape (URL-safe, no padding)."""

    def test_token_is_url_safe(self) -> None:
        # No padding chars, no `+` or `/` — only `[A-Za-z0-9_-]`.
        created_at = datetime(2026, 5, 14, 9, 30, 45, tzinfo=UTC)
        row_id = "01HZP5XQ8R2G7K9F4N1A0B3C7D"

        token = encode_cursor(created_at, row_id)

        assert "=" not in token
        assert "+" not in token
        assert "/" not in token
        # Round-trip-safe characters only
        token.replace("-", "").replace("_", "")
        for ch in token:
            assert ch.isalnum() or ch in "-_"

    def test_token_payload_is_json_datetime_isoformat_id_tuple(self) -> None:
        # Decode the base64-padded payload directly and assert the JSON shape
        # is `[<iso_datetime_string>, <id_string>]`. This pins the on-wire
        # contract so a future change to the format is loud and reviewable.
        created_at = datetime(2026, 5, 14, 9, 30, 45, 123456, tzinfo=UTC)
        row_id = "01HZP5XQ8R2G7K9F4N1A0B3C7D"

        token = encode_cursor(created_at, row_id)
        raw = base64.urlsafe_b64decode(token + "===")
        payload = json.loads(raw)

        assert isinstance(payload, list)
        assert len(payload) == 2
        assert payload[0] == created_at.isoformat()
        assert payload[1] == row_id

    def test_tokens_differ_when_inputs_differ(self) -> None:
        base = datetime(2026, 5, 14, 9, 30, 45, tzinfo=UTC)
        id_ = "01HZP5XQ8R2G7K9F4N1A0B3C7D"

        t1 = encode_cursor(base, id_)
        t2 = encode_cursor(base + timedelta(microseconds=1), id_)
        t3 = encode_cursor(base, id_ + "X")
        t4 = encode_cursor(base, id_)

        assert t1 != t2
        assert t1 != t3
        assert t1 == t4  # stable / deterministic

    def test_rejects_corrupt_token(self) -> None:
        # The cursor decoder uses raw base64 + json.loads. base64 raises
        # binascii.Error (subclass of ValueError) on bad padding/characters;
        # json.loads raises json.JSONDecodeError (also subclass of ValueError).
        with pytest.raises(ValueError):
            decode_cursor("not-a-valid-cursor!!!")


from datetime import timedelta  # noqa: E402  — kept near its sole use above
