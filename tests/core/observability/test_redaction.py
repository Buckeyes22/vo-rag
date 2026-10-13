"""Redaction policy tests (VOR-2).

The redaction level is read from ``VORAG_REDACTION_LEVEL``.
"""

from __future__ import annotations

import pytest

from vo_rag.core.observability.redaction import (
    DEFAULT_ALWAYS_SCRUB,
    RedactConfig,
    RedactionLevel,
    Redactor,
    redact,
    reset_default_redactor,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_default_redactor()


# ------------------------------------------------------------------
# Emails
# ------------------------------------------------------------------


def test_redacts_plain_email() -> None:
    assert redact("contact maintainer@example.com now") == "contact <EMAIL_ADDRESS> now"


def test_redacts_email_with_plus_and_dot() -> None:
    assert redact("email reader+filter@example.co.uk here") == "email <EMAIL_ADDRESS> here"


def test_multiple_emails_all_redacted() -> None:
    text = "a@b.com and c@d.com"
    assert redact(text) == "<EMAIL_ADDRESS> and <EMAIL_ADDRESS>"


# ------------------------------------------------------------------
# Phone numbers (US + intl)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "555-555-5555",
        "(555) 555-5555",
        "555.555.5555",
        "+1 555 555 5555",
        "+44 20 1234 5678",
    ],
)
def test_redacts_phone_formats(raw: str) -> None:
    assert "<PHONE_NUMBER>" in redact(f"call me at {raw}.")


# ------------------------------------------------------------------
# API keys
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "sk-ant-api01-" + "a" * 40,
        "sk-" + "a" * 48,
        "xoxb-" + "a" * 24,
        "AKIA" + "A" * 16,
        "ghp_" + "a" * 36,
    ],
)
def test_redacts_api_key_patterns(key: str) -> None:
    assert redact(f"token: {key}") == "token: <API_KEY>"


# ------------------------------------------------------------------
# SSN
# ------------------------------------------------------------------


def test_redacts_us_ssn() -> None:
    assert redact("SSN 123-45-6789 filed") == "SSN <US_SSN> filed"


# ------------------------------------------------------------------
# Credit card (Luhn-checked)
# ------------------------------------------------------------------


def test_redacts_valid_credit_card() -> None:
    # Valid Visa test number (Luhn-valid).
    assert redact("card 4111 1111 1111 1111 done") == "card <CREDIT_CARD> done"


def test_preserves_invalid_digit_run() -> None:
    """Random digits that fail Luhn stay."""
    assert "1234567890123456" in redact("id 1234567890123456 here")


# ------------------------------------------------------------------
# IP addresses — only on strict level
# ------------------------------------------------------------------


def test_ip_not_redacted_on_standard() -> None:
    assert "192.168.1.1" in redact("seen from 192.168.1.1 earlier")


def test_ip_redacted_on_strict() -> None:
    out = redact("seen from 192.168.1.1", level=RedactionLevel.STRICT)
    assert "192.168.1.1" not in out
    assert "<IP_ADDRESS>" in out


# ------------------------------------------------------------------
# Levels
# ------------------------------------------------------------------


def test_off_is_passthrough() -> None:
    out = redact("contact maintainer@example.com", level=RedactionLevel.OFF)
    assert out == "contact maintainer@example.com"


def test_standard_keeps_person_in_text() -> None:
    """Level STANDARD doesn't activate PERSON detection; names pass through."""
    out = redact("Ayrton Senna drove in 1988", level=RedactionLevel.STANDARD)
    assert "Ayrton Senna" in out


# ------------------------------------------------------------------
# redact_mapping
# ------------------------------------------------------------------


def test_redact_mapping_scrubs_values() -> None:
    r = Redactor(
        RedactConfig(level=RedactionLevel.STANDARD, always_scrub_in_logs=DEFAULT_ALWAYS_SCRUB)
    )
    out = r.redact_mapping(
        {
            "user_email": "maintainer@example.com",
            "token": "super-secret",
            "nested": {"phone": "555-555-5555"},
        }
    )
    assert out["user_email"] == "<EMAIL_ADDRESS>"
    assert out["token"] == "<REDACTED>"
    assert out["nested"]["phone"] == "<PHONE_NUMBER>"


def test_redact_mapping_scrubs_provider_style_secret_keys() -> None:
    r = Redactor(
        RedactConfig(level=RedactionLevel.STANDARD, always_scrub_in_logs=DEFAULT_ALWAYS_SCRUB)
    )
    out = r.redact_mapping(
        {
            "openai_api_key": "not-a-regex-shaped-secret",  # pragma: allowlist secret
            "secret_key": "plain text secret",  # pragma: allowlist secret
            "nested": {
                "access_token": "short-token-value",
                "Authorization": "Bearer short-token-value",
            },
        }
    )
    assert out["openai_api_key"] == "<REDACTED>"
    assert out["secret_key"] == "<REDACTED>"
    assert out["nested"]["access_token"] == "<REDACTED>"
    assert out["nested"]["Authorization"] == "<REDACTED>"


def test_redact_mapping_scrubs_camel_case_secret_keys() -> None:
    r = Redactor(
        RedactConfig(level=RedactionLevel.STANDARD, always_scrub_in_logs=DEFAULT_ALWAYS_SCRUB)
    )
    out = r.redact_mapping(
        {
            "apiKey": "plain-api-key",  # pragma: allowlist secret
            "openaiApiKey": "plain-openai-key",  # pragma: allowlist secret
            "nested": {"accessToken": "plain-access-token"},
        }
    )
    assert out["apiKey"] == "<REDACTED>"
    assert out["openaiApiKey"] == "<REDACTED>"
    assert out["nested"]["accessToken"] == "<REDACTED>"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("APIKey", "<REDACTED>"),
        ("apikey", "<REDACTED>"),
        ("DBPassword", "<REDACTED>"),
        ("HTTPAuthorization", "<REDACTED>"),
        ("CLIENTSECRET", "<REDACTED>"),
        ("apiKey", "<REDACTED>"),
        ("AccessToken", "<REDACTED>"),
        ("refreshtoken", "<REDACTED>"),
        ("monkey", "visible-value"),
        ("passport", "visible-value"),
        # Token *telemetry* must stay visible — the squashed comparison is
        # suffix-anchored, so these do not match the "token" scrub key.
        ("max_tokens", "visible-value"),
        ("prompt_tokens", "visible-value"),
        ("completion_tokens", "visible-value"),
        ("total_tokens", "visible-value"),
        ("tokenizer", "visible-value"),
    ],
)
def test_redact_mapping_scrubs_squashed_secret_keys_without_scrubbing_controls(
    key: str, expected: str
) -> None:
    r = Redactor(
        RedactConfig(level=RedactionLevel.STANDARD, always_scrub_in_logs=DEFAULT_ALWAYS_SCRUB)
    )

    out = r.redact_mapping({key: "visible-value"})

    assert out[key] == expected


def test_redact_mapping_preserve_fields() -> None:
    r = Redactor(
        RedactConfig(
            level=RedactionLevel.STANDARD,
            preserve_fields={"book_rag": ["author_name"]},
            always_scrub_in_logs=DEFAULT_ALWAYS_SCRUB,
        )
    )
    payload = {"author_name": "Ayrton Senna", "user_email": "x@y.com"}
    kept = r.redact_mapping(payload, pipeline="book_rag")
    assert kept["author_name"] == "Ayrton Senna"
    assert kept["user_email"] == "<EMAIL_ADDRESS>"

    # Different pipeline → no preservation.
    scrubbed = r.redact_mapping(payload, pipeline="article_extraction")
    # Value itself doesn't match PII patterns, so it passes through.
    assert scrubbed["user_email"] == "<EMAIL_ADDRESS>"


# ------------------------------------------------------------------
# Env + singleton
# ------------------------------------------------------------------


def test_env_level_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VORAG_REDACTION_LEVEL", "off")
    reset_default_redactor()
    assert redact("maintainer@example.com") == "maintainer@example.com"


def test_env_unknown_level_falls_back_to_standard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VORAG_REDACTION_LEVEL", "gibberish")
    reset_default_redactor()
    assert redact("maintainer@example.com") == "<EMAIL_ADDRESS>"
