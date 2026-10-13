"""PII redaction.

Irreversible replacement at egress. Scope:
- EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN, IP_ADDRESS, API_KEY
  detected via regex — cover the primary leak vectors.
- PERSON detection is Presidio-backed when available; on systems without
  Presidio + spaCy the level silently falls back to the regex set.

Public surface:
- `Redactor(level)` — `redact(text)` + `redact_mapping(data)`.
- `redact(text, level=...)` — module-level convenience.

The env override is `VORAG_REDACTION_LEVEL` (off|standard|strict).
"""

from __future__ import annotations

import dataclasses
import os
import re
from enum import StrEnum
from typing import Any, Final

from vo_rag.core.models import VoBaseModel


class RedactionLevel(StrEnum):
    OFF = "off"
    STANDARD = "standard"
    STRICT = "strict"


class RedactionEntity(StrEnum):
    PERSON = "PERSON"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"
    CREDIT_CARD = "CREDIT_CARD"
    US_SSN = "US_SSN"
    IP_ADDRESS = "IP_ADDRESS"
    API_KEY = "API_KEY"  # pragma: allowlist secret


# --- regex library ---------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Phone formats. Separators (space, dash, dot, or parens-surrounded area)
# are REQUIRED so pure digit runs don't false-positive against CC numbers.
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?\d{1,3}[\s.\-])?"  # optional country code
    r"\(?\d{2,4}\)?[\s.\-]"  # area (2-4 digits, optional parens)
    r"\d{3,4}[\s.\-]"  # exchange
    r"\d{4}(?!\d)"  # subscriber, not followed by more digits
)

_CREDIT_CARD_RE = re.compile(r"(?:\d[ \-]?){12,18}\d")

_US_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")

_IPV4_RE = re.compile(
    r"(?<!\d)(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}(?!\d)"
)

_API_KEY_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{40,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"ghs_[A-Za-z0-9]{20,}"),
    re.compile(r"glpat-[A-Za-z0-9_\-]{20,}"),
]


class RedactConfig(VoBaseModel):
    level: RedactionLevel = RedactionLevel.STANDARD
    preserve_fields: dict[str, list[str]] = {}
    always_scrub_in_logs: list[str] = []
    entities_by_level: dict[RedactionLevel, list[RedactionEntity]] = {}


DEFAULT_ENTITIES_BY_LEVEL: dict[RedactionLevel, list[RedactionEntity]] = {
    RedactionLevel.OFF: [],
    RedactionLevel.STANDARD: [
        RedactionEntity.EMAIL_ADDRESS,
        RedactionEntity.PHONE_NUMBER,
        RedactionEntity.CREDIT_CARD,
        RedactionEntity.US_SSN,
        RedactionEntity.API_KEY,
    ],
    RedactionLevel.STRICT: [
        RedactionEntity.PERSON,
        RedactionEntity.EMAIL_ADDRESS,
        RedactionEntity.PHONE_NUMBER,
        RedactionEntity.CREDIT_CARD,
        RedactionEntity.US_SSN,
        RedactionEntity.IP_ADDRESS,
        RedactionEntity.API_KEY,
    ],
}

DEFAULT_ALWAYS_SCRUB: list[str] = ["api_key", "secret", "token", "authorization", "password"]


class Redactor:
    def __init__(self, config: RedactConfig | None = None) -> None:
        self.config = config or RedactConfig()
        self._entities = set(
            self.config.entities_by_level.get(self.config.level)
            or DEFAULT_ENTITIES_BY_LEVEL[self.config.level]
        )

    def redact(self, text: str) -> str:
        if not text or self.config.level is RedactionLevel.OFF:
            return text
        result = text
        if RedactionEntity.EMAIL_ADDRESS in self._entities:
            result = _EMAIL_RE.sub("<EMAIL_ADDRESS>", result)
        if RedactionEntity.US_SSN in self._entities:
            result = _US_SSN_RE.sub("<US_SSN>", result)
        if RedactionEntity.CREDIT_CARD in self._entities:
            result = _redact_credit_cards(result)
        if RedactionEntity.PHONE_NUMBER in self._entities:
            result = _PHONE_RE.sub("<PHONE_NUMBER>", result)
        if RedactionEntity.IP_ADDRESS in self._entities:
            result = _IPV4_RE.sub("<IP_ADDRESS>", result)
        if RedactionEntity.API_KEY in self._entities:
            for pat in _API_KEY_PATTERNS:
                result = pat.sub("<API_KEY>", result)
        if RedactionEntity.PERSON in self._entities:
            result = _maybe_redact_person(result)
        return result

    def redact_mapping(
        self, data: dict[str, Any], *, pipeline: str | None = None
    ) -> dict[str, Any]:
        """Recursively scrub a dict. String values pass through `redact`.
        Keys in `always_scrub_in_logs` get their values replaced with
        `<REDACTED>` unconditionally."""
        always_scrub = {_normalise_key(k) for k in self.config.always_scrub_in_logs}
        preserved = (
            {k.lower() for k in self.config.preserve_fields.get(pipeline or "", [])}
            if pipeline
            else set()
        )
        result: dict[str, Any] = _walk(data, self, always_scrub, preserved)
        return result


# --- helpers ---------------------------------------------------------------


def _walk(value: Any, r: Redactor, always_scrub: set[str], preserved: set[str]) -> Any:
    coerced = _to_redactable(value)
    if coerced is not value:
        return _walk(coerced, r, always_scrub, preserved)

    if isinstance(value, dict):
        return {
            k: (
                "<REDACTED>"
                if isinstance(k, str) and _should_always_scrub_key(k, always_scrub)
                else (
                    v
                    if (isinstance(k, str) and k.lower() in preserved)
                    else _walk(v, r, always_scrub, preserved)
                )
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_walk(item, r, always_scrub, preserved) for item in value]
    if isinstance(value, tuple):
        return [_walk(item, r, always_scrub, preserved) for item in value]
    if isinstance(value, str):
        return r.redact(value)
    if isinstance(value, (int, float, bool, type(None))):
        return value
    return r.redact(str(value))


def _normalise_key(key: str) -> str:
    with_boundaries = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    # Acronym runs split before their trailing word: APIKey -> API_Key,
    # DBPassword -> DB_Password, HTTPAuthorization -> HTTP_Authorization.
    with_boundaries = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", with_boundaries)
    return re.sub(r"[^a-z0-9]+", "_", with_boundaries.lower()).strip("_")


def _should_always_scrub_key(key: str, always_scrub: set[str]) -> bool:
    normalised = _normalise_key(key)
    if normalised in always_scrub:
        return True

    # Squashed comparison is suffix-anchored, not substring containment:
    # "clientsecret" ends with "secret" (scrub), but "maxtokens"/"tokenizer"
    # do not end with "token" (telemetry counts stay visible).
    squashed = normalised.replace("_", "")
    for scrub_key in always_scrub:
        if squashed.endswith(scrub_key.replace("_", "")):
            return True

    parts = normalised.split("_")
    segments = set(parts)
    for scrub_key in always_scrub:
        scrub_segments = scrub_key.split("_")
        if len(scrub_segments) == 1:
            if scrub_key in segments:
                return True
            continue

        for index in range(0, len(parts) - len(scrub_segments) + 1):
            if parts[index : index + len(scrub_segments)] == scrub_segments:
                return True
    return False


def _to_redactable(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(mode="json")
    if not isinstance(value, type) and dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    return value


def _luhn_valid(digits: str) -> bool:
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        try:
            d = int(ch)
        except ValueError:
            return False
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_credit_cards(text: str) -> str:
    """Only replace runs that pass Luhn — avoids zapping random digit sequences."""

    def _sub(match: re.Match[str]) -> str:
        span = match.group(0)
        digits = re.sub(r"[ \-]", "", span)
        if _luhn_valid(digits):
            return "<CREDIT_CARD>"
        return span

    return _CREDIT_CARD_RE.sub(_sub, text)


def _maybe_redact_person(text: str) -> str:
    """Use Presidio when available. Presidio is an optional dependency."""
    import importlib.util

    if importlib.util.find_spec("presidio_analyzer") is None:
        return text
    try:
        engine = _get_presidio_engine()
        results = engine.analyze(text=text, entities=["PERSON"], language="en")
    except Exception:
        return text
    # Replace from the end to preserve offsets.
    for res in sorted(results, key=lambda r: r.start, reverse=True):
        text = text[: res.start] + "<PERSON>" + text[res.end :]
    return text


_PRESIDIO_ENGINE = None


def _get_presidio_engine() -> Any:
    global _PRESIDIO_ENGINE
    if _PRESIDIO_ENGINE is None:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]

        _PRESIDIO_ENGINE = AnalyzerEngine()
    return _PRESIDIO_ENGINE


# --- convenience + singleton --------------------------------------------


_DEFAULT_REDACTOR: Redactor | None = None


def _env_level() -> RedactionLevel:
    raw = os.environ.get("VORAG_REDACTION_LEVEL", "standard").strip().lower()
    try:
        return RedactionLevel(raw)
    except ValueError:
        return RedactionLevel.STANDARD


def get_default_redactor() -> Redactor:
    global _DEFAULT_REDACTOR
    if _DEFAULT_REDACTOR is None:
        _DEFAULT_REDACTOR = Redactor(
            RedactConfig(
                level=_env_level(),
                always_scrub_in_logs=list(DEFAULT_ALWAYS_SCRUB),
            )
        )
    return _DEFAULT_REDACTOR


def reset_default_redactor() -> None:
    """For tests + SIGHUP-style reload."""
    global _DEFAULT_REDACTOR
    _DEFAULT_REDACTOR = None


def redact(text: str, *, level: RedactionLevel | None = None) -> str:
    if level is not None:
        return Redactor(RedactConfig(level=level)).redact(text)
    return get_default_redactor().redact(text)


__all__ = [
    "DEFAULT_ALWAYS_SCRUB",
    "DEFAULT_ENTITIES_BY_LEVEL",
    "RedactConfig",
    "RedactionEntity",
    "RedactionLevel",
    "Redactor",
    "get_default_redactor",
    "redact",
    "reset_default_redactor",
]
