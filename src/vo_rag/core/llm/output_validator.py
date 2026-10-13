"""Post-hoc validator for LLM responses that consumed untrusted content.

Four signals:
  canary_echoed            — response contains the call's canary token
  verbatim_untrusted_echo  — response contains a >= N-char span from the
                             untrusted input
  system_prompt_leaked     — response contains a >= M-char span from the
                             system prompt
  url_exfil_attempt        — response contains a URL whose domain is not
                             in the pipeline's allowlist

A signal with `confidence >= 0.8` upgrades the artifact's trust_level to
`quarantined` at the compliance gate boundary.
"""

from __future__ import annotations

import re
from typing import Literal

from vo_rag.core.models.common import VoBaseModel

DEFAULT_VERBATIM_MIN_LEN = 120
DEFAULT_SYSTEM_LEAK_MIN_LEN = 80

_URL_RE = re.compile(r"https?://([A-Za-z0-9\-._~:%]+)(?:/[^\s)]*)?", re.IGNORECASE)

SignalKind = Literal[
    "canary_echoed",
    "system_prompt_leaked",
    "verbatim_untrusted_echo",
    "url_exfil_attempt",
    "tool_call_not_whitelisted",
]


class InjectionSignal(VoBaseModel):
    kind: SignalKind
    confidence: float
    evidence: str


class ValidationResult(VoBaseModel):
    ok: bool
    signals: list[InjectionSignal] = []


def validate_output(
    response_text: str,
    *,
    untrusted_input: str,
    system_prompt: str,
    canary: str,
    allowed_domains: list[str] | None = None,
    verbatim_min_len: int = DEFAULT_VERBATIM_MIN_LEN,
    system_leak_min_len: int = DEFAULT_SYSTEM_LEAK_MIN_LEN,
) -> ValidationResult:
    signals: list[InjectionSignal] = []

    if canary and canary in response_text:
        signals.append(
            InjectionSignal(
                kind="canary_echoed",
                confidence=1.0,
                evidence=f"canary {canary!r} present in response",
            )
        )

    echoed = _find_long_shared_substring(response_text, untrusted_input, verbatim_min_len)
    if echoed is not None:
        signals.append(
            InjectionSignal(
                kind="verbatim_untrusted_echo",
                confidence=_confidence_from_length(len(echoed), verbatim_min_len),
                evidence=f"{len(echoed)}-char verbatim echo: {echoed[:80]!r}...",
            )
        )

    leaked = _find_long_shared_substring(response_text, system_prompt, system_leak_min_len)
    if leaked is not None:
        signals.append(
            InjectionSignal(
                kind="system_prompt_leaked",
                confidence=_confidence_from_length(len(leaked), system_leak_min_len),
                evidence=f"{len(leaked)}-char system-prompt span echoed: {leaked[:80]!r}...",
            )
        )

    if allowed_domains is not None:
        allowed_set = {d.lower() for d in allowed_domains}
        exfils = _detect_url_exfil(response_text, allowed_set)
        for host in exfils:
            signals.append(
                InjectionSignal(
                    kind="url_exfil_attempt",
                    confidence=0.9,
                    evidence=f"response references non-allowlisted domain {host!r}",
                )
            )

    return ValidationResult(ok=not signals, signals=signals)


def _confidence_from_length(length: int, min_len: int) -> float:
    """Length-scaled confidence. min_len → 0.8; 2x min_len → 0.95."""
    if length < min_len:
        return 0.0
    ratio = min(length / min_len, 2.0)
    return round(0.8 + 0.15 * (ratio - 1.0), 3)


def _find_long_shared_substring(haystack: str, needle: str, min_len: int) -> str | None:
    """Return the first substring of length >= min_len from `needle` that
    also appears in `haystack`, or None."""
    if len(needle) < min_len or not haystack:
        return None
    seen: set[str] = set()
    for i in range(len(needle) - min_len + 1):
        window = needle[i : i + min_len]
        if window in seen:
            continue
        seen.add(window)
        if window in haystack:
            return window
    return None


def _detect_url_exfil(response_text: str, allowed_domains: set[str]) -> list[str]:
    found: list[str] = []
    for match in _URL_RE.finditer(response_text):
        host = match.group(1).lower()
        host = host.split("@")[-1].split(":")[0]
        if not _domain_allowed(host, allowed_domains):
            found.append(host)
    return found


def _domain_allowed(host: str, allowed: set[str]) -> bool:
    """Exact match or suffix match — 'example.com' allows 'sub.example.com'."""
    if host in allowed:
        return True
    return any(host.endswith("." + allow) for allow in allowed)


__all__ = [
    "InjectionSignal",
    "SignalKind",
    "ValidationResult",
    "validate_output",
]
