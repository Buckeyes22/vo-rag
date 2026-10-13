"""Error type + redaction helpers for the Python MCP surface.

Tool errors raised in the MCP layer are translated into MCP
``isError: True`` responses. The error message is rendered to the
client verbatim, so it must not leak:

- API keys, Garage secrets, webhook tokens (substring match against
  the same fragments the API surface redacts in
  ``vo_rag.api.utils.mask_secrets``).
- Stack traces (they reveal filesystem paths and internal
  layout).
- Raw settings payloads (the ``Settings`` object has every API
  key + DB URL + Tailscale OAuth secret on it).

The redactor wraps every tool error in a :class:`ToolError` whose
``str()`` is safe to send to the client. The original exception is
still preserved as ``__cause__`` for in-process logging.
"""

from __future__ import annotations

import re
import traceback
from typing import Any

# Field name fragments the API surface redacts in status / health
# output. We use the same allowlist so the two surfaces cannot drift.
_SECRET_FIELD_FRAGMENTS: tuple[str, ...] = (
    "api_key",
    "secret_key",
    "access_key",
    "anthropic_api_key",
    "kimi_api_key",
    "webhook_secret",
    "csrf_secret",
    "password",
    "token",
)

_MASKED = "***REDACTED***"

# A conservatively broad credential pattern. The intent is to
# replace anything that *looks* like a secret with the redacted
# token, even at the cost of a false positive (a chat-friendly
# error is worth more than a verbatim credential echo).
_CREDENTIAL_PATTERN = re.compile(
    r"""
    (?:sk|pk|ak)_live_[A-Za-z0-9]+              # generic live secret
    | sk-[A-Za-z0-9]{8,}                        # Anthropic / OpenAI key
    | AKIA[0-9A-Z]{16}                          # AWS access key
    | xox[bpars]-[A-Za-z0-9-]{8,}               # Slack
    | ghp_[A-Za-z0-9]{20,}                      # GitHub PAT
    """,
    re.VERBOSE,
)

_STACK_TRACE_PATTERN = re.compile(r"Traceback \(most recent call last\):.*?(?=\n\S|\Z)", re.DOTALL)
_FILE_PATH_PATTERN = re.compile(r"(/[A-Za-z0-9_./-]+\.py)(?::\d+)?")
_URL_WITH_CREDS_PATTERN = re.compile(r"://[^\s/@:]+:[^\s/@]+@")


def redact_error_message(message: str) -> str:
    """Return a chat-friendly copy of ``message`` with secrets +
    stack traces + credentialed URLs + filesystem paths scrubbed.
    """
    if not message:
        return message

    text = str(message)

    # 1. drop any embedded stack trace
    text = _STACK_TRACE_PATTERN.sub(_MASKED, text)

    # 2. strip credentials in URLs (user:pass@host)
    text = _URL_WITH_CREDS_PATTERN.sub("://" + _MASKED + "@", text)

    # 3. mask well-formed credential tokens
    text = _CREDENTIAL_PATTERN.sub(_MASKED, text)

    # 4. mask ``field=value`` pairs where field looks like a secret
    def _redact_kv(match: re.Match[str]) -> str:
        key = match.group(1)
        if any(fragment in key.lower() for fragment in _SECRET_FIELD_FRAGMENTS):
            return f"{key}={_MASKED}"
        return match.group(0)

    secret_kv_re = re.compile(
        r"([A-Za-z_][A-Za-z0-9_.\-]*"
        r"(?:api_key|secret_key|access_key|token|password|webhook_secret|csrf_secret))"
        r"\s*=\s*([^\s,;\"'}{]+)"
    )
    text = secret_kv_re.sub(_redact_kv, text)

    # 5. filesystem paths — leave the basename only when it would be
    # useful for the operator, but in this redactor we prefer total
    # silence over partial leakage.
    text = _FILE_PATH_PATTERN.sub(_MASKED, text)

    return text


class ToolError(Exception):
    """Raised by MCP tool implementations to signal a tool-level error.

    The MCP layer translates this into an ``isError: True``
    ``CallToolResult``; the message is run through
    :func:`redact_error_message` so the wire payload is safe to
    forward to the client.

    The original exception (if any) is preserved on ``__cause__``
    for in-process structlog capture — it never leaves the
    process.
    """

    def __init__(self, message: str, *, code: str = "tool_error") -> None:
        self.code = code
        self.original_message = message
        super().__init__(redact_error_message(message))

    def __str__(self) -> str:  # pragma: no cover - trivial
        return redact_error_message(self.original_message)


def format_exception_for_log(exc: BaseException) -> dict[str, Any]:
    """Return a structlog-friendly dict for an exception.

    The full traceback is included for in-process log capture (so
    on-call can debug). The redacted message is the one rendered
    to the client via :class:`ToolError`.
    """
    return {
        "exc_type": type(exc).__name__,
        "exc_message": redact_error_message(str(exc)),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


__all__ = [
    "ToolError",
    "format_exception_for_log",
    "redact_error_message",
]
