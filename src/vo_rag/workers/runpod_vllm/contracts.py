"""Workload→serverless contracts for the RunPod vLLM worker surface.

This module is the single seam between a workload name (``text-gen`` /
``vlm`` / ``dsi``) and a configured ``ServerlessClient`` ready to call
``/v1/chat/completions``. It does no I/O of its own — the
``ServerlessClient`` returned by ``client_for_workload`` owns the
network. The only HTTP path in the entire worker surface is the one in
``vo_rag.core.cloud.serverless``.

Functions:
    role_for_workload(workload) -> ServerlessRole
    client_for_workload(workload, settings=None) -> ServerlessClient
    build_chat_payload(request) -> dict[str, Any]

Why a separate module? ``schemas.py`` is the wire format (pure data);
``contracts.py`` is the *behavior* (how a workload becomes a client and a
request becomes an OpenAI payload). Splitting them keeps the data
shapes importable in places that don't want a network stack dependency.
"""

from __future__ import annotations

from typing import Any, Protocol

from vo_rag.core.cloud.serverless import ServerlessClient
from vo_rag.core.config.settings import ServerlessSettings, get_settings
from vo_rag.workers.runpod_vllm.schemas import (
    ServerlessRole,
    ServerlessWorkloadRequest,
)

# --------------------------------------------------------------------------- #
# Workload→role mapping
# --------------------------------------------------------------------------- #

# Explicit, not derived: the role is the *RunPod endpoint slot*, the
# workload is the *semantic job class*. They are 1:1 today but the
# mapping is owned here so future divergence (e.g. dsi traffic routed to
# a dedicated "dsi-pro" slot) is a one-line change.
_WORKLOAD_TO_ROLE: dict[str, str] = {
    "text-gen": "text",
    "vlm": "vlm",
    "dsi": "dsi",
}


def role_for_workload(workload: str) -> ServerlessRole:
    """Map a workload name to its RunPod serverless role.

    Raises:
        ValueError: workload is not one of the three supported workloads.
    """
    role = _WORKLOAD_TO_ROLE.get(workload)
    if role is None:
        raise ValueError(
            f"unknown workload: {workload!r} (expected one of {sorted(_WORKLOAD_TO_ROLE)})"
        )
    return role  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Client factory
# --------------------------------------------------------------------------- #


class _ServerlessSettingsLike(Protocol):
    """Protocol for the subset of ServerlessSettings we read.

    Lets the factory accept either a real ServerlessSettings or a test
    double (SimpleNamespace from the contract tests) without coupling
    the import surface to the live settings cache.
    """

    api_key: str
    request_timeout_s: int

    def base_url(self, role: str) -> str: ...
    def model_name(self, role: str) -> str: ...


def _resolve_settings(
    settings: _ServerlessSettingsLike | ServerlessSettings | None,
) -> _ServerlessSettingsLike | ServerlessSettings:
    """Pick the settings source. None → live get_settings() singleton."""
    if settings is None:
        return get_settings().serverless
    return settings


def client_for_workload(
    workload: str,
    settings: _ServerlessSettingsLike | ServerlessSettings | None = None,
) -> ServerlessClient:
    """Build a ``ServerlessClient`` for *workload*.

    Wires ``base_url(role)`` / ``api_key`` / ``model_name(role)`` /
    ``request_timeout_s`` from the typed settings tree. Never reads
    ``os.environ`` directly and never opens a socket — the returned
    client is configured but has not yet made any request.

    Args:
        workload: one of ``text-gen`` / ``vlm`` / ``dsi``.
        settings: optional ``ServerlessSettings`` (or test double) to
            override the live ``get_settings()`` singleton. Production
            code should pass ``None`` so the singleton wins.

    Returns:
        A ``ServerlessClient`` ready to call ``.chat_completion(...)``.
    """
    role = role_for_workload(workload)
    s = _resolve_settings(settings)
    return ServerlessClient(
        base_url=s.base_url(role),
        api_key=s.api_key,
        model=s.model_name(role),
        timeout_s=s.request_timeout_s,
    )


# --------------------------------------------------------------------------- #
# Payload builder
# --------------------------------------------------------------------------- #


def build_chat_payload(request: ServerlessWorkloadRequest) -> dict[str, Any]:
    """Turn a ``ServerlessWorkloadRequest`` into an OpenAI-shaped JSON dict.

    The ``model`` key is intentionally left empty here — it is filled in
    by ``ServerlessClient`` from its own configured model name. This keeps
    a single source of truth for "which model is this endpoint serving"
    and prevents a caller from sending the wrong model name to the wrong
    endpoint by accident.
    """
    payload: dict[str, Any] = {
        "model": "",
        "messages": [m.model_dump(mode="json") for m in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    if request.extra:
        payload.update(request.extra)
    return payload


__all__ = [
    "role_for_workload",
    "client_for_workload",
    "build_chat_payload",
    "ServerlessClient",
]
