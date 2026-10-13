"""Pydantic wire types for the RunPod serverless vLLM worker surface.

These are pure data shapes — they do no I/O, never call RunPod, and never
import any HTTP client. The transport lives in
``vo_rag.core.cloud.serverless``; this module exists so the rest of
vo-rag can refer to a stable wire format without dragging the network
stack into call sites.

The shape mirrors the OpenAI ``/v1/chat/completions`` request and
response, narrowed to the fields we actually use. ``extra="forbid"`` on
every model means a typo at the boundary surfaces as a Pydantic error
instead of a silent extra param being sent to a GPU worker.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from vo_rag.core.models.common import VoBaseModel

# --------------------------------------------------------------------------- #
# Workload and role enums (Literals — static + exhaustively checkable)
# --------------------------------------------------------------------------- #

#: The three RunPod serverless workloads.
#: ``text-gen`` / ``vlm`` / ``dsi`` — see ``vo_rag.core.cloud.router``.
RunpodWorkload = Literal["text-gen", "vlm", "dsi"]

#: The corresponding OpenAI-compatible serverless roles, one per workload.
#: The legacy ``exp`` slot still exists in ``ServerlessSettings`` for
#: backward-compat but is not a workload.
ServerlessRole = Literal["text", "vlm", "dsi"]

#: Workloads allowed to carry multimodal image parts.
#: Anything not in this set is text-only — the validator rejects
#: ``image_url`` content parts for text-only workloads so we never
#: accidentally bill a text-gen run for a vision forward pass.
_VLM_WORKLOADS: frozenset[str] = frozenset({"vlm"})


# --------------------------------------------------------------------------- #
# OpenAI content parts
# --------------------------------------------------------------------------- #


class ChatContentPart(VoBaseModel):
    """A single OpenAI multimodal content part.

    Supports the two shapes vo-rag actually sends:
    - ``{"type": "text", "text": "..."}``
    - ``{"type": "image_url", "image_url": {"url": "data:..."}}``

    Mirrors the OpenAI multimodal spec without taking on the full union —
    the runpod-vllm worker only needs text and image_url.
    """

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _text_and_image_url_match_type(self) -> ChatContentPart:
        if self.type == "text" and not self.text:
            raise ValueError("text content part requires non-empty 'text'")
        if self.type == "image_url":
            if not self.image_url or not isinstance(self.image_url, dict):
                raise ValueError("image_url content part requires 'image_url' object")
            url = self.image_url.get("url")
            if not isinstance(url, str) or not url:
                raise ValueError("image_url content part requires a non-empty 'url'")
        return self


class ChatMessage(VoBaseModel):
    """A single chat message in OpenAI format.

    ``content`` accepts either a plain string (text-only workloads) or a
    list of ``ChatContentPart`` (multimodal — vlm only). Pydantic
    coerces plain ``dict`` items at the boundary, so callers may build
    messages from raw JSON without going through ``ChatContentPart``
    explicitly.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ChatContentPart | dict[str, Any]]


# --------------------------------------------------------------------------- #
# Workload request / result
# --------------------------------------------------------------------------- #


class ServerlessWorkloadRequest(VoBaseModel):
    """A request to one of the RunPod serverless workloads.

    The fields are the OpenAI chat-completions surface, narrowed:
    - ``model`` is owned by ``ServerlessClient`` (it knows the role's
      configured model from settings), so it is intentionally NOT a field
      here. Callers pass ``client_for_workload(workload)`` and the model
      name flows from ``ServerlessSettings.model_name(role)``.
    - ``extra`` is a typed-dict escape hatch for OpenAI fields we don't
      model directly (top_p, frequency_penalty, presence_penalty, etc.).
      The harness and the production client both pass ``extra`` through
      unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    workload: RunpodWorkload
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(ge=1, le=32_768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    extra: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _vlm_only_image_url(self) -> ServerlessWorkloadRequest:
        if self.workload in _VLM_WORKLOADS:
            return self
        for msg in self.messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    part_type: str | None = None
                    if isinstance(part, ChatContentPart):
                        part_type = part.type
                    elif isinstance(part, dict):
                        raw = part.get("type")
                        if isinstance(raw, str):
                            part_type = raw
                    if part_type == "image_url":
                        raise ValueError(
                            f"workload={self.workload!r} does not allow image_url "
                            "content parts (text-only)"
                        )
        return self


class ServerlessWorkloadResult(VoBaseModel):
    """A typed result envelope for a RunPod serverless workload call.

    Mirrors the fields the rest of vo-rag actually consumes from
    ``ServerlessResponse``. Keeping a separate Pydantic shape means call
    sites that only see the workload layer don't need to import the
    transport types.
    """

    model_config = ConfigDict(extra="forbid")

    workload: RunpodWorkload
    content: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    finish_reason: str
    duration_ms: int = Field(ge=0)


__all__ = [
    "RunpodWorkload",
    "ServerlessRole",
    "ChatContentPart",
    "ChatMessage",
    "ServerlessWorkloadRequest",
    "ServerlessWorkloadResult",
]
