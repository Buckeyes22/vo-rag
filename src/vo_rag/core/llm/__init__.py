"""LLM client primitives — façade, providers, retry, structured output,
prompt registry, and the injection-defense trio (F-012).
"""

from vo_rag.core.llm.client import LLMClient
from vo_rag.core.llm.errors import (
    LLMBadRequestError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    StructuredOutputError,
)
from vo_rag.core.llm.output_validator import (
    InjectionSignal,
    SignalKind,
    ValidationResult,
    validate_output,
)
from vo_rag.core.llm.prompts import (
    Prompt,
    PromptError,
    PromptNotFound,
    PromptParseError,
    PromptRegistry,
    PromptRenderError,
)
from vo_rag.core.llm.protected import complete_json_protected
from vo_rag.core.llm.response import LLMResponse
from vo_rag.core.llm.untrusted import (
    UntrustedContentRejected,
    build_system_suffix,
    detect_canary_leakage,
    generate_canary,
    wrap_untrusted,
)

__all__ = [
    "LLMBadRequestError",
    "LLMClient",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMServerError",
    "Prompt",
    "PromptError",
    "PromptNotFound",
    "PromptParseError",
    "PromptRegistry",
    "PromptRenderError",
    "StructuredOutputError",
    "complete_json_protected",
    "validate_output",
    "InjectionSignal",
    "SignalKind",
    "ValidationResult",
    "wrap_untrusted",
    "build_system_suffix",
    "generate_canary",
    "detect_canary_leakage",
    "UntrustedContentRejected",
]
