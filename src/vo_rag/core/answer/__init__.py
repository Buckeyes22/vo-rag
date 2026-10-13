"""answer: grounded answer synthesis over retrieved evidence.

Public surface (re-exported for callers — no import gymnastics):

- :class:`AnswerSynthesizer` — protected completion over retrieved chunks.
- :class:`AnswerProfile` / :class:`ContextScope` — enums for the route layer.
- :class:`AnswerDraft` / :class:`AnswerSynthesis` — return shapes.
- :class:`AnswerConfigError` — raised when a profile is misconfigured.
- :func:`format_answer_context` / :func:`resolve_answer_collections` /
  :func:`parent_ids_for_results` / :func:`source_ids` — pure helpers
  used by the route layer to assemble inputs and persist outputs.
"""

from vo_rag.core.answer.context import (
    format_answer_context,
    parent_ids_for_results,
    resolve_answer_collections,
    source_ids,
)
from vo_rag.core.answer.synthesizer import (
    AnswerConfigError,
    AnswerDraft,
    AnswerProfile,
    AnswerSynthesis,
    AnswerSynthesizer,
    ContextScope,
)

__all__ = [
    "AnswerConfigError",
    "AnswerDraft",
    "AnswerProfile",
    "AnswerSynthesis",
    "AnswerSynthesizer",
    "ContextScope",
    "format_answer_context",
    "parent_ids_for_results",
    "resolve_answer_collections",
    "source_ids",
]
