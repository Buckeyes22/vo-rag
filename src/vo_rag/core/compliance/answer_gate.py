"""Contract for the compliance gate that guards publication and partner answers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from vo_rag.core.retrieval.query_mode import QueryMode


@dataclass(frozen=True)
class AnswerGateDecision:
    """Outcome of one gate check: whether the answer may proceed, and why."""

    allowed: bool
    reason: str


@runtime_checkable
class AnswerComplianceGate(Protocol):
    """Policy object consulted per request before synthesis in public answer modes."""

    def check(self, *, mode: QueryMode, hits: Sequence[Any], caller: str) -> AnswerGateDecision: ...


__all__ = ["AnswerComplianceGate", "AnswerGateDecision"]
