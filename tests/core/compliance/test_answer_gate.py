from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vo_rag.core.compliance.answer_gate import AnswerComplianceGate, AnswerGateDecision
from vo_rag.core.retrieval.query_mode import QueryMode


class _Gate:
    def check(self, *, mode: QueryMode, hits: Sequence[Any], caller: str) -> AnswerGateDecision:
        return AnswerGateDecision(allowed=True, reason="ok")


def test_decision_is_frozen_with_allowed_and_reason() -> None:
    decision = AnswerGateDecision(allowed=False, reason="license")
    assert (decision.allowed, decision.reason) == (False, "license")
    try:
        decision.allowed = True  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("AnswerGateDecision must be frozen")


def test_protocol_accepts_objects_with_a_check_method() -> None:
    assert isinstance(_Gate(), AnswerComplianceGate)


def test_protocol_rejects_objects_without_a_callable_check() -> None:
    assert not isinstance(object(), AnswerComplianceGate)
