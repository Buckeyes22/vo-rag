from __future__ import annotations

from typing import Any, cast, get_args

import pytest
from pydantic import ValidationError

from vo_rag.pipelines.book_rag.events import (
    INGEST_STEP_ORDER,
    ImageDescribedEvent,
    IngestProgress,
    IngestStep,
)


def test_ingest_progress_step_literal_matches_step_order() -> None:
    annotation = cast(object, IngestProgress.model_fields["step"].annotation)
    assert annotation is IngestStep
    literal_steps = list(get_args(IngestStep.__value__))

    assert literal_steps == INGEST_STEP_ORDER
    for step in INGEST_STEP_ORDER:
        assert IngestProgress(envelope_id="env-1", step=step).step == step

    with pytest.raises(ValidationError):
        IngestProgress(envelope_id="env-1", step=cast(Any, "not-a-step"))


@pytest.mark.parametrize("pct", [-0.01, 1.01])
def test_ingest_progress_pct_is_bounded_to_fraction(pct: float) -> None:
    with pytest.raises(ValidationError):
        IngestProgress(envelope_id="env-1", step="parse", pct=pct)


@pytest.mark.parametrize(
    ("event_cls", "kwargs"),
    [
        (IngestProgress, {"envelope_id": "env-1", "step": "parse"}),
        (ImageDescribedEvent, {"envelope_id": "env-1", "image_index": 0, "total": 1}),
    ],
)
def test_book_rag_events_reject_undeclared_kwargs(
    event_cls: type[IngestProgress | ImageDescribedEvent],
    kwargs: dict[str, Any],
) -> None:
    params = dict(kwargs, undeclared="extra")
    with pytest.raises(ValidationError):
        event_cls(**params)
