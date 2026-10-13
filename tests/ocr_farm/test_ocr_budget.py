"""OCR-farm per-book budget (VOR-1427).

Every worker run loads the docling/RapidOCR models (40-60 s on lin) before its
first page. On 2026-09-26 the 31-page "Engineering Design Guides" (envelope
01M3AYY7T94WQECGHC2GVYDHCY) ran past its 310 s budget
(``max(300, 31 * 10)``) and was failed as a timeout. The budget is now
``ocr_startup_s + pages * ocr_s_per_page``, never below ``ocr_min_s``.
"""

from __future__ import annotations

from typing import Any

import pytest

import vo_rag.ocr_farm.coordinator as coordinator
from tests.ocr_farm._farm_fakes import (
    BUCKET,
    FakeGarage,
    FakePool,
    Recorder,
    fake_merge,
    pdf_item,
    resolved_hosts,
)
from vo_rag.ocr_farm.coordinator import FarmTimeouts, run_farm
from vo_rag.ocr_farm.hosts import DEFAULT_HOSTS, MAX_REMOTE_TIMEOUT_S

#: Largest scanned book in the backlog, pages.
LARGEST_BOOK_PAGES = 4178


def test_budget_defaults() -> None:
    timeouts = FarmTimeouts()
    assert timeouts.ocr_startup_s == 120.0  # 2x the 60 s model load seen on lin
    assert timeouts.ocr_s_per_page == 10.0
    assert timeouts.ocr_min_s == 300.0


@pytest.mark.parametrize(
    ("pages", "budget"),
    [(0, 300.0), (1, 300.0), (18, 300.0), (19, 310.0), (31, 430.0), (100, 1120.0)],
)
def test_budget_is_startup_plus_pages_with_a_floor(pages: int, budget: float) -> None:
    assert FarmTimeouts().ocr_s(pages) == budget


def test_budget_formula_uses_every_field() -> None:
    timeouts = FarmTimeouts(ocr_startup_s=50.0, ocr_s_per_page=2.0, ocr_min_s=10.0)
    assert timeouts.ocr_s(0) == 50.0
    assert timeouts.ocr_s(7) == 64.0
    assert FarmTimeouts(ocr_startup_s=0.0, ocr_s_per_page=0.0, ocr_min_s=0.3).ocr_s(500) == 0.3


def test_the_31_page_book_that_timed_out_now_has_headroom() -> None:
    """It ran past 311 s against a 310 s budget; model load alone was ~60 s."""
    observed_s = 311.0
    budget = FarmTimeouts().ocr_s(31)
    assert budget == 430.0
    assert budget - observed_s >= 100.0
    # Even with the slowest load seen (60 s), the pages may take 11.9 s each.
    assert (budget - 60.0) / 31 > 11.5


def test_largest_book_budget_is_past_the_remote_limit_and_still_valid() -> None:
    """4,178 pages get 41,900 s. That is over MAX_REMOTE_TIMEOUT_S (the
    Wait-Process limit), which is fine: the OCR command carries no remote
    limit, so validate_remote_timeout never sees this value."""
    budget = FarmTimeouts().ocr_s(LARGEST_BOOK_PAGES)
    assert budget == 120.0 + LARGEST_BOOK_PAGES * 10.0 == 41900.0
    assert budget > MAX_REMOTE_TIMEOUT_S
    for host in DEFAULT_HOSTS:
        command = host.worker_command("a.pdf", "a.json")
        assert "timeout " not in command
        assert "farm_step" not in command
        assert "-TimeoutSeconds" not in command


async def test_largest_book_runs_as_parts_each_with_its_own_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: the 4,178-page book is split into ``SPLIT_PAGES`` parts
    (VOR-1427: one worker holding the whole book outgrew win's commit
    limit), and each part's worker ssh gets ``ocr_s(part pages)`` -- the model
    load budgeted once per part -- as its local asyncio.wait_for bound."""
    seen: list[tuple[list[str], float]] = []
    real_call = coordinator._call

    async def spy(runner: Any, argv: list[str], timeout: float) -> Any:
        seen.append((argv, timeout))
        return await real_call(runner, argv, timeout)

    monkeypatch.setattr(coordinator, "_call", spy)
    item, objects = pdf_item(LARGEST_BOOK_PAGES, text=False)
    recorder = Recorder()
    hosts = await resolved_hosts(recorder, (DEFAULT_HOSTS[1],))

    report = await run_farm(
        [item],
        hosts,
        FakeGarage(objects),
        FakePool(),
        requeue=False,
        bucket=BUCKET,
        runner=recorder,
        merge=fake_merge,
    )

    assert report.results[0].status == "ok"
    worker_budgets = [
        t for argv, t in seen if argv[:2] == ["ssh", "lin"] and "ocr_worker.py jobs/" in argv[2]
    ]
    assert worker_budgets == [4120.0] * 10 + [120.0 + 178 * 10.0]
    assert max(worker_budgets) < MAX_REMOTE_TIMEOUT_S
