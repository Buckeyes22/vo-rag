"""Tests for the static workload→location router."""

from __future__ import annotations

import pytest

from vo_rag.core.cloud.router import (
    WorkloadLocation,
    workload_location,
)


def test_text_gen_is_runpod() -> None:
    assert workload_location("text-gen") == WorkloadLocation.RUNPOD_SERVERLESS


def test_vlm_is_runpod() -> None:
    assert workload_location("vlm") == WorkloadLocation.RUNPOD_SERVERLESS


def test_dsi_is_runpod() -> None:
    assert workload_location("dsi") == WorkloadLocation.RUNPOD_SERVERLESS


def test_embedding_is_local() -> None:
    assert workload_location("embedding") == WorkloadLocation.LOCAL


def test_query_handling_is_local() -> None:
    assert workload_location("query-handling") == WorkloadLocation.LOCAL


def test_unknown_workload_raises() -> None:
    with pytest.raises(ValueError, match="unknown workload"):
        workload_location("unknown-workload")


def test_workload_location_enum_members() -> None:
    assert WorkloadLocation.RUNPOD_SERVERLESS.value == "runpod_serverless"
    assert WorkloadLocation.LOCAL.value == "local"
