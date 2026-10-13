"""Tests for the per-(run, video, stage) checkpoint blob."""

from __future__ import annotations

import pytest

from tests.pipelines.vce.fakes import FakeGarage
from vo_rag.pipelines.vce.checkpoint import CHECKPOINT_PREFIX, VceCheckpoint


@pytest.fixture
def fake_garage() -> FakeGarage:
    return FakeGarage()


class TestCheckpoint:
    async def test_record_and_load_round_trip(self, fake_garage: FakeGarage) -> None:
        ck = VceCheckpoint(fake_garage, "vorag-vce-derived", run_id="run1")
        await ck.record("yt:abc", "transcribe", {"blob_key": "transcripts/yt:abc.transcript.json"})
        loaded = await ck.load("yt:abc", "transcribe")
        assert loaded is not None
        assert loaded["blob_key"] == "transcripts/yt:abc.transcript.json"
        assert loaded["video_id"] == "yt:abc"
        assert loaded["stage"] == "transcribe"

    async def test_load_returns_none_when_absent(self, fake_garage: FakeGarage) -> None:
        ck = VceCheckpoint(fake_garage, "vorag-vce-derived", run_id="run1")
        loaded = await ck.load("yt:abc", "transcribe")
        assert loaded is None

    async def test_finalize_deletes_every_blob(self, fake_garage: FakeGarage) -> None:
        ck = VceCheckpoint(fake_garage, "vorag-vce-derived", run_id="run1")
        await ck.record("yt:abc", "transcribe", {"blob_key": "k1"})
        await ck.record("yt:abc", "frame_sample", {"frame_count": 0})
        await ck.record("yt:xyz", "transcribe", {"blob_key": "k2"})

        # Prefix is namespaced by run_id.
        expected_prefix = f"{CHECKPOINT_PREFIX}/run1"
        assert len(await fake_garage.list_objects("vorag-vce-derived", expected_prefix)) == 3

        await ck.finalize()
        assert await fake_garage.list_objects("vorag-vce-derived", expected_prefix) == []

    async def test_finalize_on_empty_run_is_safe(self, fake_garage: FakeGarage) -> None:
        ck = VceCheckpoint(fake_garage, "vorag-vce-derived", run_id="empty")
        # No records — finalize must not raise.
        await ck.finalize()
        assert (
            await fake_garage.list_objects("vorag-vce-derived", f"{CHECKPOINT_PREFIX}/empty") == []
        )

    async def test_runs_are_isolated(self, fake_garage: FakeGarage) -> None:
        """Two checkpoints with different run_ids do not see each other's blobs."""
        ck1 = VceCheckpoint(fake_garage, "vorag-vce-derived", run_id="run1")
        ck2 = VceCheckpoint(fake_garage, "vorag-vce-derived", run_id="run2")
        await ck1.record("yt:abc", "transcribe", {"blob_key": "k1"})
        ck2_loaded = await ck2.load("yt:abc", "transcribe")
        assert ck2_loaded is None
        ck1_loaded = await ck1.load("yt:abc", "transcribe")
        assert ck1_loaded is not None
        assert ck1_loaded["blob_key"] == "k1"
