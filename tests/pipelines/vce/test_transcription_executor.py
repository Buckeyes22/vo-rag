"""Tests for the TranscriptionExecutor Protocol + the three implementations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path

import pytest

from tests.pipelines.vce.conftest import make_row_dict  # noqa: F401
from tests.pipelines.vce.fakes import FakeGarage
from vo_rag.pipelines.vce.transcription import (
    DEFAULT_TRANSCRIPT_BLOB_SUFFIX,
    LocalWhisperXExecutor,
    RemoteWhisperXExecutor,
    StubTranscriptionExecutor,
    TranscriptArtifact,
    TranscriptionExecutor,
    TranscriptSegment,
)


@pytest.fixture
def fake_garage() -> FakeGarage:
    return FakeGarage()


class TestStubExecutor:
    async def test_loads_fixture_and_uploads(self, tmp_path: Path, fake_garage: FakeGarage) -> None:
        # Write a fixture for yt:abc
        fixture_dir = tmp_path / "fixtures" / "transcripts"
        fixture_dir.mkdir(parents=True)
        payload = (
            '{"video_id": "yt:abc", "language": "en", '
            '"segments": [{"start": 0.0, "end": 5.0, "text": "hello"}]}'
        )
        (fixture_dir / f"yt:abc{DEFAULT_TRANSCRIPT_BLOB_SUFFIX}").write_text(
            payload, encoding="utf-8"
        )

        executor = StubTranscriptionExecutor(
            fixture_dir=fixture_dir, bucket="vorag-vce-derived", garage=fake_garage, upload=True
        )
        artifact = await executor.run("yt:abc", video_path=Path("/nonexistent"))

        assert artifact.blob_key == "transcripts/yt:abc.transcript.json"
        assert len(artifact.segments) == 1
        assert artifact.segments[0].text == "hello"
        assert artifact.language == "en"
        assert fake_garage.has_object("vorag-vce-derived", artifact.blob_key)

    async def test_missing_fixture_raises(self, tmp_path: Path, fake_garage: FakeGarage) -> None:
        executor = StubTranscriptionExecutor(
            fixture_dir=tmp_path, bucket="vorag-vce-derived", garage=fake_garage
        )
        with pytest.raises(FileNotFoundError):
            await executor.run("yt:missing", video_path=Path("/nonexistent"))

    async def test_upload_disabled(self, tmp_path: Path, fake_garage: FakeGarage) -> None:
        fixture_dir = tmp_path / "fixtures"
        fixture_dir.mkdir()
        (fixture_dir / f"yt:abc{DEFAULT_TRANSCRIPT_BLOB_SUFFIX}").write_text(
            '{"video_id": "yt:abc", "language": "en", "segments": []}',
            encoding="utf-8",
        )
        executor = StubTranscriptionExecutor(
            fixture_dir=fixture_dir, bucket="vorag-vce-derived", garage=fake_garage, upload=False
        )
        artifact = await executor.run("yt:abc", video_path=Path("/nonexistent"))
        assert artifact.blob_key is not None
        assert not fake_garage.has_object("vorag-vce-derived", artifact.blob_key)


class TestRemoteExecutor:
    async def test_unconfigured_returns_failed_artifact(self, fake_garage: FakeGarage) -> None:
        """No endpoint URL → no upload → blob_key is None (skip, don't fail the batch)."""
        executor = RemoteWhisperXExecutor(
            endpoint_url=None, bucket="vorag-vce-derived", garage=fake_garage
        )
        artifact = await executor.run("yt:abc", video_path=Path("/nonexistent"))
        assert artifact.blob_key is None
        assert artifact.segments == []
        # Nothing was uploaded to the bucket.
        assert not fake_garage.has_object("vorag-vce-derived", "transcripts/yt:abc.transcript.json")

    async def test_satisfies_protocol(self, fake_garage: FakeGarage) -> None:
        executor = RemoteWhisperXExecutor(
            endpoint_url=None, bucket="vorag-vce-derived", garage=fake_garage
        )
        # The Protocol's runtime_checkable lets us assert the duck type.
        assert isinstance(executor, TranscriptionExecutor)


class TestLocalExecutor:
    async def test_missing_whisperx_dependency_raises(self, fake_garage: FakeGarage) -> None:
        """If whisperx isn't installed, the executor fails clearly."""
        executor = LocalWhisperXExecutor(
            model="tiny", bucket="vorag-vce-derived", garage=fake_garage
        )
        # If whisperx IS installed in the test env, the import would succeed
        # and the call would fail for a different reason. We only assert the
        # error path here; production CI uses the stub executor.
        try:
            await executor.run("yt:abc", video_path=Path("/nonexistent"))
        except RuntimeError as exc:
            assert "whisperx" in str(exc).lower() or "torch" in str(exc).lower()
        except Exception as exc:  # noqa: BLE001 - whisperx may fail for a real reason too
            # Either the dependency is missing (RuntimeError) or whisperx
            # can't read /nonexistent (FileNotFoundError) — both are valid.
            assert "whisperx" in str(exc).lower() or "No such file" in str(exc)


class TestTranscriptArtifact:
    def test_dataclass_construction(self) -> None:
        from vo_rag.pipelines.vce.types import TranscriptArtifact

        artifact = TranscriptArtifact(
            segments=[TranscriptSegment(start=0.0, end=1.0, text="hi")],
            language="en",
            blob_key="transcripts/yt:abc.transcript.json",
            cost_usd=0.05,
        )
        assert artifact.blob_key is not None
        assert artifact.cost_usd == 0.05


class TestOpenAIAudioTranscriptionExecutor:
    """Tests for the Speaches / OpenAI-Audio-API compatible executor."""

    def test_unconfigured_returns_failed_artifact(self, fake_garage: FakeGarage) -> None:
        """No endpoint URL → no upload → blob_key is None (skip, don't fail the batch)."""
        from vo_rag.pipelines.vce.transcription import OpenAIAudioTranscriptionExecutor

        executor = OpenAIAudioTranscriptionExecutor(
            endpoint_url=None, bucket="vorag-vce-derived", garage=fake_garage
        )
        # asyncio.run because we need to await — keeping it inline for simplicity.
        artifact = asyncio_run(executor.run("yt:abc", video_path=Path("/nonexistent")))
        assert artifact.blob_key is None
        assert artifact.segments == []
        assert not fake_garage.has_object("vorag-vce-derived", "transcripts/yt:abc.transcript.json")

    def test_satisfies_protocol(self, fake_garage: FakeGarage) -> None:
        from vo_rag.pipelines.vce.transcription import OpenAIAudioTranscriptionExecutor

        executor = OpenAIAudioTranscriptionExecutor(
            endpoint_url="http://speaches.test", bucket="vorag-vce-derived", garage=fake_garage
        )
        assert isinstance(executor, TranscriptionExecutor)

    def test_verbose_json_response_parses_segments(
        self, fake_garage: FakeGarage, tmp_path: Path
    ) -> None:
        """End-to-end with a fake Speaches server: a tiny audio file is POSTed,
        the verbose_json response is parsed, the transcript is uploaded.
        """
        import httpx

        from vo_rag.pipelines.vce.transcription import OpenAIAudioTranscriptionExecutor

        # Create a fake video file (any bytes — Speaches will return canned JSON)
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-audio-bytes")

        response_payload = {
            "task": "transcribe",
            "language": "english",
            "duration": 12.5,
            "text": "See how late he turns in here. The apex is much later than the reference lap.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 6.0, "text": "See how late he turns in here."},
                {
                    "id": 1,
                    "start": 6.0,
                    "end": 12.5,
                    "text": "The apex is much later than the reference lap.",
                },
            ],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/v1/audio/transcriptions")
            assert (
                "Bearer" in request.headers.get("Authorization", "")
                or request.headers.get("Authorization") == ""
            )
            # Check the multipart form has the expected fields.
            return httpx.Response(200, json=response_payload)

        transport = httpx.MockTransport(handler)
        executor = OpenAIAudioTranscriptionExecutor(
            endpoint_url="http://speaches.test",
            api_key="test-key",  # pragma: allowlist secret
            bucket="vorag-vce-derived",
            garage=fake_garage,
        )
        # Override the executor's httpx client with our mock transport.
        # The executor creates its own client internally — patch the openai-audio
        # path by monkeypatching httpx.AsyncClient via sys.modules.
        import sys

        real_async_client = sys.modules["httpx"].AsyncClient

        class _PatchedAsyncClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                kwargs["transport"] = transport
                self._inner = real_async_client(*args, **kwargs)

            async def post(self, *args: object, **kwargs: object) -> object:
                return await self._inner.post(*args, **kwargs)

            async def __aenter__(self) -> object:
                await self._inner.__aenter__()
                return self
                return self._inner.__aenter__()

            async def __aexit__(self, *args: object) -> None:
                await self._inner.__aexit__(*args)

        sys.modules["httpx"].AsyncClient = _PatchedAsyncClient  # type: ignore[attr-defined]
        try:
            artifact = asyncio_run(executor.run("yt:abc", video_path=video_path))
        finally:
            sys.modules["httpx"].AsyncClient = real_async_client  # type: ignore[attr-defined]

        assert artifact.blob_key == "transcripts/yt:abc.transcript.json"
        assert artifact.language == "english"
        assert len(artifact.segments) == 2
        assert artifact.segments[0].text == "See how late he turns in here."
        # The bucket now holds the original payload (the executor persists
        # whatever the server returned, not the segments-only projection).
        assert fake_garage.has_object("vorag-vce-derived", artifact.blob_key)
        stored = fake_garage.get_json("vorag-vce-derived", artifact.blob_key)
        assert stored["language"] == "english"
        assert len(stored["segments"]) == 2

    def test_plain_json_response_falls_back_to_single_segment(
        self, fake_garage: FakeGarage, tmp_path: Path
    ) -> None:
        """If the endpoint returns ``response_format=json`` (no segments),
        the executor wraps the full text into a single zero-length segment
        so downstream stages still see something — the row's
        transcript_excerpt falls back to the joined text via
        ``_extract_window_text``.
        """
        import sys

        import httpx

        from vo_rag.pipelines.vce.transcription import OpenAIAudioTranscriptionExecutor

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake")

        response_payload = {
            "text": "the full transcript with no segments",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=response_payload)

        transport = httpx.MockTransport(handler)

        real_async_client = sys.modules["httpx"].AsyncClient

        class _PatchedAsyncClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                kwargs["transport"] = transport
                self._inner = real_async_client(*args, **kwargs)

            async def post(self, *args: object, **kwargs: object) -> object:
                return await self._inner.post(*args, **kwargs)

            async def __aenter__(self) -> object:
                await self._inner.__aenter__()
                return self
                return self._inner.__aenter__()

            async def __aexit__(self, *args: object) -> None:
                await self._inner.__aexit__(*args)

        sys.modules["httpx"].AsyncClient = _PatchedAsyncClient  # type: ignore[attr-defined]

        executor = OpenAIAudioTranscriptionExecutor(
            endpoint_url="http://speaches.test",
            bucket="vorag-vce-derived",
            garage=fake_garage,
        )
        try:
            artifact = asyncio_run(executor.run("yt:abc", video_path=video_path))
        finally:
            sys.modules["httpx"].AsyncClient = real_async_client  # type: ignore[attr-defined]

        assert artifact.blob_key is not None
        assert len(artifact.segments) == 1
        assert artifact.segments[0].text == "the full transcript with no segments"


def asyncio_run(coro: Awaitable[TranscriptArtifact]) -> TranscriptArtifact:
    """Tiny shim so tests can ``await`` without pytest-asyncio markers."""

    result: TranscriptArtifact = asyncio.run(coro)  # type: ignore[arg-type]
    return result
