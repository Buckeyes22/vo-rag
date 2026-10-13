"""Pluggable transcription executor for the VCE pipeline.

WhisperX large-v3 is GPU-bound, the workstation has
no GPU, and the legacy failure mode was *persistence*, not transcription
quality (25 of 36 failed runs had ``output_blob: null`` — the segment text
existed in memory but never made it to the bucket). This module therefore:

1. Defines a :class:`TranscriptionExecutor` Protocol so the runner cannot
   tell the implementations apart — the runner only sees ``run() -> TranscriptArtifact``
   with a non-null ``blob_key``.
2. Ships three implementations:
   - :class:`LocalWhisperXExecutor` — CPU, ``tiny`` model. Testinfra-only smoke.
   - :class:`RemoteWhisperXExecutor` — POSTs to a configured GPU host. The
     real production executor; the endpoint URL is an env knob.
   - :class:`StubTranscriptionExecutor` — returns canned segments from a
     fixture file. Hermetic tests of downstream stages.
3. Persists first: every implementation uploads to
   ``vorag-vce-derived/transcripts/<video_id>.transcript.json`` before
   returning. The runner's ``transcribe`` stage refuses to advance without
   a non-null ``blob_key``.

Adding a fourth executor (e.g. RunPod serverless worker) is a matter of
implementing the Protocol; the runner does not change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog

from vo_rag.pipelines.vce.types import TranscriptArtifact, TranscriptSegment

log = structlog.get_logger(__name__)

# Where transcripts land. The runner reads the bucket + key
# from the artifact's blob_key; the executor is the only writer.
DEFAULT_TRANSCRIPT_BLOB_PREFIX = "transcripts"
DEFAULT_TRANSCRIPT_BLOB_SUFFIX = ".transcript.json"


@runtime_checkable
class TranscriptionExecutor(Protocol):
    """The interface the VCE runner sees.

    Implementations MUST upload the transcript to the bucket and return
    ``TranscriptArtifact(blob_key=..., segments=...)`` — never ``blob_key=None``.
    A non-null ``blob_key`` is the precondition that lets later stages run.
    """

    async def run(self, video_id: str, video_path: Path) -> TranscriptArtifact: ...


class StubTranscriptionExecutor:
    """Returns canned segments from a JSON fixture file.

    The fixture shape is the same as the bucket payload:
    ``{"video_id": ..., "language": ..., "segments": [{"start": ..., "end": ..., "text": ...}]}``

    Used by hermetic tests of the runner's downstream stages. The fixture
    MUST contain at least one segment overlapping any expected interval, or
    the export will have empty ``transcript_excerpt`` fields.

    Args:
        fixture_dir: Directory containing ``<video_id>.transcript.json`` fixtures.
        bucket: Bucket name to upload to (testinfra Garage).
        garage: A ``GarageClient`` instance for the upload.
        upload: If True (default), upload to the bucket. Set False for unit
            tests that don't need a live bucket.
    """

    def __init__(
        self,
        *,
        fixture_dir: Path | str,
        bucket: str,
        garage: Any,
        upload: bool = True,
    ) -> None:
        self._fixture_dir = Path(fixture_dir)
        self._bucket = bucket
        self._garage = garage
        self._upload = upload

    async def run(self, video_id: str, video_path: Path) -> TranscriptArtifact:
        fixture_path = self._fixture_dir / f"{video_id}{DEFAULT_TRANSCRIPT_BLOB_SUFFIX}"
        if not fixture_path.exists():
            raise FileNotFoundError(
                f"StubTranscriptionExecutor: no fixture for {video_id} at {fixture_path}"
            )
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        segments = [
            TranscriptSegment(start=float(s["start"]), end=float(s["end"]), text=str(s["text"]))
            for s in payload.get("segments", [])
        ]
        blob_key = f"{DEFAULT_TRANSCRIPT_BLOB_PREFIX}/{video_id}{DEFAULT_TRANSCRIPT_BLOB_SUFFIX}"
        if self._upload and self._garage is not None:
            await self._garage.put_bytes(
                self._bucket,
                blob_key,
                json.dumps(payload).encode("utf-8"),
                "application/json",
            )
        return TranscriptArtifact(
            segments=segments,
            language=payload.get("language"),
            blob_key=blob_key,
            cost_usd=0.0,
        )


class LocalWhisperXExecutor:
    """Local CPU WhisperX executor with the ``tiny`` model.

    This is the testinfra smoke executor. WhisperX large-v3 on CPU against
    a 60-minute video takes hours; do NOT use this in production. The
    production executor is :class:`RemoteWhisperXExecutor`, which the
    operator wires to a real GPU host via
    ``VORAG_VCE_TRANSCRIPTION_ENDPOINT_URL``.

    Args:
        model: WhisperX model name (default ``tiny``). Override via env
            ``VORAG_VCE_TRANSCRIPTION_MODEL`` for production CPU smoke.
        device: ``cpu`` (default) or ``cuda``.
        bucket: Bucket name to upload to.
        garage: A ``GarageClient`` instance.
        compute_type: WhisperX compute type (default ``int8`` for CPU).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        device: str = "cpu",
        bucket: str,
        garage: Any,
        compute_type: str = "int8",
    ) -> None:
        self._model = model or os.environ.get("VORAG_VCE_TRANSCRIPTION_MODEL", "tiny")
        self._device = device
        self._bucket = bucket
        self._garage = garage
        self._compute_type = compute_type

    async def run(self, video_id: str, video_path: Path) -> TranscriptArtifact:
        # WhisperX is an optional dependency — keep this executor dormant
        # unless the operator opts in by installing `whisperx`. Pulling it
        # in has a real cost; we don't want CI to silently pull
        # torch + ctranslate2 every run.
        try:
            import whisperx  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "LocalWhisperXExecutor requires the optional `whisperx` "
                "dependency; install with `uv pip install whisperx` or "
                "use StubTranscriptionExecutor / RemoteWhisperXExecutor."
            ) from exc

        # Lazy import torch only when actually running.
        audio = whisperx.load_audio(str(video_path))
        model = whisperx.load_model(self._model, self._device, compute_type=self._compute_type)
        result = model.transcribe(audio)
        # Align / diarize would go here in production. For the testinfra
        # smoke, raw WhisperX segments are sufficient.
        segments = [
            TranscriptSegment(start=float(s["start"]), end=float(s["end"]), text=str(s["text"]))
            for s in result.get("segments", [])
        ]
        payload = {
            "video_id": video_id,
            "language": result.get("language"),
            "segments": [s.__dict__ for s in segments],
        }
        blob_key = f"{DEFAULT_TRANSCRIPT_BLOB_PREFIX}/{video_id}{DEFAULT_TRANSCRIPT_BLOB_SUFFIX}"
        await self._garage.put_bytes(
            self._bucket,
            blob_key,
            json.dumps(payload).encode("utf-8"),
            "application/json",
        )
        return TranscriptArtifact(
            segments=segments,
            language=result.get("language"),
            blob_key=blob_key,
            cost_usd=0.0,
        )


class RemoteWhisperXExecutor:
    """HTTP executor that POSTs to a configured WhisperX endpoint.

    The endpoint is expected to:

    - Accept a multipart upload (field ``video``) plus form fields
      ``video_id`` and ``model``.
    - Return JSON ``{"video_id": ..., "language": ..., "segments": [...]}``
      with the same shape as the bucket payload.

    The operator points this at the real GPU host via
    ``VORAG_VCE_TRANSCRIPTION_ENDPOINT_URL``. Until then, this executor is a
    no-op stub that returns a failed artifact: a run that cannot transcribe a
    video marks it and moves on, rather than failing the batch.

    Args:
        endpoint_url: Absolute URL of the WhisperX HTTP endpoint.
        model: Model name expected by the endpoint.
        bucket: Bucket name to upload to.
        garage: A ``GarageClient`` instance.
        timeout_s: HTTP timeout in seconds (default 1800 = 30 min, sized
            for a 60-minute video on a slow GPU).
    """

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        model: str = "large-v3",
        bucket: str,
        garage: Any,
        timeout_s: float = 1800.0,
    ) -> None:
        self._endpoint_url = endpoint_url or os.environ.get("VORAG_VCE_TRANSCRIPTION_ENDPOINT_URL")
        self._model = model
        self._bucket = bucket
        self._garage = garage
        self._timeout_s = timeout_s

    async def run(self, video_id: str, video_path: Path) -> TranscriptArtifact:
        if not self._endpoint_url:
            # No endpoint configured — return a failed artifact (no blob_key).
            # The runner marks the video as transcribe=failed and moves on.
            log.warning(
                "vce_transcription_remote_unconfigured",
                video_id=video_id,
                hint="set VORAG_VCE_TRANSCRIPTION_ENDPOINT_URL to a GPU host",
            )
            return TranscriptArtifact(segments=[], language=None, blob_key=None, cost_usd=0.0)

        # Lazy import — `httpx` is a soft dep that only the remote executor
        # needs. Keeps the import graph small for the testinfra smoke path.
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "RemoteWhisperXExecutor requires the optional `httpx` dependency."
            ) from exc

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            with video_path.open("rb") as fh:
                response = await client.post(
                    self._endpoint_url,
                    files={"video": (video_path.name, fh, "application/octet-stream")},
                    data={"video_id": video_id, "model": self._model},
                )
        response.raise_for_status()
        payload = response.json()
        segments = [
            TranscriptSegment(start=float(s["start"]), end=float(s["end"]), text=str(s["text"]))
            for s in payload.get("segments", [])
        ]
        blob_key = f"{DEFAULT_TRANSCRIPT_BLOB_PREFIX}/{video_id}{DEFAULT_TRANSCRIPT_BLOB_SUFFIX}"
        await self._garage.put_bytes(
            self._bucket,
            blob_key,
            json.dumps(payload).encode("utf-8"),
            "application/json",
        )
        return TranscriptArtifact(
            segments=segments,
            language=payload.get("language"),
            blob_key=blob_key,
            cost_usd=0.0,
        )


# Speaches (https://speaches.ai) is a self-hosted, OpenAI-Audio-API-compatible
# transcription server. The OpenAI Audio API contract
# (POST {endpoint}/v1/audio/transcriptions, multipart with `file` + `model`)
# is implemented by Speaches, OpenAI's own /v1/audio/transcriptions endpoint,
# and a number of other providers (Groq, Together, etc.) — so this executor
# is named for the contract, not the implementation.
SPEACHES_DEFAULT_MODEL: str = "large-v3-turbo"


class OpenAIAudioTranscriptionExecutor:
    """Transcribe via the OpenAI Audio API (Speaches-compatible).

    The contract is:

        POST {endpoint}/v1/audio/transcriptions
        Content-Type: multipart/form-data
        Body fields:
          - file:        the audio/video file
          - model:       the model identifier (Speaches passes this through)
          - language:    optional ISO-639-1 (e.g. "en")
          - response_format: "verbose_json" returns segments; default "json"
                            returns the joined text only (no offsets).
          - timestamp_granularities: ["segment"] | ["word"] — Speaches supports
                            "segment" via verbose_json.

    Response shapes we handle:

        # response_format=json (default — no segments, only full text)
        {"text": "..."}

        # response_format=verbose_json
        {"task": "transcribe", "language": "en", "duration": 1234.5,
         "text": "...", "segments": [{"id": 1, "start": 0.0, "end": 5.0,
                                       "text": "..."}]}

    Speaches default model is `SPEACHES_DEFAULT_MODEL`
    (`large-v3-turbo` — faster-whisper large-v3-turbo on the lin host).
    The OpenAI Audio API itself uses ``model="whisper-1"``; set the model name
    via ``VORAG_VCE_TRANSCRIPTION_MODEL`` to match the deployment.

    Args:
        endpoint_url: Base URL of the API root (e.g. ``http://speaches:8000``
            or ``https://api.openai.com``). The executor appends
            ``/v1/audio/transcriptions``.
        model: Model name expected at the endpoint.
        bucket: Bucket name to upload the transcript artifact to.
        garage: A ``GarageClient`` instance.
        timeout_s: HTTP timeout in seconds (default 1800 = 30 min, sized
            for a 60-minute video on a slow GPU).
        api_key: Optional bearer token for endpoints that require it (e.g.
            OpenAI's hosted API). Speaches does not require auth when run
            in trusted-LAN mode.
        language: Optional ISO-639-1 language hint; ``"auto"`` skips the hint.
    """

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        model: str | None = None,
        bucket: str,
        garage: Any,
        timeout_s: float = 1800.0,
        api_key: str | None = None,
        language: str | None = None,
    ) -> None:
        self._endpoint_url = (
            endpoint_url
            or os.environ.get("VORAG_VCE_TRANSCRIPTION_ENDPOINT_URL")
            or os.environ.get("VORAG_VCE_SPEACHES_URL")
        )
        self._model = (
            model or os.environ.get("VORAG_VCE_TRANSCRIPTION_MODEL") or SPEACHES_DEFAULT_MODEL
        )
        self._bucket = bucket
        self._garage = garage
        self._timeout_s = timeout_s
        self._api_key = api_key or os.environ.get("VORAG_VCE_TRANSCRIPTION_API_KEY")
        self._language = language or os.environ.get("VORAG_VCE_TRANSCRIPTION_LANGUAGE")

    async def run(self, video_id: str, video_path: Path) -> TranscriptArtifact:
        if not self._endpoint_url:
            log.warning(
                "vce_transcription_openai_audio_unconfigured",
                video_id=video_id,
                hint=(
                    "set VORAG_VCE_TRANSCRIPTION_ENDPOINT_URL or "
                    "VORAG_VCE_SPEACHES_URL to your Speaches / OpenAI-Audio host"
                ),
            )
            return TranscriptArtifact(segments=[], language=None, blob_key=None, cost_usd=0.0)

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "OpenAIAudioTranscriptionExecutor requires the optional `httpx` dependency."
            ) from exc

        url = self._endpoint_url.rstrip("/") + "/v1/audio/transcriptions"
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        data: dict[str, str] = {
            "model": self._model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }
        if self._language and self._language != "auto":
            data["language"] = self._language

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            with video_path.open("rb") as fh:
                response = await client.post(
                    url,
                    headers=headers,
                    files={"file": (video_path.name, fh, "application/octet-stream")},
                    data=data,
                )
        response.raise_for_status()
        payload = response.json()

        # Parse segments — Speaches/OpenAI `verbose_json` returns a list under
        # "segments" with start/end/text. The bare `json` response format has
        # only "text"; in that case we fall back to a single window-segment.
        raw_segments = payload.get("segments") or []
        if raw_segments:
            segments = [
                TranscriptSegment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=str(seg.get("text", "")),
                )
                for seg in raw_segments
            ]
        else:
            full_text = str(payload.get("text", "")).strip()
            segments = [TranscriptSegment(start=0.0, end=0.0, text=full_text)] if full_text else []

        blob_key = f"{DEFAULT_TRANSCRIPT_BLOB_PREFIX}/{video_id}{DEFAULT_TRANSCRIPT_BLOB_SUFFIX}"
        await self._garage.put_bytes(
            self._bucket,
            blob_key,
            json.dumps(payload).encode("utf-8"),
            "application/json",
        )
        return TranscriptArtifact(
            segments=segments,
            language=payload.get("language"),
            blob_key=blob_key,
            cost_usd=0.0,
        )


__all__ = [
    "DEFAULT_TRANSCRIPT_BLOB_PREFIX",
    "DEFAULT_TRANSCRIPT_BLOB_SUFFIX",
    "LocalWhisperXExecutor",
    "OpenAIAudioTranscriptionExecutor",
    "RemoteWhisperXExecutor",
    "SPEACHES_DEFAULT_MODEL",
    "StubTranscriptionExecutor",
    "TranscriptArtifact",
    "TranscriptSegment",
    "TranscriptionExecutor",
]
