"""Tests for the GTE ONNX CPU reranker.

``AdaptiveReranker`` has no production caller, so the tests assert that it is
not exported.

The ONNX-model-loading happy path is marked ``integration`` and skipped when
the model directory is absent (the binary model files are not checked in).
Unit-level coverage of batching, ranking, and payload rendering is achieved by
injecting fakes for ``onnxruntime.InferenceSession`` and
``transformers.AutoTokenizer`` so CI never needs the real model files.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from vo_rag.core.reranking import gte as gte_module
from vo_rag.core.reranking.gte import GteReranker, _render_candidate
from vo_rag.core.retrieval.search import SearchResult

# --- Test doubles -------------------------------------------------------- #


def _mk(id_: str, score: float = 0.5, payload: dict[str, Any] | None = None) -> SearchResult:
    if payload is None:
        payload = {"text": "x"}
    return SearchResult(id=id_, score=score, collection="c", payload=payload)


class _FakeTokenizer:
    """Mimics the slice of the transformers API ``GteReranker`` actually uses."""

    path: str = ""
    revision: str | None = None

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, path: str, *, revision: str | None = None) -> _FakeTokenizer:
        instance = cls()
        instance.path = path
        instance.revision = revision
        return instance

    def __call__(
        self,
        queries: list[str],
        contexts: list[str],
        *,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "np",
    ) -> dict[str, Any]:
        import numpy as np

        self.calls.append(
            {
                "queries": list(queries),
                "contexts": list(contexts),
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
                "return_tensors": return_tensors,
            }
        )
        n = len(queries)
        return {
            "input_ids": np.zeros((n, 4), dtype=np.int64),
            "attention_mask": np.ones((n, 4), dtype=np.int64),
        }


class _FakeSession:
    path: str = ""
    sess_options: Any = None

    def __init__(self, scores: list[float] | None = None) -> None:
        import numpy as np

        self._scores = np.array(scores or [], dtype=np.float32).reshape(-1, 1)
        self.run_inputs: dict[str, Any] | None = None

    def run(self, _output_names: list[str] | None, inputs: dict[str, Any]) -> list[Any]:
        import numpy as np

        self.run_inputs = inputs
        n = next(iter(inputs.values())).shape[0]
        if self._scores.size == 0 or self._scores.shape[0] != n:
            # Default: emit a flat 0.5 per row so shape is always right.
            return [np.full((n, 1), 0.5, dtype=np.float32)]
        return [self._scores]


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeSession, type[_FakeTokenizer]]:
    """Patch ``onnxruntime.InferenceSession`` and ``AutoTokenizer`` for one test.

    Returns the shared fake session plus the fake tokenizer class so tests can
    inspect the session options or read back captured calls.
    """
    session = _FakeSession()

    def _fake_session(path: str, sess_options: Any = None) -> _FakeSession:
        session.path = path
        session.sess_options = sess_options
        return session

    monkeypatch.setattr(gte_module.ort, "InferenceSession", _fake_session)
    monkeypatch.setattr(gte_module, "AutoTokenizer", _FakeTokenizer)
    return session, _FakeTokenizer


# --- Empty candidates short-circuit -------------------------------------- #


def test_empty_candidates_returns_empty_list() -> None:
    """``rerank`` with no candidates must return an empty list, not raise."""

    class _Stub(GteReranker):
        def __init__(self) -> None:
            self.batch_size = 32
            self.max_length = 512

    out = _Stub().rerank("q", [], top_k=5)
    assert out == []


# --- Batching + scoring -------------------------------------------------- #


def test_rerank_sorts_by_score_descending(monkeypatch: pytest.MonkeyPatch) -> None:
    """Higher score from the ONNX session wins, even if input was low-score."""
    _install_fakes(monkeypatch)

    # Per-candidate scores; consumed in call order so multiple batches line up.
    scores_table = [0.9, 0.1, 0.5]  # candidate 1 > 3 > 2 in reranked order
    state = {"idx": 0}

    class _ScoreSession(_FakeSession):
        def run(self, _outputs: list[str] | None, inputs: dict[str, Any]) -> list[Any]:
            import numpy as np

            n = next(iter(inputs.values())).shape[0]
            chunk = scores_table[state["idx"] : state["idx"] + n]
            state["idx"] += n
            return [np.array(chunk, dtype=np.float32).reshape(-1, 1)]

    monkeypatch.setattr(gte_module.ort, "InferenceSession", lambda *a, **kw: _ScoreSession())
    reranker = GteReranker("models/fake", batch_size=2, max_length=128)
    cands = [_mk("1", score=0.1), _mk("2", score=0.1), _mk("3", score=0.1)]
    out = reranker.rerank("q", cands, top_k=3)
    assert [c.id for c in out] == ["1", "3", "2"]
    assert [c.score for c in out] == pytest.approx([0.9, 0.5, 0.1])


def test_rerank_respects_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only ``top_k`` results are returned even if more candidates score well."""
    _install_fakes(monkeypatch)
    reranker = GteReranker("models/fake", batch_size=4, max_length=64)

    # Patch the score function to return ascending scores per position.
    import numpy as np

    def _score(self: GteReranker, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.arange(len(pairs), dtype=np.float32)

    monkeypatch.setattr(GteReranker, "_score_batch", _score)
    cands = [_mk(f"id{i}") for i in range(10)]
    out = reranker.rerank("q", cands, top_k=3)
    assert len(out) == 3
    # Highest scores are the LAST three indices (score == index).
    assert [c.id for c in out] == ["id9", "id8", "id7"]


def test_rerank_batches_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_score_batch`` splits pairs into ``batch_size`` chunks; ONNX run is called per chunk."""
    _install_fakes(monkeypatch)
    seen_batch_sizes: list[int] = []

    class _BatchSpySession(_FakeSession):
        def run(self, _outputs: list[str] | None, inputs: dict[str, Any]) -> list[Any]:
            import numpy as np

            n = next(iter(inputs.values())).shape[0]
            seen_batch_sizes.append(n)
            return [np.full((n, 1), 0.5, dtype=np.float32)]

    monkeypatch.setattr(gte_module.ort, "InferenceSession", lambda *a, **kw: _BatchSpySession())
    reranker = GteReranker("models/fake", batch_size=4, max_length=64)
    pairs = [(f"q{i}", f"c{i}") for i in range(10)]
    out = reranker._score_batch(pairs)
    assert out.shape == (10,)
    assert seen_batch_sizes == [4, 4, 2], (
        f"Expected batches [4, 4, 2] for 10 pairs @ batch_size=4; got {seen_batch_sizes}"
    )


def test_rerank_passes_query_and_candidate_to_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tokenizer must see ``(query, candidate_text)`` pairs and the configured max_length."""
    _install_fakes(monkeypatch)

    class _CaptureTokenizer(_FakeTokenizer):
        def __call__(  # type: ignore[override]
            self, q: list[str], c: list[str], **kw: Any
        ) -> dict[str, Any]:
            self.calls.append({"queries": q, "contexts": c, "kwargs": kw})
            return super().__call__(q, c, **kw)

    monkeypatch.setattr(gte_module, "AutoTokenizer", _CaptureTokenizer)
    reranker = GteReranker("models/fake", batch_size=8, max_length=128)
    cands = [
        _mk("a", payload={"text": "alpha body"}),
        _mk("b", payload={"text": "beta body"}),
    ]
    reranker.rerank("hello", cands, top_k=5)
    captured = reranker.tokenizer.calls[0]
    assert captured["queries"] == ["hello", "hello"]
    assert captured["contexts"] == ["alpha body", "beta body"]
    assert captured["kwargs"]["max_length"] == 128
    assert captured["kwargs"]["padding"] is True
    assert captured["kwargs"]["truncation"] is True
    assert captured["kwargs"]["return_tensors"] == "np"


def test_rerank_preserves_dataclass_identity_for_input_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned hits must be the same dataclass type as the input (SearchResult-shaped)."""
    import numpy as np

    _install_fakes(monkeypatch)
    reranker = GteReranker("models/fake", batch_size=4, max_length=64)
    monkeypatch.setattr(
        GteReranker,
        "_score_batch",
        lambda self, pairs: np.linspace(0.1, 0.9, len(pairs), dtype=np.float32),
    )
    cands = [_mk(f"id{i}") for i in range(5)]
    out = reranker.rerank("q", cands, top_k=5)
    assert all(isinstance(c, SearchResult) for c in out)
    # Score field must be overwritten with the ONNX-derived value, not the input 0.5.
    assert {c.score for c in out} != {0.5}


# --- _render_candidate content precedence ------------------------------- #


def test_render_candidate_prefers_section_title_then_text() -> None:
    out = _render_candidate(_mk("x", payload={"section_title": "Brakes", "text": "front disc"}))
    assert out == "Brakes\nfront disc"


def test_render_candidate_falls_back_to_text_when_no_title() -> None:
    out = _render_candidate(_mk("x", payload={"text": "front disc"}))
    assert out == "front disc"


def test_render_candidate_falls_back_to_content_field() -> None:
    out = _render_candidate(_mk("x", payload={"content": "fallback body"}))
    assert out == "fallback body"


def test_render_candidate_falls_back_to_content_text_field() -> None:
    out = _render_candidate(_mk("x", payload={"content_text": "deepest fallback"}))
    assert out == "deepest fallback"


def test_render_candidate_precedence_is_section_title_then_text_then_content() -> None:
    """When all three are present, ``text`` wins over ``content``."""
    out = _render_candidate(
        _mk("x", payload={"section_title": "T", "text": "TXT", "content": "CONT"})
    )
    assert out == "T\nTXT"


def test_render_candidate_section_title_with_no_text_returns_just_title() -> None:
    out = _render_candidate(_mk("x", payload={"section_title": "Solo Title"}))
    assert out == "Solo Title"


def test_render_candidate_empty_payload_returns_empty_string() -> None:
    out = _render_candidate(_mk("x", payload={}))
    assert out == ""


# --- Failure modes (loud, no silent skip) ------------------------------- #


def test_missing_onnxruntime_raises_loud_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``onnxruntime`` is not importable at construction time, raise loud RuntimeError."""
    monkeypatch.setattr(gte_module, "ort", None)
    with pytest.raises(RuntimeError, match="onnxruntime is required"):
        GteReranker("models/fake")


def test_failed_model_load_raises_loud_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ONNX session construction blows up (e.g. model file missing), raise loudly.

    No silent ``return None`` and no skip-the-rerank semantics — the calling
    surface layer can decide whether to fail boot, but core must not hide.
    """

    def _boom(*_a: Any, **_kw: Any) -> None:
        raise FileNotFoundError("model_quantized.onnx not found at models/fake")

    monkeypatch.setattr(gte_module.ort, "InferenceSession", _boom)
    monkeypatch.setattr(gte_module, "AutoTokenizer", _FakeTokenizer)
    with pytest.raises(FileNotFoundError, match="model_quantized.onnx"):
        GteReranker("models/fake")


def test_session_options_use_intra_threads_and_ort_enable_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session is configured with the documented intra/inter threads + ORT_ENABLE_ALL."""
    session, _ = _install_fakes(monkeypatch)
    GteReranker("models/fake", intra_threads=7, inter_threads=2)
    opts = session.sess_options
    assert opts.intra_op_num_threads == 7
    assert opts.inter_op_num_threads == 2
    # ORT_ENABLE_ALL is the value onnxruntime ships for "all graph opts".
    assert opts.graph_optimization_level == gte_module.ort.GraphOptimizationLevel.ORT_ENABLE_ALL


# --- Drop assertion: AdaptiveReranker must NOT be in the public surface --- #


def test_adaptive_reranker_is_not_exported() -> None:
    """``AdaptiveReranker`` has no production caller and is not exported."""
    pkg = importlib.import_module("vo_rag.core.reranking")
    assert not hasattr(pkg, "AdaptiveReranker")
    # The sub-module must also not re-export it.
    assert not hasattr(gte_module, "AdaptiveReranker")
    # The union-unique helper the dead class used must also be gone.
    assert not hasattr(gte_module, "_union_unique")


# --- Local-path + revision security guards ----------------------------- #


def test_default_model_path_is_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default model path must look like a local filesystem path, not a Hub id.

    This locks in the bandit B615 integrity control: a future refactor cannot
    silently flip the default to a bare ``BAAI/...`` repo id and reintroduce
    an un-pinned Hub download.
    """
    session, _ = _install_fakes(monkeypatch)
    GteReranker()  # default model path
    assert session.path.startswith(("./", "/", "../")), (
        f"Default GteReranker model path must be a local path; got {session.path!r}"
    )


def test_relative_model_path_is_normalized_to_dot_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GteReranker("models/foo")`` is normalized to ``./models/foo`` for the session call."""
    session, _ = _install_fakes(monkeypatch)
    GteReranker("models/foo")
    assert session.path == "./models/foo/model_quantized.onnx"


def test_revision_is_passed_to_from_pretrained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``revision`` kwarg must reach ``AutoTokenizer.from_pretrained`` (B615 pin)."""
    _, token_cls = _install_fakes(monkeypatch)
    monkeypatch.setenv("VORAG_GTE_REVISION", "0" * 40)
    reranker = GteReranker("models/foo")
    assert reranker.tokenizer.revision == "0" * 40
    assert token_cls is _FakeTokenizer  # sanity: fakes were used


def test_revision_defaults_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``VORAG_GTE_REVISION``, a safe local-path placeholder is used."""
    monkeypatch.delenv("VORAG_GTE_REVISION", raising=False)
    _install_fakes(monkeypatch)
    reranker = GteReranker("models/foo")
    assert reranker.tokenizer.revision == gte_module._DEFAULT_LOCAL_REVISION


# --- Integration: only runs if a real ONNX model is present on disk -------- #


_INTEGRATION_MODEL_DIR = "models/gte-reranker-onnx-int8"


@pytest.mark.integration
def test_onnx_model_load_runs_if_model_present() -> None:
    """Skip unless the quantized ONNX model + tokenizer files are on disk.

    This is the only test that touches the real ``onnxruntime`` /
    ``transformers`` model loaders; the unit-level tests above inject fakes so
    CI never needs the multi-megabyte model artifacts.
    """
    import os

    if not os.path.isdir(_INTEGRATION_MODEL_DIR):
        pytest.skip(
            f"ONNX model dir {_INTEGRATION_MODEL_DIR!r} not present; "
            "reranker is exercised via fake session/tokenizer in unit tests."
        )
    reranker = GteReranker(_INTEGRATION_MODEL_DIR, intra_threads=2, inter_threads=1)
    cand = _mk("real-1", payload={"text": "the brake rotor requires resurfacing"})
    out = reranker.rerank("brake rotor resurfacing procedure", [cand], top_k=1)
    assert len(out) == 1
    assert out[0].id == "real-1"
    assert isinstance(out[0].score, float)
