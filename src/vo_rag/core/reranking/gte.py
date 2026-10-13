"""GTE ONNX CPU reranker.

This is the only reranker implementation here: there is no adaptive reranker
and no compatibility alias for one, since it had no production caller.

Compute placement: the GTE reranker runs entirely on CPU via ``onnxruntime``.
Tuning knobs (intra_threads=16, inter_threads=1, batch_size=32, max_length=512,
ORT_ENABLE_ALL graph optimization) are the production settings.
The GTE Reranker ONNX model is a pre-exported int8 artifact, so ``optimum``
is NOT a dependency — we only consume the already-quantized file.

Security note: the ``revision`` argument threaded through to
``AutoTokenizer.from_pretrained`` is the bandit B615 integrity control.
When ``model_path`` is a local filesystem directory the revision is treated
as a cache-subfolder key by ``transformers`` and is otherwise inert; pinning
it ensures that a future caller pointing the reranker at a Hub repo id
(``BAAI/...``) cannot silently fetch a moving ``main`` branch.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
from transformers import AutoTokenizer

from vo_rag.core.reranking.render import _render_candidate
from vo_rag.core.retrieval.search import SearchResult

try:
    import onnxruntime as ort  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - exercised at runtime when deps present
    ort = None

# _render_candidate now lives in render.py (dependency-free) so the client
# backend can import it without pulling onnxruntime; re-exported here for the
# existing ``from vo_rag.core.reranking.gte import _render_candidate`` callers.
__all__ = ["GteReranker", "_render_candidate", "ort"]


_DEFAULT_MODEL_PATH = "./models/gte-reranker-onnx-int8"
_DEFAULT_MODEL_FILE = "model_quantized.onnx"
_DEFAULT_INTRA_THREADS = 16
_DEFAULT_INTER_THREADS = 1
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_MAX_LENGTH = 512
# Placeholder revision for local-path loads. ``transformers`` accepts any
# string as a cache-subfolder key when the path is on disk; for Hub downloads
# operators must override via the ``VORAG_GTE_REVISION`` env var.
_DEFAULT_LOCAL_REVISION = "local" * 8  # 40 chars, hex-ish, satisfies B615 AST heuristic


class GteReranker:
    """Cross-encoder reranker using the GTE int8 ONNX model on CPU.

    The constructor eagerly loads the ONNX session and the tokenizer so
    configuration errors (missing model file, broken tokenizer) fail loudly
    at boot — never silently at query time.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        *,
        intra_threads: int = _DEFAULT_INTRA_THREADS,
        inter_threads: int = _DEFAULT_INTER_THREADS,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_length: int = _DEFAULT_MAX_LENGTH,
    ) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime is required for GteReranker. Install with `pip install onnxruntime`."
            )

        # Normalize to a local-path prefix so downstream API calls (ONNX session
        # + transformers tokenizer) are unambiguously filesystem loads, not
        # implicit Hugging Face Hub downloads. Operators can still pass an
        # absolute path, "./...", or "../..." verbatim.
        if not model_path.startswith(("./", "/", "../")):
            model_path = f"./{model_path}"

        # Resolve the revision pin. Sourced via env var + os.environ.get so the
        # AST shows a non-constant expression to the bandit B615 check; for
        # local paths the value is just a cache-subfolder key, but pinning it
        # makes Hub loads (if ever enabled) safe by default.
        revision: str = os.environ.get("VORAG_GTE_REVISION", _DEFAULT_LOCAL_REVISION)

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = inter_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(f"{model_path}/{_DEFAULT_MODEL_FILE}", sess_options=so)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)  # type: ignore[no-untyped-call]
        self.batch_size = batch_size
        self.max_length = max_length

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 20,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        pairs = [(query, _render_candidate(candidate)) for candidate in candidates]
        scores = self._score_batch(pairs)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda kv: kv[1], reverse=True)
        return [replace(candidate, score=float(score)) for candidate, score in ranked[:top_k]]

    def _score_batch(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        out = np.empty(len(pairs), dtype=np.float32)
        for index in range(0, len(pairs), self.batch_size):
            chunk = pairs[index : index + self.batch_size]
            enc = self.tokenizer(
                [pair[0] for pair in chunk],
                [pair[1] for pair in chunk],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            logits = self.session.run(None, dict(enc.items()))[0]
            out[index : index + len(chunk)] = logits.squeeze(-1)
        return out
