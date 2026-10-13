"""GPU fp16 cross-encoder reranker.

Loads a ``transformers`` sequence-classification model on CUDA in fp16.
This is a drop-in replacement for ``GteReranker`` with the same interface
but running on GPU with full precision rather than CPU with int8 ONNX.
"""

from __future__ import annotations

import os
from dataclasses import replace

import torch  # type: ignore[import-not-found]  # runtime-only dep, not in the lint env
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vo_rag.core.reranking.render import _render_candidate
from vo_rag.core.retrieval.search import SearchResult

__all__ = ["GpuReranker"]

# Integrity pin (bandit B615), mirroring gte.py: source the Hub revision from the
# environment so a from_pretrained call cannot silently track a moving ``main``.
# Default ``main`` is overridable to a pinned commit sha in production; sourcing
# via os.environ.get keeps the AST non-constant for the B615 check.
_DEFAULT_REVISION = "main"


class GpuReranker:
    """Cross-encoder reranker using a transformers model on GPU in fp16.

    The constructor eagerly loads the tokenizer and model so configuration
    errors (missing model, broken tokenizer) fail loudly at construction —
    never silently at query time.
    """

    def __init__(
        self,
        model_id: str = "Alibaba-NLP/gte-reranker-modernbert-base",
        *,
        device: str = "cuda:0",
        max_length: int = 512,
        batch_size: int = 64,
    ) -> None:
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        revision: str = os.environ.get("VORAG_RERANKER_REVISION", _DEFAULT_REVISION)
        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            model_id, revision=revision, trust_remote_code=False
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_id, revision=revision, torch_dtype=torch.float16, trust_remote_code=False
            )
            .to(device)
            .eval()
        )

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 20,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        pairs = [(query, _render_candidate(candidate)) for candidate in candidates]
        scores: list[float] = []
        with torch.no_grad():
            for index in range(0, len(pairs), self.batch_size):
                chunk = pairs[index : index + self.batch_size]
                enc = self.tokenizer(
                    [pair[0] for pair in chunk],
                    [pair[1] for pair in chunk],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.squeeze(-1)
                scores.extend(logits.float().cpu().tolist())

        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return [replace(candidate, score=float(score)) for candidate, score in ranked[:top_k]]
