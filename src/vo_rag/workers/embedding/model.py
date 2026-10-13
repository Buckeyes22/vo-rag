"""BGE-M3 model wrapper for the local embedding worker.

The wire contract is dense 1024-dim vectors, sparse Qdrant-friendly
``{indices, values}`` and ColBERT token-level vectors; the client side in
``vo_rag.core.embedding.client`` depends on it.

Design notes:

- ``FlagEmbedding.BGEM3FlagModel`` is imported LAZILY inside ``__init__``.
  This keeps the worker importable on CPU-only hosts (tests, lint, dev
  shell) without forcing the GPU stack to be installed. Tests monkeypatch
  the ``FlagEmbedding`` module attribute to swap in a fake.
- ``batch_size`` is capped at ``min(32, len(texts))`` — the BGE-M3
  reference config. Going higher risks OOM on consumer GPUs.
- Sparse lexical weights are normalised to integer token ids and float
  weights so the response is JSON-serialisable without an encoder.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def _observed_revision(value: Any) -> str | None:
    """Retain an observed config revision, never a mutable alias as a pin."""
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) else None


class BgeM3Worker:
    """Thin wrapper around ``FlagEmbedding.BGEM3FlagModel``.

    The worker exposes only the operations the BGE-M3 HTTP server needs
    (``embed``) and a ``device`` attribute the health endpoint can echo.
    The underlying ``FlagEmbedding`` model is held on ``self.model`` for
    direct access when needed (tests, instrumentation).
    """

    def __init__(
        self,
        model_path: str = "BAAI/bge-m3",
        *,
        use_fp16: bool = True,
        device: str = "cuda:0",
    ) -> None:
        # Lazy import: BGE-M3 is a GPU dependency, and the worker module
        # must stay importable in CPU-only test runs and on lint runners.
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-not-found]

        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16, devices=[device])
        self.device = device
        config = getattr(getattr(self.model, "model", None), "config", None)
        tokenizer = getattr(self.model, "tokenizer", None)
        token_config = getattr(tokenizer, "init_kwargs", {})
        token_revision = (
            token_config.get("_commit_hash") if isinstance(token_config, dict) else None
        )
        self.runtime_identity: dict[str, str | None] = {
            "model_id": "BAAI/bge-m3"
            if model_path == "BAAI/bge-m3" or "models--BAAI--bge-m3" in model_path
            else "unknown",
            "model_config_revision": _observed_revision(getattr(config, "_commit_hash", None)),
            "tokenizer_config_revision": _observed_revision(token_revision),
            "identity_status": "observed_config",
            "weights_attestation": "unknown",
        }
        if (
            self.runtime_identity["model_config_revision"] is None
            or self.runtime_identity["tokenizer_config_revision"] is None
        ):
            self.runtime_identity["identity_status"] = "unknown"

    def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
        max_length: int,
    ) -> dict[str, Any]:
        out = self.model.encode(
            texts,
            batch_size=min(32, len(texts)),
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )

        result: dict[str, Any] = {"runtime_identity": dict(self.runtime_identity)}
        tokenizer = getattr(self.model, "tokenizer", None)
        if callable(tokenizer) and sum(len(text) for text in texts) <= 262_144:
            try:
                encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
                rows = encoded["input_ids"]
                if (
                    isinstance(rows, list)
                    and len(rows) == len(texts)
                    and all(isinstance(row, list) for row in rows)
                ):
                    counts = [len(row) for row in rows]
                    result["input_token_counts"] = counts
                    result["input_truncated"] = [count > max_length for count in counts]
            except Exception as exc:  # noqa: BLE001 - advisory evidence must not interrupt embedding
                log.warning("embedding token evidence unavailable: %s", type(exc).__name__)
        if return_dense:
            result["dense"] = out["dense_vecs"].tolist()
        if return_sparse:
            # BGE-M3 lexical_weights is list[dict[str|int, float]]; normalise
            # to Qdrant's sparse vector format (integer token ids, float
            # weights) so the client can hand the result straight to a
            # Qdrant upsert without further coercion.
            result["sparse"] = [
                {
                    "indices": [int(k) for k in d],
                    "values": [float(v) for v in d.values()],
                }
                for d in out["lexical_weights"]
            ]
        if return_colbert:
            result["colbert"] = [vec.tolist() for vec in out["colbert_vecs"]]
        return result


__all__ = ["BgeM3Worker"]
