"""_release_cuda_cache (VOR-1425, VOR-1427): empty the CUDA cache, no forced GC."""

from __future__ import annotations

import gc
import sys
import types

import pytest


def test_release_empties_the_cuda_cache_without_a_full_gc(monkeypatch: pytest.MonkeyPatch) -> None:
    from vo_rag.core.parsing.parsers import docling as docling_parser

    emptied: list[bool] = []
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True, empty_cache=lambda: emptied.append(True)
        )
    )

    def _no_gc(*_a: object, **_k: object) -> int:
        raise AssertionError("a full gc.collect() after every parse costs 1-2.5 s per book")

    # The fake torch lives only around the call: where real
    # torch is installed, pytest-randomly reseeds it (thinc ->
    # ``torch.manual_seed`` -> ``import torch.cuda``) in the teardown hook,
    # before fixture finalizers run. A fake still in sys.modules there makes
    # the reseed raise, the monkeypatch finalizer never runs, and the fake
    # leaks into every later test.
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", fake_torch)
        patch.setattr(gc, "collect", _no_gc)
        docling_parser._release_cuda_cache()

    assert emptied == [True]
