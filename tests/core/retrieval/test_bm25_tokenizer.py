"""Tests for the production BM25 tokenizer leaf.

The **persistence contract** is the load-bearing thing under test: token ids
MUST be ``int.from_bytes(hashlib.blake2b(token, digest_size=4).digest(), "big")``
both at write time and at query time. Changing the digest size, algorithm, or
endianness means every existing Qdrant point has to be re-ingested.

Hard-coded token-id expectations below lock the blake2b-4 contract in place.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys

from vo_rag.core.retrieval.bm25_tokenizer import (
    _stable_token_id,
    tokenize,
    tokens_to_bm25_sparse,
)

# Hard-coded persistence-contract ids. Pinned to the current tokenizer;
# drift here = full re-ingest.
EXPECTED_TOKEN_IDS: dict[str, int] = {
    "eq._3.14": 8_170_999,  # preserved Eq./Equation/Fig. pattern
    "sae_j2464": 937_479_371,  # preserved SAE/ISO/IEC pattern
    "iso_26262": 2_212_737_921,  # preserved SAE/ISO/IEC pattern
    "figure_7.2": 1_395_873_718,  # preserved Fig./Figure pattern
    "rtx3060": 3_602_333_998,  # camelcase-num pattern (no separator)
}


def test_preserve_eq() -> None:
    toks = tokenize("Eq. 3.14 is the key")
    assert "eq._3.14" in toks


def test_preserve_sae_iso() -> None:
    assert "sae_j2464" in tokenize("reference SAE J2464 for this")
    assert "iso_26262" in tokenize("ISO 26262 compliance")


def test_preserve_figure() -> None:
    assert "figure_7.2" in tokenize("see Figure 7.2 above")


def test_camelcase_num_pattern() -> None:
    toks = tokenize("running on RTX3060 gpus")
    assert "rtx3060" in toks


def test_stopwords_filtered() -> None:
    toks = tokenize("the car and the driver")
    for sw in ("the", "and"):
        assert sw not in toks


def test_min_token_length() -> None:
    assert "a" not in tokenize("a big cat")


def test_deterministic_across_calls() -> None:
    t1 = tokenize("SAE J2464 Eq. 3.14 rev.2")
    t2 = tokenize("SAE J2464 Eq. 3.14 rev.2")
    assert t1 == t2


def test_max_tokens_per_doc_cap() -> None:
    """A 20K-word document must be capped at 16_384 tokens."""
    long = "word " * 20_000
    toks = tokenize(long)
    assert len(toks) == 16_384


# --- PERSISTENCE CONTRACT: blake2b-4 token-id stability -------------------- #


def test_stable_token_id_reproducible_across_processes() -> None:
    """The blake2b-4 id must be stable across processes / Python versions."""
    tok = "eq._3.14"
    local = _stable_token_id(tok)
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from vo_rag.core.retrieval.bm25_tokenizer import _stable_token_id;"
            " print(_stable_token_id('eq._3.14'))",
        ],
        text=True,
    ).strip()
    assert int(out) == local


def test_stable_token_id_uses_blake2b_digest_size_4() -> None:
    """Direct contract: ``int.from_bytes(blake2b(tok, digest_size=4), 'big')``."""
    tok = "suspension"
    expected = int.from_bytes(
        hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest(),
        "big",
    )
    assert _stable_token_id(tok) == expected


def test_persistence_contract_hardcoded_ids() -> None:
    """Hard-coded token-id values. Any drift = re-ingest everything."""
    for tok, expected_id in EXPECTED_TOKEN_IDS.items():
        actual = _stable_token_id(tok)
        assert actual == expected_id, (
            f"BM25 token-id drift for {tok!r}: "
            f"expected {expected_id}, got {actual}. "
            f"Changing this means re-ingesting every Qdrant point."
        )


def test_persistence_contract_persists_across_subprocess() -> None:
    """A fresh Python process must produce the same ids for the pinned tokens."""
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from vo_rag.core.retrieval.bm25_tokenizer import _stable_token_id;"
            " print(','.join(str(_stable_token_id(t)) "
            " for t in ['eq._3.14','sae_j2464','iso_26262','figure_7.2','rtx3060']))",
        ],
        text=True,
    ).strip()
    actual = [int(x) for x in out.split(",")]
    expected = [
        EXPECTED_TOKEN_IDS[t]
        for t in (
            "eq._3.14",
            "sae_j2464",
            "iso_26262",
            "figure_7.2",
            "rtx3060",
        )
    ]
    assert actual == expected, (
        f"BM25 token-id drift across processes: {actual} != {expected}. Re-ingest required."
    )


def test_tokens_to_bm25_sparse_frequencies() -> None:
    toks = ["alpha", "beta", "alpha", "gamma", "alpha"]
    sp = tokens_to_bm25_sparse(toks)
    assert sp[_stable_token_id("alpha")] == 3.0
    assert sp[_stable_token_id("beta")] == 1.0
    assert sp[_stable_token_id("gamma")] == 1.0


def test_tokens_to_bm25_sparse_uses_stable_ids() -> None:
    """Sparse payload ids must equal the hard-coded persistence ids."""
    toks = ["eq._3.14", "sae_j2464"]
    sp = tokens_to_bm25_sparse(toks)
    assert sp[EXPECTED_TOKEN_IDS["eq._3.14"]] == 1.0
    assert sp[EXPECTED_TOKEN_IDS["sae_j2464"]] == 1.0
