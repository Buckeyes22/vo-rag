from __future__ import annotations

from collections.abc import Sequence

from vo_rag.core.chunking.semantic_drift import SemanticDriftSplitter


def _word_count(text: str) -> int:
    return len(text.split())


def _unused_embed_fn(texts: list[str]) -> list[Sequence[float]]:
    raise AssertionError(f"embed_fn should not be called for {texts!r}")


def test_two_tight_clusters_split_once_at_cluster_seam() -> None:
    text = (
        "Brake release settles corner entry. Trail braking keeps rotation stable. "
        "Steering traces confirm the apex. Tomato stock simmers overnight. "
        "Fresh herbs brighten the broth. Warm bowls finish dinner."
    )

    def embed(texts: list[str]) -> list[Sequence[float]]:
        return [
            [1.0, 0.0] if text.startswith(("Brake", "Trail", "Steering")) else [0.0, 1.0]
            for text in texts
        ]

    splitter = SemanticDriftSplitter(
        embed_fn=embed,
        counter=_word_count,
        max_tokens=100,
        breakpoint_percentile=50.0,
        buffer_size=0,
    )

    assert splitter.split_text(text, "[Book]") == [
        (
            "Brake release settles corner entry. Trail braking keeps rotation stable. "
            "Steering traces confirm the apex."
        ),
        "Tomato stock simmers overnight. Fresh herbs brighten the broth. Warm bowls finish dinner.",
    ]


def test_uniform_cluster_returns_single_piece_when_under_ceiling() -> None:
    text = (
        "Brake pressure builds confidence. Trail braking keeps rotation stable. "
        "Apex timing improves corner exit."
    )

    def embed(texts: list[str]) -> list[Sequence[float]]:
        return [[1.0, 0.0] for _ in texts]

    splitter = SemanticDriftSplitter(
        embed_fn=embed,
        counter=_word_count,
        max_tokens=100,
        buffer_size=0,
    )

    assert splitter.split_text(text, "[Book]") == [text]


def test_ceiling_repacks_single_cluster_at_sentence_boundaries() -> None:
    text = (
        "One two three. Four five six. Seven eight nine. "
        "Ten eleven twelve. Thirteen fourteen fifteen."
    )

    def embed(texts: list[str]) -> list[Sequence[float]]:
        return [[1.0, 0.0] for _ in texts]

    splitter = SemanticDriftSplitter(
        embed_fn=embed,
        counter=_word_count,
        max_tokens=7,
        buffer_size=0,
    )

    pieces = splitter.split_text(text, "[Book]")

    assert len(pieces) > 1
    assert all(_word_count(piece) <= 7 for piece in pieces)
    assert " ".join(pieces) == text


def test_empty_single_sentence_and_lone_overlong_sentence_are_not_embedded() -> None:
    splitter = SemanticDriftSplitter(
        embed_fn=_unused_embed_fn,
        counter=_word_count,
        max_tokens=3,
    )
    overlong = "This single sentence exceeds the configured token ceiling."

    assert splitter.split_text("", "[Book]") == []
    assert splitter.split_text("A single sentence.", "[Book]") == ["A single sentence."]
    assert splitter.split_text(overlong, "[Book]") == [overlong]


def test_split_text_is_deterministic_for_identical_inputs() -> None:
    text = (
        "Brake release settles corner entry. Trail braking keeps rotation stable. "
        "Tomato stock simmers overnight. Fresh herbs brighten the broth."
    )

    def embed(texts: list[str]) -> list[Sequence[float]]:
        return [[1.0, 0.0] if "brake" in text.lower() else [0.0, 1.0] for text in texts]

    splitter = SemanticDriftSplitter(
        embed_fn=embed,
        counter=_word_count,
        max_tokens=100,
        breakpoint_percentile=50.0,
        buffer_size=0,
    )

    first = splitter.split_text(text, "[Book]")
    second = splitter.split_text(text, "[Book]")

    assert second == first


def test_buffered_sentence_contexts_are_embedded_in_one_batch() -> None:
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> list[Sequence[float]]:
        calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    splitter = SemanticDriftSplitter(
        embed_fn=embed,
        counter=_word_count,
        max_tokens=100,
        buffer_size=1,
    )

    assert splitter.split_text("First sentence. Second sentence. Third sentence.", "[Book]") == [
        "First sentence. Second sentence. Third sentence."
    ]
    assert calls == [
        [
            "First sentence. Second sentence.",
            "First sentence. Second sentence. Third sentence.",
            "Second sentence. Third sentence.",
        ]
    ]
