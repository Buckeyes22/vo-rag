"""Contract tests for BookRagConfig.

Verifies Pydantic v2 strict defaults and overrides.
"""

from __future__ import annotations

from vo_rag.pipelines.book_rag.config import BookRagConfig
from vo_rag.pipelines.book_rag.domain_vocab import DOMAIN_VOCABULARY


def test_default_sterilize_enabled_is_false() -> None:
    cfg = BookRagConfig()
    assert cfg.sterilize_enabled is False


def test_default_vlm_concurrency_is_1() -> None:
    cfg = BookRagConfig()
    assert cfg.vlm_concurrency == 1


def test_default_min_vlm_describe_yield_is_0_90() -> None:
    cfg = BookRagConfig()
    assert cfg.min_vlm_describe_yield == 0.90


def test_default_min_visual_sterile_yield_is_0_70() -> None:
    cfg = BookRagConfig()
    assert cfg.min_visual_sterile_yield == 0.70


def test_default_domain_vocab_is_motorsport_pack() -> None:
    cfg = BookRagConfig()
    assert cfg.domain_vocab == DOMAIN_VOCABULARY


def test_custom_domain_vocab() -> None:
    cfg = BookRagConfig(domain_vocab=["custom", "tags"])
    assert cfg.domain_vocab == ["custom", "tags"]


def test_sterilize_enabled_can_be_enabled_explicitly() -> None:
    cfg = BookRagConfig(sterilize_enabled=True)
    assert cfg.sterilize_enabled is True


def test_custom_concurrency() -> None:
    cfg = BookRagConfig(vlm_concurrency=8)
    assert cfg.vlm_concurrency == 8


def test_custom_yield_thresholds() -> None:
    cfg = BookRagConfig(min_vlm_describe_yield=0.85, min_visual_sterile_yield=0.60)
    assert cfg.min_vlm_describe_yield == 0.85
    assert cfg.min_visual_sterile_yield == 0.60
