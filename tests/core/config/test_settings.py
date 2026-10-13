"""Contract tests for src/vo_rag/core/config/settings.py."""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from vo_rag.core.config.settings import (
    Settings,
    _interpolate,
    get_settings,
    reset_settings_cache,
)

MINIMAL_YAML = textwrap.dedent("""
    api:
      host: "0.0.0.0"
      port: 8000
      api_key: "${VORAG_API_KEY:devkey}"
      cors_origins:
        - "${EXTRA_ORIGIN:http://localhost:5173}"
      request_timeout_s: 60
    database:
      url: "${VORAG_POSTGRES_URL:postgresql://localhost/vorag}"
      migrations_dir: "src/vo_rag/core/db/migrations"
    qdrant:
      url: "http://localhost:6335"
    garage:
      endpoint: "http://localhost:3900"
    embedding:
      batch_url: "http://localhost:8090"
      query_url: "http://localhost:8090"
    vlm:
      url: "http://localhost:8080"
      enabled: false
    serverless:
      api_key: "test-runpod-key"
      vlm_endpoint_id: "vlm-test-id"
      text_endpoint_id: "text-test-id"
    reranker:
      model_path: "models/gte"
    triage:
      local_model_url: "http://localhost:11434"
    observability:
      langfuse_host: "http://localhost:3101"
    compliance:
      mode: "monitoring"
""")


@pytest.fixture
def settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    p = tmp_path / "settings.yaml"
    p.write_text(MINIMAL_YAML, encoding="utf-8")
    monkeypatch.setenv("VORAG_SETTINGS_PATH", str(p))
    reset_settings_cache()
    yield p
    reset_settings_cache()


# ---------- _interpolate unit tests ----------


def test_interp_substitutes_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOO", "bar")
    assert _interpolate("${FOO}") == "bar"
    assert _interpolate("x-${FOO}-y") == "x-bar-y"


def test_interp_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE", raising=False)
    assert _interpolate("${NOPE:hello}") == "hello"
    assert _interpolate("${NOPE:}") == ""
    assert _interpolate("${NOPE}") == ""  # empty when no default


def test_interp_walks_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOO", "X")
    out = _interpolate({"a": ["${FOO}", "z"], "b": {"c": "${FOO}-${MISSING:def}"}})
    assert out == {"a": ["X", "z"], "b": {"c": "X-def"}}


def test_interp_passthrough_for_non_strings() -> None:
    assert _interpolate(42) == 42
    assert _interpolate(True) is True
    assert _interpolate(None) is None


# ---------- get_settings / cache / reset ----------


def test_get_settings_loads_file(settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "hunter2")
    reset_settings_cache()
    s = get_settings()
    assert isinstance(s, Settings)
    assert s.api.api_key == "hunter2"
    assert s.answer.enabled is False
    assert s.answer.quality_model == "claude-sonnet-4-6"


def test_get_settings_default_when_env_unset(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VORAG_API_KEY", raising=False)
    reset_settings_cache()
    s = get_settings()
    assert s.api.api_key == "devkey"


def test_get_settings_is_cached(settings_file: Path) -> None:
    a = get_settings()
    b = get_settings()
    assert a is b


def test_reset_cache_picks_up_env_change(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VORAG_API_KEY", "first")
    reset_settings_cache()
    assert get_settings().api.api_key == "first"
    monkeypatch.setenv("VORAG_API_KEY", "second")
    # Without reset, cached value holds.
    assert get_settings().api.api_key == "first"
    reset_settings_cache()
    assert get_settings().api.api_key == "second"


def test_list_items_interpolate(settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTRA_ORIGIN", "http://localhost:3080")
    reset_settings_cache()
    s = get_settings()
    assert "http://localhost:3080" in s.api.cors_origins


def test_unknown_top_level_key_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(MINIMAL_YAML + "\nbogus: {}\n", encoding="utf-8")
    monkeypatch.setenv("VORAG_SETTINGS_PATH", str(bad))
    reset_settings_cache()
    with pytest.raises(ValidationError):
        get_settings()


def test_unknown_nested_key_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # MINIMAL_YAML is dedented: sections at col 0, fields at col 2.
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        MINIMAL_YAML.replace(
            "compliance:\n  mode:",
            'compliance:\n  not_a_field: "x"\n  mode:',
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VORAG_SETTINGS_PATH", str(bad))
    reset_settings_cache()
    with pytest.raises(ValidationError):
        get_settings()


def test_missing_required_section_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Drop the `database:` section (header at col 0 + its indented children).
    out: list[str] = []
    skipping = False
    for line in MINIMAL_YAML.splitlines():
        if line.startswith("database:"):
            skipping = True
            continue
        if skipping and line and not line[0].isspace():
            skipping = False
        if not skipping:
            out.append(line)
    bad = tmp_path / "bad.yaml"
    bad.write_text("\n".join(out), encoding="utf-8")
    monkeypatch.setenv("VORAG_SETTINGS_PATH", str(bad))
    reset_settings_cache()
    with pytest.raises(ValidationError):
        get_settings()


# ---------- DSI serverless role + backward-compat ----------


DSI_YAML = textwrap.dedent("""
    api:
      host: "0.0.0.0"
      port: 8000
      api_key: "devkey"
      request_timeout_s: 60
    database:
      url: "postgresql://localhost/vorag"
      migrations_dir: "src/vo_rag/core/db/migrations"
    qdrant:
      url: "http://localhost:6335"
    garage:
      endpoint: "http://localhost:3900"
    embedding:
      batch_url: "http://localhost:8090"
      query_url: "http://localhost:8090"
    vlm:
      url: "http://localhost:8080"
      enabled: false
    serverless:
      api_key: "test-runpod-key"
      vlm_endpoint_id: "${SERVERLESS_VLM_ENDPOINT_ID:vlm-endpoint}"
      vlm_model_name: "${SERVERLESS_VLM_MODEL_NAME:qwen3-vl-32b}"
      text_endpoint_id: "${SERVERLESS_TEXT_ENDPOINT_ID:text-endpoint}"
      text_model_name: "${SERVERLESS_TEXT_MODEL_NAME:qwen3-6-27b}"
      dsi_endpoint_id: "${SERVERLESS_DSI_ENDPOINT_ID:dsi-endpoint}"
      dsi_model_name: "${SERVERLESS_DSI_MODEL_NAME:qwen3-dsi-7b}"
      exp_endpoint_id: "${SERVERLESS_EXP_ENDPOINT_ID:exp-endpoint}"
      exp_model_name: "${SERVERLESS_EXP_MODEL_NAME:qwen3-exp-9b}"
      base_url_override: "${SERVERLESS_BASE_URL_OVERRIDE:}"
    reranker:
      model_path: "models/gte"
    triage:
      local_model_url: "http://localhost:11434"
    observability:
      langfuse_host: "http://localhost:3101"
    compliance:
      mode: "monitoring"
""")


@pytest.fixture
def dsi_settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    p = tmp_path / "dsi.yaml"
    p.write_text(DSI_YAML, encoding="utf-8")
    monkeypatch.setenv("VORAG_SETTINGS_PATH", str(p))
    reset_settings_cache()
    yield p
    reset_settings_cache()


def test_serverless_dsi_role_base_url_and_model_name(
    dsi_settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SERVERLESS_DSI_ENDPOINT_ID / SERVERLESS_DSI_MODEL_NAME flow into
    ``base_url('dsi')`` / ``model_name('dsi')`` the same way the other serverless
    roles do."""
    monkeypatch.setenv("SERVERLESS_DSI_ENDPOINT_ID", "env-dsi")
    monkeypatch.setenv("SERVERLESS_DSI_MODEL_NAME", "env-dsi-model")
    reset_settings_cache()
    s = get_settings()
    assert s.serverless.base_url("dsi") == ("https://api.runpod.ai/v2/env-dsi/openai/v1")
    assert s.serverless.model_name("dsi") == "env-dsi-model"


def test_serverless_dsi_role_from_yaml_when_env_unset(dsi_settings_file: Path) -> None:
    """When SERVERLESS_DSI_* env vars are unset, the YAML literals take over
    via ${SERVERLESS_DSI_*:default} interpolation."""
    s = get_settings()
    assert s.serverless.base_url("dsi") == "https://api.runpod.ai/v2/dsi-endpoint/openai/v1"
    assert s.serverless.model_name("dsi") == "qwen3-dsi-7b"


def test_serverless_text_vlm_exp_remain_backward_compatible(
    dsi_settings_file: Path,
) -> None:
    """Existing text / vlm / exp roles keep working — exp is
    retained only for backward-compat until it is deliberately removed."""
    s = get_settings()
    assert s.serverless.base_url("text") == "https://api.runpod.ai/v2/text-endpoint/openai/v1"
    assert s.serverless.base_url("vlm") == "https://api.runpod.ai/v2/vlm-endpoint/openai/v1"
    assert s.serverless.base_url("exp") == "https://api.runpod.ai/v2/exp-endpoint/openai/v1"
    assert s.serverless.model_name("text") == "qwen3-6-27b"
    assert s.serverless.model_name("vlm") == "qwen3-vl-32b"
    assert s.serverless.model_name("exp") == "qwen3-exp-9b"


def test_serverless_base_url_override_routes_text_and_dsi_verbatim(
    dsi_settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dsi_settings_file.write_text(
        DSI_YAML.replace(
            'text_endpoint_id: "${SERVERLESS_TEXT_ENDPOINT_ID:text-endpoint}"',
            'text_endpoint_id: ""',
        ).replace(
            'dsi_endpoint_id: "${SERVERLESS_DSI_ENDPOINT_ID:dsi-endpoint}"',
            'dsi_endpoint_id: ""',
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SERVERLESS_BASE_URL_OVERRIDE", "http://lin:8000/v1")
    monkeypatch.delenv("SERVERLESS_TEXT_ENDPOINT_ID", raising=False)
    monkeypatch.delenv("SERVERLESS_DSI_ENDPOINT_ID", raising=False)
    reset_settings_cache()

    s = get_settings()

    assert s.serverless.base_url("text") == "http://lin:8000/v1"
    assert s.serverless.base_url("dsi") == "http://lin:8000/v1"


def test_serverless_base_url_ignore_override_uses_runpod_template(
    dsi_settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SERVERLESS_BASE_URL_OVERRIDE", "http://lin:8000/v1")
    reset_settings_cache()

    s = get_settings()

    assert s.serverless.base_url("text") == "http://lin:8000/v1"
    assert (
        s.serverless.base_url("text", ignore_override=True)
        == "https://api.runpod.ai/v2/text-endpoint/openai/v1"
    )


def test_serverless_base_url_empty_override_keeps_endpoint_template(
    dsi_settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SERVERLESS_BASE_URL_OVERRIDE", "")
    reset_settings_cache()

    s = get_settings()

    assert s.serverless.base_url("text") == "https://api.runpod.ai/v2/text-endpoint/openai/v1"
    assert s.serverless.base_url("dsi") == "https://api.runpod.ai/v2/dsi-endpoint/openai/v1"


def test_serverless_dsi_role_missing_endpoint_raises(
    dsi_settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset dsi_endpoint_id (YAML defaulting to empty) must surface the same
    configuration error as the other serverless roles — fail loud, not silent."""
    # Strip dsi_endpoint_id from the YAML so the interpolated value is "".
    bad = dsi_settings_file.with_name("dsi_missing.yaml")
    bad.write_text(
        DSI_YAML.replace(
            'dsi_endpoint_id: "${SERVERLESS_DSI_ENDPOINT_ID:dsi-endpoint}"',
            'dsi_endpoint_id: ""',
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VORAG_SETTINGS_PATH", str(bad))
    reset_settings_cache()
    s = get_settings()
    with pytest.raises(ValueError, match="dsi"):
        s.serverless.base_url("dsi")


def test_serverless_unknown_role_raises(dsi_settings_file: Path) -> None:
    s = get_settings()
    with pytest.raises(KeyError):
        s.serverless.base_url("not-a-role")
    with pytest.raises(KeyError):
        s.serverless.model_name("not-a-role")


def test_settings_has_no_acquisition_section() -> None:
    from vo_rag.core.config.settings import Settings

    assert "books_acquisition" not in Settings.model_fields


def test_shipped_settings_yaml_loads_without_acquisition() -> None:
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
    assert "books_acquisition" not in raw


_DEAD_SECTIONS = ("hatchet", "github", "repo", "training", "web_search", "ocr")


def test_shipped_settings_yaml_has_no_dead_sections() -> None:
    from pathlib import Path

    import yaml

    from vo_rag.core.config.settings import Settings

    raw = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
    for section in _DEAD_SECTIONS:
        assert section not in Settings.model_fields, section
        assert section not in raw, section
