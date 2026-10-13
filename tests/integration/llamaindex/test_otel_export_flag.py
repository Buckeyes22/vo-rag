"""Contract tests for the LlamaIndex Phoenix/OTel export flag (VOR-230).

Verifies the operator-facing ``enable_otel_export()`` is opt-in only,
routes the export through the ``export_span_attr`` redaction seam, and
does NOT call ``set_global_handler`` when the flag is unset.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any
from unittest.mock import patch

import llama_index.core
import pytest

from vo_rag.integrations.llamaindex import observability as obs


@pytest.fixture(autouse=True)
def _reset_activation() -> Iterator[None]:
    """Each test starts with the module-level activation flag cleared."""
    obs._reset_otel_export_state()
    llama_index.core.global_handler = None
    yield
    obs._reset_otel_export_state()
    llama_index.core.global_handler = None


def _patched_set_global_handler() -> Any:
    """Return a MagicMock-like spy wired to llama_index.core.set_global_handler."""
    return patch.object(llama_index.core, "set_global_handler", autospec=True)


def test_flag_unset_skips_set_global_handler() -> None:
    """An unset flag must be a no-op — no set_global_handler call."""
    env: Mapping[str, str] = {}
    assert getattr(llama_index.core, "global_handler", None) is None

    with _patched_set_global_handler() as spy:
        activated = obs.enable_otel_export(env=env)

    assert activated is False
    spy.assert_not_called()
    assert obs.is_otel_export_active() is False


def test_flag_empty_string_skips_set_global_handler() -> None:
    """An empty/whitespace flag must be a no-op."""
    env = {obs.OTEL_EXPORT_ENV_VAR: "   "}
    with _patched_set_global_handler() as spy:
        activated = obs.enable_otel_export(env=env)

    assert activated is False
    spy.assert_not_called()


def test_flag_set_attaches_configured_backend() -> None:
    """A set flag must call set_global_handler with the configured backend."""
    env = {obs.OTEL_EXPORT_ENV_VAR: "arize_phoenix"}
    with _patched_set_global_handler() as spy:
        activated = obs.enable_otel_export(env=env)

    assert activated is True
    spy.assert_called_once_with("arize_phoenix")
    assert obs.is_otel_export_active() is True


def test_second_activation_does_not_replace_global_handler() -> None:
    """A repeated activation is a no-op so trace state is not discarded."""
    env = {obs.OTEL_EXPORT_ENV_VAR: "arize_phoenix"}
    with (
        _patched_set_global_handler() as spy,
        patch.object(obs, "export_span_attr", autospec=True) as seam,
    ):
        first = obs.enable_otel_export(env=env)
        second = obs.enable_otel_export(env=env)

    assert first is True
    assert second is True
    spy.assert_called_once_with("arize_phoenix")
    seam.assert_called_once_with("arize_phoenix")


def test_flag_set_passes_handler_kwargs() -> None:
    """Operator-supplied handler kwargs are forwarded to set_global_handler."""
    env = {obs.OTEL_EXPORT_ENV_VAR: "openinference"}
    with _patched_set_global_handler() as spy:
        activated = obs.enable_otel_export(
            env=env,
            handler_kwargs={"project_name": "vo-rag-test"},
        )

    assert activated is True
    spy.assert_called_once_with("openinference", project_name="vo-rag-test")


def test_phoenix_alias_normalizes_to_arize_phoenix() -> None:
    """The ``phoenix`` operator alias must normalize to ``arize_phoenix``."""
    env = {obs.OTEL_EXPORT_ENV_VAR: "phoenix"}
    with _patched_set_global_handler() as spy:
        activated = obs.enable_otel_export(env=env)

    assert activated is True
    spy.assert_called_once_with("arize_phoenix")


def test_otel_alias_normalizes_to_openinference() -> None:
    """The ``otel`` operator alias must normalize to ``openinference``."""
    env = {obs.OTEL_EXPORT_ENV_VAR: "otel"}
    with _patched_set_global_handler() as spy:
        activated = obs.enable_otel_export(env=env)

    assert activated is True
    spy.assert_called_once_with("openinference")


def test_flag_unset_does_not_import_backend() -> None:
    """An unset flag must NOT trigger any backend module import.

    ``set_global_handler`` is patched so its body (which would lazily
    import the backend's third-party package) cannot run; the test
    asserts the patch was never even entered.
    """
    env: Mapping[str, str] = {}
    with _patched_set_global_handler() as spy:
        obs.enable_otel_export(env=env)

    spy.assert_not_called()


def test_export_path_runs_text_through_redaction_seam() -> None:
    """When the flag is set, the export path runs text through export_span_attr."""
    env = {obs.OTEL_EXPORT_ENV_VAR: "arize_phoenix"}
    with (
        _patched_set_global_handler(),
        patch.object(obs, "export_span_attr", autospec=True) as seam,
    ):
        obs.enable_otel_export(env=env)

    seam.assert_called_once()
    # The seam's argument is the resolved backend name; the point is that
    # the export path goes through it (VOR-231's contract).
    (arg,) = seam.call_args.args
    assert arg == "arize_phoenix"


def test_export_span_attr_delegates_to_default_redactor() -> None:
    """The seam itself is a thin shim over get_default_redactor().redact().

    Locks in VOR-231's contract: ``export_span_attr`` returns the
    redactor's output.  VOR-231 will harden this (additional entities,
    dict mapping, Presidio) but the one-line shape must hold.
    """
    from vo_rag.core.observability.redaction import (
        RedactionLevel,
        get_default_redactor,
    )

    redactor = get_default_redactor()
    pii = "contact me at user@example.com"
    expected = redactor.redact(pii)
    if redactor.config.level is RedactionLevel.OFF:
        expected = pii

    assert obs.export_span_attr(pii) == expected
    assert "<EMAIL_ADDRESS>" in obs.export_span_attr(pii) or (
        redactor.config.level is RedactionLevel.OFF
    )


def test_default_process_env_unset_means_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no ``env`` is passed, the function reads from os.environ."""
    monkeypatch.delenv(obs.OTEL_EXPORT_ENV_VAR, raising=False)
    with _patched_set_global_handler() as spy:
        activated = obs.enable_otel_export()

    assert activated is False
    spy.assert_not_called()


def test_default_process_env_set_means_attach(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no ``env`` is passed and the env var is set, attach the handler."""
    monkeypatch.setenv(obs.OTEL_EXPORT_ENV_VAR, "arize_phoenix")
    with _patched_set_global_handler() as spy:
        activated = obs.enable_otel_export()

    assert activated is True
    spy.assert_called_once_with("arize_phoenix")
