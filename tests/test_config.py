"""Configuration loading, environment expansion and secret redaction.

The redaction tests carry the most weight here. A provider error body routinely
echoes the request headers back, and every report path in this package runs free
text through :func:`~llmverify.config.redact` on the way out, so a key that
survives that function reaches a file a user is about to attach to a support
ticket.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from llmverify.config import (
    BudgetConfig,
    ProviderConfig,
    RunConfig,
    load_provider,
    load_run_config,
    redact,
    register_secret,
)
from llmverify.errors import ConfigError


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_load_provider_reads_a_minimal_file(tmp_path: Path) -> None:
    path = write(
        tmp_path / "provider.yaml",
        "name: acme\napi: openai\nbase_url: https://api.example.com/v1\nmodel: some-model\n",
    )
    provider = load_provider(path)
    assert provider.name == "acme"
    assert provider.model == "some-model"
    assert provider.target_model == "some-model"
    assert provider.max_concurrency == 4


def test_claimed_model_overrides_the_lookup_identity(tmp_path: Path) -> None:
    path = write(
        tmp_path / "provider.yaml",
        "name: acme\nmodel: opus5\nclaimed_model: claude-opus-5\n",
    )
    assert load_provider(path).target_model == "claude-opus-5"


def test_env_expansion_substitutes_into_nested_structures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOCK_HOST", "api.example.com")
    monkeypatch.setenv("MOCK_TENANT", "acme-42")
    path = write(
        tmp_path / "provider.yaml",
        "name: acme\n"
        "model: m\n"
        "base_url: https://${MOCK_HOST}/v1\n"
        "headers:\n"
        "  x-tenant: ${MOCK_TENANT}\n",
    )
    provider = load_provider(path)
    assert provider.base_url == "https://api.example.com/v1"
    assert provider.headers == {"x-tenant": "acme-42"}


def test_a_missing_variable_is_a_config_error_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MOCK_ABSENT", raising=False)
    path = write(tmp_path / "provider.yaml", "name: a\nmodel: m\nbase_url: ${MOCK_ABSENT}\n")
    with pytest.raises(ConfigError, match="MOCK_ABSENT"):
        load_provider(path)


def test_unknown_keys_are_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "provider.yaml", "name: a\nmodel: m\nnot_a_field: 1\n")
    with pytest.raises(ConfigError):
        load_provider(path)


def test_a_relative_base_url_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "provider.yaml", "name: a\nmodel: m\nbase_url: example.com/v1\n")
    with pytest.raises(ConfigError, match="absolute URL"):
        load_provider(path)


def test_missing_empty_and_non_mapping_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_provider(tmp_path / "nope.yaml")
    with pytest.raises(ConfigError, match="empty"):
        load_provider(write(tmp_path / "empty.yaml", ""))
    with pytest.raises(ConfigError, match="mapping"):
        load_provider(write(tmp_path / "list.yaml", "- a\n- b\n"))
    with pytest.raises(ConfigError, match="valid YAML"):
        load_provider(write(tmp_path / "bad.yaml", "a: [1,\n"))


def test_the_run_section_is_read_from_a_provider_file(tmp_path: Path) -> None:
    path = write(
        tmp_path / "provider.yaml",
        "name: a\n"
        "model: m\n"
        "run:\n"
        "  seed: 7\n"
        "  layers: [0, 1]\n"
        "  budget:\n"
        "    max_samples: 25\n",
    )
    # The provider loader drops the run section rather than choking on it.
    assert load_provider(path).model == "m"

    run = load_run_config(path)
    assert run.seed == 7
    assert run.layers == (0, 1)
    assert run.budget.max_samples == 25


def test_a_baseline_provider_nested_in_the_run_section_is_validated(tmp_path: Path) -> None:
    path = write(
        tmp_path / "run.yaml",
        "run:\n"
        "  baseline:\n"
        "    name: first-party\n"
        "    api: anthropic\n"
        "    model: claude-opus-5\n",
    )
    run = load_run_config(path)
    assert run.baseline is not None
    assert run.baseline.api == "anthropic"


def test_load_run_config_without_a_path_returns_the_defaults() -> None:
    run = load_run_config(None)
    assert run.layers == (0, 1, 2, 3)
    assert run.anti_evasion is True
    assert run.budget.alpha < run.budget.beta


def test_budget_bounds_are_enforced() -> None:
    with pytest.raises(ValueError):
        BudgetConfig(alpha=0.9)
    with pytest.raises(ValueError):
        BudgetConfig(min_effect_pp=0.0)
    with pytest.raises(ValueError):
        RunConfig(prior_odds=1.0, budget=BudgetConfig(beta=0.0))


def test_provider_configs_are_frozen() -> None:
    provider = ProviderConfig(name="a", model="m")
    with pytest.raises(ValidationError):
        provider.model = "other"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


def test_resolve_api_key_reads_the_environment_and_registers_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOCK_PROVIDER_KEY", "abcdefgh-super-secret-value-1234")
    provider = ProviderConfig(name="a", model="m", api_key_env="MOCK_PROVIDER_KEY")
    assert provider.resolve_api_key() == "abcdefgh-super-secret-value-1234"
    assert "abcdefgh-super-secret-value-1234" not in redact(
        "authorization failed for abcdefgh-super-secret-value-1234"
    )


def test_resolve_api_key_complains_about_the_variable_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MOCK_UNSET_KEY", raising=False)
    provider = ProviderConfig(name="a", model="m", api_key_env="MOCK_UNSET_KEY")
    with pytest.raises(ConfigError, match=r"\$MOCK_UNSET_KEY"):
        provider.resolve_api_key()


def test_no_api_key_env_means_no_key() -> None:
    assert ProviderConfig(name="a", model="m").resolve_api_key() is None


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def test_registered_secrets_are_replaced_wherever_they_appear() -> None:
    register_secret("hunter2-hunter2-hunter2")
    text = "prefix hunter2-hunter2-hunter2 suffix hunter2-hunter2-hunter2"
    cleaned = redact(text)
    assert "hunter2-hunter2-hunter2" not in cleaned
    assert cleaned.count("***REDACTED***") == 2


def test_short_strings_are_never_registered() -> None:
    """A seven-character 'secret' would redact half the English language."""
    register_secret("model")
    register_secret(None)
    assert redact("the model answered") == "the model answered"


def test_bearer_tokens_are_caught_even_when_never_registered() -> None:
    cleaned = redact("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in cleaned
    assert "Bearer ***REDACTED***" in cleaned


def test_vendor_key_shapes_are_caught_by_pattern() -> None:
    for key in (
        "sk-abcdefghijklmnopqrstuvwxyz",
        "xai-abcdefghijklmnopqrstuvwxyz",
        "gsk-abcdefghijklmnopqrstuvwxyz",
    ):
        assert key not in redact(f"the endpoint echoed {key} back at us")


def test_redact_removes_a_key_from_a_nested_structure() -> None:
    """The paths that carry a key are nested, so this is the shape that matters.

    :func:`redact` works on text, and every emitter -- ``RunResult.to_dict``, the
    console reporter, the HTML writer -- walks its structure and passes the
    strings through it. Serialising first and redacting the whole document is
    the same operation seen from outside, and it is what a caller writing an
    artefact actually does.
    """
    register_secret("sk-nested-secret-value-0001")
    payload: dict[str, Any] = {
        "provider": {
            "headers": {"authorization": "Bearer sk-nested-secret-value-0001"},
            "notes": ["retry with sk-nested-secret-value-0001"],
        },
        "evidence": [{"detail": "upstream said: sk-nested-secret-value-0001"}],
    }
    cleaned = json.loads(redact(json.dumps(payload)))
    assert "sk-nested-secret-value-0001" not in json.dumps(cleaned)
    assert cleaned["provider"]["headers"]["authorization"].endswith("***REDACTED***")
    assert cleaned["evidence"][0]["detail"].endswith("***REDACTED***")


def test_redaction_leaves_ordinary_text_alone() -> None:
    text = "claude-opus-5 scored 91.8 on GPQA Diamond; 198 items is not enough."
    assert redact(text) == text
