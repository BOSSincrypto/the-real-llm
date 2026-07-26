"""Provider and run configuration.

Secrets are never stored in a config file. A config names an *environment
variable* (``api_key_env``); the value is resolved at runtime and redacted from
every report, log line and HTML artefact. This is enforced by
:func:`redact`, which every output path runs over free text before emitting it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ConfigError

__all__ = [
    "BudgetConfig",
    "ProviderConfig",
    "RunConfig",
    "load_provider",
    "load_run_config",
    "redact",
]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Populated as api keys are resolved, so reports can scrub them.
_SECRETS: set[str] = set()


def register_secret(value: str | None) -> None:
    """Mark a string as sensitive so :func:`redact` will strip it."""
    if value and len(value) >= 8:
        _SECRETS.add(value)


def redact(text: str) -> str:
    """Replace every registered secret in ``text`` with a stable placeholder.

    Also catches the common bearer-token shapes so that a key which arrived via
    some path we did not register (a header echoed back by the provider, say)
    still does not reach a report.
    """
    out = text
    for secret in _SECRETS:
        out = out.replace(secret, "***REDACTED***")
    out = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{12,}", r"\1***REDACTED***", out)
    out = re.sub(r"\b(sk|xai|gsk|api)-[A-Za-z0-9._\-]{12,}\b", "***REDACTED***", out)
    return out


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` references in strings."""
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            var = m.group(1)
            if var not in os.environ:
                raise ConfigError(f"config references ${{{var}}} but it is not set")
            return os.environ[var]

        return _ENV_PATTERN.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class ProviderConfig(BaseModel):
    """How to reach one endpoint, and what it claims to be serving."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Human label used in reports.")
    api: str = Field(
        default="openai",
        description="Adapter name: 'openai', 'anthropic', 'gemini', or a plugin.",
    )
    base_url: str | None = Field(
        default=None,
        description="Endpoint root. Defaults to the adapter's first-party URL.",
    )
    model: str = Field(description="Model identifier to send in requests.")
    claimed_model: str | None = Field(
        default=None,
        description=(
            "Canonical id of the model the provider claims to serve, used to look "
            "up reference data. Defaults to `model`. Set it when the provider uses "
            "its own naming, e.g. model='opus5' claimed_model='claude-opus-5'."
        ),
    )
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable holding the API key. Never the key itself.",
    )
    auth_scheme: Literal["bearer", "x-api-key", "query", "none"] | None = Field(
        default=None,
        description="Override how the key is presented. Defaults to the adapter's norm.",
    )
    headers: dict[str, str] = Field(default_factory=dict)
    query_params: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 180.0
    connect_timeout_s: float = 20.0
    max_concurrency: int = Field(default=4, ge=1, le=64)
    max_retries: int = Field(default=3, ge=0, le=10)
    verify_tls: bool = True
    proxy: str | None = None

    reasoning_effort: str | None = Field(
        default=None,
        description=(
            "Pinned reasoning effort. Reference benchmark scores are effort-"
            "conditional, so leaving this unset makes benchmark comparison "
            "approximate at best."
        ),
    )
    thinking_budget: int | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)

    #: Per-million-token prices the provider advertises. Used by the performance
    #: probe to compare advertised economics against first-party pricing.
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None

    @model_validator(mode="after")
    def _defaults(self) -> ProviderConfig:
        if self.base_url is not None and not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must be an absolute URL, got {self.base_url!r}")
        return self

    @property
    def target_model(self) -> str:
        """The canonical model id this endpoint is being held to."""
        return self.claimed_model or self.model

    def resolve_api_key(self) -> str | None:
        """Read the key from the environment and register it for redaction."""
        if not self.api_key_env:
            return None
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ConfigError(
                f"provider {self.name!r} expects the API key in ${self.api_key_env}, "
                "but that variable is empty or unset"
            )
        register_secret(key)
        return key


class BudgetConfig(BaseModel):
    """Ceilings for one run.

    ``max_cost_usd`` is enforced against the *estimated* spend computed from
    reference pricing, because providers do not report cost. It is therefore an
    approximation, and the runner errs on the side of stopping early.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_cost_usd: float | None = 0.50
    max_samples: int | None = 600
    max_wall_s: float | None = 900.0
    #: Type I error for the sequential test: probability of wrongly accusing a
    #: provider that is serving the real model. Set tight by default.
    alpha: float = Field(default=0.01, gt=0, lt=0.5)
    #: Type II error: probability of missing a genuine substitution.
    beta: float = Field(default=0.05, gt=0, lt=0.5)
    #: The accuracy shortfall, in percentage points, that counts as a real
    #: degradation worth detecting.
    min_effect_pp: float = Field(default=8.0, gt=0, le=100)


class RunConfig(BaseModel):
    """Everything about a verification run that is not the provider itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layers: tuple[int, ...] = (0, 1, 2, 3)
    probes: tuple[str, ...] = ()
    exclude_probes: tuple[str, ...] = ()
    benchmarks: tuple[str, ...] = ()
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    #: Seed for every randomised choice: item sampling, option shuffling,
    #: paraphrase selection, probe ordering. Makes a run reproducible.
    seed: int = 20260726
    anti_evasion: bool = True
    prior_odds: float = 1.0
    #: Optional first-party endpoint for side-by-side A/B comparison. When set,
    #: distribution and benchmark probes gain a measured baseline instead of
    #: relying only on published numbers.
    baseline: ProviderConfig | None = None
    cache_dir: Path = Field(default_factory=lambda: _default_cache_dir())
    hf_token_env: str = "HF_TOKEN"


def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "llmverify"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if raw is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return raw


def load_provider(path: str | Path) -> ProviderConfig:
    """Load a provider config from YAML, expanding ``${ENV_VAR}`` references."""
    data = _expand_env(_read_yaml(Path(path)))
    data.pop("run", None)
    try:
        return ProviderConfig.model_validate(data)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(f"{path}: {exc}") from exc


def load_run_config(path: str | Path | None) -> RunConfig:
    """Load run settings, optionally from the ``run:`` key of a provider file."""
    if path is None:
        return RunConfig()
    data = _expand_env(_read_yaml(Path(path)))
    section = data.get("run", data)
    if "baseline" in section and isinstance(section["baseline"], dict):
        section = dict(section)
        section["baseline"] = ProviderConfig.model_validate(section["baseline"])
    try:
        return RunConfig.model_validate(section)
    except Exception as exc:
        raise ConfigError(f"{path}: {exc}") from exc
