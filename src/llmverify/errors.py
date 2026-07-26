"""Exception hierarchy.

Probes are expected to raise these rather than provider-library exceptions, so
that the runner can tell "the provider refused this capability" (informative,
and often evidence in itself) apart from "the network broke" (not evidence).
"""

from __future__ import annotations

__all__ = [
    "AdapterError",
    "AuthError",
    "BudgetExhausted",
    "ConfigError",
    "DatasetError",
    "LLMVerifyError",
    "ProviderError",
    "RateLimited",
    "ReferenceDataError",
    "UnsupportedCapability",
]


class LLMVerifyError(Exception):
    """Base class for everything this package raises."""


class ConfigError(LLMVerifyError):
    """A provider config or CLI invocation is invalid."""


class AdapterError(LLMVerifyError):
    """No adapter could be resolved, or an adapter is misconfigured."""


class ProviderError(LLMVerifyError):
    """The provider returned an error response.

    Carries the HTTP status and body so probes can reason about *why* a request
    was refused -- an OpenAI-shaped endpoint rejecting ``seed`` with a 400 says
    something quite different from one dropping it silently.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.headers = headers or {}


class AuthError(ProviderError):
    """401/403 from the provider."""


class RateLimited(ProviderError):
    """429 or a provider-specific throttling response."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kw: object) -> None:
        super().__init__(message, **kw)  # type: ignore[arg-type]
        self.retry_after = retry_after


class UnsupportedCapability(LLMVerifyError):
    """The endpoint cannot do what a probe needs (no logprobs, no vision, ...).

    This is a normal, expected outcome. It downgrades a probe to ``UNSUPPORTED``
    rather than failing the run.
    """


class BudgetExhausted(LLMVerifyError):
    """The run hit its token or cost ceiling."""


class DatasetError(LLMVerifyError):
    """A benchmark dataset could not be fetched, authenticated, or parsed."""


class ReferenceDataError(LLMVerifyError):
    """The reference snapshot is missing, malformed, or lacks the model."""
