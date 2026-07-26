"""llmverify -- check whether a custom LLM provider serves the model it claims.

    from llmverify import verify_provider
    result = await verify_provider(provider_config, run_config)

The command line entry point is ``llmverify``; see ``README.md``.

What this tool can and cannot establish is stated plainly in
``docs/limitations.md``, and the short version belongs here too: no purely
software-side method can *prove* which weights an endpoint ran. A provider that
routes a fraction of traffic to the genuine model defeats every statistical test
in this package, and only hardware attestation closes that gap. What llmverify
does is make substitution expensive to hide and cheap to detect, and show its
work when it accuses someone.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "Evidence",
    "ProviderConfig",
    "RunConfig",
    "RunResult",
    "Verdict",
    "__version__",
    "verify_provider",
]


def __getattr__(name: str) -> object:
    # Lazy re-exports keep `import llmverify` cheap: the CLI's --help path
    # should not pay for httpx, pydantic model building and the probe registry.
    if name in ("ProviderConfig", "RunConfig"):
        from . import config

        return getattr(config, name)
    if name in ("Evidence", "Verdict"):
        from . import evidence

        return getattr(evidence, name)
    if name == "RunResult":
        from .results import RunResult

        return RunResult
    if name == "verify_provider":
        from .runner import verify_provider

        return verify_provider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
