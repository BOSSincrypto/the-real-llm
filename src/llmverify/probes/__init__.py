"""Probe framework: context, budget, registry.

A probe is a self-contained experiment that turns observations of an endpoint
into :class:`~llmverify.evidence.Evidence`. Probes are grouped into layers:

===== ==================================================================
Layer Content
===== ==================================================================
0     Metadata only. Free -- no generation. Echoed model id, response id
      shape, usage object shape, catalogue entries, response headers.
1     Cheap behavioural fingerprints. Seconds and fractions of a cent:
      parameter-support matrix, tokenizer ratios, token accounting,
      knowledge cutoff, thinking-block signatures, self-identification.
2     Capability probes. Long context, tool calling, structured output,
      vision, multilingual behaviour, throughput and price sanity.
3     Statistical. Logprob-distribution tests where exposed, and
      accuracy benchmarks with sequential early stopping.
===== ==================================================================

A probe must never raise for an expected negative result. "The endpoint has no
logprobs" is an ``UNSUPPORTED`` evidence item, not a crash.
"""

from __future__ import annotations

import abc
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from ..config import ProviderConfig, RunConfig
from ..errors import BudgetExhausted
from ..evidence import Evidence, EvidenceStatus

if TYPE_CHECKING:
    from ..adapters.base import Adapter
    from ..reference.schema import ModelRecord, ReferenceSnapshot

__all__ = [
    "Budget",
    "Probe",
    "ProbeContext",
    "all_probes",
    "get_probe",
    "register_probe",
    "select_probes",
]


@dataclass(slots=True)
class Budget:
    """Tracks spend and stops the run when a ceiling is reached.

    Cost is *estimated* from reference pricing, because providers do not report
    it. When pricing is unknown the estimate is zero and only the sample and
    wall-clock ceilings bind.
    """

    max_cost_usd: float | None = None
    max_samples: int | None = None
    max_wall_s: float | None = None
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None

    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    samples: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def charge(
        self, input_tokens: int | None, output_tokens: int | None, *, samples: int = 1
    ) -> float:
        """Record usage and return the estimated marginal cost."""
        tin, tout = input_tokens or 0, output_tokens or 0
        self.input_tokens += tin
        self.output_tokens += tout
        self.samples += samples
        cost = 0.0
        if self.price_in_per_mtok is not None:
            cost += tin / 1_000_000 * self.price_in_per_mtok
        if self.price_out_per_mtok is not None:
            cost += tout / 1_000_000 * self.price_out_per_mtok
        self.spent_usd += cost
        return cost

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def exhausted(self) -> bool:
        return (
            (self.max_cost_usd is not None and self.spent_usd >= self.max_cost_usd)
            or (self.max_samples is not None and self.samples >= self.max_samples)
            or (self.max_wall_s is not None and self.elapsed_s >= self.max_wall_s)
        )

    @property
    def remaining_samples(self) -> int | None:
        if self.max_samples is None:
            return None
        return max(0, self.max_samples - self.samples)

    def check(self) -> None:
        if self.exhausted:
            raise BudgetExhausted(
                f"budget exhausted: ${self.spent_usd:.4f} / {self.samples} samples / "
                f"{self.elapsed_s:.0f}s"
            )


@dataclass(slots=True)
class ProbeContext:
    """Everything a probe needs, and nothing it does not."""

    adapter: Adapter
    provider: ProviderConfig
    run: RunConfig
    budget: Budget
    snapshot: ReferenceSnapshot | None = None
    reference: ModelRecord | None = None
    #: First-party endpoint for A/B comparison, when the user supplied keys.
    baseline: Adapter | None = None
    baseline_provider: ProviderConfig | None = None
    #: Scratch space shared between probes within one run, so that e.g. the
    #: metadata probe's warm-up response can be reused instead of re-paid for.
    shared: dict[str, Any] = field(default_factory=dict)

    def rng(self, salt: str) -> random.Random:
        """A deterministic RNG namespaced by ``salt``.

        Namespacing matters: probes must not consume from a shared stream, or
        skipping one probe would change every later probe's sampling.
        """
        return random.Random(f"{self.run.seed}:{salt}")

    @property
    def has_baseline(self) -> bool:
        return self.baseline is not None

    def skipped(self, probe: str, reason: str, *, family: str = "misc") -> Evidence:
        return Evidence(
            probe=probe,
            label="skipped",
            llr=0.0,
            family=family,
            status=EvidenceStatus.SKIPPED,
            detail=reason,
        )

    def unsupported(self, probe: str, reason: str, *, family: str = "misc") -> Evidence:
        return Evidence(
            probe=probe,
            label="unsupported",
            llr=0.0,
            family=family,
            status=EvidenceStatus.UNSUPPORTED,
            detail=reason,
        )


class Probe(abc.ABC):
    """One experiment against an endpoint."""

    name: ClassVar[str]
    layer: ClassVar[int]
    family: ClassVar[str] = "misc"
    #: One sentence shown in ``llmverify probes`` and in reports.
    description: ClassVar[str] = ""
    #: Rough number of generation requests, for budget planning and ordering.
    estimated_requests: ClassVar[int] = 1
    #: Probes with a lower order run first. Cheap and decisive probes go early
    #: so that an obvious substitution can short-circuit an expensive run.
    order: ClassVar[int] = 100

    def applicable(self, ctx: ProbeContext) -> bool:
        """Whether this probe can say anything useful about this endpoint."""
        return True

    @abc.abstractmethod
    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        """Execute and return evidence. Must not raise for expected negatives."""


_PROBES: dict[str, type[Probe]] = {}


def register_probe(cls: type[Probe]) -> type[Probe]:
    """Register a probe class. Usable as a decorator."""
    _PROBES[cls.name] = cls
    return cls


def _import_builtin_probes() -> None:
    from . import (  # noqa: F401
        api_surface,
        benchmark,
        determinism,
        evasion,
        knowledge_cutoff,
        logprobs,
        long_context,
        metadata,
        multilingual,
        performance,
        self_report,
        structured_output,
        thinking_signature,
        token_accounting,
        tokenizer,
        tool_calling,
        vision,
    )


def all_probes() -> dict[str, type[Probe]]:
    """Every registered probe, built-in and plugin."""
    from importlib.metadata import entry_points

    _import_builtin_probes()
    try:
        eps = entry_points(group="llmverify.probes")
    except TypeError:  # pragma: no cover
        eps = entry_points().get("llmverify.probes", [])  # type: ignore[assignment]
    for ep in eps:
        if ep.name in _PROBES:
            continue
        try:
            cls = ep.load()
        except Exception:
            continue
        if isinstance(cls, type) and issubclass(cls, Probe):
            _PROBES[cls.name] = cls
    return dict(_PROBES)


def get_probe(name: str) -> type[Probe]:
    from ..errors import ConfigError

    probes = all_probes()
    if name not in probes:
        raise ConfigError(f"unknown probe {name!r}; try `llmverify probes` to list them")
    return probes[name]


def select_probes(run: RunConfig) -> list[Probe]:
    """Instantiate the probes selected by a run config, in execution order."""
    registry = all_probes()
    if run.probes:
        chosen = [get_probe(n) for n in run.probes]
    else:
        chosen = [c for c in registry.values() if c.layer in run.layers]
    excluded = set(run.exclude_probes)
    chosen = [c for c in chosen if c.name not in excluded]
    return [c() for c in sorted(chosen, key=lambda c: (c.layer, c.order, c.name))]
