"""The artefact a run produces.

Kept separate from :mod:`llmverify.evidence` because reporting, comparison and
serialisation all consume it, and none of them should have to import the
scoring machinery to do so.
"""

from __future__ import annotations

import datetime as dt
import platform
from dataclasses import asdict, dataclass, field
from typing import Any

from . import __version__
from .config import redact
from .evidence import Evidence, EvidenceStatus, Verdict, VerdictReport

__all__ = ["ProbeTiming", "RunResult", "compare_results"]


@dataclass(slots=True)
class ProbeTiming:
    probe: str
    layer: int
    duration_s: float
    requests: int
    status: str
    error: str | None = None


@dataclass(slots=True)
class RunResult:
    """Everything about one verification run, ready to serialise."""

    provider_name: str
    requested_model: str
    claimed_model: str
    api_family: str
    base_url: str | None
    verdict: VerdictReport
    started_at: dt.datetime
    finished_at: dt.datetime
    #: Which reference snapshot the thresholds came from.
    reference_as_of: dt.date | None = None
    reference_found: bool = False
    timings: list[ProbeTiming] = field(default_factory=list)
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    samples: int = 0
    seed: int = 0
    layers: tuple[int, ...] = ()
    baseline_used: bool = False
    warnings: list[str] = field(default_factory=list)
    tool_version: str = __version__

    @property
    def duration_s(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def evidence(self) -> list[Evidence]:
        return self.verdict.evidence

    @property
    def usable_evidence(self) -> list[Evidence]:
        return [e for e in self.evidence if e.status is EvidenceStatus.OK]

    def to_dict(self) -> dict[str, Any]:
        """A stable, secret-free JSON representation.

        Every free-text field passes through :func:`~llmverify.config.redact`,
        because provider error bodies routinely echo request headers back.
        """

        def clean(value: Any) -> Any:
            if isinstance(value, str):
                return redact(value)
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            return value

        return {
            "schema_version": 1,
            "tool": {"name": "llmverify", "version": self.tool_version},
            "run": {
                "started_at": self.started_at.isoformat(),
                "finished_at": self.finished_at.isoformat(),
                "duration_s": round(self.duration_s, 3),
                "seed": self.seed,
                "layers": list(self.layers),
                "platform": platform.platform(),
            },
            "provider": {
                "name": self.provider_name,
                "requested_model": self.requested_model,
                "claimed_model": self.claimed_model,
                "api_family": self.api_family,
                "base_url": clean(self.base_url),
                "baseline_used": self.baseline_used,
            },
            "reference": {
                "as_of": self.reference_as_of.isoformat() if self.reference_as_of else None,
                "model_found": self.reference_found,
            },
            "verdict": {
                "value": self.verdict.verdict.value,
                "probability_genuine": round(self.verdict.probability, 6),
                "total_log_odds_nats": round(self.verdict.total_llr, 4),
                "total_bans": round(self.verdict.total_bans, 4),
                "prior_odds": self.verdict.prior_odds,
                "family_totals": {
                    k: round(v, 4) for k, v in sorted(self.verdict.family_totals.items())
                },
                "notes": [clean(n) for n in self.verdict.notes],
            },
            "budget": {
                "estimated_cost_usd": round(self.spent_usd, 6),
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "samples": self.samples,
            },
            "evidence": [
                {
                    "probe": e.probe,
                    "label": e.label,
                    "family": e.family,
                    "status": e.status.value,
                    "llr_nats": round(e.llr, 4),
                    "bans": round(e.bans, 4),
                    "cap": round(e.cap, 4),
                    "detail": clean(e.detail),
                    "data": clean(e.data),
                    "cost_usd": round(e.cost_usd, 6),
                    "tokens": e.tokens,
                    "duration_s": round(e.duration_s, 3),
                }
                for e in self.evidence
            ],
            "timings": [clean(asdict(t)) for t in self.timings],
            "warnings": [clean(w) for w in self.warnings],
        }


def compare_results(results: list[RunResult]) -> dict[str, Any]:
    """Cross-provider summary for ``llmverify compare``.

    Only providers run against the same claimed model, seed and probe selection
    are directly comparable; mismatches are surfaced as warnings rather than
    silently averaged away.
    """
    warnings: list[str] = []
    models = {r.claimed_model for r in results}
    if len(models) > 1:
        warnings.append(
            "Providers were checked against different claimed models "
            f"({', '.join(sorted(models))}); per-provider verdicts remain valid but "
            "the side-by-side columns are not directly comparable."
        )
    seeds = {r.seed for r in results}
    if len(seeds) > 1:
        warnings.append("Runs used different seeds, so item samples differ between providers.")

    order = {
        Verdict.MATCH: 0,
        Verdict.LIKELY_MATCH: 1,
        Verdict.INCONCLUSIVE: 2,
        Verdict.LIKELY_MISMATCH: 3,
        Verdict.MISMATCH: 4,
        Verdict.EVASION: 5,
    }
    ranked = sorted(
        results, key=lambda r: (order[r.verdict.verdict], -r.verdict.probability)
    )
    return {
        "schema_version": 1,
        "claimed_models": sorted(models),
        "warnings": warnings,
        "providers": [
            {
                "name": r.provider_name,
                "claimed_model": r.claimed_model,
                "verdict": r.verdict.verdict.value,
                "probability_genuine": round(r.verdict.probability, 6),
                "total_bans": round(r.verdict.total_bans, 3),
                "estimated_cost_usd": round(r.spent_usd, 6),
                "duration_s": round(r.duration_s, 2),
            }
            for r in ranked
        ],
    }
