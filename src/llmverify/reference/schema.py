"""Schema for the reference snapshot.

The snapshot answers "what should the real model look like?". It is versioned
in git rather than fetched at runtime so that a verdict is reproducible and
reviewable: when the tool accuses a provider of substitution, a human can read
the diff that changed the threshold.

Every number carries its ``source`` and ``as_of`` date. A benchmark score
without stated evaluation conditions is close to meaningless -- labs report the
same benchmark at different reasoning efforts, harnesses and tool settings, and
the spread between those conditions routinely exceeds the gap between models.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BenchmarkScore",
    "FamilySignature",
    "ModelRecord",
    "Pricing",
    "ReferenceSnapshot",
    "TokenAccounting",
]

Confidence = Literal["primary", "independent", "secondary", "unverified"]


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None
    source: str | None = None


class BenchmarkScore(BaseModel):
    """One published score, with the conditions that make it comparable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: str
    score: float = Field(description="Accuracy in percent, or the benchmark's native unit.")
    unit: Literal["percent", "elo", "score"] = "percent"
    #: Reasoning effort / thinking budget the score was measured at. Comparing
    #: across efforts is invalid; the runner refuses to when this is unset and
    #: the provider pins a different effort.
    effort: str | None = None
    tools: bool | None = None
    shots: int | None = None
    #: Whether the publisher used benchmark-optimised settings.
    optimized: bool | None = None
    harness: str | None = None
    n_items: int | None = None
    source: str
    source_url: str | None = None
    as_of: dt.date
    confidence: Confidence = "secondary"
    notes: str | None = None


class TokenAccounting(BaseModel):
    """Deterministic token-overhead fingerprints.

    Anthropic publishes the exact token cost of the tool-use system prompt for
    each model. Sending one request with a single tool and reading back
    ``usage.input_tokens`` therefore identifies the model generation with no
    statistics, no logprobs and essentially no cost -- the strongest cheap
    signal available for the Claude family.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: System-prompt overhead in tokens with ``tool_choice`` auto/none.
    tool_overhead_auto: int | None = None
    #: ...and with ``tool_choice`` any/tool.
    tool_overhead_forced: int | None = None
    #: Overhead added by the provider's server-side bash tool, when offered.
    bash_tool_overhead: int | None = None
    #: Token count for the canonical probe string in
    #: ``llmverify.probes.tokenizer.CANONICAL_TEXT``. Distinguishes tokenizer
    #: generations within one vendor.
    canonical_text_tokens: int | None = None
    source: str | None = None
    as_of: dt.date | None = None


class FamilySignature(BaseModel):
    """What a first-party endpoint of a given API family looks like.

    Used to catch the crudest substitutions: an endpoint claiming to serve a
    Claude model while accepting ``seed`` and returning ``logprobs`` is not
    talking to Anthropic, whatever its ``model`` field says.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: str
    response_id_prefix: str | None = None
    supports_logprobs: bool | None = None
    supports_seed: bool | None = None
    has_system_fingerprint: bool | None = None
    #: Keys expected to appear in the usage object.
    usage_keys: tuple[str, ...] = ()
    #: Values ``finish_reason`` / ``stop_reason`` may take.
    finish_reasons: tuple[str, ...] = ()
    notes: str | None = None


class ModelRecord(BaseModel):
    """Everything the verifier knows about one claimed model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Canonical API model identifier.")
    vendor: str
    family: str = Field(description="API family: openai | anthropic | gemini | openai_compat.")
    display_name: str | None = None
    aliases: tuple[str, ...] = ()
    released: dt.date | None = None
    #: The cutoff the model *behaves* as if it has, which is what a probe can
    #: measure. Often earlier than the training cutoff the vendor states.
    knowledge_cutoff: dt.date | None = None
    training_cutoff: dt.date | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    reasoning: bool | None = None
    open_weights: bool = False
    modalities: tuple[str, ...] = ("text",)
    default_effort: str | None = None
    effort_ladder: tuple[str, ...] = ()
    pricing: Pricing | None = None
    token_accounting: TokenAccounting | None = None
    scores: tuple[BenchmarkScore, ...] = ()
    #: Median output tokens/second observed on first-party infrastructure, when
    #: known. Only ever weak evidence -- hardware differs legitimately.
    typical_output_tps: float | None = None
    notes: str | None = None

    def score_for(
        self, benchmark: str, *, effort: str | None = None, tools: bool | None = None
    ) -> BenchmarkScore | None:
        """Best matching published score, preferring exact condition matches."""
        candidates = [s for s in self.scores if s.benchmark == benchmark]
        if not candidates:
            return None
        if effort is not None:
            exact = [s for s in candidates if s.effort == effort]
            if exact:
                candidates = exact
        if tools is not None:
            exact = [s for s in candidates if s.tools == tools]
            if exact:
                candidates = exact
        rank = {"primary": 0, "independent": 1, "secondary": 2, "unverified": 3}
        return sorted(candidates, key=lambda s: (rank[s.confidence], -s.as_of.toordinal()))[0]

    def score_range_for(
        self, benchmark: str, *, effort: str | None = None, tools: bool | None = None
    ) -> tuple[float, float, int] | None:
        """Lowest and highest published score under matching conditions, and the count.

        Independent evaluators disagree about the same model at the same
        settings. Claude Opus 5 on ARC-AGI-2 at max effort is 90.4 according to
        ARC Prize and 88.3 according to Epoch AI -- both independent, both
        current, 2.1 points apart.

        Picking one and calling it *the* reference score decides, by accident of
        sort order, whether an honest endpoint starts the comparison two points
        in the hole. Callers running a hypothesis test should take the
        conservative end of this range instead, so that disagreement between
        sources widens the benefit of the doubt rather than silently becoming
        the tool's own bias.
        """
        candidates = [s for s in self.scores if s.benchmark == benchmark]
        if effort is not None:
            candidates = [s for s in candidates if s.effort == effort] or candidates
        if tools is not None:
            candidates = [s for s in candidates if s.tools == tools] or candidates
        if not candidates:
            return None
        values = [s.score for s in candidates]
        return min(values), max(values), len(values)

    def matches(self, identifier: str) -> bool:
        ident = identifier.strip().lower()
        return ident == self.id.lower() or ident in {a.lower() for a in self.aliases}


class ReferenceSnapshot(BaseModel):
    """The full reference dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    as_of: dt.date
    generated_by: str = "manual"
    sources: dict[str, str] = Field(default_factory=dict)
    families: tuple[FamilySignature, ...] = ()
    models: tuple[ModelRecord, ...] = ()

    def find(self, identifier: str) -> ModelRecord | None:
        for record in self.models:
            if record.matches(identifier):
                return record
        return None

    def family_signature(self, family: str) -> FamilySignature | None:
        for sig in self.families:
            if sig.family == family:
                return sig
        return None

    def to_yaml_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
