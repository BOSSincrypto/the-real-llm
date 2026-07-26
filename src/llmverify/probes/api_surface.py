"""The parameter-support matrix: the cheapest strong signal in the tool.

Eight short requests establish which request parameters an endpoint accepts,
rejects, or accepts and silently drops. That matrix is far harder to fake than
anything in the response envelope, because honouring a parameter requires the
serving stack to actually implement it, and refusing one requires a validator
that knows the protocol.

Three distinctions carry almost all the weight.

**Accepted versus ignored.** A 2xx alone proves nothing. Where an effect can be
observed -- logprobs came back, the response really is JSON, the raw body really
carries ``prompt_logprobs`` -- the probe checks for it, and a 2xx with no effect
is reported as ``IGNORED``. Several ignored parameters at once is the signature
of a LiteLLM front end running ``drop_params=True``. That is a fact about the
serving stack, not about the weights, and it is reported at zero LLR.

**Logprobs against a Claude claim.** Anthropic's protocol has no logprobs and no
seed; this was confirmed both from the request schema and from the fact that no
Anthropic model advertises either parameter in OpenRouter's parameter
vocabulary. A translating proxy can accept a ``seed`` field and throw it away,
so acceptance there is weak. It cannot manufacture per-token logprobs without
running a model that produces them, so an endpoint claiming a Claude model and
returning real logprobs is strong evidence against the claim -- and, unusually
for this package, the evidence is *not* weakened by the endpoint speaking an
OpenAI-compatible protocol, because the observation is about what computed the
tokens rather than about how they were packaged.

**prompt_logprobs.** Logprobs over the *input* tokens is a vLLM extension.
SGLang spells its equivalent ``return_logprob``/``top_logprobs_num`` and has no
``prompt_logprobs`` at all, and it appears nowhere in the first-party
protocols. Support for it therefore says vLLM, which is neutral for an
open-weight model and hard to reconcile with a claim to serve closed weights.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from ..adapters.base import ProbeOutcome
from ..config import redact
from ..errors import BudgetExhausted
from ..evidence import MODERATE, STRONG, WEAK, Evidence, EvidenceStatus
from ..reference.schema import FamilySignature
from ..types import ChatRequest, ChatResponse, Message, ParamSupport, Role
from . import Probe, ProbeContext, register_probe

__all__ = ["ApiSurfaceProbe"]

_TRIVIAL_PROMPT = "Reply with the single word: ok"

_JSON_PROMPT = "Reply with a JSON object naming one colour."

_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"colour": {"type": "string"}},
    "required": ["colour"],
    "additionalProperties": False,
}


def _returned_logprobs(response: ChatResponse) -> bool:
    """Whether the response actually carries per-token logprob entries."""
    return bool(response.logprobs)


def _returned_prompt_logprobs(response: ChatResponse) -> bool:
    """Whether the raw body carries vLLM's input-token logprob block."""
    if "prompt_logprobs" in response.raw:
        return response.raw["prompt_logprobs"] is not None
    choices = response.raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0].get("prompt_logprobs") is not None
    return False


def _returned_schema_object(response: ChatResponse) -> bool:
    """Whether the completion is a JSON object matching the requested schema."""
    text = response.text.strip()
    if not text:
        return False
    try:
        parsed = json.loads(text)
    except ValueError:
        return False
    return isinstance(parsed, dict) and "colour" in parsed


@dataclass(frozen=True, slots=True)
class _ParamSpec:
    """One parameter to try, and how to tell whether it took effect."""

    parameter: str
    patch: dict[str, Any]
    detect: Callable[[ChatResponse], bool] | None = None
    prompt: str = _TRIVIAL_PROMPT
    max_tokens: int = 8
    #: Why no effect detector exists, when there is none. Reported verbatim so
    #: an ``ACCEPTED`` that could not be confirmed is never read as confirmed.
    undetectable: str = ""


#: Probed in this order so that the two parameters carrying real weight are
#: measured before any budget ceiling can bite.
_PARAMETERS: tuple[_ParamSpec, ...] = (
    _ParamSpec(
        "logprobs",
        {"logprobs": True, "top_logprobs": 5},
        detect=_returned_logprobs,
    ),
    _ParamSpec(
        "seed",
        {"seed": 20260726},
        undetectable=(
            "a seed's effect cannot be observed from a single call, so a 2xx here means "
            "the field was accepted, not that it was honoured"
        ),
    ),
    _ParamSpec(
        "prompt_logprobs",
        {"prompt_logprobs": 1},
        detect=_returned_prompt_logprobs,
    ),
    _ParamSpec(
        "response_format_json_schema",
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "probe", "schema": _JSON_SCHEMA, "strict": True},
            }
        },
        detect=_returned_schema_object,
        prompt=_JSON_PROMPT,
        max_tokens=48,
    ),
    _ParamSpec(
        "logit_bias",
        {"logit_bias": {"1734": -100}},
        undetectable=(
            "confirming a bias took effect would need the endpoint's tokenizer, which is "
            "what a later probe measures"
        ),
    ),
    _ParamSpec(
        "min_p",
        {"min_p": 0.05},
        undetectable="a sampling cutoff has no single-call signature",
    ),
    _ParamSpec(
        "top_k",
        {"top_k": 5},
        undetectable="a sampling cutoff has no single-call signature",
    ),
    _ParamSpec(
        "presence_penalty",
        {"presence_penalty": 0.5},
        undetectable="a penalty has no single-call signature at this length",
    ),
)


@register_probe
class ApiSurfaceProbe(Probe):
    """Measure which request parameters an endpoint honours, and read the pattern."""

    name: ClassVar[str] = "api_surface"
    layer: ClassVar[int] = 1
    family: ClassVar[str] = "api_surface"
    order: ClassVar[int] = 20
    estimated_requests: ClassVar[int] = len(_PARAMETERS)
    description: ClassVar[str] = (
        "Parameter-support matrix -- seed, logprobs, logit_bias, min_p, top_k, "
        "penalties, json_schema and vLLM's prompt_logprobs -- against the family "
        "signature for the claimed model."
    )

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        started = time.perf_counter()
        outcomes, truncated = await self._measure(ctx)
        elapsed = time.perf_counter() - started

        if not outcomes:
            return [
                self._ev(
                    "parameter_matrix",
                    0.0,
                    status=EvidenceStatus.TRUNCATED,
                    detail="the budget was exhausted before any parameter could be probed.",
                    duration_s=elapsed,
                )
            ]

        signature = (
            ctx.snapshot.family_signature(ctx.adapter.family.value)
            if ctx.snapshot is not None
            else None
        )
        matrix = {
            name: {
                "support": outcome.support.value,
                "http_status": outcome.status,
                "detail": redact(outcome.detail)[:200],
                "body_excerpt": redact(outcome.body_excerpt)[:200],
            }
            for name, outcome in outcomes.items()
        }

        evidence = [self._summary(ctx, outcomes, matrix, signature, truncated, elapsed)]
        evidence.extend(self._divergences(ctx, outcomes, signature))
        return evidence

    # ------------------------------------------------------------- measurement

    async def _measure(self, ctx: ProbeContext) -> tuple[dict[str, ProbeOutcome], bool]:
        """Probe each parameter in turn, stopping cleanly when the budget ends.

        Sequential rather than concurrent: a fan-out of eight identical requests
        invites the rate limiting that would look like a capability failure, and
        the ordering above puts the two decisive parameters first so a truncated
        run still says something.
        """
        outcomes: dict[str, ProbeOutcome] = {}
        for spec in _PARAMETERS:
            try:
                ctx.budget.check()
            except BudgetExhausted:
                return outcomes, True

            request = ChatRequest(
                messages=(Message(Role.USER, spec.prompt),), max_tokens=spec.max_tokens
            )
            try:
                outcome = await ctx.adapter.probe_parameter(
                    request,
                    parameter=spec.parameter,
                    payload_patch=spec.patch,
                    detect_effect=spec.detect,
                )
            except Exception as exc:
                outcome = ProbeOutcome(
                    spec.parameter, ParamSupport.UNKNOWN, detail=redact(str(exc))[:200]
                )
            # The adapter does not return the response, so only the sample count
            # can be charged honestly; token spend for this probe is unknown.
            ctx.budget.charge(None, None)
            outcomes[spec.parameter] = outcome
        return outcomes, False

    # ------------------------------------------------------------ interpretation

    def _summary(
        self,
        ctx: ProbeContext,
        outcomes: dict[str, ProbeOutcome],
        matrix: dict[str, dict[str, Any]],
        signature: FamilySignature | None,
        truncated: bool,
        elapsed: float,
    ) -> Evidence:
        """One row per parameter, plus a verdict on signature consistency."""
        data: dict[str, Any] = {
            "matrix": matrix,
            "wire_family": ctx.adapter.family.value,
            "claimed_model_family": ctx.reference.family if ctx.reference else None,
            "undetectable": {
                spec.parameter: spec.undetectable for spec in _PARAMETERS if spec.undetectable
            },
        }
        summary = ", ".join(
            f"{name}={outcome.support.value}" for name, outcome in outcomes.items()
        )

        consistent = self._signature_consistency(outcomes, signature)
        data["signature_consistent"] = consistent

        status = EvidenceStatus.TRUNCATED if truncated else EvidenceStatus.OK

        if consistent is True:
            llr = 0.5 * WEAK
            verdict = (
                f"consistent with the {signature.family} family signature -- which "
                "describes the protocol being spoken, not the claimed model's own API."
                if signature is not None
                else ""
            )
        else:
            llr = 0.0
            verdict = (
                "divergences from the family signature are itemised separately."
                if consistent is False
                else "no family signature was available to compare against."
            )

        return self._ev(
            "parameter_matrix",
            llr,
            status=status,
            detail=f"{summary}. {verdict}".strip(),
            data=data,
            duration_s=elapsed,
        )

    def _signature_consistency(
        self, outcomes: dict[str, ProbeOutcome], signature: FamilySignature | None
    ) -> bool | None:
        """Whether logprob and seed support match the family's declared expectations."""
        if signature is None:
            return None
        checks: list[bool] = []
        if signature.supports_logprobs is not None:
            observed = outcomes.get("logprobs")
            if observed is not None:
                honoured = observed.support is ParamSupport.ACCEPTED
                checks.append(honoured == signature.supports_logprobs)
        if signature.supports_seed is not None:
            observed = outcomes.get("seed")
            if observed is not None:
                accepted = observed.support is ParamSupport.ACCEPTED
                checks.append(accepted == signature.supports_seed)
        if not checks:
            return None
        return all(checks)

    def _divergences(
        self,
        ctx: ProbeContext,
        outcomes: dict[str, ProbeOutcome],
        signature: FamilySignature | None,
    ) -> list[Evidence]:
        """One evidence item per pattern in the matrix that means something."""
        evidence: list[Evidence] = []
        claimed_family = ctx.reference.family if ctx.reference is not None else None

        logprobs = outcomes.get("logprobs")
        seed = outcomes.get("seed")

        if claimed_family == "anthropic":
            if logprobs is not None and logprobs.support is ParamSupport.ACCEPTED:
                evidence.append(
                    self._ev(
                        "anthropic_claim_returns_logprobs",
                        -STRONG,
                        cap=STRONG,
                        detail=(
                            f"the endpoint claims {ctx.provider.target_model!r} and returned "
                            "real per-token logprobs. Anthropic's protocol has no logprobs "
                            "at all -- confirmed from the request schema and from the fact "
                            "that no Anthropic model advertises the parameter anywhere -- "
                            "and a translating proxy cannot synthesise them without running "
                            "a model that produces them."
                        ),
                        data={"top_logprobs_requested": 5},
                    )
                )
            if seed is not None and seed.support is ParamSupport.ACCEPTED:
                evidence.append(
                    self._ev(
                        "anthropic_claim_accepts_seed",
                        -WEAK,
                        detail=(
                            f"the endpoint claims {ctx.provider.target_model!r} and accepted a "
                            "seed, which Anthropic's protocol does not define. Only weak: a "
                            "proxy that accepts the field and drops it is indistinguishable "
                            "from one that honours it in a single call."
                        ),
                    )
                )

        if (
            claimed_family == "openai"
            and logprobs is not None
            and logprobs.support is ParamSupport.REJECTED
        ):
            reasoning = bool(ctx.reference is not None and ctx.reference.reasoning)
            evidence.append(
                self._ev(
                    "openai_claim_rejects_logprobs",
                    -0.3 * WEAK if reasoning else -WEAK,
                    detail=(
                        f"the endpoint claims {ctx.provider.target_model!r} but rejected "
                        f"logprobs with HTTP {logprobs.status}. Reasoning models appear to "
                        "block logprobs in practice -- no current OpenAI reasoning model "
                        "advertises the parameter -- but this is UNVERIFIED as an absolute "
                        "rule, so the finding is weakened rather than dropped."
                        + (" The claimed model is a reasoning model." if reasoning else "")
                    ),
                    data={"reasoning_model": reasoning, "status": logprobs.status},
                )
            )

        prompt_logprobs = outcomes.get("prompt_logprobs")
        if prompt_logprobs is not None and prompt_logprobs.support is ParamSupport.ACCEPTED:
            evidence.append(self._vllm_evidence(ctx))

        ignored = sorted(
            name
            for name, outcome in outcomes.items()
            if outcome.support is ParamSupport.IGNORED
        )
        if len(ignored) >= 2:
            evidence.append(
                self._ev(
                    "silently_dropped_parameters",
                    0.0,
                    detail=(
                        f"{len(ignored)} parameters returned 2xx with no observable effect "
                        f"({', '.join(ignored)}). That is the LiteLLM drop_params signature: "
                        "a front end configured to discard what its upstream cannot handle. "
                        "It describes the serving stack, not the weights, so it carries no "
                        "weight for or against the claim."
                    ),
                    data={"ignored": ignored},
                )
            )

        if signature is None:
            evidence.append(
                self._ev(
                    "family_signature",
                    0.0,
                    status=EvidenceStatus.SKIPPED,
                    detail=(
                        "the reference snapshot records no family signature for the protocol "
                        "this endpoint speaks, so the matrix stands on its own."
                    ),
                )
            )
        return evidence

    def _vllm_evidence(self, ctx: ProbeContext) -> Evidence:
        """Read support for ``prompt_logprobs`` against what is being claimed."""
        detail = (
            "the endpoint returned logprobs over the input tokens. ``prompt_logprobs`` is "
            "a vLLM extension: SGLang spells its equivalent return_logprob/"
            "top_logprobs_num and has no such parameter, and none of the first-party "
            "protocols define it."
        )
        if ctx.reference is None:
            return self._ev(
                "prompt_logprobs_supported",
                0.0,
                detail=(
                    f"{detail} The claimed model is absent from the reference snapshot, so "
                    "whether vLLM serving is compatible with the claim cannot be judged."
                ),
            )
        if ctx.reference.open_weights:
            return self._ev(
                "prompt_logprobs_supported",
                0.5 * WEAK,
                detail=(
                    f"{detail} {ctx.reference.id!r} has open weights, so vLLM serving is "
                    "exactly what one would expect; mildly supportive at most, since it "
                    "says nothing about which checkpoint was loaded."
                ),
            )
        return self._ev(
            "prompt_logprobs_supported",
            -STRONG,
            cap=STRONG,
            detail=(
                f"{detail} {ctx.reference.id!r} is closed-weight, so its weights are not "
                "available to load into vLLM. Something else is generating these tokens."
            ),
        )

    # ------------------------------------------------------------------ helpers

    def _ev(
        self,
        label: str,
        llr: float,
        *,
        cap: float = MODERATE,
        status: EvidenceStatus = EvidenceStatus.OK,
        detail: str = "",
        data: dict[str, Any] | None = None,
        duration_s: float = 0.0,
    ) -> Evidence:
        return Evidence(
            probe=self.name,
            label=label,
            llr=llr,
            cap=cap,
            family=self.family,
            status=status,
            detail=detail,
            data=data or {},
            duration_s=duration_s,
        )
