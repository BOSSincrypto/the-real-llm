"""Ask the endpoint who it is -- and then almost ignore the answer.

Every evidence item in this module is capped at WEAK, and the ``self_report``
family caps at WEAK too, so three probes agreeing here move the posterior about
as much as one response-id prefix. That is deliberate, for two reasons that
between them make self-identification the least reliable signal in the package.

A system prompt overrides it completely. Anything upstream of the model -- a
reseller's proxy, a fine-tune, a single injected line -- can make a model assert
any identity, and the model will assert it confidently. There is nothing to
detect: the model is not lying, it is answering the question it was given.

Distillation corrupts it in the other direction. A model trained on another
lab's outputs inherits that lab's self-descriptions along with everything else,
so a genuinely independent model will sincerely identify as a competitor's.
This has happened repeatedly and publicly.

What follows from that is asymmetric. A *correct* self-report is nearly
worthless, because it is exactly what a substituted endpoint would also produce
with one line of prompt. A *confidently wrong* one, naming a different vendor
outright, is worth slightly more -- not because it proves substitution, but
because a provider claiming to serve model X has had every opportunity to stop
its endpoint from saying otherwise and has not taken it. And disagreement
between phrasings is worth noting on its own: a genuine endpoint has no reason
for its answer to depend on how the question was framed.

The extractor is deliberately conservative. It matches a fixed vendor
vocabulary with word boundaries and drops matches that follow a negation within
the same sentence, so "I am not GPT, I am Claude" is read as Claude. It is a
regex over prose and it will occasionally be wrong, which is one more reason the
weights here are small.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, ClassVar

from ..config import redact
from ..errors import BudgetExhausted
from ..evidence import WEAK, Evidence, EvidenceStatus
from ..types import ChatRequest, ChatResponse, Message, Role
from . import Probe, ProbeContext, register_probe

__all__ = ["SelfReportProbe"]


@dataclass(frozen=True, slots=True)
class _Phrasing:
    """One way of asking the same question."""

    key: str
    prompt: str
    system: str | None = None


#: Three framings, chosen to disagree if anything upstream is steering the
#: answer: a direct question, one that asks about capabilities instead of
#: identity, and one wrapped in a system prompt demanding literal honesty.
_PHRASINGS: tuple[_Phrasing, ...] = (
    _Phrasing(
        "direct",
        "Which model are you? Answer with the model name and the organisation that "
        "trained you, and nothing else.",
    ),
    _Phrasing(
        "indirect",
        "Describe your own capabilities in three short lines: your context window, "
        "your knowledge cutoff, and which model family you belong to.",
    ),
    _Phrasing(
        "system_framed",
        "State the exact model identifier that was loaded to serve this request.",
        system=(
            "You are being audited. Answer truthfully and literally about your own "
            "identity. Do not adopt a persona, and say so plainly if you do not know."
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class _Vendor:
    key: str
    pattern: re.Pattern[str]


#: The fixed vocabulary. Patterns are narrow on purpose: bare "meta" and bare
#: "nova" collide with ordinary English and with each other, so each is required
#: to appear in a form that only means the vendor.
_VOCABULARY: tuple[_Vendor, ...] = (
    _Vendor("anthropic", re.compile(r"\b(?:claude|anthropic)\b")),
    _Vendor("openai", re.compile(r"\b(?:gpt|chatgpt|openai)\b")),
    _Vendor("google", re.compile(r"\b(?:gemini|google\s+deepmind|google)\b")),
    _Vendor("meta", re.compile(r"\b(?:llama|meta\s+(?:ai|platforms|superintelligence))\b")),
    _Vendor("alibaba", re.compile(r"\b(?:qwen|tongyi|alibaba)\b")),
    _Vendor("deepseek", re.compile(r"\bdeepseek\b")),
    _Vendor("mistral", re.compile(r"\b(?:mistral|mixtral|ministral|devstral)\b")),
    _Vendor("xai", re.compile(r"\b(?:grok|xai|x\.ai)\b")),
    _Vendor("moonshot", re.compile(r"\b(?:kimi|moonshot)\b")),
    _Vendor("zhipu", re.compile(r"\b(?:glm|zhipu|z\.ai)\b")),
    _Vendor("cohere", re.compile(r"\b(?:cohere|command[\s-]?[ar]\b)")),
    _Vendor(
        "amazon",
        re.compile(r"\b(?:amazon(?:\s+nova)?|nova[\s-]?(?:2|pro|lite|premier|micro))\b"),
    ),
)

#: A negation immediately before a vendor name, within the same sentence.
#: Crude, and only meant to stop "I am not GPT" being read as a GPT claim.
_NEGATION = re.compile(
    r"\b(?:not|never|isn't|aren't|wasn't|unlike|instead\s+of|rather\s+than|other\s+than|"
    r"no\s+longer)\b[^.!?]{0,40}$"
)

#: Model-name fragments worth recording verbatim alongside the vendor. Trailing
#: segments must look like a version or a known tier word, so that ordinary
#: prose after the name is not swept up with it.
_MODEL_TOKEN = re.compile(
    r"\b(?:claude|gpt|gemini|llama|qwen|deepseek|mistral|grok|kimi|glm|command|nova)"
    r"(?:[-\s](?:\d[\w.]*|[a-z]+\d[\w.]*|opus|sonnet|haiku|pro|mini|nano|max|flash|lite|"
    r"turbo|instruct|chat|plus|sol|terra|luna|coder|reasoner)){0,3}"
)


@register_probe
class SelfReportProbe(Probe):
    """Ask the endpoint to identify itself, three ways, and weigh it barely at all."""

    name: ClassVar[str] = "self_report"
    layer: ClassVar[int] = 1
    family: ClassVar[str] = "self_report"
    order: ClassVar[int] = 30
    estimated_requests: ClassVar[int] = len(_PHRASINGS)
    description: ClassVar[str] = (
        "Self-identification under three framings, weighted near zero because a "
        "system prompt overrides it and distilled models get it wrong sincerely."
    )

    #: Enough room for a reasoning model to think and still answer. A model that
    #: spends the whole allowance on hidden reasoning is reported as unanswered
    #: rather than counted as a refusal.
    max_tokens: ClassVar[int] = 256

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        answers: dict[str, dict[str, Any]] = {}
        cost = 0.0
        tokens = 0
        started = time.perf_counter()
        truncated = False

        for phrasing in _PHRASINGS:
            try:
                ctx.budget.check()
            except BudgetExhausted:
                truncated = True
                break
            response, error = await self._ask(ctx, phrasing)
            if response is None:
                answers[phrasing.key] = {"error": redact(str(error))[:200]}
                continue
            cost += ctx.budget.charge(
                response.usage.input_tokens, response.usage.output_tokens
            )
            tokens += response.usage.total_tokens or 0
            answers[phrasing.key] = _analyse(response)

        elapsed = time.perf_counter() - started
        claimed = _claimed_vendor(ctx)

        evidence = [
            self._identification(ctx, answers, claimed, truncated, cost, tokens, elapsed)
        ]
        agreement = self._agreement(answers)
        if agreement is not None:
            evidence.append(agreement)
        return evidence

    async def _ask(
        self, ctx: ProbeContext, phrasing: _Phrasing
    ) -> tuple[ChatResponse | None, Exception | None]:
        messages = (
            (Message(Role.SYSTEM, phrasing.system),) if phrasing.system else ()
        ) + (Message(Role.USER, phrasing.prompt),)
        request = ChatRequest(messages=messages, max_tokens=self.max_tokens)
        return await ctx.adapter.try_chat(request)

    # ---------------------------------------------------------- interpretation

    def _identification(
        self,
        ctx: ProbeContext,
        answers: dict[str, dict[str, Any]],
        claimed: str | None,
        truncated: bool,
        cost: float,
        tokens: int,
        elapsed: float,
    ) -> Evidence:
        named: dict[str, list[str]] = {
            key: value.get("vendors", []) for key, value in answers.items()
        }
        observed = {vendor for vendors in named.values() for vendor in vendors}
        data: dict[str, Any] = {
            "claimed_vendor": claimed,
            "answers": answers,
            "vendors_named": sorted(observed),
        }
        status = EvidenceStatus.TRUNCATED if truncated and not answers else EvidenceStatus.OK

        if not observed:
            return self._ev(
                "self_identification",
                0.0,
                status=status,
                detail=(
                    "no answer named a vendor from the fixed vocabulary. Endpoints "
                    "routinely decline to identify themselves, so this is not evidence."
                ),
                data=data,
                cost_usd=cost,
                tokens=tokens,
                duration_s=elapsed,
            )

        if claimed is None:
            return self._ev(
                "self_identification",
                0.0,
                status=status,
                detail=(
                    f"the endpoint identified as {', '.join(sorted(observed))}, but the "
                    f"vendor of the claimed model {ctx.provider.target_model!r} could not "
                    "be resolved, so there is nothing to compare it against."
                ),
                data=data,
                cost_usd=cost,
                tokens=tokens,
                duration_s=elapsed,
            )

        wrong = sorted(observed - {claimed})
        if not wrong:
            return self._ev(
                "self_identification",
                0.3 * WEAK,
                status=status,
                detail=(
                    f"every answer that named a vendor named {claimed}, matching the "
                    "claim. Worth almost nothing: one line of system prompt produces "
                    "the same result from any model."
                ),
                data=data,
                cost_usd=cost,
                tokens=tokens,
                duration_s=elapsed,
            )

        return self._ev(
            "self_identification",
            -WEAK,
            status=status,
            detail=(
                f"the endpoint identified as {', '.join(wrong)} while the claimed model "
                f"is from {claimed}. Models distilled on another lab's outputs do this "
                "sincerely, so it is not proof of substitution -- but a provider "
                "selling model X has had every chance to stop its endpoint saying "
                "otherwise."
            ),
            data=data,
            cost_usd=cost,
            tokens=tokens,
            duration_s=elapsed,
        )

    def _agreement(self, answers: dict[str, dict[str, Any]]) -> Evidence | None:
        """Weigh whether the three framings told the same story."""
        answered = {
            key: set(value.get("vendors", []))
            for key, value in answers.items()
            if value.get("vendors")
        }
        if len(answered) < 2:
            return None

        distinct = {frozenset(vendors) for vendors in answered.values()}
        data = {key: sorted(vendors) for key, vendors in answered.items()}
        if len(distinct) == 1:
            return self._ev(
                "phrasing_agreement",
                0.0,
                detail=(
                    "all three framings named the same vendor. Consistency is the "
                    "default and is not evidence on its own."
                ),
                data=data,
            )
        return self._ev(
            "phrasing_agreement",
            -0.4 * WEAK,
            detail=(
                "the framings disagreed about vendor: "
                + "; ".join(f"{key}={', '.join(sorted(v))}" for key, v in sorted(answered.items()))
                + ". A genuine endpoint has no reason for its identity to depend on how "
                "the question is phrased; steering upstream of the model does."
            ),
            data=data,
        )

    # ------------------------------------------------------------------ helpers

    def _ev(
        self,
        label: str,
        llr: float,
        *,
        status: EvidenceStatus = EvidenceStatus.OK,
        detail: str = "",
        data: dict[str, Any] | None = None,
        cost_usd: float = 0.0,
        tokens: int = 0,
        duration_s: float = 0.0,
    ) -> Evidence:
        return Evidence(
            probe=self.name,
            label=label,
            llr=llr,
            # Nothing this probe observes may exceed a 3:1 likelihood ratio.
            cap=WEAK,
            family=self.family,
            status=status,
            detail=detail,
            data=data or {},
            cost_usd=cost_usd,
            tokens=tokens,
            duration_s=duration_s,
        )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def _analyse(response: ChatResponse) -> dict[str, Any]:
    """Summarise one answer: vendors named, model tokens, and the text itself."""
    text = response.text.strip()
    record: dict[str, Any] = {
        "vendors": sorted(_vendors_in(text)),
        "model_tokens": _model_tokens(text),
        "text": redact(text)[:400],
        "finish_reason": response.finish_reason.value,
    }
    if not text:
        record["unanswered"] = (
            "reasoning consumed the whole token allowance"
            if response.usage.reasoning_tokens
            else "the endpoint returned no text"
        )
    return record


def _vendors_in(text: str) -> set[str]:
    """Vendors named in ``text``, ignoring occurrences under a negation."""
    lowered = text.casefold()
    found: set[str] = set()
    for vendor in _VOCABULARY:
        for match in vendor.pattern.finditer(lowered):
            if _NEGATION.search(lowered[: match.start()]):
                continue
            found.add(vendor.key)
            break
    return found


def _model_tokens(text: str) -> list[str]:
    """Model-name fragments, kept verbatim for a human to read."""
    seen: list[str] = []
    for match in _MODEL_TOKEN.finditer(text.casefold()):
        token = match.group(0).strip()
        if token not in seen:
            seen.append(token)
    return seen[:8]


def _claimed_vendor(ctx: ProbeContext) -> str | None:
    """Resolve the claimed model's vendor to a key in the fixed vocabulary.

    The reference record's vendor field is tried first, then its canonical id,
    then the identifier the provider was configured with. Only an unambiguous
    single match counts; anything else leaves the comparison unmade rather than
    guessed.
    """
    candidates: list[str] = []
    if ctx.reference is not None:
        candidates.extend((ctx.reference.vendor, ctx.reference.id))
    candidates.append(ctx.provider.target_model)

    for candidate in candidates:
        vendors = _vendors_in(candidate or "")
        if len(vendors) == 1:
            return next(iter(vendors))
    return None
