"""Layer-0 metadata: what one minimal request already gives away.

This probe costs a single sixteen-token completion and one catalogue read, and
it leaves its response in ``ctx.shared["warmup_response"]`` so that no later
probe has to pay for the same information twice.

Nothing it measures is hard to fake. An echoed model id is a string the provider
chose; a response-id prefix is four characters. The probe runs first anyway,
for two reasons. Faking *all* of it consistently is work a careless reseller
does not do, and the observations it records -- usage-object shape, catalogue
shape, declared quantization, infrastructure headers -- are the context every
later probe is read against. Every LLR here is deliberately small, and the
``metadata`` family caps at MODERATE, so this probe cannot convict anyone alone.

One distinction runs through the whole module. The response *envelope* -- id
prefix, usage keys, finish-reason vocabulary, ``system_fingerprint`` -- is
produced by whatever software answered the HTTP request, not by the weights. It
is evidence about model identity only when the claimed model's own first-party
API is the protocol being spoken. A reseller that fronts Claude behind an
OpenAI-compatible route rewrites the envelope entirely, and legitimately so;
penalising it for that would be a false accusation. Envelope findings are
recorded either way, but they carry weight only in the native case.
"""

from __future__ import annotations

import re
import time
from typing import Any, ClassVar

from ..config import redact
from ..errors import BudgetExhausted
from ..evidence import MODERATE, STRONG, WEAK, Evidence, EvidenceStatus
from ..reference.schema import FamilySignature
from ..types import ChatRequest, ChatResponse, Message, Role
from . import Probe, ProbeContext, register_probe

__all__ = ["MetadataProbe"]

#: Headers worth recording. They identify infrastructure -- a CDN, a load
#: balancer, a request-id scheme -- and infrastructure is not weights, so every
#: one of them is reported with zero LLR.
_INFRASTRUCTURE_HEADERS: tuple[str, ...] = (
    "server",
    "via",
    "cf-ray",
    "cf-cache-status",
    "x-served-by",
    "x-request-id",
    "request-id",
    "x-amzn-requestid",
    "x-envoy-upstream-service-time",
)

#: Catalogue-entry keys verified against a first-party ``GET /v1/models``
#: response. Only Anthropic's shape was confirmed when this reference was
#: taken, so it is the only family whose catalogue shape carries an LLR;
#: everything else is recorded and left at zero rather than guessed at.
_CATALOGUE_SHAPES: dict[str, tuple[str, ...]] = {
    "anthropic": ("id", "display_name", "created_at"),
}

#: Reduced-precision labels and how much weight a declaration of one carries
#: against a claim to serve a frontier model. Published benchmark numbers are
#: measured on the publisher's own serving stack, so a provider declaring a
#: narrower numeric format is declaring a different artefact from the one the
#: reference scores describe.
_QUANTIZATION_PENALTY: dict[str, float] = {
    "fp4": MODERATE,
    "int4": MODERATE,
    "fp8": WEAK,
    "int8": WEAK,
}

_FULL_PRECISION = frozenset({"fp16", "bf16", "fp32"})

#: Trailing snapshot dates and version tags. Stripping these is what makes
#: ``claude-opus-5-20260723`` and ``claude-opus-5`` compare equal.
_VERSION_SUFFIX = re.compile(r"[-_](?:\d{8}|\d{6}|\d{4}-\d{2}-\d{2}|v\d+)$")

#: Bedrock/Vertex-style dotted prefixes: ``us.anthropic.claude-...``.
_DOTTED_PREFIX = re.compile(r"^[a-z0-9]+\.")


@register_probe
class MetadataProbe(Probe):
    """Free identity evidence from one minimal request plus the catalogue."""

    name: ClassVar[str] = "metadata"
    layer: ClassVar[int] = 0
    family: ClassVar[str] = "metadata"
    order: ClassVar[int] = 10
    estimated_requests: ClassVar[int] = 1
    description: ClassVar[str] = (
        "Echoed model id, response-id prefix, usage and catalogue shape, declared "
        "quantization and infrastructure headers, from one minimal request."
    )

    #: Small on purpose: the envelope is the measurement, not the text. A
    #: reasoning model may spend the whole allowance on hidden reasoning and
    #: return no visible text at all, which does not affect anything here.
    max_tokens: ClassVar[int] = 16
    prompt: ClassVar[str] = "Reply with the single word: ok"

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        evidence: list[Evidence] = []

        response: ChatResponse | None = ctx.shared.get("warmup_response")
        if response is None:
            response, warmup = await self._warm_up(ctx)
            evidence.append(warmup)

        signature = self._signature(ctx)
        weight, caveat = _envelope_weight(ctx)

        if response is not None:
            evidence.append(self._model_echo(ctx, response))
            evidence.extend(self._envelope(response, signature, weight, caveat))
            evidence.append(self._headers(response))

        evidence.append(await self._catalogue(ctx, response, weight, caveat))

        declared = await self._quantization(ctx, response)
        if declared is not None:
            evidence.append(declared)

        return evidence

    # ------------------------------------------------------------------ warm-up

    async def _warm_up(self, ctx: ProbeContext) -> tuple[ChatResponse | None, Evidence]:
        """Send the one request this probe pays for and publish it for reuse."""
        try:
            ctx.budget.check()
        except BudgetExhausted as exc:
            return None, self._ev(
                "warmup", 0.0, status=EvidenceStatus.TRUNCATED, detail=str(exc)
            )

        request = ChatRequest(
            messages=(Message(Role.USER, self.prompt),), max_tokens=self.max_tokens
        )
        started = time.perf_counter()
        response, error = await ctx.adapter.try_chat(request)
        elapsed = time.perf_counter() - started

        if response is None:
            return None, self._ev(
                "warmup",
                0.0,
                status=EvidenceStatus.ERROR,
                detail=f"the minimal request failed: {redact(str(error))[:300]}",
                data={"error_type": type(error).__name__ if error else None},
                duration_s=elapsed,
            )

        ctx.shared["warmup_response"] = response
        ctx.shared["warmup_request"] = request
        cost = ctx.budget.charge(response.usage.input_tokens, response.usage.output_tokens)

        return response, self._ev(
            "warmup",
            0.0,
            detail=(
                f"one {self.max_tokens}-token request succeeded in {elapsed:.2f}s; "
                "the response is shared with every later probe."
            ),
            data={
                "http_status": response.http_status,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "finish_reason": response.finish_reason.value,
            },
            cost_usd=cost,
            tokens=response.usage.total_tokens or 0,
            duration_s=elapsed,
        )

    # -------------------------------------------------------------- model echo

    def _model_echo(self, ctx: ProbeContext, response: ChatResponse) -> Evidence:
        """Weigh the model identifier the provider chose to report back."""
        reported = response.model_reported
        requested = ctx.provider.model
        target = ctx.provider.target_model
        data: dict[str, Any] = {
            "reported": reported,
            "requested": requested,
            "claimed": target,
        }

        if not reported:
            return self._ev(
                "model_echo",
                0.0,
                status=EvidenceStatus.UNSUPPORTED,
                detail="the endpoint reported no model identifier at all.",
                data=data,
            )

        data["reported_canonical"] = _canonical_model_id(reported)

        if reported in (requested, target) or reported.lower() in {
            requested.lower(),
            target.lower(),
        }:
            return self._ev(
                "model_echo",
                WEAK,
                detail=(
                    f"the endpoint echoed {reported!r} exactly. This is the cheapest "
                    "field in the response to forge, so it is capped at weak."
                ),
                data=data,
            )

        if ctx.reference is not None and ctx.reference.matches(reported):
            return self._ev(
                "model_echo",
                WEAK,
                detail=(
                    f"{reported!r} is a known alias of {ctx.reference.id!r}. Weak, and "
                    "for the same reason: the field is a string the provider picks."
                ),
                data=data,
            )

        if _same_model(reported, target) or _same_model(reported, requested):
            return self._ev(
                "model_echo",
                0.0,
                detail=(
                    f"{reported!r} differs from {target!r} only by a vendor prefix or a "
                    "snapshot date. That is a legitimate way to name the same model -- "
                    "OpenRouter's canonical_slug values look exactly like this -- so it "
                    "is neither support nor refutation."
                ),
                data=data,
            )

        other = ctx.snapshot.find(reported) if ctx.snapshot is not None else None
        if other is not None and not other.matches(target):
            data["reported_matches_record"] = other.id
            return self._ev(
                "model_echo",
                -STRONG,
                cap=STRONG,
                detail=(
                    f"the endpoint reported {reported!r}, which the reference snapshot "
                    f"identifies as {other.id!r} from {other.vendor}, a different model "
                    f"from the claimed {target!r}."
                ),
                data=data,
            )

        return self._ev(
            "model_echo",
            -WEAK,
            detail=(
                f"the endpoint reported {reported!r}, which is neither the claimed "
                f"{target!r} nor any model in the reference snapshot. It may be an "
                "internal routing name rather than a substitution."
            ),
            data=data,
        )

    # ---------------------------------------------------------------- envelope

    def _envelope(
        self,
        response: ChatResponse,
        signature: FamilySignature | None,
        weight: float,
        caveat: str,
    ) -> list[Evidence]:
        """Compare the response envelope with the first-party family signature."""
        if signature is None:
            return [
                self._ev(
                    "envelope",
                    0.0,
                    status=EvidenceStatus.SKIPPED,
                    detail=(
                        "the reference snapshot records no family signature for the "
                        "protocol this endpoint speaks, so there is nothing to compare "
                        "the envelope against."
                    ),
                )
            ]
        return [
            self._response_id(response, signature, weight, caveat),
            self._usage_shape(response, signature, weight, caveat),
            self._finish_reason(response, signature, weight, caveat),
            self._fingerprint_fields(response, signature, weight, caveat),
        ]

    def _response_id(
        self,
        response: ChatResponse,
        signature: FamilySignature,
        weight: float,
        caveat: str,
    ) -> Evidence:
        expected = signature.response_id_prefix
        observed = response.response_id
        data = {
            "response_id_shape": _id_shape(observed) if observed else None,
            "expected_prefix": expected,
            "family": signature.family,
        }
        if not expected:
            return self._ev(
                "response_id_prefix",
                0.0,
                status=EvidenceStatus.SKIPPED,
                detail=f"no id prefix is recorded for the {signature.family} family.",
                data=data,
            )
        if not observed:
            return self._ev(
                "response_id_prefix",
                _weigh(-WEAK, weight),
                detail=_with_caveat(
                    f"the response carried no id, where {signature.family} endpoints "
                    f"prefix theirs with {expected!r}.",
                    weight,
                    caveat,
                ),
                data=data,
            )
        if observed.startswith(expected):
            return self._ev(
                "response_id_prefix",
                _weigh(0.5 * WEAK, weight),
                detail=_with_caveat(
                    f"the response id starts with {expected!r}, matching the "
                    f"{signature.family} family.",
                    weight,
                    caveat,
                ),
                data=data,
            )
        return self._ev(
            "response_id_prefix",
            _weigh(-MODERATE, weight),
            cap=MODERATE,
            detail=_with_caveat(
                f"the response id has shape {data['response_id_shape']}, which does not "
                f"begin with the {signature.family} prefix {expected!r}.",
                weight,
                caveat,
            ),
            data=data,
        )

    def _usage_shape(
        self,
        response: ChatResponse,
        signature: FamilySignature,
        weight: float,
        caveat: str,
    ) -> Evidence:
        expected = signature.usage_keys
        observed = response.usage.raw
        data: dict[str, Any] = {
            "observed_keys": sorted(observed),
            "expected_keys": list(expected),
        }
        if not expected:
            return self._ev(
                "usage_shape",
                0.0,
                status=EvidenceStatus.SKIPPED,
                detail=f"no usage-key set is recorded for the {signature.family} family.",
                data=data,
            )
        if not observed:
            return self._ev(
                "usage_shape",
                _weigh(-WEAK, weight),
                detail=_with_caveat(
                    "the response carried no usage object at all.", weight, caveat
                ),
                data=data,
            )

        missing = [key for key in expected if not _has_path(observed, key)]
        extra = sorted(set(observed) - {key.split(".", 1)[0] for key in expected})
        data["missing"] = missing
        data["unexpected"] = extra

        if not missing:
            return self._ev(
                "usage_shape",
                _weigh(0.4 * WEAK, weight),
                detail=_with_caveat(
                    f"the usage object carries every key the {signature.family} family "
                    f"reports"
                    + (f"; it also carries {', '.join(extra)}." if extra else "."),
                    weight,
                    caveat,
                ),
                data=data,
            )

        # Scaled by how much of the expected shape is absent: one optional key
        # missing is a shrug, the whole shape missing is a different stack.
        share = len(missing) / len(expected)
        return self._ev(
            "usage_shape",
            _weigh(-WEAK * share, weight),
            detail=_with_caveat(
                f"the usage object is missing {len(missing)} of {len(expected)} keys the "
                f"{signature.family} family reports ({', '.join(missing)})"
                + (f"; unexpected keys: {', '.join(extra)}." if extra else "."),
                weight,
                caveat,
            ),
            data=data,
        )

    def _finish_reason(
        self,
        response: ChatResponse,
        signature: FamilySignature,
        weight: float,
        caveat: str,
    ) -> Evidence:
        raw = _raw_finish_reason(response)
        vocabulary = signature.finish_reasons
        data: dict[str, Any] = {
            "raw": raw,
            "normalised": response.finish_reason.value,
            "vocabulary": list(vocabulary),
            "native_finish_reason": _dig(response.raw, "choices", 0, "native_finish_reason"),
        }
        if not vocabulary or raw is None:
            return self._ev(
                "finish_reason_vocabulary",
                0.0,
                status=EvidenceStatus.SKIPPED,
                detail=(
                    "no finish-reason vocabulary is recorded for this family, or the "
                    "endpoint reported none."
                ),
                data=data,
            )
        if raw.casefold() in {value.casefold() for value in vocabulary}:
            return self._ev(
                "finish_reason_vocabulary",
                _weigh(0.25 * WEAK, weight),
                detail=_with_caveat(
                    f"the stop reason {raw!r} is in the {signature.family} vocabulary.",
                    weight,
                    caveat,
                ),
                data=data,
            )
        return self._ev(
            "finish_reason_vocabulary",
            _weigh(-WEAK, weight),
            detail=_with_caveat(
                f"the stop reason {raw!r} is outside the {signature.family} vocabulary "
                f"({', '.join(vocabulary)}).",
                weight,
                caveat,
            ),
            data=data,
        )

    def _fingerprint_fields(
        self,
        response: ChatResponse,
        signature: FamilySignature,
        weight: float,
        caveat: str,
    ) -> Evidence:
        """Weigh ``system_fingerprint``; record ``service_tier`` alongside it.

        Only ``system_fingerprint`` has a declared expectation in the family
        signature, so it is the only one of the two that moves the LLR.
        ``service_tier`` is recorded because it discriminates stacks, not
        because the reference says what it ought to be.
        """
        present = "system_fingerprint" in response.raw
        value = response.raw.get("system_fingerprint")
        expected = signature.has_system_fingerprint
        data: dict[str, Any] = {
            "system_fingerprint_present": present,
            "system_fingerprint_null": present and value is None,
            "service_tier": response.raw.get("service_tier"),
            "expected_system_fingerprint": expected,
        }

        if expected is None:
            return self._ev(
                "fingerprint_fields",
                0.0,
                detail=(
                    "recorded only: the reference states no expectation about "
                    f"system_fingerprint for the {signature.family} family."
                ),
                data=data,
            )

        # A key present but null is what a proxy emits when its upstream has no
        # fingerprint to pass on, so it counts as absence of the value.
        has_value = present and value is not None
        if has_value == expected:
            return self._ev(
                "fingerprint_fields",
                _weigh(0.3 * WEAK, weight),
                detail=_with_caveat(
                    "system_fingerprint is "
                    + ("present" if expected else "absent")
                    + f", matching the {signature.family} family.",
                    weight,
                    caveat,
                ),
                data=data,
            )
        return self._ev(
            "fingerprint_fields",
            _weigh(-WEAK, weight),
            detail=_with_caveat(
                "system_fingerprint is "
                + ("absent" if expected else "present")
                + f", where the {signature.family} family does the opposite.",
                weight,
                caveat,
            ),
            data=data,
        )

    # --------------------------------------------------------------- catalogue

    async def _catalogue(
        self,
        ctx: ProbeContext,
        response: ChatResponse | None,
        weight: float,
        caveat: str,
    ) -> Evidence:
        """Ask the endpoint what it claims to offer, and in what shape."""
        try:
            models = await ctx.adapter.list_models()
        except Exception as exc:
            return self._ev(
                "catalogue",
                0.0,
                status=EvidenceStatus.UNSUPPORTED,
                detail=f"the endpoint exposes no usable model catalogue: {redact(str(exc))[:200]}",
            )

        ids = [str(entry.get("id") or entry.get("name") or "") for entry in models]
        listed = [i for i in ids if i]
        present = any(
            i == ctx.provider.model or _same_model(i, ctx.provider.model) for i in listed
        )
        shape = sorted({key for entry in models[:8] for key in entry})

        data: dict[str, Any] = {
            "catalogue_size": len(models),
            "requested_model": ctx.provider.model,
            "requested_model_listed": present,
            "entry_keys": shape,
        }

        stack = self._serving_stack(ctx, models, response)
        if stack is not None:
            data["serving_stack"] = stack

        if not listed:
            return self._ev(
                "catalogue",
                0.0,
                status=EvidenceStatus.UNSUPPORTED,
                detail="the endpoint returned an empty model catalogue.",
                data=data,
            )

        expected_shape = _CATALOGUE_SHAPES.get(ctx.adapter.family.value)
        shape_missing = (
            [key for key in expected_shape if key not in shape] if expected_shape else []
        )
        data["shape_missing"] = shape_missing

        if not present:
            return self._ev(
                "catalogue",
                -0.5 * WEAK,
                detail=(
                    f"the catalogue lists {len(listed)} models and {ctx.provider.model!r} "
                    "is not among them, although the endpoint serves requests for it. "
                    "Routing proxies commonly do this, so it is only weak evidence."
                ),
                data=data,
            )

        if shape_missing:
            return self._ev(
                "catalogue",
                _weigh(-WEAK, weight),
                detail=_with_caveat(
                    f"{ctx.provider.model!r} is listed, but the entries lack "
                    f"{', '.join(shape_missing)}, which a first-party "
                    f"{ctx.adapter.family.value} catalogue carries.",
                    weight,
                    caveat,
                ),
                data=data,
            )

        return self._ev(
            "catalogue",
            0.5 * WEAK,
            detail=(
                f"{ctx.provider.model!r} appears in a catalogue of {len(listed)} models"
                + (f"; serving stack looks like {stack}." if stack else ".")
            ),
            data=data,
        )

    def _serving_stack(
        self,
        ctx: ProbeContext,
        models: list[dict[str, Any]],
        response: ChatResponse | None,
    ) -> str | None:
        """Best-effort stack label, when the adapter offers one."""
        detect = getattr(ctx.adapter, "detect_serving_stack", None)
        if detect is None:
            return None
        try:
            return detect(models, response)
        except Exception:
            return None

    # ------------------------------------------------------------ quantization

    async def _quantization(
        self, ctx: ProbeContext, response: ChatResponse | None
    ) -> Evidence | None:
        """Report the numeric formats an OpenRouter endpoint declares for itself.

        This is filed under ``capability`` rather than ``metadata`` because a
        narrower numeric format is a claim about the artefact being served, not
        about how the response is packaged.
        """
        fetch = getattr(ctx.adapter, "fetch_openrouter_endpoints", None)
        if fetch is None:
            return None
        try:
            endpoints = await fetch(ctx.provider.model)
        except Exception:
            endpoints = None
        if not endpoints:
            return None

        selected = response.raw.get("provider") if response is not None else None
        declared: dict[str, str] = {}
        for entry in endpoints:
            provider = entry.get("provider_name")
            quantization = entry.get("quantization")
            if isinstance(provider, str) and isinstance(quantization, str):
                declared[provider] = quantization.lower()

        data: dict[str, Any] = {
            "declared_quantization": declared,
            "selected_provider": selected if isinstance(selected, str) else None,
            "unknown_count": sum(1 for q in declared.values() if q == "unknown"),
        }
        if not declared:
            return self._ev(
                "declared_quantization",
                0.0,
                family="capability",
                status=EvidenceStatus.UNSUPPORTED,
                detail="the OpenRouter endpoint list declared no quantization at all.",
                data=data,
            )

        relevant = (
            {selected: declared[selected]}
            if isinstance(selected, str) and selected in declared
            else declared
        )
        reduced = {
            provider: quantization
            for provider, quantization in relevant.items()
            if quantization in _QUANTIZATION_PENALTY
        }
        unknown_note = (
            " Roughly a third of OpenRouter endpoints declare 'unknown', and no label "
            "is audited by OpenRouter, so 'unknown' is not evidence either way."
        )

        if reduced:
            penalty = max(_QUANTIZATION_PENALTY[q] for q in reduced.values())
            listed = ", ".join(f"{p}: {q}" for p, q in sorted(reduced.items()))
            return self._ev(
                "declared_quantization",
                -penalty,
                cap=MODERATE,
                family="capability",
                detail=(
                    f"the endpoint declares reduced precision ({listed}) while serving "
                    f"{ctx.provider.target_model!r}. Published scores for that model were "
                    "measured on the publisher's own serving stack, so a narrower format "
                    "is a different artefact from the one the reference describes."
                    + unknown_note
                ),
                data=data,
            )

        if any(q in _FULL_PRECISION for q in relevant.values()):
            return self._ev(
                "declared_quantization",
                0.25 * WEAK,
                family="capability",
                detail=(
                    "the endpoint declares full-precision serving ("
                    + ", ".join(f"{p}: {q}" for p, q in sorted(relevant.items()))
                    + "). Self-reported and unaudited, hence barely any weight."
                    + unknown_note
                ),
                data=data,
            )

        return self._ev(
            "declared_quantization",
            0.0,
            family="capability",
            detail=(
                "every relevant endpoint declares 'unknown' quantization."
                + unknown_note
            ),
            data=data,
        )

    # ----------------------------------------------------------------- headers

    def _headers(self, response: ChatResponse) -> Evidence:
        """Record infrastructure headers with zero weight, deliberately."""
        observed: dict[str, Any] = {}
        for key in _INFRASTRUCTURE_HEADERS:
            value = response.http_headers.get(key)
            if value is None:
                continue
            observed[key] = _id_shape(value) if key.endswith(("request-id", "requestid")) else value

        return self._ev(
            "infrastructure_headers",
            0.0,
            detail=(
                "recorded, not weighed: "
                + (", ".join(sorted(observed)) if observed else "none of the usual headers")
                + ". These identify a CDN, a proxy and a request-id scheme, which is "
                "infrastructure rather than weights."
            ),
            data={"headers": observed},
        )

    # ------------------------------------------------------------------ helpers

    def _signature(self, ctx: ProbeContext) -> FamilySignature | None:
        if ctx.snapshot is None:
            return None
        return ctx.snapshot.family_signature(ctx.adapter.family.value)

    def _ev(
        self,
        label: str,
        llr: float,
        *,
        cap: float = WEAK,
        family: str | None = None,
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
            cap=cap,
            family=family or self.family,
            status=status,
            detail=detail,
            data=data or {},
            cost_usd=cost_usd,
            tokens=tokens,
            duration_s=duration_s,
        )


# --------------------------------------------------------------------------- #
# Envelope weighting
# --------------------------------------------------------------------------- #


def _envelope_weight(ctx: ProbeContext) -> tuple[float, str]:
    """How much the response envelope says about *model* identity here.

    Returns a multiplier and, when it is zero, the sentence explaining why.
    """
    spoken = ctx.adapter.family.value
    if ctx.reference is None:
        return 0.0, (
            f"{ctx.provider.target_model!r} is absent from the reference snapshot, so "
            "whether this protocol is its first-party API is unknown; the envelope is "
            "recorded but not weighed."
        )
    if ctx.reference.family == spoken:
        return 1.0, ""
    return 0.0, (
        f"the claimed model's first-party API is {ctx.reference.family!r} while this "
        f"endpoint speaks {spoken!r}; a compatible proxy rewrites the envelope as a "
        "matter of course, so the envelope is recorded but not weighed."
    )


def _weigh(llr: float, weight: float) -> float:
    """Scale an envelope finding, returning a clean zero when it is not weighed."""
    return llr * weight if weight else 0.0


def _with_caveat(detail: str, weight: float, caveat: str) -> str:
    if weight or not caveat:
        return detail
    return f"{detail} Not weighed: {caveat}"


# --------------------------------------------------------------------------- #
# Identifier handling
# --------------------------------------------------------------------------- #


def _canonical_model_id(name: str) -> str:
    """Reduce a model identifier to the stem two namings can be compared on.

    Removes an OpenRouter variant suffix (``:free``), an ``author/`` prefix, a
    Bedrock-style dotted region and vendor prefix, and a trailing snapshot date
    or version tag. The last one deserves a caveat: for the Claude 4.6
    generation onward a dateless id names a pinned snapshot rather than an
    evergreen alias, so equality after stripping means "the same model line",
    not necessarily the same snapshot.
    """
    ident = name.strip().lower()
    ident = ident.split(":", 1)[0]
    ident = ident.rsplit("/", 1)[-1]
    for _ in range(3):
        stripped = _DOTTED_PREFIX.sub("", ident)
        if stripped == ident:
            break
        ident = stripped
    while True:
        stripped = _VERSION_SUFFIX.sub("", ident)
        if stripped == ident:
            return stripped
        ident = stripped


def _same_model(left: str, right: str) -> bool:
    canonical = _canonical_model_id(left)
    return bool(canonical) and canonical == _canonical_model_id(right)


def _id_shape(value: str) -> str:
    """A compact structural pattern for an opaque identifier.

    The pattern, not the value, is what identifies a request-id scheme, and it
    can be published in a report without leaking anything about a real request.
    """
    parts: list[str] = []
    for chunk in re.split(r"([^A-Za-z0-9])", value[:96]):
        if not chunk:
            continue
        if len(chunk) == 1 and not chunk.isalnum():
            parts.append(chunk)
        elif re.fullmatch(r"[0-9a-f]{8,}", chunk):
            parts.append(f"hex{{{len(chunk)}}}")
        elif chunk.isdigit():
            parts.append(f"9{{{len(chunk)}}}")
        elif chunk.isalpha():
            parts.append(f"{'a' if chunk.islower() else 'A'}{{{len(chunk)}}}")
        else:
            parts.append(f"w{{{len(chunk)}}}")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Response inspection
# --------------------------------------------------------------------------- #


def _dig(node: Any, *path: str | int) -> Any:
    """Nested lookup that also walks list indices, unlike ``ChatResponse.raw_get``."""
    for key in path:
        if isinstance(key, int):
            if not isinstance(node, list) or key >= len(node):
                return None
            node = node[key]
        elif isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return None
    return node


def _has_path(node: dict[str, Any], dotted: str) -> bool:
    """Whether a dotted usage key such as ``cache_creation.ephemeral_5m`` exists."""
    return _dig(node, *dotted.split(".")) is not None


def _raw_finish_reason(response: ChatResponse) -> str | None:
    """The stop reason exactly as the provider spelled it.

    :class:`~llmverify.types.FinishReason` deliberately normalises across
    protocols, which throws away the vocabulary that identifies one.
    """
    for path in (
        ("stop_reason",),
        ("choices", 0, "finish_reason"),
        ("candidates", 0, "finishReason"),
    ):
        value = _dig(response.raw, *path)
        if isinstance(value, str) and value:
            return value
    return None
