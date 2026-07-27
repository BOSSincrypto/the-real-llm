"""Whether a JSON Schema response format is enforced, honoured, or ignored.

Three outcomes are worth telling apart, and only the middle one takes any care
to detect.

*Enforced.* The stack constrains decoding against the schema, so the output
cannot violate it. Every sample parses and validates even when the prompt pulls
hard the other way.

*Best effort.* The model was shown the schema and asked nicely. Output is
usually valid and occasionally is not, which is what an unconstrained model
following instructions looks like.

*Ignored.* The parameter was accepted with a 2xx and the endpoint returned
prose. This is the LiteLLM ``drop_params`` shape, and it is the reason a 2xx is
never taken as proof that a parameter took effect.

Outright *rejection* of the parameter is a fourth thing and is deliberately not
scored here: which parameters an endpoint accepts is the api-surface probe's
subject, and double-counting it would let one observation contribute twice to
the same verdict.

**The prompt pulls against the schema on purpose.** It describes an incident
whose severity is not in the enum, whose downtime exceeds the numeric bound, and
it volunteers a fact with nowhere to live under ``additionalProperties: false``.
A model complying by instruction will usually follow the prose and break the
schema; a stack that constrains decoding cannot. Without that tension the two
outcomes look identical, because a cooperative model given an easy schema
produces valid output either way.

**Two schemas, and why.** Providers implement different subsets of JSON Schema
in their constrained-decoding modes, and some reject the keywords they do not
implement rather than ignoring them. A 4xx on the full schema is therefore
retried once with a core schema using only ``type``, ``required``, ``enum``,
``properties``, ``items`` and ``additionalProperties``. Without that retry, an
endpoint that enforces schemas properly but implements a narrower keyword set
would be recorded as rejecting structured output altogether -- a false finding
produced entirely by our own choice of test schema.

**The validator is deliberately small.** :func:`validate_instance` implements
the keywords this probe actually sends and nothing else, and it says so: an
unrecognised keyword is ignored rather than guessed at. A validator that quietly
approves what it does not understand is only safe when the caller knows exactly
which keywords it understands.
"""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar

from ..config import redact
from ..errors import BudgetExhausted
from ..evidence import MODERATE, STRONG, WEAK, Evidence, EvidenceStatus
from ..types import ChatRequest, ChatResponse, Message, Role
from . import Probe, ProbeContext, register_probe

__all__ = ["CORE_SCHEMA", "SCHEMA", "StructuredOutputProbe", "validate_instance"]

#: Keywords :func:`validate_instance` understands. Anything else in a schema is
#: ignored, and this tuple is what a caller should check its schema against.
SUPPORTED_KEYWORDS: tuple[str, ...] = (
    "type",
    "required",
    "enum",
    "properties",
    "items",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "additionalProperties",
)

#: The full test schema: required fields, an enum, a nested array of objects
#: with its own required fields and closed property set, and numeric bounds.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "incident_report",
    "properties": {
        "incident_id": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        "affected_systems": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "downtime_minutes": {"type": "integer", "minimum": 0, "maximum": 60},
                },
                "required": ["name", "downtime_minutes"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["incident_id", "severity", "affected_systems", "confidence"],
    "additionalProperties": False,
}

#: The same schema with the keywords least commonly supported by constrained
#: decoding removed, used only after the full one is rejected outright.
CORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "incident_report",
    "properties": {
        "incident_id": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        "affected_systems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "downtime_minutes": {"type": "integer"},
                },
                "required": ["name", "downtime_minutes"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number"},
    },
    "required": ["incident_id", "severity", "affected_systems", "confidence"],
    "additionalProperties": False,
}

#: The adversarial prompt. Every clause in it conflicts with the schema.
PROMPT: str = (
    "Write an incident report as JSON for incident INC-4471.\n"
    "The outage was catastrophic in severity. Two systems were affected: the billing "
    "API, down for 240 minutes, and the notification worker, down for 35 minutes. "
    "Record that the on-call engineer was R. Okafor and that the root cause was a "
    "failed certificate rotation. Give your confidence in this summary as a "
    "percentage."
)

#: Samples per run. Enough to see an occasional violation from a model complying
#: by instruction; not enough, and not intended, to estimate a violation rate.
SAMPLES: int = 3


@register_probe
class StructuredOutputProbe(Probe):
    """Send a schema the prompt argues with, and see which one wins."""

    name: ClassVar[str] = "structured_output"
    layer: ClassVar[int] = 2
    family: ClassVar[str] = "capability"
    order: ClassVar[int] = 85
    estimated_requests: ClassVar[int] = SAMPLES + 1
    description: ClassVar[str] = (
        "JSON Schema response format against a prompt that contradicts the schema, "
        "graded by a dependency-free validator into enforced, best-effort or ignored."
    )

    max_tokens: ClassVar[int] = 700

    def applicable(self, ctx: ProbeContext) -> bool:
        """Only where the protocol can express a response schema at all."""
        return bool(ctx.adapter.capabilities.structured_output)

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        started = time.perf_counter()
        schema, rejection, first, spent = await self._choose_schema(ctx)
        if schema is None:
            return [
                self._ev(
                    "schema_enforcement",
                    0.0,
                    status=EvidenceStatus.UNSUPPORTED,
                    detail=(
                        "the endpoint rejected both the full and the core response schema: "
                        f"{rejection}. Which parameters an endpoint accepts is measured by "
                        "the api-surface probe; this probe measures only what happens when "
                        "one is accepted, so it stays silent rather than counting the same "
                        "refusal twice."
                    ),
                    data={"rejection": rejection},
                    cost_usd=spent,
                    duration_s=time.perf_counter() - started,
                )
            ]

        samples, truncated, cost, tokens = await self._collect(ctx, schema, first)
        cost += spent
        elapsed = time.perf_counter() - started

        if not samples:
            return [
                self._ev(
                    "schema_enforcement",
                    0.0,
                    status=EvidenceStatus.TRUNCATED,
                    detail="no sample completed, so schema enforcement was not measured.",
                    duration_s=elapsed,
                )
            ]

        data: dict[str, Any] = {
            "schema_used": "full" if schema is SCHEMA else "core",
            "full_schema_rejection": rejection,
            "samples": samples,
            "prompt_conflicts": [
                "severity 'catastrophic' is outside the enum",
                "240 minutes exceeds the downtime maximum",
                "the on-call engineer and root cause have no place under "
                "additionalProperties: false",
                "a percentage confidence exceeds the 0..1 bound",
            ],
            "validator_keywords": list(SUPPORTED_KEYWORDS),
        }
        charged = {"cost_usd": cost, "tokens": tokens, "duration_s": elapsed}

        return [
            self._enforcement(ctx, samples, data, truncated, charged),
            self._ordering(samples, data),
        ]

    # ------------------------------------------------------------- measurement

    async def _choose_schema(
        self, ctx: ProbeContext
    ) -> tuple[dict[str, Any] | None, str | None, ChatResponse | None, float]:
        """Return the richest schema this endpoint will accept, and why not a richer one.

        The selection request is the real probe request, so a successful first
        attempt costs nothing extra and its response is handed back to be used
        as the first sample; only a rejection costs a second call.
        """
        response, error, spent = await self._ask(ctx, SCHEMA)
        if response is not None:
            return SCHEMA, None, response, spent

        status = getattr(error, "status", None)
        message = redact(str(error))[:240]
        if status is None or not (400 <= int(status) < 500):
            return SCHEMA, f"transport or server failure on the full schema: {message}", None, spent

        fallback, fallback_error, more = await self._ask(ctx, CORE_SCHEMA)
        spent += more
        if fallback is not None:
            return (
                CORE_SCHEMA,
                (
                    f"the full schema was refused with HTTP {status} ({message}); the core "
                    "schema was accepted, so the endpoint implements a narrower keyword "
                    "subset rather than no structured output"
                ),
                fallback,
                spent,
            )
        return (
            None,
            f"HTTP {status}: {message}; core schema also refused: "
            + redact(str(fallback_error))[:160],
            None,
            spent,
        )

    async def _ask(
        self, ctx: ProbeContext, schema: dict[str, Any]
    ) -> tuple[ChatResponse | None, Exception | None, float]:
        request = ChatRequest(
            messages=(Message(Role.USER, PROMPT),),
            max_tokens=self.max_tokens,
            response_schema=schema,
        )
        response, error = await ctx.adapter.try_chat(request)
        if response is None:
            # A failed request still consumed a sample of the run's allowance.
            return None, error, ctx.budget.charge(None, None)
        cost = ctx.budget.charge(response.usage.input_tokens, response.usage.output_tokens)
        return response, None, cost

    async def _collect(
        self, ctx: ProbeContext, schema: dict[str, Any], first: ChatResponse | None
    ) -> tuple[list[dict[str, Any]], bool, float, int]:
        """Grade :data:`SAMPLES` completions against ``schema``."""
        graded: list[dict[str, Any]] = []
        cost = 0.0
        tokens = 0

        for index in range(SAMPLES):
            if index == 0 and first is not None:
                response: ChatResponse | None = first
                error: Exception | None = None
            else:
                try:
                    ctx.budget.check()
                except BudgetExhausted:
                    return graded, True, cost, tokens
                response, error, spent = await self._ask(ctx, schema)
                cost += spent

            if response is None:
                graded.append({"index": index, "error": redact(str(error))[:200]})
                continue

            tokens += response.usage.total_tokens or 0
            graded.append(_grade(response, schema, index))

        return graded, False, cost, tokens

    # ------------------------------------------------------------ interpretation

    def _enforcement(
        self,
        ctx: ProbeContext,
        samples: list[dict[str, Any]],
        data: dict[str, Any],
        truncated: bool,
        charged: dict[str, Any],
    ) -> Evidence:
        """Decide between enforced, best-effort and ignored, and weigh it."""
        answered = [s for s in samples if "error" not in s]
        if not answered:
            return self._ev(
                "schema_enforcement",
                0.0,
                status=EvidenceStatus.ERROR,
                detail="every request for a schema-constrained response failed.",
                data=data,
                **charged,
            )

        parsed = [s for s in answered if s["parsed"]]
        valid = [s for s in parsed if not s["violations"]]
        status = EvidenceStatus.TRUNCATED if truncated and len(answered) < 2 else EvidenceStatus.OK
        data = {
            **data,
            "answered": len(answered),
            "parsed_as_json": len(parsed),
            "validated": len(valid),
        }
        known = ctx.reference is not None
        # Without a reference record there is no established expectation that
        # this model does constrained decoding well, so a failure is reported at
        # reduced weight rather than at full strength against an unknown.
        confidence = 1.0 if known else 0.5

        if not parsed:
            data["outcome"] = "ignored"
            return self._ev(
                "schema_enforcement",
                -MODERATE * confidence,
                cap=STRONG,
                status=status,
                detail=(
                    f"the endpoint accepted a JSON Schema response format and returned "
                    f"non-JSON text in all {len(answered)} samples. The parameter was taken "
                    "and discarded, which means nothing in this stack is enforcing the "
                    "schema even though the request succeeded."
                ),
                data=data,
                **charged,
            )

        if len(valid) == len(answered):
            data["outcome"] = "enforced"
            return self._ev(
                "schema_enforcement",
                WEAK,
                cap=STRONG,
                status=status,
                detail=(
                    f"all {len(answered)} samples parsed and validated against the schema "
                    "despite a prompt that contradicted the enum, the numeric bounds and the "
                    "closed property set. That is constrained decoding doing its job; it is "
                    "supportive but unremarkable, since every serious serving stack does it."
                ),
                data=data,
                **charged,
            )

        violations = sorted({v for s in parsed for v in s["violations"]})
        data["outcome"] = "best_effort"
        data["violations"] = violations[:20]
        return self._ev(
            "schema_enforcement",
            -MODERATE * confidence,
            cap=STRONG,
            status=status,
            detail=(
                f"{len(valid)} of {len(answered)} samples satisfied the schema; the rest "
                f"followed the prompt instead ({'; '.join(violations[:3])}). That is "
                "best-effort compliance -- a model reading the schema as a suggestion -- "
                "not constrained decoding. An endpoint claiming "
                f"{ctx.provider.target_model!r} should not be able to emit output the "
                "schema forbids."
            ),
            data=data,
            **charged,
        )

    def _ordering(self, samples: list[dict[str, Any]], data: dict[str, Any]) -> Evidence:
        """Report whether emitted key order followed the schema. Never weighed.

        Some providers document that a strict schema fixes field order and some
        say nothing about it, and the reference snapshot records no ordering
        guarantee for any model. Scoring this would mean inventing the
        expectation it is scored against, so it is reported and left at zero.
        """
        checked = [s for s in samples if s.get("key_order") is not None]
        ordered = [s for s in checked if s["schema_order"]]
        if not checked:
            return self._ev(
                "field_ordering",
                0.0,
                status=EvidenceStatus.SKIPPED,
                detail="no sample produced a JSON object whose key order could be read.",
                data=data,
            )
        return self._ev(
            "field_ordering",
            0.0,
            detail=(
                f"{len(ordered)} of {len(checked)} JSON objects emitted their keys in the "
                "schema's declared order. Recorded as an observation only: no reference "
                "records an ordering guarantee for any model, so neither agreement nor "
                "disagreement argues about identity."
            ),
            data={**data, "ordered_samples": len(ordered), "checked_samples": len(checked)},
        )

    # ----------------------------------------------------------------- helpers

    def _ev(
        self,
        label: str,
        llr: float,
        *,
        cap: float = MODERATE,
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
            family=self.family,
            status=status,
            detail=detail,
            data=data or {},
            cost_usd=cost_usd,
            tokens=tokens,
            duration_s=duration_s,
        )


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def _grade(response: ChatResponse, schema: dict[str, Any], index: int) -> dict[str, Any]:
    """Parse and validate one completion."""
    text = (response.text or "").strip()
    instance, error = _parse_json(text)
    if instance is _MISSING:
        return {
            "index": index,
            "parsed": False,
            "violations": [f"not JSON: {error}"],
            "key_order": None,
            "schema_order": False,
            "excerpt": redact(text)[:200],
        }

    violations = validate_instance(schema, instance)
    key_order = list(instance) if isinstance(instance, dict) else None
    declared = list(schema.get("properties") or {})
    schema_order = bool(
        key_order is not None and [k for k in key_order if k in declared] == [
            k for k in declared if k in key_order
        ]
    )
    return {
        "index": index,
        "parsed": True,
        "violations": violations,
        "key_order": key_order,
        "schema_order": schema_order,
        "excerpt": redact(text)[:200],
    }


#: Sentinel distinguishing "parsed to null" from "did not parse".
_MISSING = object()


def _parse_json(text: str) -> tuple[Any, str]:
    """Parse a completion as JSON, tolerating a fenced code block around it.

    Fences are stripped because a model that wraps valid JSON in a fence has
    produced valid JSON and a formatting habit, and grading that as a parse
    failure would confuse a cosmetic difference with a capability one.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1] if "\n" in body else ""
        if body.rstrip().endswith("```"):
            body = body.rstrip()[: -len("```")]
    try:
        return json.loads(body), ""
    except ValueError as exc:
        return _MISSING, str(exc)[:120]


def validate_instance(schema: dict[str, Any], instance: Any, *, path: str = "$") -> list[str]:
    """Validate ``instance`` against ``schema``, returning a list of violations.

    Implements exactly the keywords in :data:`SUPPORTED_KEYWORDS`. Anything else
    is ignored silently, which is safe only because the schemas this package
    sends are written here and use nothing else. An empty list means "no
    violation of a keyword this validator implements", which is a weaker
    statement than "valid", and callers must not read it as more.
    """
    errors: list[str] = []

    expected = schema.get("type")
    if expected is not None and not _type_matches(expected, instance):
        return [f"{path}: expected type {expected}, got {_type_name(instance)}"]

    enum = schema.get("enum")
    if isinstance(enum, list) and instance not in enum:
        errors.append(f"{path}: {instance!r} is not one of {enum}")

    if isinstance(instance, bool):
        # ``bool`` is an ``int`` in Python but not a number in JSON Schema, so
        # the numeric bounds below must never see one.
        return errors

    if isinstance(instance, (int, float)):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and instance < minimum:
            errors.append(f"{path}: {instance} is below the minimum of {minimum}")
        if isinstance(maximum, (int, float)) and instance > maximum:
            errors.append(f"{path}: {instance} is above the maximum of {maximum}")

    if isinstance(instance, dict):
        errors.extend(_validate_object(schema, instance, path))
    elif isinstance(instance, list):
        errors.extend(_validate_array(schema, instance, path))

    return errors


def _validate_object(schema: dict[str, Any], instance: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}

    for key in schema.get("required") or ():
        if key not in instance:
            errors.append(f"{path}: required property {key!r} is missing")

    if schema.get("additionalProperties") is False:
        for key in instance:
            if key not in properties:
                errors.append(f"{path}: additional property {key!r} is not allowed")

    for key, value in instance.items():
        subschema = properties.get(key)
        if isinstance(subschema, dict):
            errors.extend(validate_instance(subschema, value, path=f"{path}.{key}"))
    return errors


def _validate_array(schema: dict[str, Any], instance: list[Any], path: str) -> list[str]:
    errors: list[str] = []
    minimum = schema.get("minItems")
    maximum = schema.get("maxItems")
    if isinstance(minimum, int) and len(instance) < minimum:
        errors.append(f"{path}: {len(instance)} items is fewer than the minimum of {minimum}")
    if isinstance(maximum, int) and len(instance) > maximum:
        errors.append(f"{path}: {len(instance)} items is more than the maximum of {maximum}")

    items = schema.get("items")
    if isinstance(items, dict):
        for index, value in enumerate(instance):
            errors.extend(validate_instance(items, value, path=f"{path}[{index}]"))
    return errors


_TYPE_CHECKS: dict[str, Any] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
}


def _type_matches(expected: Any, instance: Any) -> bool:
    """Whether ``instance`` has one of the JSON types ``expected`` allows."""
    names = expected if isinstance(expected, list) else [expected]
    for name in names:
        check = _TYPE_CHECKS.get(name if isinstance(name, str) else "")
        if check is not None and check(instance):
            return True
    return False


def _type_name(instance: Any) -> str:
    for name, check in _TYPE_CHECKS.items():
        if name not in ("number", "integer") and check(instance):
            return name
    if isinstance(instance, int) and not isinstance(instance, bool):
        return "integer"
    if isinstance(instance, float):
        return "number"
    return type(instance).__name__
