"""Five short tool-use experiments, read as a behavioural profile.

Tool calling is the richest cheap fingerprint an endpoint offers, because every
family has its own habits: whether prose accompanies a call, whether two
independent lookups come back as two calls or one call followed by a second
turn, how a forced choice is honoured, what happens to a property the schema
forbids. None of that is specified by the protocol, so all of it is a
consequence of the weights and of the stack around them.

**The stack matters more than the weights here, and the weighting says so.**
The field evidence on this is unambiguous: identical open weights served by
different providers produced wildly different tool reliability, and the cause
was serving software -- old builds, wrong defaults, home-grown parsers for the
model's call syntax -- far more often than quantization or a different
checkpoint. Malformed ``arguments_raw`` in particular is a parser bug: the model
emitted something the harness could not turn back into JSON. So a malformed
result here is reported with almost no weight and with that explanation in the
evidence detail, while the observations that survive a bad parser -- no call at
all, a call emitted under ``tool_choice: "none"``, a value outside a declared
enum -- carry what little weight this probe has.

**The reference records no per-model tool expectations**, so most of this probe
is a profile rather than a verdict. Two findings are model-independent enough to
weigh: an endpoint that emits no tool call for a single unambiguous prompt is
not serving a current frontier model, and an endpoint that calls a tool after
being told not to has a validator that does not implement the protocol. The
``tool_use`` family caps at MODERATE, which is the right ceiling for a signal
this confounded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..config import redact
from ..errors import BudgetExhausted
from ..evidence import MODERATE, WEAK, Evidence, EvidenceStatus
from ..types import ChatRequest, ChatResponse, Message, Role, ToolSpec
from . import Probe, ProbeContext, register_probe
from .structured_output import validate_instance

__all__ = ["RESERVE_TOOL", "TEMPERATURE_TOOL", "ToolCallingProbe"]

#: The obvious tool. One required string argument and one optional enum, which
#: is the shape every model has seen thousands of times, so failing to call it
#: is a statement about the endpoint rather than about the schema.
TEMPERATURE_TOOL = ToolSpec(
    name="get_current_temperature",
    description="Look up the current outdoor temperature in a named city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, for example 'Oslo'."},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
        "additionalProperties": False,
    },
)

#: The awkward tool: a required enum, a required nested object with its own
#: bounds, and a closed property set the prompt invites the model to violate.
RESERVE_TOOL = ToolSpec(
    name="reserve_room",
    description="Reserve a meeting room for a block of time.",
    parameters={
        "type": "object",
        "properties": {
            "room": {"type": "string", "enum": ["aurora", "borealis", "cascade"]},
            "window": {
                "type": "object",
                "properties": {
                    "start_hour": {"type": "integer", "minimum": 0, "maximum": 23},
                    "duration_minutes": {"type": "integer", "minimum": 15, "maximum": 240},
                },
                "required": ["start_hour", "duration_minutes"],
                "additionalProperties": False,
            },
            "attendees": {"type": "integer", "minimum": 1, "maximum": 12},
        },
        "required": ["room", "window", "attendees"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True, slots=True)
class _Case:
    """One tool-use experiment."""

    key: str
    prompt: str
    tools: tuple[ToolSpec, ...]
    tool_choice: str | None
    max_tokens: int = 512


#: Ordered so that the two cases carrying weight are measured first and a
#: budget-truncated run still says something.
CASES: tuple[_Case, ...] = (
    _Case(
        "single_call",
        "What is the current outdoor temperature in Oslo right now? Use the tool.",
        (TEMPERATURE_TOOL,),
        "auto",
    ),
    _Case(
        "tool_choice_none",
        "What is the current outdoor temperature in Oslo? Answer from your own "
        "knowledge in one sentence and do not call any tool.",
        (TEMPERATURE_TOOL,),
        "none",
    ),
    _Case(
        "parallel_calls",
        "I need the current outdoor temperature in Oslo and in Lima. They are "
        "independent lookups; get both.",
        (TEMPERATURE_TOOL,),
        "auto",
    ),
    _Case(
        "forced_choice",
        "Reserve the Borealis room for six people, starting at 14:00, for an hour and "
        "a half. Note that we will also need a projector in the room.",
        (RESERVE_TOOL,),
        "reserve_room",
    ),
    _Case(
        "complex_schema",
        "Book Borealis for six of us at 14:00 for ninety minutes, and record that a "
        "projector and a whiteboard are required.",
        (RESERVE_TOOL,),
        "auto",
    ),
)


@dataclass(slots=True)
class _Observation:
    """What one case produced."""

    key: str
    called: bool = False
    call_count: int = 0
    names: list[str] = field(default_factory=list)
    valid_json: bool | None = None
    schema_violations: list[str] = field(default_factory=list)
    prose_chars: int = 0
    finish_reason: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "called": self.called,
            "call_count": self.call_count,
            "tool_names": self.names,
            "arguments_valid_json": self.valid_json,
            "schema_violations": self.schema_violations[:8],
            "prose_chars": self.prose_chars,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


@register_probe
class ToolCallingProbe(Probe):
    """Emission, arity, argument well-formedness, tool_choice, and schema adherence."""

    name: ClassVar[str] = "tool_calling"
    layer: ClassVar[int] = 2
    family: ClassVar[str] = "tool_use"
    order: ClassVar[int] = 80
    estimated_requests: ClassVar[int] = len(CASES)
    description: ClassVar[str] = (
        "Tool-call emission, parallel calls, tool_choice compliance, argument "
        "well-formedness and adherence to an enum/nested/closed schema."
    )

    def applicable(self, ctx: ProbeContext) -> bool:
        """Only where the protocol can carry tool definitions."""
        return bool(ctx.adapter.capabilities.tools)

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        started = time.perf_counter()
        observations, truncated, cost, tokens = await self._measure(ctx)
        elapsed = time.perf_counter() - started

        if not observations:
            return [
                self._ev(
                    "tool_call_profile",
                    0.0,
                    status=EvidenceStatus.TRUNCATED,
                    detail="the budget ended before any tool-use case could be run.",
                    duration_s=elapsed,
                )
            ]

        profile = {key: obs.as_dict() for key, obs in observations.items()}
        ctx.shared["tool_call_profile"] = profile
        data: dict[str, Any] = {
            "profile": profile,
            "cases_run": list(observations),
            "truncated": truncated,
        }
        charged = {"cost_usd": cost, "tokens": tokens, "duration_s": elapsed}

        evidence = [self._emission(ctx, observations, data, charged)]
        for item in (
            self._arguments(observations, data),
            self._choice_compliance(observations, data),
            self._schema_adherence(observations, data),
            self._shape(observations, data),
        ):
            if item is not None:
                evidence.append(item)
        return evidence

    # ------------------------------------------------------------- measurement

    async def _measure(
        self, ctx: ProbeContext
    ) -> tuple[dict[str, _Observation], bool, float, int]:
        """Run each case once, sequentially, stopping cleanly on budget."""
        observations: dict[str, _Observation] = {}
        cost = 0.0
        tokens = 0

        for case in CASES:
            try:
                ctx.budget.check()
            except BudgetExhausted:
                return observations, True, cost, tokens

            request = ChatRequest(
                messages=(Message(Role.USER, case.prompt),),
                max_tokens=case.max_tokens,
                tools=case.tools,
                tool_choice=case.tool_choice,
            )
            response, error = await ctx.adapter.try_chat(request)
            if response is None:
                ctx.budget.charge(None, None)
                observations[case.key] = _Observation(
                    key=case.key, error=redact(str(error))[:220]
                )
                continue

            cost += ctx.budget.charge(
                response.usage.input_tokens, response.usage.output_tokens
            )
            tokens += response.usage.total_tokens or 0
            observations[case.key] = _observe(case, response)

        return observations, False, cost, tokens

    # ------------------------------------------------------------ interpretation

    def _emission(
        self,
        ctx: ProbeContext,
        observations: dict[str, _Observation],
        data: dict[str, Any],
        charged: dict[str, Any],
    ) -> Evidence:
        """Did the endpoint call the obvious tool at all?"""
        single = observations.get("single_call")
        if single is None or single.error is not None:
            return self._ev(
                "tool_call_emitted",
                0.0,
                status=EvidenceStatus.ERROR,
                detail=(
                    "the single-tool case did not complete: "
                    f"{single.error if single is not None else 'not run'}."
                ),
                data=data,
                **charged,
            )

        if single.called and single.names[:1] == [TEMPERATURE_TOOL.name]:
            return self._ev(
                "tool_call_emitted",
                0.5 * WEAK,
                detail=(
                    f"the endpoint emitted {single.call_count} call(s) to "
                    f"{TEMPERATURE_TOOL.name} for a prompt that plainly needed one. "
                    "Expected of any current model, so this supports the claim only "
                    "mildly."
                ),
                data=data,
                **charged,
            )

        if single.called:
            return self._ev(
                "tool_call_emitted",
                -WEAK,
                detail=(
                    f"the endpoint called {', '.join(single.names) or 'an unnamed tool'} "
                    f"rather than {TEMPERATURE_TOOL.name}, which is the only tool it was "
                    "given. Calling a tool that was not offered points at a translating "
                    "layer rewriting the tool block."
                ),
                data=data,
                **charged,
            )

        return self._ev(
            "tool_call_emitted",
            -MODERATE,
            detail=(
                "the endpoint emitted no tool call for a single unambiguous prompt with a "
                f"single obvious tool, answering with {single.prose_chars} characters of "
                "prose instead. No model marketed at this tier fails that, so either the "
                "weights are not what is claimed or the serving stack is not passing the "
                "tool definitions through."
            ),
            data=data,
            **charged,
        )

    def _arguments(
        self, observations: dict[str, _Observation], data: dict[str, Any]
    ) -> Evidence | None:
        """Whether ``arguments_raw`` parsed. Weighted almost to nothing, on purpose."""
        judged = {
            key: obs.valid_json for key, obs in observations.items() if obs.valid_json is not None
        }
        if not judged:
            return None
        malformed = sorted(key for key, ok in judged.items() if not ok)
        if not malformed:
            return self._ev(
                "arguments_wellformed",
                0.25 * WEAK,
                detail=(
                    f"tool-call arguments parsed as JSON in all {len(judged)} cases that "
                    "produced a call."
                ),
                data={**data, "cases_judged": sorted(judged)},
            )
        return self._ev(
            "arguments_wellformed",
            -0.25 * WEAK,
            cap=WEAK,
            detail=(
                f"tool-call arguments did not parse as JSON in {len(malformed)} case(s) "
                f"({', '.join(malformed)}). Deliberately weighed at almost nothing: "
                "malformed arguments are overwhelmingly a serving-stack defect -- a "
                "home-grown parser for the model's call syntax, or an old build -- and the "
                "field evidence is that buggy parsers, not quantization or different "
                "weights, dominate tool-call unreliability. It says something is wrong with "
                "the harness, not with the checkpoint."
            ),
            data={**data, "malformed_cases": malformed},
        )

    def _choice_compliance(
        self, observations: dict[str, _Observation], data: dict[str, Any]
    ) -> Evidence | None:
        """``tool_choice`` obedience in both directions."""
        none_case = observations.get("tool_choice_none")
        forced = observations.get("forced_choice")
        findings: list[str] = []
        llr = 0.0

        if none_case is not None and none_case.error is None:
            if none_case.called:
                findings.append(
                    "a tool was called under tool_choice 'none', which a conforming "
                    "implementation makes impossible rather than merely discouraged"
                )
                llr -= WEAK
            else:
                findings.append("tool_choice 'none' was respected")

        if forced is not None and forced.error is None:
            if forced.called and forced.names[:1] == [RESERVE_TOOL.name]:
                findings.append("a forced tool_choice produced exactly that tool")
            elif forced.called:
                findings.append(
                    f"a forced tool_choice produced {', '.join(forced.names)} instead of "
                    f"{RESERVE_TOOL.name}"
                )
                llr -= WEAK
            else:
                findings.append(
                    "a forced tool_choice produced no call at all, which the parameter "
                    "exists to prevent"
                )
                llr -= WEAK

        if not findings:
            return None
        return self._ev(
            "tool_choice_respected",
            llr,
            detail=(
                "; ".join(findings)
                + ". Both directions describe the endpoint's request validator, which is "
                "part of the serving stack rather than of the weights, so neither is weighed "
                "above weak."
            ),
            data=data,
        )

    def _schema_adherence(
        self, observations: dict[str, _Observation], data: dict[str, Any]
    ) -> Evidence | None:
        """Enum, nested object and closed-property-set adherence in the arguments."""
        cases = {
            key: obs
            for key, obs in observations.items()
            if key in ("forced_choice", "complex_schema") and obs.valid_json
        }
        if not cases:
            return None

        violations = sorted({v for obs in cases.values() for v in obs.schema_violations})
        if not violations:
            return self._ev(
                "tool_schema_adherence",
                0.5 * WEAK,
                detail=(
                    f"arguments satisfied the awkward schema in all {len(cases)} case(s): "
                    "the enum value was one of the three offered, the nested time window "
                    "carried both required integers within their bounds, and nothing was "
                    "smuggled in as an extra property despite the prompt asking for a "
                    "projector."
                ),
                data=data,
            )
        return self._ev(
            "tool_schema_adherence",
            -WEAK,
            detail=(
                f"tool arguments violated the declared schema: {'; '.join(violations[:3])}. "
                "A model that invents an enum member or an undeclared property is not being "
                "constrained by anything, which is ordinary for tool calls -- most stacks do "
                "not constrain them -- so this is weighed lightly and reported mainly as a "
                "profile."
            ),
            data={**data, "violations": violations[:12]},
        )

    def _shape(
        self, observations: dict[str, _Observation], data: dict[str, Any]
    ) -> Evidence | None:
        """Parallel-call arity and prose-alongside-call habits. Reported, not weighed.

        Both vary between model families and between versions of one family, and
        the reference snapshot records no expectation for either. Scoring them
        would mean inventing the expectation being scored against.
        """
        parallel = observations.get("parallel_calls")
        if parallel is None:
            return None
        with_prose = sorted(
            key for key, obs in observations.items() if obs.called and obs.prose_chars > 0
        )
        return self._ev(
            "tool_use_shape",
            0.0,
            detail=(
                f"two independent lookups produced {parallel.call_count} call(s) in one "
                f"turn; {len(with_prose)} of the cases that called a tool also emitted prose "
                "alongside it. Recorded as a family fingerprint: no reference records what "
                "either should be for the claimed model, so neither is weighed."
            ),
            data={
                **data,
                "parallel_call_count": parallel.call_count,
                "cases_with_prose": with_prose,
            },
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


def _observe(case: _Case, response: ChatResponse) -> _Observation:
    """Read one response into the profile fields the probe scores."""
    calls = response.tool_calls
    schemas = {tool.name: tool.parameters for tool in case.tools}

    valid_json: bool | None = None
    violations: list[str] = []
    for call in calls:
        parsed_ok = call.arguments is not None
        valid_json = parsed_ok if valid_json is None else (valid_json and parsed_ok)
        schema = schemas.get(call.name)
        if parsed_ok and isinstance(schema, dict) and call.arguments is not None:
            violations.extend(validate_instance(schema, call.arguments))

    return _Observation(
        key=case.key,
        called=bool(calls),
        call_count=len(calls),
        names=[call.name for call in calls],
        valid_json=valid_json,
        schema_violations=violations,
        prose_chars=len((response.text or "").strip()),
        finish_reason=response.finish_reason.value,
    )
