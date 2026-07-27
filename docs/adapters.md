# Writing an adapter

An adapter is the only place in llmverify that knows a wire format. Every probe
is written against `llmverify.types`, so supporting a new provider protocol means
writing one subclass and registering it. Nothing else changes.

If your provider speaks the OpenAI Chat Completions protocol — vLLM, SGLang,
LiteLLM, OpenRouter, Ollama, LM Studio and essentially every reseller do — you
do not need an adapter at all. Set `api: openai` and point `base_url` at it.

## The contract

Subclass `llmverify.adapters.base.Adapter` and provide four class attributes and
three methods.

| member | required | what it is |
|---|---|---|
| `name` | yes | registry key, matched against `ProviderConfig.api` |
| `family` | yes | `ApiFamily.OPENAI` / `ANTHROPIC` / `GEMINI` / `UNKNOWN` |
| `default_base_url` | yes | first-party endpoint, used when the config omits one |
| `default_auth_scheme` | no | `bearer` (default), `x-api-key`, `query` or `none` |
| `chat_path` | no | path appended to `base_url`; defaults to `/chat/completions` |
| `capabilities` | no | a `Capabilities` instance; see below |
| `build_payload` | yes | `ChatRequest` → this protocol's JSON body |
| `parse_response` | yes | `HttpResult` → `ChatResponse` |
| `parse_stream` | yes | collected SSE frames → `ChatResponse` |

`Capabilities` describes what the *protocol* can express, not what a given
endpoint honours. The gap between the two is precisely what the `api_surface`
probe measures, so declare the protocol honestly and let the probe find the
endpoint's behaviour.

The base class already gives you the HTTP client with retry and timing,
authentication, `chat`, `try_chat`, `list_models`, `probe_parameter` and async
context-manager lifecycle. Override `count_tokens` if your provider has a
tokenizer endpoint — that is the cleanest possible tokenizer fingerprint,
because it removes generation from the measurement entirely.

### Two rules that are not optional

**Omit, never null.** A field the request left as `None` must be *absent* from
the payload, not present as `null`. Endpoints answer the two differently — some
reject an unknown-typed `null`, some coerce it, some ignore it — and the
`api_surface` probe reads exactly that difference to tell a parameter that was
*rejected* from one that was silently *dropped*. Nulling breaks the probe.

**Lose nothing.** Everything the endpoint volunteered must survive into
`ChatResponse.raw`, with the usage object's shape intact. Compatible stacks
differ from each other mostly in fields a normal client throws away, so those
fields are the evidence.

## A complete worked example

Suppose Nimbus Inference serves models over a protocol of its own: `POST
/v1/generate`, a bearer token, turns instead of messages, and a `counts` object
instead of `usage`.

```python
"""Adapter for Nimbus Inference's /v1/generate protocol."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from llmverify.adapters._http import HttpResult
from llmverify.adapters.base import Adapter, Capabilities, register_adapter
from llmverify.types import (
    ApiFamily,
    ChatRequest,
    ChatResponse,
    FinishReason,
    Timing,
    Usage,
)

__all__ = ["NimbusAdapter"]

_STOP_REASONS: dict[str, FinishReason] = {
    "complete": FinishReason.STOP,
    "limit": FinishReason.LENGTH,
    "filtered": FinishReason.CONTENT_FILTER,
}


@register_adapter
class NimbusAdapter(Adapter):
    """Nimbus Inference's ``/v1/generate`` protocol."""

    name: ClassVar[str] = "nimbus"
    family: ClassVar[ApiFamily] = ApiFamily.UNKNOWN
    default_base_url: ClassVar[str] = "https://api.nimbus.example/v1"
    default_auth_scheme: ClassVar[str] = "bearer"
    chat_path: ClassVar[str] = "/generate"

    capabilities: ClassVar[Capabilities] = Capabilities(
        logprobs=False,
        seed=True,
        tools=False,
        vision=False,
        structured_output=False,
        reasoning_effort=True,
        list_models_endpoint=True,
        streaming=False,
    )

    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "turns": [
                {"speaker": message.role.value, "text": message.text}
                for message in request.messages
            ],
            "limit": request.max_tokens,
        }
        # Unset fields are omitted, never sent as null: the api-surface probe
        # tells "rejected" from "silently dropped" by exactly this difference.
        for wire_name, value in (
            ("randomness", request.temperature),
            ("nucleus", request.top_p),
            ("seed", request.seed),
            ("effort", request.reasoning_effort),
        ):
            if value is not None:
                payload[wire_name] = value
        if request.stop:
            payload["halt_on"] = list(request.stop)
        # Merged last and verbatim, including malformed values: probes send
        # those deliberately to see how the endpoint complains.
        payload.update(request.extra_body)
        return payload

    def parse_response(self, result: HttpResult) -> ChatResponse:
        body = result.json if isinstance(result.json, dict) else {}
        counts = body.get("counts") if isinstance(body.get("counts"), dict) else {}
        return ChatResponse(
            text=str(body.get("text") or ""),
            model_reported=_as_str(body.get("served_model")),
            response_id=_as_str(body.get("request_id")),
            finish_reason=_STOP_REASONS.get(str(body.get("stop")), FinishReason.OTHER),
            usage=Usage(
                input_tokens=_as_int(counts.get("in")),
                output_tokens=_as_int(counts.get("out")),
                # The usage object's shape discriminates API families even when
                # the numbers agree, so it is kept verbatim.
                raw=dict(counts),
            ),
            timing=Timing(total_s=result.total_s, ttft_s=result.ttft_s),
            raw=dict(body),
            http_status=result.status,
            http_headers=dict(result.headers),
            api_family=self.family,
        )

    def parse_stream(self, result: HttpResult) -> ChatResponse:
        text: list[str] = []
        last: dict[str, Any] = {}
        for event in result.events:
            if isinstance(event.get("text"), str):
                text.append(event["text"])
            last = {**last, **event}
        merged = HttpResult(
            status=result.status,
            headers=result.headers,
            json={**last, "text": "".join(text)},
            text=json.dumps(last),
            total_s=result.total_s,
            ttft_s=result.ttft_s,
        )
        return self.parse_response(merged)


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
```

That is the whole adapter. Every probe that does not need a capability Nimbus
lacks now works against it.

## Registering it

`@register_adapter` is enough when your module is imported. For an adapter that
should be discoverable by name from the command line, publish an entry point in
the package that ships it:

```toml
# pyproject.toml of llmverify-nimbus
[project]
name = "llmverify-nimbus"
version = "0.1.0"
dependencies = ["llmverify>=0.1"]

[project.entry-points."llmverify.adapters"]
nimbus = "llmverify_nimbus.adapter:NimbusAdapter"
```

`available_adapters()` loads that group lazily, skips any entry point that fails
to import — a broken plugin must not break the tool — and refuses to shadow a
built-in name. After `pip install llmverify-nimbus`:

```console
$ llmverify check --api nimbus --base-url https://api.nimbus.example/v1 \
      --model nimbus-large-1 --claimed-model claude-opus-5 \
      --api-key-env NIMBUS_API_KEY --layers 0,1
```

or in a provider YAML:

```yaml
name: nimbus
api: nimbus
base_url: https://api.nimbus.example/v1
model: nimbus-large-1
claimed_model: claude-opus-5
api_key_env: NIMBUS_API_KEY
```

## Two other plugin groups

Probes and benchmarks are discovered the same way, with the same
skip-on-failure and no-shadowing rules.

```toml
[project.entry-points."llmverify.probes"]
my_probe = "my_package.probes:MyProbe"

[project.entry-points."llmverify.benchmarks"]
my_benchmark = "my_package.benchmarks:MyBenchmark"
```

### A probe

Subclass `llmverify.probes.Probe`, set `name`, `layer`, `family`, `description`,
`estimated_requests` and `order`, and implement `async def run(ctx) ->
list[Evidence]`. Optionally override `applicable(ctx)`.

```python
from __future__ import annotations

from typing import ClassVar

from llmverify.evidence import Evidence, WEAK
from llmverify.probes import Probe, ProbeContext, register_probe
from llmverify.types import ChatRequest, Message, Role


@register_probe
class GreetingProbe(Probe):
    """One trivial round trip, as a template."""

    name: ClassVar[str] = "greeting"
    layer: ClassVar[int] = 1
    family: ClassVar[str] = "misc"
    order: ClassVar[int] = 200
    estimated_requests: ClassVar[int] = 1
    description: ClassVar[str] = "Whether the endpoint answers a one-word prompt."

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        ctx.budget.check()
        request = ChatRequest(
            messages=(Message(Role.USER, "Reply with the single word: ok"),),
            max_tokens=8,
        )
        response, error = await ctx.adapter.try_chat(request)
        if error is not None:
            return [ctx.unsupported(self.name, f"no usable response: {error}")]
        ctx.budget.charge(
            response.usage.input_tokens, response.usage.output_tokens
        )
        answered = response.text.strip().lower().startswith("ok")
        return [
            Evidence(
                probe=self.name,
                label="greeting",
                llr=WEAK if answered else -WEAK,
                cap=WEAK,
                family=self.family,
                detail=f"the endpoint replied {response.text.strip()!r}.",
                data={"text": response.text},
            )
        ]
```

Three rules. A probe must **never raise for an expected negative result**: "this
endpoint has no logprobs" is `ctx.unsupported(...)`, not a crash. `llr` must be
signed from the point of view of the provider's *claim* — positive supports it,
negative refutes it. And use `ctx.rng(salt)` rather than the global `random`
module, so that skipping one probe does not change every later probe's sampling.

Anything a probe writes to `ctx.shared` is visible to later probes; the runner
serialises any probe that touches it, so no locking is needed.

### A benchmark

Subclass `llmverify.benchmarks.base.Benchmark`, declare `name`,
`reference_key`, the HuggingFace spec (or `hf_dataset = None` for a generated
set), `licence`, `gated`, `discriminative`, `score_spread` and `description`, and
implement `async def load(loader, *, limit=None) -> list[BenchmarkItem]`. The
base class handles rendering (verbatim, paraphrased, shuffled), the
`ANSWER: <x>` extractor and exact-match grading; override `render`, `grade` or
`extract_answer` only when your items need something else.

Grading must be deterministic and must never call another model. An LLM judge
would add a second trust assumption to a tool whose entire purpose is to check a
trust assumption.

Two fields carry more weight than their size suggests. `score_spread` is the
standard deviation of published scores across current models, in percentage
points; the runner uses it to rank benchmarks by information per request. Set
`discriminative = False` when your benchmark cannot separate current frontier
models — GPQA Diamond does exactly this — and say so rather than letting the
runner spend a budget on a comparison it cannot win.

## Testing an adapter

The fastest check is a local mock speaking your protocol, plus a layer-0/1 run:

```python
import asyncio

import my_package.adapter  # noqa: F401  -- registers the adapter
from llmverify.config import ProviderConfig, RunConfig
from llmverify.runner import verify_provider

provider = ProviderConfig(
    name="nimbus",
    api="nimbus",
    base_url="http://127.0.0.1:8932/v1",
    model="nimbus-large-1",
    claimed_model="claude-opus-5",
    auth_scheme="none",
)
result = asyncio.run(verify_provider(provider, RunConfig(layers=(0, 1))))
print(result.verdict.verdict.value, result.verdict.probability)
```

Check three things in the output. Every probe should reach `ok`, `skipped` or
`unsupported` rather than `error` — an `error` is usually a parse bug in the
adapter, not a finding about the endpoint. `ChatResponse.raw` should contain
every field your mock sent. And `usage` should be populated, since the budget,
the tokenizer probe and the token-accounting probe all read it.
