"""Provider-neutral request/response data model.

Every adapter translates its wire format into these types, and every probe is
written against them. Nothing above the adapter layer may touch a raw payload
except through :attr:`ChatResponse.raw`, which is preserved verbatim precisely
so that identity probes can inspect provider-specific fields.

The design rule is: *never lose information*. A verifier's whole job is to
notice small discrepancies, so a field an adapter cannot map is kept in ``raw``
rather than dropped.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ApiFamily",
    "ChatRequest",
    "ChatResponse",
    "ContentPart",
    "FinishReason",
    "ImagePart",
    "Message",
    "ParamSupport",
    "Role",
    "TextPart",
    "ThinkingBlock",
    "Timing",
    "TokenLogprob",
    "ToolCall",
    "ToolSpec",
    "Usage",
]


class Role(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ApiFamily(str, enum.Enum):
    """Wire-protocol family.

    This is *not* the same thing as the model vendor: an OpenAI-compatible
    reseller can serve Claude weights, and that mismatch between advertised
    weights and observed wire family is itself a piece of evidence.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    UNKNOWN = "unknown"


class FinishReason(str, enum.Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    REFUSAL = "refusal"
    PAUSE_TURN = "pause_turn"
    OTHER = "other"


# --------------------------------------------------------------------------- #
# Content parts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TextPart:
    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    """Inline image. ``data`` is raw bytes; adapters base64-encode as needed."""

    data: bytes
    media_type: str = "image/png"


ContentPart = TextPart | ImagePart


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str | tuple[ContentPart, ...]
    #: Set on assistant turns that are being replayed back to the provider.
    tool_calls: tuple[ToolCall, ...] = ()
    #: Set on ``Role.TOOL`` turns.
    tool_call_id: str | None = None
    #: Opaque assistant reasoning being replayed verbatim (see ThinkingBlock).
    thinking: tuple[ThinkingBlock, ...] = ()

    @property
    def text(self) -> str:
        """Flatten content to text, ignoring non-text parts."""
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text for p in self.content if isinstance(p, TextPart))


def user(text: str) -> Message:
    return Message(Role.USER, text)


def system(text: str) -> Message:
    return Message(Role.SYSTEM, text)


def assistant(text: str) -> Message:
    return Message(Role.ASSISTANT, text)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    #: JSON Schema for the tool's arguments.
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    #: Arguments exactly as the provider emitted them. Kept as a string because
    #: *whether the provider emits valid JSON* is itself a fingerprint.
    arguments_raw: str
    #: Parsed arguments, or ``None`` when ``arguments_raw`` did not parse.
    arguments: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """A reasoning block.

    ``signature`` is Anthropic's cryptographic attestation that the block was
    produced by genuine Anthropic inference. It is the single strongest identity
    signal available for Claude models, because a reseller cannot forge one
    without actually routing the request through Anthropic.
    """

    text: str
    signature: str | None = None
    redacted: bool = False


# --------------------------------------------------------------------------- #
# Logprobs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    token: str
    logprob: float
    #: ``(token, logprob)`` pairs for the top alternatives at this position,
    #: highest first. Empty when the provider returned no alternatives.
    top: tuple[tuple[str, float], ...] = ()


# --------------------------------------------------------------------------- #
# Usage and timing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    #: The provider's usage object verbatim. Its *shape* discriminates API
    #: families even when the numbers agree.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@dataclass(frozen=True, slots=True)
class Timing:
    """Wall-clock measurements for one request.

    ``ttft_s`` is only populated for streamed requests.
    """

    total_s: float
    ttft_s: float | None = None
    queue_s: float | None = None

    def output_tokens_per_s(self, output_tokens: int | None) -> float | None:
        if not output_tokens:
            return None
        gen = self.total_s - (self.ttft_s or 0.0)
        return output_tokens / gen if gen > 0 else None


# --------------------------------------------------------------------------- #
# Request / response
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """A single chat completion request.

    Fields left as ``None`` are omitted from the wire payload entirely. This
    matters: sending ``"seed": null`` to a provider that rejects unknown-typed
    fields produces a different error than omitting it, which would corrupt the
    parameter-support probe.
    """

    messages: tuple[Message, ...]
    max_tokens: int = 1024
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    tools: tuple[ToolSpec, ...] = ()
    #: ``"auto" | "none" | "required" | "<tool name>"``
    tool_choice: str | None = None
    #: JSON Schema for a constrained response, if the provider supports it.
    response_schema: dict[str, Any] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    #: Normalised reasoning control. Adapters map this onto the provider's own
    #: ladder (OpenAI ``reasoning.effort``, Anthropic ``effort``, Gemini
    #: ``thinkingConfig``). Pinning it is mandatory for benchmark comparability.
    reasoning_effort: str | None = None
    #: Token budget for reasoning, where the provider exposes one.
    thinking_budget: int | None = None
    stream: bool = False
    #: Escape hatch merged into the outgoing payload verbatim. Used by probes
    #: that deliberately send provider-specific or malformed parameters.
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)

    def replace(self, **changes: Any) -> ChatRequest:
        from dataclasses import replace as _replace

        return _replace(self, **changes)


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """A completion, normalised but lossless."""

    text: str
    #: The model identifier the provider *claims* to have served. Never trust
    #: it; it is trivially spoofed and is only one piece of evidence.
    model_reported: str | None
    response_id: str | None
    finish_reason: FinishReason
    usage: Usage
    timing: Timing
    tool_calls: tuple[ToolCall, ...] = ()
    thinking: tuple[ThinkingBlock, ...] = ()
    logprobs: tuple[TokenLogprob, ...] = ()
    #: Provider-specific fields worth fingerprinting: ``system_fingerprint``,
    #: ``service_tier``, ``modelVersion``, ``provider_name``, and so on.
    raw: dict[str, Any] = field(default_factory=dict)
    http_status: int = 200
    http_headers: dict[str, str] = field(default_factory=dict)
    api_family: ApiFamily = ApiFamily.UNKNOWN

    def raw_get(self, *path: str) -> Any:
        """Safe nested lookup into :attr:`raw`."""
        node: Any = self.raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node


class ParamSupport(str, enum.Enum):
    """Outcome of probing whether a provider honours a request parameter.

    The distinction between ``REJECTED`` and ``IGNORED`` is load-bearing: a
    LiteLLM front end with ``drop_params=True`` silently ignores parameters the
    upstream cannot handle, whereas a genuine first-party API rejects them.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IGNORED = "ignored"
    UNKNOWN = "unknown"
