"""The OpenAI-compatible Chat Completions protocol.

This is the wire format nearly everything speaks: OpenAI itself, vLLM, SGLang,
LiteLLM, OpenRouter, Ollama, LM Studio, and essentially every reseller. Getting
it right therefore matters more than any other adapter, and "right" here means
two things an ordinary client library does not care about.

**Omission, never nulling.** A field the request left as ``None`` is absent from
the payload, not present as ``null``. Endpoints answer the two differently --
some reject an unknown-typed ``null``, some coerce it, some ignore it -- and the
api-surface probe reads exactly that difference to tell a parameter that was
*rejected* from one that was silently *dropped*.

**No information lost.** Everything the endpoint volunteered survives into
:attr:`~llmverify.types.ChatResponse.raw`: ``system_fingerprint``,
``service_tier``, ``provider``, OpenRouter's ``openrouter_metadata``, and the
usage object with its shape intact. Compatible stacks differ from each other
mostly in fields a normal client throws away, so those fields are the evidence.
"""

from __future__ import annotations

import base64
import json
import math
import re
from typing import Any, ClassVar
from urllib.parse import urlsplit

from ..types import (
    ApiFamily,
    ChatRequest,
    ChatResponse,
    ContentPart,
    FinishReason,
    ImagePart,
    Message,
    Role,
    TextPart,
    ThinkingBlock,
    Timing,
    TokenLogprob,
    ToolCall,
    Usage,
)
from ._http import HttpResult
from .base import Adapter, Capabilities, register_adapter

__all__ = ["OpenAICompatAdapter"]

OPENROUTER_HOST = "openrouter.ai"

_TOOL_CHOICE_KEYWORDS = frozenset({"auto", "none", "required"})

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}

_SCHEMA_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


@register_adapter
class OpenAICompatAdapter(Adapter):
    """Adapter for ``POST /chat/completions`` and its many dialects.

    Subclasses exist to pin dialect choices rather than to re-implement
    anything: the two knobs that genuinely differ between compatible stacks --
    which output-length field is accepted, and how reasoning effort is spelled
    -- are class attributes, so a plugin for a specific reseller is a four-line
    subclass.
    """

    name: ClassVar[str] = "openai"
    family: ClassVar[ApiFamily] = ApiFamily.OPENAI
    default_base_url: ClassVar[str] = "https://api.openai.com/v1"
    default_auth_scheme: ClassVar[str] = "bearer"
    capabilities: ClassVar[Capabilities] = Capabilities(
        logprobs=True,
        seed=True,
        tools=True,
        vision=True,
        structured_output=True,
        reasoning_effort=True,
        thinking_signature=False,
        count_tokens_endpoint=False,
        list_models_endpoint=True,
        streaming=True,
    )

    #: Which output-length field to emit: ``"both"``, ``"max_tokens"``,
    #: ``"max_completion_tokens"`` or ``"none"``. Sending both by default is
    #: the only choice that works everywhere -- newer OpenAI models reject
    #: ``max_tokens`` outright, while much of the compatible ecosystem has
    #: never implemented ``max_completion_tokens``.
    max_tokens_field: ClassVar[str] = "both"

    #: Key recognised in :attr:`~llmverify.types.ChatRequest.extra_body` to
    #: override :attr:`max_tokens_field` for one request. It is removed from the
    #: payload rather than sent, so a probe can try each spelling in turn and
    #: see which one the endpoint actually honours.
    MAX_TOKENS_DIRECTIVE: ClassVar[str] = "llmverify_max_tokens_field"

    #: How reasoning effort is spelled: ``"flat"`` sends ``reasoning_effort``,
    #: ``"nested"`` sends ``reasoning: {"effort": ...}``, ``"both"`` sends both,
    #: ``"none"`` sends neither. Flat is the default because it is what the
    #: compatible ecosystem implements.
    reasoning_effort_style: ClassVar[str] = "flat"

    #: Ask for a usage object on the final stream frame. Without it most
    #: implementations stream no usage at all, which would cost the token
    #: accounting probe its input.
    stream_include_usage: ClassVar[bool] = True

    # ------------------------------------------------------------------ request

    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        """Render a :class:`~llmverify.types.ChatRequest` as a JSON body.

        Optional fields are omitted when unset. ``extra_body`` is merged last
        and verbatim -- including nulls, wrong types and unknown keys -- because
        probes deliberately send malformed values to see how the endpoint
        complains.
        """
        extra = dict(request.extra_body)
        length_field = str(extra.pop(self.MAX_TOKENS_DIRECTIVE, self.max_tokens_field)).lower()

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._message_payload(m) for m in request.messages],
        }

        if length_field in ("both", "max_tokens"):
            payload["max_tokens"] = request.max_tokens
        if length_field in ("both", "max_completion_tokens"):
            payload["max_completion_tokens"] = request.max_tokens

        for key, value in (
            ("temperature", request.temperature),
            ("top_p", request.top_p),
            ("seed", request.seed),
        ):
            if value is not None:
                payload[key] = value

        if request.stop:
            payload["stop"] = list(request.stop)

        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]

        if request.tool_choice is not None:
            payload["tool_choice"] = (
                request.tool_choice
                if request.tool_choice in _TOOL_CHOICE_KEYWORDS
                else {"type": "function", "function": {"name": request.tool_choice}}
            )

        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(request.response_schema),
                    "schema": request.response_schema,
                    "strict": True,
                },
            }

        # ``top_logprobs`` is meaningless on this protocol without ``logprobs``,
        # so asking for one implies the other. A probe that wants the invalid
        # combination can still set ``logprobs`` false through extra_body.
        if request.logprobs or request.top_logprobs is not None:
            payload["logprobs"] = True
        if request.top_logprobs is not None:
            payload["top_logprobs"] = request.top_logprobs

        reasoning: dict[str, Any] = {}
        if request.reasoning_effort is not None:
            if self.reasoning_effort_style in ("flat", "both"):
                payload["reasoning_effort"] = request.reasoning_effort
            if self.reasoning_effort_style in ("nested", "both"):
                reasoning["effort"] = request.reasoning_effort
        if request.thinking_budget is not None:
            # Chat Completions has no first-party thinking budget; the nested
            # reasoning object is the extension the compatible ecosystem uses
            # for it. Unverified against first-party OpenAI, and an endpoint
            # that does not know the field will reject or ignore it -- which is
            # itself a usable observation.
            reasoning["max_tokens"] = request.thinking_budget
        if reasoning:
            payload["reasoning"] = reasoning

        if request.stream:
            payload["stream"] = True
            if self.stream_include_usage:
                payload["stream_options"] = {"include_usage": True}

        payload.update(extra)
        return payload

    def _message_payload(self, message: Message) -> dict[str, Any]:
        """Render one message, including tool results and replayed reasoning."""
        if message.role is Role.TOOL:
            payload: dict[str, Any] = {"role": "tool", "content": message.text}
            if message.tool_call_id is not None:
                payload["tool_call_id"] = message.tool_call_id
            return payload

        content = _content_payload(message.content)
        payload = {"role": message.role.value}
        # An assistant turn that is only tool calls carries no content at all;
        # sending an empty string there upsets stricter implementations.
        if content or not message.tool_calls:
            payload["content"] = content

        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_raw},
                }
                for call in message.tool_calls
            ]

        if message.thinking:
            text = "".join(block.text for block in message.thinking)
            if text:
                payload["reasoning_content"] = text
            signed = [b for b in message.thinking if b.signature]
            if signed:
                # A signature is only meaningful to the upstream that issued it.
                # Dropping it would quietly turn a verifiable replayed turn into
                # an unverifiable one, so it is passed through in the shape used
                # by proxies that front a signature-emitting API.
                payload["thinking_blocks"] = [
                    {"type": "thinking", "thinking": b.text, "signature": b.signature}
                    for b in signed
                ]

        return payload

    # ----------------------------------------------------------------- response

    def parse_response(self, result: HttpResult) -> ChatResponse:
        """Normalise a non-streamed completion without discarding anything."""
        body = result.json if isinstance(result.json, dict) else {}
        choice = _first_choice(body)
        message = choice.get("message")
        if not isinstance(message, dict):
            message = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}

        return ChatResponse(
            text=_text_from_content(message.get("content")),
            model_reported=_as_str(body.get("model")),
            response_id=_as_str(body.get("id")),
            finish_reason=_finish_reason(choice.get("finish_reason")),
            usage=_parse_usage(body.get("usage")),
            timing=Timing(total_s=result.total_s, ttft_s=result.ttft_s),
            tool_calls=_parse_tool_calls(message),
            thinking=_parse_thinking(message),
            logprobs=_parse_logprobs(choice.get("logprobs")),
            raw=dict(body),
            http_status=result.status,
            http_headers=dict(result.headers),
            api_family=ApiFamily.OPENAI,
        )

    def parse_stream(self, result: HttpResult) -> ChatResponse:
        """Reassemble a completion from its SSE frames.

        Streaming is not just a transport detail here: it is the only way to
        measure time-to-first-token, and the frame layout (how many chunks, how
        tool-call arguments are split, whether usage arrives at all) differs
        between serving stacks.
        """
        text: list[str] = []
        reasoning: list[str] = []
        logprob_entries: list[Any] = []
        calls: dict[int, dict[str, Any]] = {}
        order: list[int] = []
        finish_raw: Any = None
        usage_raw: Any = None
        top: dict[str, Any] = {}

        for event in result.events:
            for key, value in event.items():
                if key in ("choices", "usage"):
                    continue
                # Keep the first sighting of a key so its presence is recorded
                # even when later frames omit it, but let a real value replace a
                # null -- OpenRouter always sends system_fingerprint, null until
                # the upstream supplies one.
                if key not in top or (top[key] is None and value is not None):
                    top[key] = value
            if isinstance(event.get("usage"), dict):
                usage_raw = event["usage"]

            choice = _first_choice(event)
            if not choice:
                continue
            if choice.get("finish_reason") is not None:
                finish_raw = choice["finish_reason"]

            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = choice.get("message") if isinstance(choice.get("message"), dict) else {}

            chunk = _text_from_content(delta.get("content"))
            if chunk:
                text.append(chunk)
            thought = _reasoning_text(delta)
            if thought:
                reasoning.append(thought)

            logprobs = choice.get("logprobs")
            if isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list):
                logprob_entries.extend(logprobs["content"])

            fragments = delta.get("tool_calls")
            if isinstance(fragments, list):
                for fragment in fragments:
                    _accumulate_tool_call(fragment, calls, order)

        raw = dict(top)
        if usage_raw is not None:
            raw["usage"] = usage_raw
        # Frame count is otherwise lost on reassembly, and a proxy that buffers
        # the whole completion into one frame looks very different from one that
        # streams token by token.
        raw["_llmverify_stream_chunks"] = len(result.events)

        thinking = "".join(reasoning)
        return ChatResponse(
            text="".join(text),
            model_reported=_as_str(top.get("model")),
            response_id=_as_str(top.get("id")),
            finish_reason=_finish_reason(finish_raw),
            usage=_parse_usage(usage_raw),
            timing=Timing(total_s=result.total_s, ttft_s=result.ttft_s),
            tool_calls=tuple(_finish_tool_call(calls[i]) for i in order),
            thinking=(ThinkingBlock(text=thinking),) if thinking else (),
            logprobs=tuple(_token_logprob(e) for e in logprob_entries if isinstance(e, dict)),
            raw=raw,
            http_status=result.status,
            http_headers=dict(result.headers),
            api_family=ApiFamily.OPENAI,
        )

    # ------------------------------------------------------------ introspection

    async def fetch_openrouter_endpoints(self, model: str) -> list[dict[str, Any]] | None:
        """Read OpenRouter's per-provider endpoint list for ``model``.

        This is how the tool obtains a provider's *self-declared* quantization,
        alongside its context length and supported parameters. Values observed
        on 2026-07-26 include ``fp4``, ``fp8``, ``int4`` and ``unknown``; a
        third of endpoints declare ``unknown``, and none of the labels are
        audited by OpenRouter, so this is a claim to be checked rather than a
        fact to be trusted.

        Returns ``None`` when the configured endpoint is not OpenRouter, when
        ``model`` is not an ``author/slug`` identifier, or when the call fails
        for any reason -- an unavailable catalogue is not evidence.
        """
        if not self._is_openrouter():
            return None
        author, _, slug = model.partition("/")
        # Variant suffixes such as ":free" or ":nitro" are routing hints, not
        # part of the slug the endpoints route is keyed on.
        slug = slug.split(":", 1)[0]
        if not author or not slug:
            return None

        url = f"https://{OPENROUTER_HOST}/api/v1/models/{author}/{slug}/endpoints"
        try:
            result = await self.http.request("GET", url)
        except Exception:
            return None
        if not result.ok or not isinstance(result.json, dict):
            return None

        data = result.json.get("data")
        endpoints = data.get("endpoints") if isinstance(data, dict) else None
        if not isinstance(endpoints, list):
            return None
        return [e for e in endpoints if isinstance(e, dict)]

    def detect_serving_stack(
        self, models: list[dict[str, Any]], sample: ChatResponse | None = None
    ) -> str | None:
        """Best-effort label for the software serving this endpoint.

        Three tiers of signal, in descending order of certainty.

        *Certain*: the configured host is OpenAI's or OpenRouter's; or something
        names itself, in a response header or a catalogue entry's ``owned_by``.

        *Verified shape*: OpenRouter's catalogue entries all carry
        ``canonical_slug`` and a ``pricing`` block, and it exposes
        ``openrouter_metadata`` on responses.

        *Heuristic*: keys such as ``max_model_len`` in the catalogue, an echoed
        ``prompt_logprobs`` (a vLLM-distinctive parameter -- SGLang spells the
        equivalent ``return_logprob``/``top_logprobs_num`` and has no
        ``prompt_logprobs`` at all), a missing ``system_fingerprint`` where
        OpenAI and OpenRouter both emit one, and the well-known local ports of
        Ollama and LM Studio. These are not verified against every version, so
        a heuristic label is only returned when two independent hints agree and
        nothing contradicts them.

        Returns ``None`` when the evidence does not reach that bar. Guessing
        would be worse than silence: this label feeds a verdict about whether
        someone is lying.
        """
        host = self._host()
        if host == OPENROUTER_HOST or host.endswith(f".{OPENROUTER_HOST}"):
            return "openrouter"
        if host == "api.openai.com":
            return "openai"

        headers = dict(sample.http_headers) if sample is not None else {}
        raw = sample.raw if sample is not None else {}

        for key, value in headers.items():
            label = _token_label(f"{key} {value}")
            if label:
                return label

        for entry in models:
            for field in ("owned_by", "object", "created_by", "served_by", "provider"):
                value = entry.get(field)
                if isinstance(value, str):
                    label = _token_label(value)
                    if label:
                        return label

        if "openrouter_metadata" in raw:
            return "openrouter"
        if models and all("canonical_slug" in m and "pricing" in m for m in models):
            return "openrouter"

        hints: dict[str, int] = {}

        def hint(label: str) -> None:
            hints[label] = hints.get(label, 0) + 1

        if any("max_model_len" in m for m in models):
            hint("vllm")
        if any(isinstance(m.get("permission"), list) and m["permission"] for m in models):
            hint("vllm")
        if "prompt_logprobs" in raw:
            hint("vllm")
        if sample is not None and "system_fingerprint" not in raw:
            hint("vllm")
            hint("sglang")

        port = self._port()
        if _is_local(host):
            if port == 11434:
                hint("ollama")
            if any(":" in str(m.get("id", "")) for m in models):
                hint("ollama")
            if port == 1234:
                hint("lm-studio")
            if any("quantization" in m and "publisher" in m for m in models):
                hint("lm-studio")

        if not hints:
            return None
        best = max(hints.values())
        winners = [label for label, count in hints.items() if count == best]
        if best < 2 or len(winners) != 1:
            return None
        return winners[0]

    # ------------------------------------------------------------------ helpers

    def _host(self) -> str:
        return (urlsplit(self.http.base_url).hostname or "").lower()

    def _port(self) -> int | None:
        try:
            return urlsplit(self.http.base_url).port
        except ValueError:
            return None

    def _is_openrouter(self) -> bool:
        host = self._host()
        return host == OPENROUTER_HOST or host.endswith(f".{OPENROUTER_HOST}")


# --------------------------------------------------------------------------- #
# Payload construction
# --------------------------------------------------------------------------- #


def _content_payload(content: str | tuple[ContentPart, ...]) -> str | list[dict[str, Any]]:
    """Render message content, using the multimodal part list only when needed.

    Plain strings stay strings: some compatible servers accept only that shape
    for text, and a needless part list would make a text-only probe fail for
    reasons unrelated to what it is measuring.
    """
    if isinstance(content, str):
        return content

    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            encoded = base64.b64encode(part.data).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{part.media_type};base64,{encoded}"},
                }
            )
    return parts


def _schema_name(schema: dict[str, Any]) -> str:
    """Pick a response-format name the endpoint will accept.

    The name is restricted to an identifier-like alphabet, so a schema title
    containing spaces or punctuation cannot turn a structured-output probe into
    a validation error about the name.
    """
    title = schema.get("title")
    if isinstance(title, str) and _SCHEMA_NAME_RE.fullmatch(title):
        return title
    return "response"


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


def _first_choice(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _text_from_content(value: Any) -> str:
    """Flatten a content field that may be a string or a list of parts."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""

    chunks: list[str] = []
    for part in value:
        if isinstance(part, str):
            chunks.append(part)
        elif isinstance(part, dict):
            for key in ("text", "content", "summary_text"):
                nested = part.get(key)
                if isinstance(nested, str):
                    chunks.append(nested)
                    break
    return "".join(chunks)


def _reasoning_text(node: dict[str, Any]) -> str:
    """Extract reasoning text from the fields reasoning-model proxies add.

    ``reasoning_content`` and ``reasoning`` both occur in the wild, and
    ``reasoning`` is variously a string, an object, or a list of summary parts.
    """
    for key in ("reasoning_content", "reasoning"):
        value = node.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            for nested_key in ("content", "text", "summary"):
                text = _text_from_content(value.get(nested_key))
                if text:
                    return text
        if isinstance(value, list):
            text = _text_from_content(value)
            if text:
                return text
    return ""


def _parse_thinking(message: dict[str, Any]) -> tuple[ThinkingBlock, ...]:
    """Recover reasoning blocks, keeping any signature intact.

    A signature is the strongest identity signal there is, so a proxy that
    passes one through must not have it dropped here.
    """
    raw_blocks = message.get("thinking_blocks")
    if isinstance(raw_blocks, list):
        blocks = []
        for entry in raw_blocks:
            if not isinstance(entry, dict):
                continue
            text = entry.get("thinking")
            if not isinstance(text, str):
                text = entry.get("text") if isinstance(entry.get("text"), str) else ""
            signature = entry.get("signature")
            blocks.append(
                ThinkingBlock(
                    text=text,
                    signature=signature if isinstance(signature, str) else None,
                    redacted=entry.get("type") == "redacted_thinking",
                )
            )
        if blocks:
            return tuple(blocks)

    text = _reasoning_text(message)
    return (ThinkingBlock(text=text),) if text else ()


def _parse_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    """Read tool calls, preserving the argument string exactly as sent.

    Whether a provider emits parseable JSON arguments is a fingerprint in its
    own right, so ``arguments`` is populated only when ``arguments_raw``
    actually parses into an object.
    """
    calls: list[ToolCall] = []
    entries = message.get("tool_calls")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            function = entry.get("function")
            if not isinstance(function, dict):
                continue
            raw_args, parsed = _tool_arguments(function.get("arguments"))
            calls.append(
                ToolCall(
                    id=str(entry.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments_raw=raw_args,
                    arguments=parsed,
                )
            )

    legacy = message.get("function_call")
    if not calls and isinstance(legacy, dict):
        raw_args, parsed = _tool_arguments(legacy.get("arguments"))
        calls.append(
            ToolCall(
                id="",
                name=str(legacy.get("name") or ""),
                arguments_raw=raw_args,
                arguments=parsed,
            )
        )
    return tuple(calls)


def _tool_arguments(value: Any) -> tuple[str, dict[str, Any] | None]:
    """Return ``(verbatim string, parsed object or None)`` for tool arguments."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value, None
        return value, parsed if isinstance(parsed, dict) else None
    if isinstance(value, dict):
        # A server that sent an object rather than a string has already lost the
        # verbatim form; re-serialising is the closest honest reconstruction.
        return json.dumps(value, ensure_ascii=False), value
    if value is None:
        return "", None
    return str(value), None


def _accumulate_tool_call(
    fragment: Any, calls: dict[int, dict[str, Any]], order: list[int]
) -> None:
    """Fold one streamed tool-call delta into the accumulator.

    Arguments arrive as fragments that must be concatenated in order. The
    ``index`` field is what associates them; implementations that omit it emit
    fragments sequentially, so a fragment bearing a new id starts a new call and
    everything else extends the newest one.
    """
    if not isinstance(fragment, dict):
        return

    index = fragment.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        fragment_id = fragment.get("id")
        if order and (not fragment_id or calls[order[-1]]["id"] == fragment_id):
            index = order[-1]
        else:
            index = max(order) + 1 if order else 0

    call = calls.setdefault(index, {"id": "", "name": "", "arguments": []})
    if index not in order:
        order.append(index)

    call_id = fragment.get("id")
    if isinstance(call_id, str) and call_id:
        call["id"] = call_id

    function = fragment.get("function")
    if not isinstance(function, dict):
        return
    name = function.get("name")
    if isinstance(name, str) and name and not call["name"]:
        call["name"] = name
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        call["arguments"].append(arguments)
    elif isinstance(arguments, dict):
        call["arguments"].append(json.dumps(arguments, ensure_ascii=False))


def _finish_tool_call(call: dict[str, Any]) -> ToolCall:
    raw_args = "".join(call["arguments"])
    _, parsed = _tool_arguments(raw_args) if raw_args else ("", None)
    return ToolCall(id=call["id"], name=call["name"], arguments_raw=raw_args, arguments=parsed)


def _parse_logprobs(value: Any) -> tuple[TokenLogprob, ...]:
    """Read the per-token logprob table, including the legacy dialect.

    Untracked tokens are reported as ``-9999.0`` by real OpenAI. That sentinel
    is kept as sent: normalising it away would erase the very thing that makes
    it recognisable.
    """
    if not isinstance(value, dict):
        return ()
    entries = value.get("content")
    if isinstance(entries, list):
        return tuple(_token_logprob(e) for e in entries if isinstance(e, dict))
    return _legacy_logprobs(value)


def _token_logprob(entry: dict[str, Any]) -> TokenLogprob:
    top: list[tuple[str, float]] = []
    alternatives = entry.get("top_logprobs")
    if isinstance(alternatives, list):
        for alternative in alternatives:
            if not isinstance(alternative, dict):
                continue
            token = alternative.get("token")
            logprob = _as_float(alternative.get("logprob"))
            if isinstance(token, str) and logprob is not None:
                top.append((token, logprob))

    token = entry.get("token")
    logprob = _as_float(entry.get("logprob"))
    return TokenLogprob(
        token=token if isinstance(token, str) else "",
        # A missing or non-numeric logprob is malformed. NaN records that
        # honestly and lets downstream statistics drop the position explicitly,
        # which inventing a number would not.
        logprob=logprob if logprob is not None else math.nan,
        top=tuple(top),
    )


def _legacy_logprobs(value: dict[str, Any]) -> tuple[TokenLogprob, ...]:
    """Parse the older completions-style logprob block some servers still emit."""
    tokens = value.get("tokens")
    logprobs = value.get("token_logprobs")
    if not isinstance(tokens, list) or not isinstance(logprobs, list):
        return ()

    alternatives = value.get("top_logprobs")
    out: list[TokenLogprob] = []
    for position, token in enumerate(tokens):
        logprob = _as_float(logprobs[position]) if position < len(logprobs) else None
        top: list[tuple[str, float]] = []
        if isinstance(alternatives, list) and position < len(alternatives):
            candidate = alternatives[position]
            if isinstance(candidate, dict):
                for alt_token, alt_logprob in candidate.items():
                    parsed = _as_float(alt_logprob)
                    if parsed is not None:
                        top.append((str(alt_token), parsed))
                top.sort(key=lambda pair: -pair[1])
        out.append(
            TokenLogprob(
                token=str(token),
                logprob=logprob if logprob is not None else math.nan,
                top=tuple(top),
            )
        )
    return tuple(out)


def _parse_usage(value: Any) -> Usage:
    """Map the usage object, keeping it verbatim in :attr:`Usage.raw`.

    The shape is a discriminator in itself: Chat Completions reports
    ``prompt_tokens``/``completion_tokens`` with a
    ``completion_tokens_details`` block, while the Responses surface reports
    ``input_tokens``/``output_tokens`` with ``output_tokens_details``. Both are
    read here, because compatible proxies mirror either one, and ``raw``
    preserves which was actually sent.
    """
    if not isinstance(value, dict):
        return Usage()

    output_details = value.get("completion_tokens_details")
    input_details = value.get("prompt_tokens_details")
    responses_details = value.get("output_tokens_details")

    return Usage(
        input_tokens=_pick_int(value, "prompt_tokens", "input_tokens"),
        output_tokens=_pick_int(value, "completion_tokens", "output_tokens"),
        reasoning_tokens=_first(
            _pick_int(output_details, "reasoning_tokens"),
            _pick_int(responses_details, "reasoning_tokens"),
        ),
        cache_read_tokens=_first(
            _pick_int(input_details, "cached_tokens"),
            _pick_int(value, "cache_read_input_tokens"),
        ),
        cache_write_tokens=_first(
            _pick_int(value, "cache_creation_input_tokens"),
            _pick_int(input_details, "cache_creation_tokens"),
        ),
        raw=dict(value),
    )


def _finish_reason(value: Any) -> FinishReason:
    if not isinstance(value, str):
        return FinishReason.OTHER
    return _FINISH_REASONS.get(value.strip().lower(), FinishReason.OTHER)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _pick_int(node: Any, *keys: str) -> int | None:
    if not isinstance(node, dict):
        return None
    for key in keys:
        value = node.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _first(*values: int | None) -> int | None:
    return next((v for v in values if v is not None), None)


# --------------------------------------------------------------------------- #
# Serving-stack identification
# --------------------------------------------------------------------------- #

#: Substrings that identify a serving stack when a server puts one in a header
#: or a catalogue entry. Only literal self-identification counts here, which is
#: why model ids are not searched: a LiteLLM route named ``ollama/llama3`` says
#: what it proxies, not what is serving it.
_STACK_TOKENS: dict[str, str] = {
    "openrouter": "openrouter",
    "litellm": "litellm",
    "sglang": "sglang",
    "vllm": "vllm",
    "ollama": "ollama",
    "lm-studio": "lm-studio",
    "lmstudio": "lm-studio",
    "llama.cpp": "llama.cpp",
    "llamacpp": "llama.cpp",
    "text-generation-inference": "tgi",
    "tensorrt": "tensorrt-llm",
}

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"})


def _token_label(text: str) -> str | None:
    lowered = text.lower()
    for token in sorted(_STACK_TOKENS, key=len, reverse=True):
        if token in lowered:
            return _STACK_TOKENS[token]
    return None


def _is_local(host: str) -> bool:
    return host in _LOCAL_HOSTS or host.endswith(".local") or host.startswith("192.168.")
