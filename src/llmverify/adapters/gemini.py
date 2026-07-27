"""The Google Gemini ``generateContent`` protocol.

Three things distinguish this protocol from the other two, and all three are
useful to a verifier.

**The model is in the URL, not the body.** A request goes to
``/models/{model}:generateContent``, so an endpoint that accepts a ``model``
field in the body is not running Google's routing. This is why
:attr:`GeminiAdapter.chat_path` is computed per instance rather than fixed.

**Turns are ``contents`` with an assistant role spelled ``model``**, and every
part is a typed object -- ``text``, ``inlineData``, ``functionCall``,
``functionResponse``.

**Identity travels in ``modelVersion``.** Gemini does not echo the model id that
was requested; it reports the version it actually served, which is the field
worth fingerprinting, so it is what :attr:`ChatResponse.model_reported` is
populated from.

Authentication works either way: ``?key=`` on the query string (the default
here) or an ``x-goog-api-key`` header. Which one an endpoint accepts is a small
piece of evidence in its own right, so both are supported.
"""

from __future__ import annotations

import base64
import json
from typing import Any, ClassVar
from urllib.parse import quote

from ..errors import UnsupportedCapability
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
    ToolCall,
    ToolSpec,
    Usage,
)
from ._http import HttpResult, raise_for_status
from .base import Adapter, Capabilities, register_adapter

__all__ = ["GeminiAdapter"]

_FINISH_REASONS: dict[str, FinishReason] = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
    "BLOCKLIST": FinishReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "SPII": FinishReason.CONTENT_FILTER,
}

_TOOL_MODES: dict[str, str] = {
    "auto": "AUTO",
    "any": "ANY",
    "required": "ANY",
    "none": "NONE",
}

_ROUTE_ABSENT = frozenset({404, 405, 501})

#: Catalogue pages to follow before giving up. Bounded so that a misbehaving
#: endpoint cannot turn a metadata probe into an unbounded crawl.
_MAX_MODEL_PAGES = 10


@register_adapter
class GeminiAdapter(Adapter):
    """Adapter for ``POST /models/{model}:generateContent``."""

    name: ClassVar[str] = "gemini"
    family: ClassVar[ApiFamily] = ApiFamily.GEMINI
    default_base_url: ClassVar[str] = "https://generativelanguage.googleapis.com/v1beta"
    default_auth_scheme: ClassVar[str] = "query"
    capabilities: ClassVar[Capabilities] = Capabilities(
        # Declared False, not because Gemini is known to lack logprobs but
        # because it is not known to have them: `responseLogprobs` and
        # `avgLogprobs` could not be confirmed against the live API reference on
        # 2026-07-26, and no closed Gemini model advertises logprobs in
        # OpenRouter's parameter vocabulary (only the open Gemma models do).
        # Claiming a capability the protocol may not have would make the
        # logprob probe report a spurious failure against a genuine endpoint.
        logprobs=False,
        # Verified indirectly: Gemini models advertise `seed` in OpenRouter's
        # supported-parameter data.
        seed=True,
        tools=True,
        vision=True,
        structured_output=True,
        reasoning_effort=True,
        thinking_signature=False,
        count_tokens_endpoint=True,
        list_models_endpoint=True,
        streaming=True,
    )

    #: Key inside ``thinkingConfig`` that carries the reasoning-effort level.
    #: Gemini's ladder is ``high``/``medium``/``low``/``minimal`` and, unlike
    #: OpenAI's and Anthropic's, it is mandatory -- there is no documented
    #: default, so a benchmark run that leaves ``reasoning_effort`` unset is not
    #: comparable to any published score. The spelling is a class attribute
    #: because the public thinking guide fetched on 2026-07-26 showed the
    #: flattened ``generation_config.thinking_level`` form while the reference
    #: documents the nested ``thinkingConfig`` object, and the two could not be
    #: reconciled; an endpoint that rejects one can be retried with the other.
    thinking_level_field: ClassVar[str] = "thinkingLevel"

    # ---------------------------------------------------------------- routing

    @property
    def model_id(self) -> str:
        """Bare model identifier, with the catalogue's ``models/`` prefix removed."""
        model = self.config.model
        return model[len("models/") :] if model.startswith("models/") else model

    @property
    def model_resource(self) -> str:
        """Fully qualified resource name, as ``countTokens`` expects it."""
        return f"models/{self.model_id}"

    def _model_path(self, method: str) -> str:
        return f"/models/{quote(self.model_id, safe='')}:{method}"

    @property
    def chat_path(self) -> str:  # type: ignore[override]
        return self._model_path("generateContent")

    @property
    def stream_path(self) -> str:
        return self._model_path("streamGenerateContent")

    def _auth_headers(self) -> dict[str, str]:
        scheme = self.config.auth_scheme or self.default_auth_scheme
        if scheme != "x-api-key":
            return super()._auth_headers()
        # Google spells the key header ``x-goog-api-key``; the generic
        # ``x-api-key`` the base class would emit is ignored, which would look
        # like an auth failure rather than a configuration one.
        headers = dict(self.config.headers)
        if self.api_key and not any(k.lower() == "x-goog-api-key" for k in headers):
            headers["x-goog-api-key"] = self.api_key
        return headers

    async def try_chat(self, request: ChatRequest) -> tuple[ChatResponse | None, Exception | None]:
        """Send a request, choosing the streaming or unary method by name.

        Overridden because Gemini encodes the operation in the URL rather than
        in the body: streaming is a different method (``streamGenerateContent``)
        plus ``alt=sse``, not a ``stream: true`` flag.
        """
        payload = self.build_payload(request)
        headers = dict(request.extra_headers)
        try:
            if request.stream:
                result = await self.http.stream(
                    "POST",
                    self.stream_path,
                    json_body=payload,
                    headers=headers or None,
                    params={"alt": "sse"},
                )
            else:
                result = await self.http.request(
                    "POST", self.chat_path, json_body=payload, headers=headers or None
                )
        except Exception as exc:
            return None, exc

        if not result.ok:
            try:
                raise_for_status(result, context=f"{self.name} chat")
            except Exception as exc:
                return None, exc

        try:
            parsed = self.parse_stream(result) if request.stream else self.parse_response(result)
        except Exception as exc:
            return None, exc
        return parsed, None

    # ------------------------------------------------------------------ request

    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        """Render a :class:`~llmverify.types.ChatRequest` as a generateContent body.

        Unset fields are omitted rather than nulled, and ``extra_body`` is
        merged last and verbatim at the top level -- a probe patching anything
        inside ``generationConfig`` must therefore supply that whole object.
        """
        contents, system_parts = self._conversation(request.messages)

        payload: dict[str, Any] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        generation: dict[str, Any] = {"maxOutputTokens": request.max_tokens}
        for key, value in (
            ("temperature", request.temperature),
            ("topP", request.top_p),
            ("seed", request.seed),
        ):
            if value is not None:
                generation[key] = value
        if request.stop:
            generation["stopSequences"] = list(request.stop)
        if request.response_schema is not None:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = request.response_schema

        thinking: dict[str, Any] = {}
        if request.reasoning_effort is not None:
            thinking[self.thinking_level_field] = request.reasoning_effort
        if request.thinking_budget is not None:
            thinking["thinkingBudget"] = request.thinking_budget
        if thinking:
            # Thought parts are withheld unless asked for, and a probe that
            # cannot see them cannot tell a reasoning model from one that merely
            # bills for reasoning tokens.
            thinking["includeThoughts"] = True
            generation["thinkingConfig"] = thinking
        payload["generationConfig"] = generation

        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                        for tool in request.tools
                    ]
                }
            ]

        if request.tool_choice is not None:
            mode = _TOOL_MODES.get(request.tool_choice)
            calling: dict[str, Any] = {"mode": mode or "ANY"}
            if mode is None:
                calling["allowedFunctionNames"] = [request.tool_choice]
            payload["toolConfig"] = {"functionCallingConfig": calling}

        # request.logprobs is not sent: the field could not be confirmed to
        # exist, and guessing a spelling would make the api-surface probe read
        # an unknown-field error as a refusal of logprobs.
        payload.update(request.extra_body)
        return payload

    def _conversation(
        self, messages: tuple[Message, ...]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split messages into ``contents`` turns and the system instruction."""
        contents: list[dict[str, Any]] = []
        system_parts: list[dict[str, Any]] = []

        for message in messages:
            if message.role is Role.SYSTEM:
                system_parts.extend(_parts(message.content))
            elif message.role is Role.TOOL:
                contents.append({"role": "user", "parts": [_function_response(message)]})
            elif message.role is Role.ASSISTANT:
                contents.append({"role": "model", "parts": _model_parts(message)})
            else:
                contents.append({"role": "user", "parts": _parts(message.content)})

        return contents, system_parts

    # ----------------------------------------------------------------- response

    def parse_response(self, result: HttpResult) -> ChatResponse:
        """Normalise a non-streamed candidate, keeping the body intact in ``raw``."""
        body = result.json if isinstance(result.json, dict) else {}
        candidate = _first_candidate(body)
        text, tool_calls, thinking = _split_parts(_candidate_parts(candidate))

        return ChatResponse(
            text=text,
            # Not the requested id: modelVersion is what the endpoint says it
            # actually served, which is the claim worth checking.
            model_reported=_as_str(body.get("modelVersion")),
            response_id=_as_str(body.get("responseId")),
            finish_reason=_finish_reason(candidate.get("finishReason"), tool_calls),
            usage=_parse_usage(body.get("usageMetadata")),
            timing=Timing(total_s=result.total_s, ttft_s=result.ttft_s),
            tool_calls=tool_calls,
            thinking=thinking,
            raw=dict(body),
            http_status=result.status,
            http_headers=dict(result.headers),
            api_family=ApiFamily.GEMINI,
        )

    def parse_stream(self, result: HttpResult) -> ChatResponse:
        """Reassemble a candidate from its ``alt=sse`` chunks.

        Each chunk is a whole ``GenerateContentResponse`` carrying the next few
        parts, so assembly is concatenation rather than delta application. The
        reconstructed candidate is written back into ``raw`` in the shape a
        unary response would have, so probes need not care which transport ran.
        """
        parts: list[dict[str, Any]] = []
        top: dict[str, Any] = {}
        usage: Any = None
        finish: Any = None
        safety: Any = None

        for event in result.events:
            for key, value in event.items():
                if key in ("candidates", "usageMetadata"):
                    continue
                if key not in top or (top[key] is None and value is not None):
                    top[key] = value
            if isinstance(event.get("usageMetadata"), dict):
                # Gemini repeats cumulative usage on every chunk; the last one
                # is the complete account.
                usage = event["usageMetadata"]

            candidate = _first_candidate(event)
            if not candidate:
                continue
            if candidate.get("finishReason") is not None:
                finish = candidate["finishReason"]
            if candidate.get("safetyRatings") is not None:
                safety = candidate["safetyRatings"]
            parts.extend(_candidate_parts(candidate))

        text, tool_calls, thinking = _split_parts(parts)

        candidate_raw: dict[str, Any] = {"content": {"role": "model", "parts": parts}}
        if finish is not None:
            candidate_raw["finishReason"] = finish
        if safety is not None:
            candidate_raw["safetyRatings"] = safety

        raw = dict(top)
        raw["candidates"] = [candidate_raw]
        if usage is not None:
            raw["usageMetadata"] = usage
        raw["_llmverify_stream_chunks"] = len(result.events)

        return ChatResponse(
            text=text,
            model_reported=_as_str(top.get("modelVersion")),
            response_id=_as_str(top.get("responseId")),
            finish_reason=_finish_reason(finish, tool_calls),
            usage=_parse_usage(usage),
            timing=Timing(total_s=result.total_s, ttft_s=result.ttft_s),
            tool_calls=tool_calls,
            thinking=thinking,
            raw=raw,
            http_status=result.status,
            http_headers=dict(result.headers),
            api_family=ApiFamily.GEMINI,
        )

    # ----------------------------------------------------------------- optional

    async def count_tokens(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolSpec, ...] = (),
        *,
        tool_choice: str | None = None,
    ) -> int:
        """Ask the endpoint for its own token count of these messages.

        Generation is removed from the measurement entirely, so the number is a
        pure reading of the tokenizer and of whatever system prompt the model
        carries. Raises :class:`~llmverify.errors.UnsupportedCapability` when
        the route is absent or the response is not shaped as documented.
        """
        request = ChatRequest(messages=messages, tools=tools, tool_choice=tool_choice)
        payload = self.build_payload(request)
        payload.pop("generationConfig", None)

        # A bare ``contents`` list is the documented minimal form; anything that
        # also needs a system instruction or tools has to be wrapped, because
        # those fields only exist inside a generateContentRequest.
        if len(payload) > 1:
            body: dict[str, Any] = {
                "generateContentRequest": {"model": self.model_resource, **payload}
            }
        else:
            body = {"contents": payload["contents"]}

        result = await self.http.request("POST", self._model_path("countTokens"), json_body=body)
        if result.status in _ROUTE_ABSENT:
            raise UnsupportedCapability(
                f"{self.name} endpoint has no countTokens route for {self.model_id!r} "
                f"(HTTP {result.status})"
            )
        raise_for_status(result, context=f"{self.name} count tokens")

        data = result.json if isinstance(result.json, dict) else {}
        total = data.get("totalTokens")
        if isinstance(total, bool) or not isinstance(total, int):
            raise UnsupportedCapability(
                f"{self.name} countTokens returned no integer totalTokens: {result.text[:200]!r}"
            )
        return total

    async def list_models(self) -> list[dict[str, Any]]:
        """Return the catalogue with a ``models/``-free ``id`` added to each entry.

        Gemini names models ``models/gemini-...`` while every caller in this
        package compares bare ids, so the prefix is stripped into an extra key
        rather than by rewriting ``name`` -- the original is part of the
        catalogue's shape, and shape is evidence.
        """
        entries: list[dict[str, Any]] = []
        params: dict[str, str] = {"pageSize": "1000"}

        for _ in range(_MAX_MODEL_PAGES):
            result = await self.http.request("GET", "/models", params=params)
            raise_for_status(result, context=f"{self.name} list models")
            body = result.json if isinstance(result.json, dict) else {}

            for entry in body.get("models") or ():
                if not isinstance(entry, dict):
                    continue
                item = dict(entry)
                name = item.get("name")
                if isinstance(name, str):
                    item.setdefault(
                        "id", name[len("models/") :] if name.startswith("models/") else name
                    )
                entries.append(item)

            token = body.get("nextPageToken")
            if not isinstance(token, str) or not token:
                break
            params = {**params, "pageToken": token}

        return entries


# --------------------------------------------------------------------------- #
# Payload construction
# --------------------------------------------------------------------------- #


def _parts(content: str | tuple[ContentPart, ...]) -> list[dict[str, Any]]:
    """Render message content as Gemini parts."""
    if isinstance(content, str):
        return [{"text": content}] if content else []

    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            if part.text:
                parts.append({"text": part.text})
        elif isinstance(part, ImagePart):
            parts.append(
                {
                    "inlineData": {
                        "mimeType": part.media_type,
                        "data": base64.b64encode(part.data).decode("ascii"),
                    }
                }
            )
    return parts


def _model_parts(message: Message) -> list[dict[str, Any]]:
    """Render an assistant turn, replaying reasoning and tool calls."""
    parts: list[dict[str, Any]] = []
    for block in message.thinking:
        thought: dict[str, Any] = {"text": block.text, "thought": True}
        if block.signature is not None:
            # Only ever set when the endpoint itself supplied one on the way
            # out; this replays that value under the key it arrived on rather
            # than asserting anything about how Gemini validates it.
            thought["thoughtSignature"] = block.signature
        parts.append(thought)

    parts.extend(_parts(message.content))

    for call in message.tool_calls:
        parts.append(
            {
                "functionCall": {
                    "name": call.name,
                    "args": call.arguments if call.arguments is not None else {},
                }
            }
        )
    return parts


def _function_response(message: Message) -> dict[str, Any]:
    """Render a tool result.

    Gemini keys a tool result by the *function's name* rather than by a call id,
    so a replayed tool turn carries that name in ``tool_call_id``. The
    ``response`` value must be an object; a result that is not already JSON is
    wrapped, and the wrapper key is arbitrary because the field is free-form.
    """
    text = message.text
    payload: Any = None
    if text:
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
    return {
        "functionResponse": {
            "name": message.tool_call_id or "",
            "response": payload if isinstance(payload, dict) else {"result": text},
        }
    }


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


def _first_candidate(body: dict[str, Any]) -> dict[str, Any]:
    candidates = body.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        return candidates[0]
    return {}


def _candidate_parts(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    content = candidate.get("content")
    if not isinstance(content, dict):
        return []
    parts = content.get("parts")
    if not isinstance(parts, list):
        return []
    return [p for p in parts if isinstance(p, dict)]


def _split_parts(
    parts: list[dict[str, Any]],
) -> tuple[str, tuple[ToolCall, ...], tuple[ThinkingBlock, ...]]:
    """Fold a part list into text, tool calls and reasoning blocks."""
    chunks: list[str] = []
    calls: list[ToolCall] = []
    thinking: list[ThinkingBlock] = []

    for part in parts:
        if part.get("thought") is True:
            signature = part.get("thoughtSignature")
            thinking.append(
                ThinkingBlock(
                    text=part.get("text") if isinstance(part.get("text"), str) else "",
                    signature=signature if isinstance(signature, str) else None,
                )
            )
            continue

        call = part.get("functionCall")
        if isinstance(call, dict):
            arguments = call.get("args") if isinstance(call.get("args"), dict) else None
            name = str(call.get("name") or "")
            calls.append(
                ToolCall(
                    # Gemini identifies a call by name unless it volunteers an
                    # id, and the name is what a tool result must be keyed on.
                    id=str(call.get("id") or name),
                    name=name,
                    arguments_raw=json.dumps(
                        arguments if arguments is not None else call.get("args"),
                        ensure_ascii=False,
                    ),
                    arguments=arguments,
                )
            )
            continue

        text = part.get("text")
        if isinstance(text, str):
            chunks.append(text)

    return "".join(chunks), tuple(calls), tuple(thinking)


def _parse_usage(value: Any) -> Usage:
    """Map ``usageMetadata``, keeping it verbatim in :attr:`Usage.raw`.

    ``candidatesTokenCount`` excludes ``thoughtsTokenCount``, so the two are
    reported separately here rather than summed; the token-accounting probe
    needs the split, and ``raw`` keeps ``totalTokenCount`` for cross-checking.
    """
    if not isinstance(value, dict):
        return Usage()
    return Usage(
        input_tokens=_as_int(value.get("promptTokenCount")),
        output_tokens=_as_int(value.get("candidatesTokenCount")),
        reasoning_tokens=_as_int(value.get("thoughtsTokenCount")),
        cache_read_tokens=_as_int(value.get("cachedContentTokenCount")),
        raw=dict(value),
    )


def _finish_reason(value: Any, tool_calls: tuple[ToolCall, ...]) -> FinishReason:
    """Map ``finishReason``, normalising a tool-calling turn.

    Gemini reports ``STOP`` when a turn ends in a function call, where the other
    two families report a distinct tool-call reason. Probes compare across
    families, so the normalised value follows the parts actually returned; the
    verbatim ``finishReason`` survives in ``raw``.
    """
    if not isinstance(value, str):
        return FinishReason.TOOL_CALLS if tool_calls else FinishReason.OTHER
    reason = _FINISH_REASONS.get(value.strip().upper(), FinishReason.OTHER)
    if tool_calls and reason is FinishReason.STOP:
        return FinishReason.TOOL_CALLS
    return reason


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None
