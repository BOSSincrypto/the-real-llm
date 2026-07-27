"""An in-process provider that is dishonest in specific, configurable ways.

Every adapter in this package exists to talk HTTP to a stranger, so testing them
against monkeypatched coroutines would test the wrong half of the code. This
module serves the three wire protocols over a real socket on an ephemeral port,
so a probe under test builds a real payload, sends it through
:class:`~llmverify.adapters._http.HttpClient`, and parses a real response.

The value is not the transport, though. It is the personas. Each one reproduces
a failure mode that has actually been observed in the field, and each is the
thing exactly one probe was written to catch:

``HONEST``
    Answers correctly, echoes the model it was asked for, reports the token
    accounting the claimed model publishes, and emits the response shape of the
    family whose protocol it speaks.
``SUBSTITUTED``
    Answers at a materially lower accuracy, and optionally reports a different
    model id. The plain case the benchmark probe exists for.
``QUANTIZED``
    Slightly worse overall, and corrupts CJK output -- the shape reported from
    FP4/INT4 routing in the field, where aggregate accuracy barely moves but
    non-Latin scripts come back as replacement characters.
``EVADING``
    Answers correctly when the input string-matches an entry in its lookup
    table and badly otherwise. This is the cheapest attack a provider can run
    and it defeats every absolute accuracy measurement; the paraphrased arm of
    the evasion probe is the only thing that sees it.
``TRUNCATING``
    Advertises a large context window and silently clips the prompt at a much
    smaller one. ``truncation_keeps`` selects which end survives: ``"head"``
    reproduces a hard clip that loses the question at the end of the prompt,
    ``"tail"`` reproduces a rolling window, which retrieves a needle planted
    near the end and never one planted near the start.
``LITELLM``
    Answers every request 2xx, accepts every parameter, and honours only the
    ones it implements. The ``drop_params=True`` signature.
``BROKEN``
    Malformed tool-call arguments, no usage object, and logprob alternatives in
    ascending order. Not dishonesty -- just a stack that does not hold to its
    own schema, which the adapters must survive without inventing data.

All non-ASCII strings are written as escapes. The grading in several probes is
byte-exact, and a source file carrying literal CJK, RTL runs and combining marks
is one editor normalisation away from testing something other than it claims.
"""

from __future__ import annotations

import base64
import contextlib
import enum
import hashlib
import hmac
import json
import re
import threading
import time
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit

__all__ = ["MockConfig", "MockProvider", "Persona", "RecordedRequest"]


class Persona(str, enum.Enum):
    HONEST = "honest"
    SUBSTITUTED = "substituted"
    QUANTIZED = "quantized"
    EVADING = "evading"
    TRUNCATING = "truncating"
    LITELLM = "litellm"
    BROKEN = "broken"


#: Secret behind the fake thinking-block signatures. A signature is an HMAC over
#: the block's text, so replaying a block unchanged verifies and replaying a
#: tampered one does not -- which is the property the real signature has and the
#: only property the thinking-signature probe depends on.
_SIGNING_KEY = b"llmverify-mock-signing-key"


@dataclass(slots=True)
class RecordedRequest:
    """One request the server handled, kept so tests can inspect the wire form."""

    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: Any

    def json_body(self) -> dict[str, Any]:
        return self.body if isinstance(self.body, dict) else {}


@dataclass
class MockConfig:
    """Everything about how the fake provider behaves.

    Mutable on purpose: a test flips one field between requests to reproduce a
    provider that changes behaviour mid-run.
    """

    persona: Persona = Persona.HONEST
    #: Model id echoed back. ``None`` echoes whatever was requested.
    model_id: str | None = None
    #: Probability that a gradable question is answered correctly, applied
    #: deterministically per item so a rerun asks and answers the same things.
    accuracy: float = 1.0
    #: Accuracy applied to inputs the ``EVADING`` persona does not recognise.
    evasion_accuracy: float = 0.05
    #: The lookup table ``EVADING`` matches against, exactly as a provider
    #: running a corpus-containment check would hold it.
    known_items: tuple[str, ...] = ()

    #: Published-style token accounting. The tool-use system prompt costs
    #: ``tool_overhead_auto`` tokens with tool_choice auto and
    #: ``tool_overhead_forced`` when it is forced; every tool definition costs
    #: ``tool_definition_tokens`` on top.
    tool_overhead_auto: int = 286
    tool_overhead_forced: int = 406
    tool_definition_tokens: int = 24
    #: Fixed per-request envelope, standing in for a chat template.
    envelope_tokens: int = 7
    chars_per_token: int = 4

    accept_seed: bool = True
    accept_logprobs: bool = True
    return_logprobs: bool = True
    #: Reject request fields the protocol does not define, the way a first-party
    #: validator does. Turning this on is what makes the Anthropic persona
    #: refuse ``seed`` and ``logprobs``.
    reject_unknown_params: bool = False
    #: Parameters rejected when ``reject_unknown_params`` is set.
    unknown_params: tuple[str, ...] = ("seed", "logprobs", "top_logprobs", "prompt_logprobs")
    #: Honour ``response_format``. When false the endpoint accepts the field and
    #: answers in prose anyway.
    honour_response_format: bool = True

    latency_s: float = 0.0
    emit_thinking: bool = False
    sign_thinking: bool = True
    validate_signature: bool = True

    #: Real input ceiling, in tokens. ``None`` means the advertised window is
    #: the real one.
    context_limit_tokens: int | None = None
    truncation_keeps: Literal["head", "tail"] = "head"
    advertised_context: int = 1_000_000

    #: Whether the endpoint actually looks at inline images. A persona with this
    #: off accepts image content and answers from the text alone, which is what a
    #: text-only model wearing a multimodal name does.
    answers_vision: bool = True
    corrupt_cjk: bool = False
    omit_usage: bool = False
    malformed_tool_arguments: bool = False
    invalid_logprob_order: bool = False

    system_fingerprint: str | None = "fp_mock_0001"
    owned_by: str = "mockprovider"
    #: Force every completion to fail with this status, for error-path tests.
    fail_status: int | None = None
    fail_body: str = '{"error": {"message": "mock failure"}}'

    @classmethod
    def for_persona(cls, persona: Persona, **overrides: Any) -> MockConfig:
        """Build the preset for ``persona``, with any field overridden."""
        base = cls(persona=persona)
        preset: dict[str, Any] = {}
        if persona is Persona.SUBSTITUTED:
            preset = {"accuracy": 0.05, "tool_overhead_auto": 675, "tool_overhead_forced": 804}
        elif persona is Persona.QUANTIZED:
            preset = {"accuracy": 0.7, "corrupt_cjk": True}
        elif persona is Persona.EVADING:
            preset = {"accuracy": 1.0, "evasion_accuracy": 0.05}
        elif persona is Persona.TRUNCATING:
            preset = {"context_limit_tokens": 8_000, "advertised_context": 1_000_000}
        elif persona is Persona.LITELLM:
            preset = {
                "return_logprobs": False,
                "honour_response_format": False,
                "owned_by": "litellm",
                "system_fingerprint": None,
            }
        elif persona is Persona.BROKEN:
            preset = {
                "omit_usage": True,
                "malformed_tool_arguments": True,
                "invalid_logprob_order": True,
            }
        return replace(base, **{**preset, **overrides})


# --------------------------------------------------------------------------- #
# Deterministic answering
# --------------------------------------------------------------------------- #

_ECHO_MARKER = "Repeat the following text back exactly"
#: Only the question carries this phrasing; the planted line does not, so a
#: prompt whose tail was clipped away leaves the endpoint with nothing to answer.
_NEEDLE_QUESTION = re.compile(r"exactly one line recording the registry key for locker (\d+)")
_NEEDLE_LINE = re.compile(r"MEMO the registry key for locker (\d+) is ([A-Z0-9]+)")
#: Four digits or more, so the answer-format instruction cannot be mistaken for
#: the arithmetic item itself.
_SUM = re.compile(r"(\d{4,})\s*\+\s*(\d{4,})")

#: ``(prompt substring, correct answer)`` for the multilingual probe's tasks.
#: Written as escapes; see the module docstring.
_LANGUAGE_ANSWERS: tuple[tuple[str, str], ...] = (
    ("into Spanish", "El gato bebe agua en la casa."),
    (
        "into Russian",
        "\u041a\u043e\u0442 \u043f\u044c\u0451\u0442 \u0432\u043e\u0434\u0443 "
        "\u0432 \u0434\u043e\u043c\u0435.",
    ),
    (
        "Simplified Chinese",
        "\u732b\u5728\u623f\u5b50\u91cc\u559d\u6c34\u3002",
    ),
    (
        "into Arabic",
        "\u0627\u0644\u0642\u0637 \u064a\u0634\u0631\u0628 \u0627\u0644\u0645\u0627\u0621 "
        "\u0641\u064a \u0627\u0644\u0645\u0646\u0632\u0644.",
    ),
    # "What is the capital of France?" in Russian.
    ("\u0441\u0442\u043e\u043b\u0438\u0446\u0430", "\u041f\u0430\u0440\u0438\u0436"),
    # "How many months in a year?" in Chinese.
    ("\u4e00\u5e74\u6709\u591a\u5c11\u4e2a\u6708", "12"),
    # "How many days in a week?" in Arabic.
    ("\u0623\u064a\u0627\u0645 \u0627\u0644\u0623\u0633\u0628\u0648\u0639", "7"),
)

_STRUCTURAL_ANSWERS: tuple[tuple[str, str], ...] = (
    ("capital city of Australia", "Canberra"),
    ("shortest day", "Jupiter"),
    ("powered aeroplane flight", "1903"),
    ("chemical symbol for tungsten", "W"),
    ("ocean lies between Africa and Australia", "the Indian Ocean"),
)


def _unit_interval(*parts: str) -> float:
    """A stable pseudo-random number in ``[0, 1)`` keyed by ``parts``.

    Hashing rather than an RNG stream so that the same item gets the same
    treatment however many other requests preceded it. That is what makes the
    ``EVADING`` persona's verbatim and paraphrased arms comparable, and what
    makes a rerun of the whole suite ask and answer identically.
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF  # kana
        or 0x3400 <= code <= 0x4DBF  # CJK extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK unified
        or 0xAC00 <= code <= 0xD7AF  # hangul syllables
        or 0xF900 <= code <= 0xFAFF  # CJK compatibility
    )


def _corrupt_cjk(text: str) -> str:
    """Replace every CJK codepoint with U+FFFD, leaving other scripts intact."""
    return "".join("\ufffd" if _is_cjk(char) else char for char in text)


def sign_thinking(text: str) -> str:
    """The signature this server issues for a thinking block."""
    mac = hmac.new(_SIGNING_KEY, text.encode("utf-8"), hashlib.sha256)
    return "sig_" + base64.b64encode(mac.digest()).decode("ascii")


@dataclass(slots=True)
class _Reply:
    """What the brain decided to say."""

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    thinking: str = ""
    finish_reason: str = "stop"


class _Brain:
    """Turns a prompt into an answer, at the persona's accuracy."""

    def __init__(self, config: MockConfig) -> None:
        self.config = config

    def recognises(self, prompt: str) -> bool:
        """Whether a lookup table of known benchmark items would match ``prompt``.

        Substring containment, which is the check a dishonest provider can
        afford to run on every request and the one
        :func:`llmverify.benchmarks.paraphrase.defeats_substring_match` is
        written to break.
        """
        return any(item and item in prompt for item in self.config.known_items)

    def _answers_correctly(self, prompt: str, key: str) -> bool:
        config = self.config
        accuracy = config.accuracy
        if config.persona is Persona.EVADING and not self.recognises(prompt):
            accuracy = config.evasion_accuracy
        return _unit_interval(key) < accuracy

    def reply(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        forced_tool: str | None,
        images: list[bytes] | None = None,
    ) -> _Reply:
        thinking = "Considering the request." if self.config.emit_thinking else ""

        if images and self.config.answers_vision:
            seen = answer_vision(prompt, images)
            if seen is not None:
                return _Reply(text=f"ANSWER: {seen}", thinking=thinking)

        if tools and forced_tool is not None:
            return _Reply(
                text="",
                tool_calls=self._tool_calls(tools, forced_tool),
                thinking=thinking,
                finish_reason="tool_calls",
            )

        text = self._text(prompt)
        return _Reply(text=text, thinking=thinking)

    def _tool_calls(self, tools: list[dict[str, Any]], forced: str) -> list[dict[str, Any]]:
        name = forced if forced not in ("any", "required") else _tool_name(tools[0])
        arguments = "{" if self.config.malformed_tool_arguments else '{"city": "Paris"}'
        return [{"id": "call_mock_0001", "name": name, "arguments": arguments}]

    def _text(self, prompt: str) -> str:
        for handler in (
            self._echo,
            self._needle,
            self._arithmetic,
            self._language,
            self._structural,
            self._identity,
            self._prose,
        ):
            answer = handler(prompt)
            if answer is not None:
                return self._maybe_corrupt(answer)
        return "ok"

    def _maybe_corrupt(self, text: str) -> str:
        return _corrupt_cjk(text) if self.config.corrupt_cjk else text

    def _echo(self, prompt: str) -> str | None:
        if _ECHO_MARKER not in prompt:
            return None
        _head, _, body = prompt.partition("\n\n")
        return body.strip()

    def _needle(self, prompt: str) -> str | None:
        question = _NEEDLE_QUESTION.search(prompt)
        if question is None:
            return None
        locker = question.group(1)
        for match in _NEEDLE_LINE.finditer(prompt):
            if match.group(1) == locker:
                return f"ANSWER: {match.group(2)}"
        # The line is not in what arrived, which is the whole point of the
        # truncating persona: the endpoint answers, it just cannot answer this.
        return "ANSWER: UNKNOWN"

    def _arithmetic(self, prompt: str) -> str | None:
        match = _SUM.search(prompt)
        if match is None:
            return None
        left, right = int(match.group(1)), int(match.group(2))
        total = left + right
        if not self._answers_correctly(prompt, f"sum:{left}+{right}"):
            total += 7
        return f"ANSWER: {total}"

    def _language(self, prompt: str) -> str | None:
        for marker, answer in _LANGUAGE_ANSWERS:
            if marker in prompt:
                if not self._answers_correctly(prompt, f"lang:{marker}"):
                    return "I am not able to answer that."
                return answer
        return None

    def _structural(self, prompt: str) -> str | None:
        for marker, answer in _STRUCTURAL_ANSWERS:
            if marker in prompt:
                return f"ANSWER: {answer}"
        return None

    def _identity(self, prompt: str) -> str | None:
        lowered = prompt.lower()
        if "which model" in lowered or "who are you" in lowered or "identify yourself" in lowered:
            return f"I am {self.config.model_id or 'an assistant'}."
        return None

    def _prose(self, prompt: str) -> str | None:
        """Answer a request for prose with prose.

        The determinism probe asks for two sentences and reads a short
        byte-identical reply as a cache answering instead of a model. Returning
        "ok" there would make every persona look like a lookup table, which is a
        finding about this fixture rather than about the endpoint under test.
        """
        if "two sentences" not in prompt.lower():
            return None
        return (
            "The lantern turns slowly above a grey sea and the fog thins as the light "
            "reaches the water. Gulls settle on the gallery rail while the keeper writes "
            "the hour in the log."
        )


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return str(tool.get("name") or "tool")


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #


class MockProvider:
    """A running fake provider. Use as a context manager or call :meth:`close`."""

    def __init__(self, config: MockConfig | None = None) -> None:
        self.config = config if config is not None else MockConfig()
        self.requests: list[RecordedRequest] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._server.provider = self  # type: ignore[attr-defined]
        # A short poll interval so that shutting the server down is immediate.
        # The default half-second would otherwise be paid once per test.
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------ address

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def root(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def openai_url(self) -> str:
        return f"{self.root}/v1"

    @property
    def anthropic_url(self) -> str:
        return f"{self.root}/v1"

    @property
    def gemini_url(self) -> str:
        return f"{self.root}/v1beta"

    # ---------------------------------------------------------------- lifecycle

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> MockProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ helpers

    @property
    def brain(self) -> _Brain:
        return _Brain(self.config)

    def bodies(self, *, path_contains: str = "") -> list[dict[str, Any]]:
        """Every JSON request body seen, optionally filtered by path."""
        return [
            record.json_body()
            for record in self.requests
            if path_contains in record.path and isinstance(record.body, dict)
        ]

    def reset(self) -> None:
        self.requests.clear()

    # ------------------------------------------------------------- token counts

    def count_tokens(
        self, texts: list[str], tools: list[dict[str, Any]], *, forced: bool
    ) -> int:
        """The token count this endpoint reports for a request.

        Deliberately additive and floor-divided so that the differencing the
        token-accounting probe does comes out exact: two prompts differing by a
        repetition differ by exactly the repetition's tokens, and one tool costs
        exactly ``tool_definition_tokens``.
        """
        config = self.config
        total = config.envelope_tokens
        total += sum(len(text) for text in texts) // config.chars_per_token
        if tools:
            total += config.tool_overhead_forced if forced else config.tool_overhead_auto
            total += config.tool_definition_tokens * len(tools)
        return total

    def apply_context_limit(self, text: str) -> tuple[str, int]:
        """Clip ``text`` to the real window, returning it and the reported count.

        The reported count is the count of what survived, which is what silent
        truncation looks like from outside: the endpoint answers happily and its
        own usage object is the only place the loss shows up.
        """
        config = self.config
        tokens = len(text) // config.chars_per_token
        limit = config.context_limit_tokens
        if limit is None or tokens <= limit:
            return text, tokens
        budget = limit * config.chars_per_token
        clipped = text[:budget] if config.truncation_keeps == "head" else text[-budget:]
        return clipped, limit


class _Handler(BaseHTTPRequestHandler):
    """Routes the three protocols. One instance per request, as usual."""

    protocol_version = "HTTP/1.1"
    server_version = "llmverify-mock/1.0"

    # ------------------------------------------------------------------ plumbing

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence the default stderr access log."""

    @property
    def provider(self) -> MockProvider:
        return self.server.provider  # type: ignore[attr-defined]

    @property
    def config(self) -> MockConfig:
        return self.provider.config

    def _read(self) -> tuple[str, dict[str, list[str]], Any]:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body: Any = json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            body = raw.decode("utf-8", "replace")
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        self.provider.requests.append(
            RecordedRequest(
                method=self.command,
                path=parts.path,
                query=query,
                headers={k.lower(): v for k, v in self.headers.items()},
                body=body,
            )
        )
        return parts.path, query, body

    def _send(self, status: int, payload: Any, *, raw: str | None = None) -> None:
        body = (raw if raw is not None else json.dumps(payload)).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("x-mock-persona", self.config.persona.value)
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, frames: list[str]) -> None:
        """Stream frames and close, so no length has to be known in advance."""
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        for frame in frames:
            self.wfile.write(f"data: {frame}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _error(self, status: int, message: str) -> None:
        self._send(status, {"error": {"message": message, "type": "invalid_request_error"}})

    # -------------------------------------------------------------------- routes

    def do_GET(self) -> None:
        path, _query, _body = self._read()
        if self.config.fail_status is not None:
            self._send(self.config.fail_status, None, raw=self.config.fail_body)
            return
        if path.endswith("/models"):
            self._send(200, self._catalogue(path))
            return
        self._error(404, f"no route for GET {path}")

    def do_POST(self) -> None:
        path, query, body = self._read()
        payload = body if isinstance(body, dict) else {}

        if self.config.latency_s:
            time.sleep(self.config.latency_s)

        if path.endswith("/chat/completions"):
            self._openai(payload)
        elif path.endswith("/messages/count_tokens"):
            self._anthropic_count(payload)
        elif path.endswith("/messages"):
            self._anthropic(payload)
        elif ":generateContent" in path or ":streamGenerateContent" in path:
            self._gemini(path, payload, stream=":streamGenerateContent" in path or "sse" in query)
        elif ":countTokens" in path:
            self._gemini_count(payload)
        else:
            self._error(404, f"no route for POST {path}")

    # ----------------------------------------------------------------- catalogue

    def _catalogue(self, path: str) -> dict[str, Any]:
        config = self.config
        model = config.model_id or self._last_requested_model() or "mock-model"
        if "/v1beta" in path:
            return {
                "models": [
                    {
                        "name": f"models/{model}",
                        "displayName": model,
                        "inputTokenLimit": config.advertised_context,
                        "outputTokenLimit": 8192,
                    }
                ]
            }
        if self._is_anthropic_style():
            return {
                "data": [
                    {
                        "id": model,
                        "type": "model",
                        "display_name": model,
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ],
                "has_more": False,
            }
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 1_767_225_600,
                    "owned_by": config.owned_by,
                }
            ],
        }

    def _last_requested_model(self) -> str | None:
        """The model id the client most recently asked for, if it has asked.

        Without this a config that leaves ``model_id`` unset would list a
        catalogue that does not contain the model the endpoint happily serves,
        which is a finding in its own right and not one every test wants.
        """
        for record in reversed(self.provider.requests):
            model = record.json_body().get("model")
            if isinstance(model, str) and model:
                return model
        return None

    def _is_anthropic_style(self) -> bool:
        """Whether the client has been speaking the Messages protocol.

        The catalogue route is the same path in both protocols, so the shape is
        chosen from what this client has already sent rather than guessed.
        """
        return any("/messages" in record.path for record in self.provider.requests)

    # -------------------------------------------------------------------- openai

    def _openai(self, payload: dict[str, Any]) -> None:
        if self.config.fail_status is not None:
            self._send(self.config.fail_status, None, raw=self.config.fail_body)
            return
        rejected = self._rejected_param(payload)
        if rejected is not None:
            self._error(400, f"Unsupported parameter: {rejected!r}")
            return

        messages = payload.get("messages") or []
        texts = [_openai_message_text(m) for m in messages if isinstance(m, dict)]
        images = openai_images(messages)
        prompt = "\n".join(texts)
        prompt, reported_input = self._clip(prompt, texts)

        tools = [t for t in (payload.get("tools") or []) if isinstance(t, dict)]
        forced = _openai_forced_tool(payload.get("tool_choice"))
        reply = self.provider.brain.reply(
            prompt, tools=tools, forced_tool=forced, images=images
        )
        text = self._shape_for_schema(payload, reply.text)

        message: dict[str, Any] = {"role": "assistant", "content": text}
        if reply.tool_calls:
            message["content"] = None
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for call in reply.tool_calls
            ]
        if reply.thinking:
            message["reasoning_content"] = reply.thinking

        choice: dict[str, Any] = {
            "index": 0,
            "message": message,
            "finish_reason": reply.finish_reason,
            "logprobs": None,
        }
        if payload.get("logprobs") and self.config.return_logprobs:
            choice["logprobs"] = {"content": self._logprobs(text)}

        body: dict[str, Any] = {
            "id": "chatcmpl-mock0000000000000001",
            "object": "chat.completion",
            "created": 1_785_000_000,
            "model": self.config.model_id or payload.get("model") or "mock-model",
            "choices": [choice],
        }
        if self.config.system_fingerprint is not None:
            body["system_fingerprint"] = self.config.system_fingerprint
        if not self.config.omit_usage:
            output = max(1, len(text) // self.config.chars_per_token)
            body["usage"] = {
                "prompt_tokens": reported_input + self._tool_tokens(tools, payload),
                "completion_tokens": output,
                "total_tokens": reported_input + output,
                "completion_tokens_details": {"reasoning_tokens": 0},
            }

        if payload.get("stream"):
            self._send_sse(_openai_stream_frames(body, text))
            return
        self._send(200, body)

    def _shape_for_schema(self, payload: dict[str, Any], text: str) -> str:
        """Answer as JSON when a schema was requested and this persona honours it."""
        response_format = payload.get("response_format")
        if not isinstance(response_format, dict):
            return text
        if not self.config.honour_response_format:
            return text
        schema = _nested(response_format, "json_schema", "schema")
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(properties, dict):
            return json.dumps({"result": text})
        return json.dumps({name: _placeholder(spec) for name, spec in properties.items()})

    def _logprobs(self, text: str) -> list[dict[str, Any]]:
        tokens = (text or "ok").split() or ["ok"]
        entries: list[dict[str, Any]] = []
        for index, token in enumerate(tokens[:8]):
            base = -0.05 - 0.01 * index
            alternatives = [
                {"token": token, "logprob": base},
                {"token": f"{token}_alt", "logprob": base - 1.5},
                {"token": f"{token}_alt2", "logprob": base - 3.0},
            ]
            if self.config.invalid_logprob_order:
                alternatives.reverse()
            entries.append({"token": token, "logprob": base, "top_logprobs": alternatives})
        return entries

    def _rejected_param(self, payload: dict[str, Any]) -> str | None:
        config = self.config
        if not config.accept_seed and "seed" in payload:
            return "seed"
        if not config.accept_logprobs and payload.get("logprobs"):
            return "logprobs"
        if config.reject_unknown_params:
            for name in config.unknown_params:
                if name in payload:
                    return name
        return None

    def _tool_tokens(self, tools: list[dict[str, Any]], payload: dict[str, Any]) -> int:
        if not tools:
            return 0
        forced = _openai_forced_tool(payload.get("tool_choice")) is not None
        config = self.config
        overhead = config.tool_overhead_forced if forced else config.tool_overhead_auto
        return overhead + config.tool_definition_tokens * len(tools)

    def _clip(self, prompt: str, texts: list[str]) -> tuple[str, int]:
        """Apply the real context window and report what survived."""
        clipped, tokens = self.provider.apply_context_limit(prompt)
        if self.config.context_limit_tokens is None:
            return clipped, self.provider.count_tokens(texts, [], forced=False)
        return clipped, tokens + self.config.envelope_tokens

    # ----------------------------------------------------------------- anthropic

    def _anthropic(self, payload: dict[str, Any]) -> None:
        if self.config.fail_status is not None:
            self._send(self.config.fail_status, None, raw=self.config.fail_body)
            return
        rejected = self._rejected_param(payload)
        if rejected is not None:
            self._error(400, f"{rejected}: Extra inputs are not permitted")
            return

        invalid = self._invalid_replayed_signature(payload)
        if invalid is not None:
            self._error(400, f"The thinking block signature is invalid: {invalid}")
            return

        texts = _anthropic_texts(payload)
        images = anthropic_images(payload)
        prompt = "\n".join(texts)
        prompt, reported_input = self._clip(prompt, texts)
        tools = [t for t in (payload.get("tools") or []) if isinstance(t, dict)]
        forced = _anthropic_forced_tool(payload.get("tool_choice"))
        reply = self.provider.brain.reply(
            prompt, tools=tools, forced_tool=forced, images=images
        )

        content: list[dict[str, Any]] = []
        if reply.thinking:
            block: dict[str, Any] = {"type": "thinking", "thinking": reply.thinking}
            if self.config.sign_thinking:
                block["signature"] = sign_thinking(reply.thinking)
            content.append(block)
        if reply.tool_calls:
            for call in reply.tool_calls:
                try:
                    arguments = json.loads(call["arguments"])
                except ValueError:
                    arguments = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": arguments,
                    }
                )
        else:
            content.append({"type": "text", "text": reply.text})

        stop_reason = "tool_use" if reply.tool_calls else "end_turn"
        body: dict[str, Any] = {
            "id": "msg_mock0000000000000001",
            "type": "message",
            "role": "assistant",
            "model": self.config.model_id or payload.get("model") or "mock-model",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "service_tier": "standard",
        }
        if not self.config.omit_usage:
            output = max(1, len(reply.text) // self.config.chars_per_token)
            body["usage"] = {
                "input_tokens": reported_input + self._anthropic_tool_tokens(payload, tools),
                "output_tokens": output,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 0,
                    "ephemeral_1h_input_tokens": 0,
                },
                "server_tool_use": {"web_search_requests": 0},
                "output_tokens_details": {"thinking_tokens": 0},
            }

        if payload.get("stream"):
            self._send_sse(_anthropic_stream_frames(body, reply.text))
            return
        self._send(200, body)

    def _anthropic_tool_tokens(
        self, payload: dict[str, Any], tools: list[dict[str, Any]]
    ) -> int:
        if not tools:
            return 0
        forced = _anthropic_forced_tool(payload.get("tool_choice")) is not None
        config = self.config
        overhead = config.tool_overhead_forced if forced else config.tool_overhead_auto
        return overhead + config.tool_definition_tokens * len(tools)

    def _anthropic_count(self, payload: dict[str, Any]) -> None:
        texts = _anthropic_texts(payload)
        tools = [t for t in (payload.get("tools") or []) if isinstance(t, dict)]
        forced = _anthropic_forced_tool(payload.get("tool_choice")) is not None
        self._send(200, {"input_tokens": self.provider.count_tokens(texts, tools, forced=forced)})

    def _invalid_replayed_signature(self, payload: dict[str, Any]) -> str | None:
        """Refuse a replayed thinking block whose signature does not verify."""
        if not self.config.validate_signature:
            return None
        for message in payload.get("messages") or []:
            if not isinstance(message, dict):
                continue
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "thinking":
                    continue
                signature = block.get("signature")
                if not isinstance(signature, str):
                    return "missing"
                if not hmac.compare_digest(
                    signature, sign_thinking(str(block.get("thinking") or ""))
                ):
                    return signature[:24]
        return None

    # -------------------------------------------------------------------- gemini

    def _gemini(self, path: str, payload: dict[str, Any], *, stream: bool) -> None:
        if self.config.fail_status is not None:
            self._send(self.config.fail_status, None, raw=self.config.fail_body)
            return
        texts = _gemini_texts(payload)
        images = gemini_images(payload)
        prompt = "\n".join(texts)
        prompt, reported_input = self._clip(prompt, texts)
        tools = _gemini_tools(payload)
        forced = _gemini_forced_tool(payload)
        reply = self.provider.brain.reply(
            prompt, tools=tools, forced_tool=forced, images=images
        )

        parts: list[dict[str, Any]] = []
        if reply.thinking:
            parts.append({"text": reply.thinking, "thought": True})
        if reply.tool_calls:
            for call in reply.tool_calls:
                try:
                    arguments = json.loads(call["arguments"])
                except ValueError:
                    arguments = {}
                parts.append({"functionCall": {"name": call["name"], "args": arguments}})
        else:
            parts.append({"text": reply.text})

        output = max(1, len(reply.text) // self.config.chars_per_token)
        body: dict[str, Any] = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": parts},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "modelVersion": self.config.model_id or _gemini_model(path),
            "responseId": "resp-mock-0001",
        }
        if not self.config.omit_usage:
            body["usageMetadata"] = {
                "promptTokenCount": reported_input,
                "candidatesTokenCount": output,
                "totalTokenCount": reported_input + output,
            }
        if stream:
            self._send_sse([json.dumps(body)])
            return
        self._send(200, body)

    def _gemini_count(self, payload: dict[str, Any]) -> None:
        request = payload.get("generateContentRequest")
        inner = request if isinstance(request, dict) else payload
        texts = _gemini_texts(inner)
        tools = _gemini_tools(inner)
        forced = _gemini_forced_tool(inner) is not None
        self._send(
            200, {"totalTokens": self.provider.count_tokens(texts, tools, forced=forced)}
        )


# --------------------------------------------------------------------------- #
# Payload readers
# --------------------------------------------------------------------------- #


def _nested(node: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _placeholder(spec: Any) -> Any:
    kind = spec.get("type") if isinstance(spec, dict) else None
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.5
    if kind == "boolean":
        return True
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return "blue"


def _openai_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for part in content:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "".join(chunks)


def _openai_forced_tool(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice in ("required", "any") else None
    if isinstance(tool_choice, dict):
        name = _nested(tool_choice, "function", "name")
        return name if isinstance(name, str) else "required"
    return None


def _anthropic_texts(payload: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    system = payload.get("system")
    if isinstance(system, str) and system:
        texts.append(system)
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for block in content or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                texts.append(block["text"])
    return texts


def _anthropic_forced_tool(tool_choice: Any) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    kind = tool_choice.get("type")
    if kind == "tool" and isinstance(tool_choice.get("name"), str):
        return tool_choice["name"]
    if kind == "any":
        return "any"
    return None


def _gemini_texts(payload: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for part in _nested(payload, "systemInstruction", "parts") or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    for turn in payload.get("contents") or []:
        if not isinstance(turn, dict):
            continue
        for part in turn.get("parts") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return texts


def _gemini_tools(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for entry in payload.get("tools") or []:
        if not isinstance(entry, dict):
            continue
        for declaration in entry.get("functionDeclarations") or []:
            if isinstance(declaration, dict):
                tools.append(declaration)
    return tools


def _gemini_forced_tool(payload: dict[str, Any]) -> str | None:
    calling = _nested(payload, "toolConfig", "functionCallingConfig")
    if not isinstance(calling, dict):
        return None
    if calling.get("mode") != "ANY":
        return None
    allowed = calling.get("allowedFunctionNames")
    if isinstance(allowed, list) and allowed:
        return str(allowed[0])
    return "any"


def _gemini_model(path: str) -> str:
    tail = path.rsplit("/", 1)[-1]
    return tail.split(":", 1)[0]


# --------------------------------------------------------------------------- #
# Stream framing
# --------------------------------------------------------------------------- #


def _openai_stream_frames(body: dict[str, Any], text: str) -> list[str]:
    """Split a completion into role, content and usage frames."""
    head = {
        "id": body["id"],
        "object": "chat.completion.chunk",
        "created": body["created"],
        "model": body["model"],
        "system_fingerprint": body.get("system_fingerprint"),
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    frames = [json.dumps(head)]
    chunks = [text[i : i + 24] for i in range(0, len(text), 24)] or [""]
    for chunk in chunks:
        frames.append(
            json.dumps(
                {
                    "id": body["id"],
                    "object": "chat.completion.chunk",
                    "model": body["model"],
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
            )
        )
    tail: dict[str, Any] = {
        "id": body["id"],
        "object": "chat.completion.chunk",
        "model": body["model"],
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    if "usage" in body:
        tail["usage"] = body["usage"]
    frames.append(json.dumps(tail))
    return frames


def _anthropic_stream_frames(body: dict[str, Any], text: str) -> list[str]:
    """Emit the event sequence the Messages protocol defines, signatures included."""
    start = {k: v for k, v in body.items() if k != "content"}
    start["content"] = []
    frames = [json.dumps({"type": "message_start", "message": start})]

    for index, block in enumerate(body["content"]):
        kind = block.get("type")
        frames.append(
            json.dumps(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": (
                        {"type": kind, "thinking": ""}
                        if kind == "thinking"
                        else {"type": kind, "text": ""}
                        if kind == "text"
                        else {**block, "input": {}}
                    ),
                }
            )
        )
        if kind == "thinking":
            frames.append(
                json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "thinking_delta", "thinking": block["thinking"]},
                    }
                )
            )
            if block.get("signature"):
                frames.append(
                    json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "signature_delta",
                                "signature": block["signature"],
                            },
                        }
                    )
                )
        elif kind == "text":
            for chunk in [text[i : i + 24] for i in range(0, len(text), 24)] or [""]:
                frames.append(
                    json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": chunk},
                        }
                    )
                )
        elif kind == "tool_use":
            frames.append(
                json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(block.get("input") or {}),
                        },
                    }
                )
            )
        frames.append(json.dumps({"type": "content_block_stop", "index": index}))

    delta: dict[str, Any] = {
        "type": "message_delta",
        "delta": {"stop_reason": body["stop_reason"], "stop_sequence": None},
    }
    if "usage" in body:
        delta["usage"] = body["usage"]
    frames.append(json.dumps(delta))
    frames.append(json.dumps({"type": "message_stop"}))
    return frames


# --------------------------------------------------------------------------- #
# Vision
# --------------------------------------------------------------------------- #
#
# The vision probe distinguishes an endpoint that rejects images from one that
# accepts them and cannot see, so a mock that merely accepts them is
# indistinguishable from a text-only model wearing a multimodal name -- exactly
# the substitution the probe exists to catch. To test the probe rather than
# tautologically confirm it, the honest persona has to actually look.
#
# The images are decoded and measured, never guessed at from the prompt. The
# probe writes 8-bit truecolour PNGs with filter type 0 on every scanline, so
# decoding is a zlib inflate and a stride calculation.

_PALETTE_RGB: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("red", (215, 35, 35)),
    ("green", (30, 155, 60)),
    ("blue", (35, 70, 200)),
    ("yellow", (240, 210, 45)),
    ("purple", (130, 50, 175)),
    ("orange", (240, 135, 30)),
)


class _Image:
    """A decoded 8-bit truecolour PNG."""

    __slots__ = ("_raw", "_stride", "height", "width")

    def __init__(self, width: int, height: int, raw: bytes) -> None:
        self.width = width
        self.height = height
        self._raw = raw
        self._stride = 1 + width * 3

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        base = y * self._stride + 1 + x * 3
        return self._raw[base], self._raw[base + 1], self._raw[base + 2]


def decode_png(data: bytes) -> _Image | None:
    """Decode the narrow PNG dialect the vision probe emits, or None."""
    import zlib

    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    offset = 8
    width = height = 0
    idat = bytearray()
    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        if kind == b"IHDR":
            width = int.from_bytes(body[0:4], "big")
            height = int.from_bytes(body[4:8], "big")
            depth, colour_type, _, _, interlace = body[8:13]
            if (depth, colour_type, interlace) != (8, 2, 0):
                return None
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
        offset += 12 + length
    if not width or not height:
        return None
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error:
        return None
    if len(raw) < height * (1 + width * 3) or any(
        raw[y * (1 + width * 3)] != 0 for y in range(height)
    ):
        return None  # a filter other than None, which this decoder does not do
    return _Image(width, height, raw)


def _nearest_colour(rgb: tuple[int, int, int]) -> str:
    return min(
        _PALETTE_RGB,
        key=lambda entry: sum((a - b) ** 2 for a, b in zip(entry[1], rgb, strict=True)),
    )[0]


def answer_vision(prompt: str, images: list[bytes]) -> str | None:
    """Answer one of the probe's three questions by measuring the image.

    Returns ``None`` when the prompt is not a vision task or the image cannot be
    decoded, so the caller falls through to its ordinary text behaviour.
    """
    image = next((img for img in (decode_png(b) for b in images) if img is not None), None)
    if image is None:
        return None
    lowered = prompt.lower()

    if "one solid colour" in lowered:
        return _nearest_colour(image.pixel(image.width // 2, image.height // 2))

    if "count the filled squares" in lowered:
        # Four-by-four layout; a cell counts as filled when its centre is inked.
        cell = image.width // 4
        background = image.pixel(1, 1)
        filled = sum(
            image.pixel(column * cell + cell // 2, row * cell + cell // 2) != background
            for row in range(4)
            for column in range(4)
        )
        return str(filled)

    if "3 by 3 grid" in lowered:
        cell = image.width // 3
        centres = [
            (row, column, image.pixel(column * cell + cell // 2, row * cell + cell // 2))
            for row in range(3)
            for column in range(3)
        ]
        counts: dict[tuple[int, int, int], int] = {}
        for _, _, rgb in centres:
            counts[rgb] = counts.get(rgb, 0) + 1
        odd = min(centres, key=lambda entry: counts[entry[2]])
        return f"{odd[0] + 1},{odd[1] + 1}"

    return None


def openai_images(messages: list[Any]) -> list[bytes]:
    """Inline image bytes from an OpenAI-shaped message list."""
    out: list[bytes] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = (part.get("image_url") or {}).get("url", "")
            _, _, payload = url.partition("base64,")
            if payload:
                with contextlib.suppress(ValueError, TypeError):
                    out.append(base64.b64decode(payload))
    return out


def anthropic_images(payload: dict[str, Any]) -> list[bytes]:
    """Inline image bytes from an Anthropic-shaped request body."""
    out: list[bytes] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image":
                continue
            source = block.get("source") or {}
            if source.get("type") == "base64" and source.get("data"):
                with contextlib.suppress(ValueError, TypeError):
                    out.append(base64.b64decode(source["data"]))
    return out


def gemini_images(payload: dict[str, Any]) -> list[bytes]:
    """Inline image bytes from a Gemini-shaped request body."""
    out: list[bytes] = []
    for content in payload.get("contents") or []:
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            inline = part.get("inlineData") if isinstance(part, dict) else None
            if isinstance(inline, dict) and inline.get("data"):
                with contextlib.suppress(ValueError, TypeError):
                    out.append(base64.b64decode(inline["data"]))
    return out
