"""Payload construction and response parsing for all three protocols.

Two rules run through every test here.

**Omission, never nulling.** A field the request left unset must be absent from
the wire payload, not present as ``null``. The api-surface probe reads the
difference between a rejected parameter and a silently dropped one, and an
adapter that emitted ``"seed": null`` would turn every endpoint into one that
"accepts" seed.

**No information lost.** Whatever the endpoint volunteered has to survive into
``raw``, and whatever it got wrong has to survive as it was sent: unparseable
tool arguments stay unparsed, a missing usage object stays missing, and a
signature is never normalised. Those are the observations the identity probes
are built on.

Every round trip goes over a real socket to the mock provider, so the transport,
the retry policy and the parsers are all in the path.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from conftest import CLAIMED_MODEL, anthropic_config, gemini_config, openai_config
from llmverify.adapters._http import HttpResult, raise_for_status
from llmverify.adapters.anthropic import AnthropicAdapter
from llmverify.adapters.base import available_adapters, get_adapter
from llmverify.adapters.gemini import GeminiAdapter
from llmverify.adapters.openai_compat import OpenAICompatAdapter
from llmverify.errors import AdapterError, AuthError, ProviderError, RateLimited
from llmverify.types import (
    ApiFamily,
    ChatRequest,
    FinishReason,
    ImagePart,
    Message,
    ParamSupport,
    Role,
    TextPart,
    ThinkingBlock,
    ToolCall,
    ToolSpec,
)
from mockserver import MockProvider, Persona, sign_thinking

TOOL = ToolSpec(
    name="get_weather",
    description="Return the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

SCHEMA: dict[str, Any] = {
    "title": "colour_answer",
    "type": "object",
    "properties": {"colour": {"type": "string"}},
    "required": ["colour"],
}

#: A one-pixel PNG, so the multimodal path carries real bytes.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg=="
)


def result(body: Any, *, status: int = 200) -> HttpResult:
    return HttpResult(status=status, headers={}, json=body, text=json.dumps(body), total_s=0.1)


def nulls(node: Any, path: str = "") -> list[str]:
    """Every path in ``node`` whose value is ``None``."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if value is None:
                found.append(here)
            found.extend(nulls(value, here))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(nulls(value, f"{path}[{index}]"))
    return found


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_the_three_first_party_adapters_are_registered() -> None:
    registry = available_adapters()
    assert registry["openai"] is OpenAICompatAdapter
    assert registry["anthropic"] is AnthropicAdapter
    assert registry["gemini"] is GeminiAdapter


def test_an_unknown_adapter_names_the_ones_that_exist(mock_provider: MockProvider) -> None:
    config = openai_config(mock_provider, api="not-an-adapter")
    with pytest.raises(AdapterError, match="anthropic"):
        get_adapter(config)


def test_capabilities_record_the_protocol_not_the_endpoint() -> None:
    assert AnthropicAdapter.capabilities.logprobs is False
    assert AnthropicAdapter.capabilities.seed is False
    assert AnthropicAdapter.capabilities.thinking_signature is True
    assert AnthropicAdapter.capabilities.count_tokens_endpoint is True
    assert OpenAICompatAdapter.capabilities.logprobs is True
    assert OpenAICompatAdapter.capabilities.seed is True
    assert GeminiAdapter.capabilities.logprobs is False
    assert GeminiAdapter.capabilities.seed is True


# --------------------------------------------------------------------------- #
# The omit-None rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("api", ["openai", "anthropic", "gemini"])
def test_an_unset_field_is_absent_rather_than_null(
    mock_provider: MockProvider, api: str
) -> None:
    config = openai_config(mock_provider, api=api)
    adapter = get_adapter(config)
    payload = adapter.build_payload(ChatRequest(messages=(Message(Role.USER, "hi"),)))

    assert nulls(payload) == []
    for absent in ("seed", "temperature", "top_p", "tools", "tool_choice", "logprobs"):
        assert absent not in json.dumps(payload)


@pytest.mark.parametrize("api", ["openai", "anthropic", "gemini"])
def test_a_fully_specified_request_still_carries_no_nulls(
    mock_provider: MockProvider, api: str
) -> None:
    config = openai_config(mock_provider, api=api)
    adapter = get_adapter(config)
    payload = adapter.build_payload(
        ChatRequest(
            messages=(Message(Role.SYSTEM, "be terse"), Message(Role.USER, "hi")),
            max_tokens=64,
            temperature=0.0,
            top_p=0.9,
            seed=7,
            stop=("STOP",),
            tools=(TOOL,),
            tool_choice="auto",
            response_schema=SCHEMA,
            logprobs=True,
            top_logprobs=5,
            reasoning_effort="high",
            thinking_budget=1024,
        )
    )
    assert nulls(payload) == []


@pytest.mark.parametrize("api", ["openai", "anthropic", "gemini"])
def test_extra_body_is_merged_verbatim_including_nulls(
    mock_provider: MockProvider, api: str
) -> None:
    """Probes send malformed values on purpose; the adapter must not sanitise them."""
    config = openai_config(mock_provider, api=api)
    adapter = get_adapter(config)
    payload = adapter.build_payload(
        ChatRequest(
            messages=(Message(Role.USER, "hi"),),
            extra_body={"prompt_logprobs": 1, "deliberately_null": None},
        )
    )
    assert payload["prompt_logprobs"] == 1
    assert nulls(payload) == ["deliberately_null"]


# --------------------------------------------------------------------------- #
# OpenAI payloads
# --------------------------------------------------------------------------- #


def test_openai_payload_shape(mock_provider: MockProvider) -> None:
    adapter = get_adapter(openai_config(mock_provider))
    payload = adapter.build_payload(
        ChatRequest(
            messages=(Message(Role.USER, "hi"),),
            tools=(TOOL,),
            tool_choice="get_weather",
            response_schema=SCHEMA,
            top_logprobs=3,
            reasoning_effort="high",
        )
    )
    assert payload["model"] == CLAIMED_MODEL
    # Both length spellings, which is the only setting that works everywhere.
    assert payload["max_tokens"] == payload["max_completion_tokens"]
    assert payload["tools"][0]["function"]["name"] == "get_weather"
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }
    assert payload["response_format"]["json_schema"]["name"] == "colour_answer"
    # Asking for alternatives implies asking for logprobs at all.
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 3
    assert payload["reasoning_effort"] == "high"


def test_openai_multimodal_content_becomes_a_data_uri(mock_provider: MockProvider) -> None:
    adapter = get_adapter(openai_config(mock_provider))
    payload = adapter.build_payload(
        ChatRequest(
            messages=(
                Message(
                    Role.USER,
                    (TextPart("what colour?"), ImagePart(PNG, media_type="image/png")),
                ),
            )
        )
    )
    parts = payload["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "what colour?"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1]) == PNG


def test_openai_plain_text_stays_a_string(mock_provider: MockProvider) -> None:
    """Several compatible servers accept only the string form for text."""
    adapter = get_adapter(openai_config(mock_provider))
    payload = adapter.build_payload(ChatRequest(messages=(Message(Role.USER, "hi"),)))
    assert payload["messages"][0]["content"] == "hi"


def test_openai_replays_tool_calls_and_signed_reasoning(mock_provider: MockProvider) -> None:
    adapter = get_adapter(openai_config(mock_provider))
    call = ToolCall(id="call_1", name="get_weather", arguments_raw='{"city": "Paris"}')
    payload = adapter.build_payload(
        ChatRequest(
            messages=(
                Message(Role.USER, "weather?"),
                Message(
                    Role.ASSISTANT,
                    "",
                    tool_calls=(call,),
                    thinking=(ThinkingBlock(text="considering", signature="sig_abc"),),
                ),
                Message(Role.TOOL, '{"c": 21}', tool_call_id="call_1"),
            )
        )
    )
    assistant = payload["messages"][1]
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'
    assert assistant["reasoning_content"] == "considering"
    assert assistant["thinking_blocks"][0]["signature"] == "sig_abc"
    assert payload["messages"][2] == {
        "role": "tool",
        "content": '{"c": 21}',
        "tool_call_id": "call_1",
    }


def test_openai_max_tokens_field_can_be_pinned_per_request(
    mock_provider: MockProvider,
) -> None:
    adapter = get_adapter(openai_config(mock_provider))
    payload = adapter.build_payload(
        ChatRequest(
            messages=(Message(Role.USER, "hi"),),
            extra_body={OpenAICompatAdapter.MAX_TOKENS_DIRECTIVE: "max_completion_tokens"},
        )
    )
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 1024
    assert OpenAICompatAdapter.MAX_TOKENS_DIRECTIVE not in payload


# --------------------------------------------------------------------------- #
# OpenAI parsing
# --------------------------------------------------------------------------- #


def test_openai_parses_a_full_completion(mock_provider: MockProvider) -> None:
    adapter = get_adapter(openai_config(mock_provider))
    response = adapter.parse_response(
        result(
            {
                "id": "chatcmpl-1",
                "model": "served-model",
                "system_fingerprint": "fp_1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "blue",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                        "logprobs": {
                            "content": [
                                {
                                    "token": "blue",
                                    "logprob": -0.1,
                                    "top_logprobs": [
                                        {"token": "blue", "logprob": -0.1},
                                        {"token": "red", "logprob": -2.4},
                                    ],
                                }
                            ]
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 3,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                    "prompt_tokens_details": {"cached_tokens": 5},
                },
            }
        )
    )
    assert response.api_family is ApiFamily.OPENAI
    assert response.text == "blue"
    assert response.model_reported == "served-model"
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.tool_calls[0].arguments == {"city": "Paris"}
    assert response.logprobs[0].top == (("blue", -0.1), ("red", -2.4))
    assert response.usage.input_tokens == 11
    assert response.usage.reasoning_tokens == 2
    assert response.usage.cache_read_tokens == 5
    assert response.usage.total_tokens == 14
    # Nothing volunteered is discarded.
    assert response.raw["system_fingerprint"] == "fp_1"


def test_openai_keeps_unparseable_tool_arguments_as_sent(
    mock_provider: MockProvider,
) -> None:
    """Whether a stack emits valid JSON is itself a fingerprint."""
    adapter = get_adapter(openai_config(mock_provider))
    response = adapter.parse_response(
        result(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c",
                                    "function": {"name": "f", "arguments": '{"city": '},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
    )
    assert response.tool_calls[0].arguments_raw == '{"city": '
    assert response.tool_calls[0].arguments is None


def test_openai_records_a_missing_logprob_as_not_a_number(
    mock_provider: MockProvider,
) -> None:
    adapter = get_adapter(openai_config(mock_provider))
    body = {"choices": [{"message": {"content": "x"}, "logprobs": {"content": [{"token": "x"}]}}]}
    response = adapter.parse_response(result(body))
    assert response.logprobs[0].logprob != response.logprobs[0].logprob  # NaN


def test_openai_reads_the_legacy_logprob_block(mock_provider: MockProvider) -> None:
    adapter = get_adapter(openai_config(mock_provider))
    response = adapter.parse_response(
        result(
            {
                "choices": [
                    {
                        "message": {"content": "ab"},
                        "logprobs": {
                            "tokens": ["a", "b"],
                            "token_logprobs": [-0.2, -0.4],
                            "top_logprobs": [{"a": -0.2, "c": -3.0}, {"b": -0.4}],
                        },
                    }
                ]
            }
        )
    )
    assert [entry.token for entry in response.logprobs] == ["a", "b"]
    assert response.logprobs[0].top == (("a", -0.2), ("c", -3.0))


def test_openai_survives_a_response_with_no_usage(mock_provider: MockProvider) -> None:
    adapter = get_adapter(openai_config(mock_provider))
    response = adapter.parse_response(result({"choices": [{"message": {"content": "x"}}]}))
    assert response.usage.input_tokens is None
    assert response.usage.total_tokens is None
    assert response.finish_reason is FinishReason.OTHER


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


def test_anthropic_lifts_the_system_prompt_out_of_the_messages(
    mock_provider: MockProvider,
) -> None:
    adapter = get_adapter(anthropic_config(mock_provider))
    payload = adapter.build_payload(
        ChatRequest(
            messages=(
                Message(Role.SYSTEM, "first"),
                Message(Role.SYSTEM, "second"),
                Message(Role.USER, "hi"),
            )
        )
    )
    assert payload["system"] == "first\n\nsecond"
    assert [turn["role"] for turn in payload["messages"]] == ["user"]


def test_anthropic_folds_consecutive_turns_of_the_same_role(
    mock_provider: MockProvider,
) -> None:
    adapter = get_adapter(anthropic_config(mock_provider))
    payload = adapter.build_payload(
        ChatRequest(
            messages=(
                Message(Role.USER, "one"),
                Message(Role.USER, "two"),
                Message(Role.ASSISTANT, "ok"),
                Message(Role.TOOL, "a", tool_call_id="t1"),
                Message(Role.TOOL, "b", tool_call_id="t2"),
            )
        )
    )
    assert [turn["role"] for turn in payload["messages"]] == ["user", "assistant", "user"]
    assert len(payload["messages"][0]["content"]) == 2
    assert len(payload["messages"][2]["content"]) == 2


def test_anthropic_never_sends_seed_or_logprobs_of_its_own_accord(
    mock_provider: MockProvider,
) -> None:
    """The protocol has neither, and inventing a spelling would break a probe."""
    adapter = get_adapter(anthropic_config(mock_provider))
    payload = adapter.build_payload(
        ChatRequest(messages=(Message(Role.USER, "hi"),), seed=7, logprobs=True, top_logprobs=5)
    )
    assert "seed" not in json.dumps(payload)
    assert "logprobs" not in json.dumps(payload)


def test_anthropic_replays_thinking_first_and_keeps_the_signature(
    mock_provider: MockProvider,
) -> None:
    """A signature is only valid in the position and form the API emitted it."""
    adapter = get_adapter(anthropic_config(mock_provider))
    payload = adapter.build_payload(
        ChatRequest(
            messages=(
                Message(Role.USER, "hi"),
                Message(
                    Role.ASSISTANT,
                    "the answer",
                    thinking=(
                        ThinkingBlock(text="step one", signature="sig_xyz"),
                        ThinkingBlock(text="opaque", redacted=True),
                    ),
                    tool_calls=(
                        ToolCall(
                            id="c1",
                            name="get_weather",
                            arguments_raw='{"city": "Paris"}',
                            arguments={"city": "Paris"},
                        ),
                    ),
                ),
            )
        )
    )
    blocks = payload["messages"][1]["content"]
    assert [block["type"] for block in blocks] == [
        "thinking",
        "redacted_thinking",
        "text",
        "tool_use",
    ]
    assert blocks[0]["signature"] == "sig_xyz"
    assert blocks[1]["data"] == "opaque"


def test_anthropic_maps_tool_choice_and_output_config(mock_provider: MockProvider) -> None:
    adapter = get_adapter(anthropic_config(mock_provider))
    forced = adapter.build_payload(
        ChatRequest(
            messages=(Message(Role.USER, "hi"),),
            tools=(TOOL,),
            tool_choice="required",
            reasoning_effort="high",
            response_schema=SCHEMA,
            thinking_budget=2048,
        )
    )
    assert forced["tool_choice"] == {"type": "any"}
    assert forced["output_config"]["effort"] == "high"
    assert forced["output_config"]["format"]["type"] == "json_schema"
    assert forced["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert forced["tools"][0]["input_schema"] == TOOL.parameters

    named = adapter.build_payload(
        ChatRequest(messages=(Message(Role.USER, "hi"),), tools=(TOOL,), tool_choice="get_weather")
    )
    assert named["tool_choice"] == {"type": "tool", "name": "get_weather"}


def test_anthropic_sends_the_version_header(mock_provider: MockProvider) -> None:
    adapter = get_adapter(anthropic_config(mock_provider))
    assert adapter.http._client.headers["anthropic-version"] == AnthropicAdapter.api_version


def test_anthropic_parses_content_blocks_and_its_usage_shape(
    mock_provider: MockProvider,
) -> None:
    adapter = get_adapter(anthropic_config(mock_provider))
    response = adapter.parse_response(
        result(
            {
                "id": "msg_1",
                "model": "claude-mock",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig_1"},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "blue"},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {"city": "Paris"}},
                ],
                "stop_reason": "tool_use",
                "service_tier": "standard",
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 4,
                    "cache_read_input_tokens": 6,
                    "cache_creation_input_tokens": 2,
                    "output_tokens_details": {"thinking_tokens": 3},
                },
            }
        )
    )
    assert response.api_family is ApiFamily.ANTHROPIC
    assert response.response_id == "msg_1"
    assert response.text == "blue"
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.thinking[0].signature == "sig_1"
    assert response.thinking[1].redacted is True
    assert response.tool_calls[0].arguments == {"city": "Paris"}
    assert response.usage.cache_read_tokens == 6
    assert response.usage.cache_write_tokens == 2
    assert response.usage.reasoning_tokens == 3
    assert response.raw["service_tier"] == "standard"


def test_anthropic_maps_every_stop_reason_it_documents(mock_provider: MockProvider) -> None:
    adapter = get_adapter(anthropic_config(mock_provider))
    expected = {
        "end_turn": FinishReason.STOP,
        "stop_sequence": FinishReason.STOP,
        "max_tokens": FinishReason.LENGTH,
        "model_context_window_exceeded": FinishReason.LENGTH,
        "tool_use": FinishReason.TOOL_CALLS,
        "refusal": FinishReason.REFUSAL,
        "pause_turn": FinishReason.PAUSE_TURN,
    }
    for raw, reason in expected.items():
        parsed = adapter.parse_response(result({"content": [], "stop_reason": raw}))
        assert parsed.finish_reason is reason


def test_anthropic_renders_images_as_base64_source_blocks(
    mock_provider: MockProvider,
) -> None:
    adapter = get_adapter(anthropic_config(mock_provider))
    payload = adapter.build_payload(
        ChatRequest(
            messages=(Message(Role.USER, (TextPart("what?"), ImagePart(PNG, "image/png"))),)
        )
    )
    blocks = payload["messages"][0]["content"]
    assert blocks[1]["source"]["media_type"] == "image/png"
    assert base64.b64decode(blocks[1]["source"]["data"]) == PNG


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #


def test_gemini_puts_the_model_in_the_url_not_the_body(mock_provider: MockProvider) -> None:
    adapter = get_adapter(gemini_config(mock_provider))
    payload = adapter.build_payload(ChatRequest(messages=(Message(Role.USER, "hi"),)))
    assert "model" not in payload
    assert adapter.chat_path == f"/models/{CLAIMED_MODEL}:generateContent"
    assert adapter.stream_path.endswith(":streamGenerateContent")


def test_gemini_payload_shape(mock_provider: MockProvider) -> None:
    adapter = get_adapter(gemini_config(mock_provider))
    payload = adapter.build_payload(
        ChatRequest(
            messages=(Message(Role.SYSTEM, "be terse"), Message(Role.USER, "hi")),
            temperature=0.2,
            seed=3,
            stop=("END",),
            tools=(TOOL,),
            tool_choice="get_weather",
            response_schema=SCHEMA,
            reasoning_effort="high",
            thinking_budget=512,
        )
    )
    assert payload["systemInstruction"]["parts"] == [{"text": "be terse"}]
    assert payload["contents"][0]["role"] == "user"
    generation = payload["generationConfig"]
    assert generation["temperature"] == 0.2
    assert generation["seed"] == 3
    assert generation["stopSequences"] == ["END"]
    assert generation["responseMimeType"] == "application/json"
    assert generation["thinkingConfig"]["thinkingLevel"] == "high"
    assert generation["thinkingConfig"]["includeThoughts"] is True
    assert payload["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"
    assert payload["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["get_weather"],
    }


def test_gemini_spells_the_assistant_role_model(mock_provider: MockProvider) -> None:
    adapter = get_adapter(gemini_config(mock_provider))
    payload = adapter.build_payload(
        ChatRequest(
            messages=(
                Message(Role.USER, "hi"),
                Message(
                    Role.ASSISTANT,
                    "sure",
                    thinking=(ThinkingBlock(text="hmm", signature="thought_sig"),),
                    tool_calls=(
                        ToolCall(id="f", name="f", arguments_raw="{}", arguments={"a": 1}),
                    ),
                ),
                Message(Role.TOOL, '{"ok": true}', tool_call_id="f"),
            )
        )
    )
    assert payload["contents"][1]["role"] == "model"
    parts = payload["contents"][1]["parts"]
    assert parts[0] == {"text": "hmm", "thought": True, "thoughtSignature": "thought_sig"}
    assert parts[2]["functionCall"] == {"name": "f", "args": {"a": 1}}
    assert payload["contents"][2]["parts"][0]["functionResponse"]["response"] == {"ok": True}


def test_gemini_reports_the_version_it_served(mock_provider: MockProvider) -> None:
    """``modelVersion`` is the claim worth checking, not the id that was asked for."""
    adapter = get_adapter(gemini_config(mock_provider))
    response = adapter.parse_response(
        result(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "thinking", "thought": True},
                                {"text": "blue"},
                                {"functionCall": {"name": "f", "args": {"city": "Paris"}}},
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 9,
                    "candidatesTokenCount": 2,
                    "thoughtsTokenCount": 4,
                    "cachedContentTokenCount": 1,
                    "totalTokenCount": 15,
                },
                "modelVersion": "gemini-mock-001",
                "responseId": "r1",
            }
        )
    )
    assert response.api_family is ApiFamily.GEMINI
    assert response.model_reported == "gemini-mock-001"
    assert response.text == "blue"
    assert response.thinking[0].text == "thinking"
    assert response.tool_calls[0].id == "f"
    # STOP alongside a function call is normalised, and the raw value survives.
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.raw["candidates"][0]["finishReason"] == "STOP"
    assert response.usage.reasoning_tokens == 4
    assert response.usage.cache_read_tokens == 1


def test_gemini_renders_images_as_inline_data(mock_provider: MockProvider) -> None:
    adapter = get_adapter(gemini_config(mock_provider))
    payload = adapter.build_payload(
        ChatRequest(messages=(Message(Role.USER, (ImagePart(PNG, "image/png"),)),))
    )
    inline = payload["contents"][0]["parts"][0]["inlineData"]
    assert inline["mimeType"] == "image/png"
    assert base64.b64decode(inline["data"]) == PNG


# --------------------------------------------------------------------------- #
# Round trips over the wire
# --------------------------------------------------------------------------- #


async def test_openai_round_trip(mock_provider: MockProvider) -> None:
    async with get_adapter(openai_config(mock_provider)) as adapter:
        response = await adapter.chat(
            ChatRequest(messages=(Message(Role.USER, "Compute 123456 + 654321."),))
        )
    assert response.http_status == 200
    assert response.model_reported == CLAIMED_MODEL
    assert response.response_id.startswith("chatcmpl-")
    assert "ANSWER: 777777" in response.text
    assert response.usage.input_tokens is not None
    # The payload that actually went over the socket carries no nulls either.
    sent = mock_provider.bodies(path_contains="chat/completions")[0]
    assert nulls(sent) == []


async def test_anthropic_round_trip(mock_provider: MockProvider) -> None:
    async with get_adapter(anthropic_config(mock_provider)) as adapter:
        response = await adapter.chat(
            ChatRequest(messages=(Message(Role.USER, "Compute 123456 + 654321."),))
        )
    assert response.api_family is ApiFamily.ANTHROPIC
    assert response.response_id.startswith("msg_")
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.raw["cache_creation"]["ephemeral_5m_input_tokens"] == 0


async def test_gemini_round_trip(mock_provider: MockProvider) -> None:
    async with get_adapter(gemini_config(mock_provider)) as adapter:
        response = await adapter.chat(
            ChatRequest(messages=(Message(Role.USER, "Compute 123456 + 654321."),))
        )
    assert response.api_family is ApiFamily.GEMINI
    assert response.model_reported == CLAIMED_MODEL
    assert "ANSWER: 777777" in response.text


async def test_openai_stream_reassembly(mock_provider: MockProvider) -> None:
    async with get_adapter(openai_config(mock_provider)) as adapter:
        response = await adapter.chat(
            ChatRequest(
                messages=(Message(Role.USER, "Compute 123456 + 654321."),), stream=True
            )
        )
    assert "ANSWER: 777777" in response.text
    assert response.model_reported == CLAIMED_MODEL
    assert response.usage.output_tokens is not None
    assert response.raw["_llmverify_stream_chunks"] > 1
    assert response.timing.ttft_s is not None


async def test_anthropic_stream_reassembles_a_signature_from_its_deltas(
    make_provider: Any,
) -> None:
    """Signatures arrive as their own frames and must be concatenated back on."""
    server = make_provider(Persona.HONEST, emit_thinking=True)
    async with get_adapter(anthropic_config(server)) as adapter:
        response = await adapter.chat(
            ChatRequest(messages=(Message(Role.USER, "hi"),), stream=True)
        )
    assert response.thinking
    assert response.thinking[0].signature == sign_thinking(response.thinking[0].text)
    assert response.raw["content"][0]["signature"] == response.thinking[0].signature
    assert response.finish_reason is FinishReason.STOP


async def test_anthropic_rejects_a_tampered_signature_on_replay(
    make_provider: Any,
) -> None:
    """The property the whole cryptographic probe rests on."""
    server = make_provider(Persona.HONEST, emit_thinking=True)
    async with get_adapter(anthropic_config(server)) as adapter:
        first = await adapter.chat(ChatRequest(messages=(Message(Role.USER, "hi"),)))
        block = first.thinking[0]

        replay = ChatRequest(
            messages=(
                Message(Role.USER, "hi"),
                Message(Role.ASSISTANT, first.text, thinking=(block,)),
                Message(Role.USER, "again"),
            )
        )
        good, error = await adapter.try_chat(replay)
        assert error is None and good is not None

        tampered = ThinkingBlock(text=block.text, signature="sig_notarealvalue")
        bad, error = await adapter.try_chat(
            replay.replace(
                messages=(
                    Message(Role.USER, "hi"),
                    Message(Role.ASSISTANT, first.text, thinking=(tampered,)),
                    Message(Role.USER, "again"),
                )
            )
        )
    assert bad is None
    assert isinstance(error, ProviderError)
    assert error.status == 400


async def test_count_tokens_over_the_wire(mock_provider: MockProvider) -> None:
    async with get_adapter(anthropic_config(mock_provider)) as adapter:
        bare = await adapter.count_tokens((Message(Role.USER, "Say ok."),))
        with_tool = await adapter.count_tokens(
            (Message(Role.USER, "Say ok."),), (TOOL,), tool_choice="auto"
        )
        forced = await adapter.count_tokens(
            (Message(Role.USER, "Say ok."),), (TOOL,), tool_choice="required"
        )
    assert with_tool - bare == mock_provider.config.tool_overhead_auto + (
        mock_provider.config.tool_definition_tokens
    )
    assert forced - with_tool == (
        mock_provider.config.tool_overhead_forced - mock_provider.config.tool_overhead_auto
    )


async def test_list_models_returns_the_catalogue(mock_provider: MockProvider) -> None:
    async with get_adapter(openai_config(mock_provider)) as adapter:
        models = await adapter.list_models()
    assert [entry["id"] for entry in models] == [CLAIMED_MODEL]

    async with get_adapter(gemini_config(mock_provider)) as adapter:
        gemini_models = await adapter.list_models()
    # The catalogue's own ``name`` survives alongside the bare id every caller uses.
    assert gemini_models[0]["name"] == f"models/{CLAIMED_MODEL}"
    assert gemini_models[0]["id"] == CLAIMED_MODEL


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


async def test_probe_parameter_tells_accepted_from_ignored_from_rejected(
    make_provider: Any,
) -> None:
    honest = make_provider(Persona.HONEST)
    dropping = make_provider(Persona.LITELLM)
    strict = make_provider(Persona.HONEST, accept_logprobs=False)
    request = ChatRequest(messages=(Message(Role.USER, "hi"),), max_tokens=8)
    patch = {"logprobs": True, "top_logprobs": 5}

    def saw_logprobs(response: Any) -> bool:
        return bool(response.logprobs)

    async with get_adapter(openai_config(honest)) as adapter:
        accepted = await adapter.probe_parameter(
            request, parameter="logprobs", payload_patch=patch, detect_effect=saw_logprobs
        )
    async with get_adapter(openai_config(dropping)) as adapter:
        ignored = await adapter.probe_parameter(
            request, parameter="logprobs", payload_patch=patch, detect_effect=saw_logprobs
        )
    async with get_adapter(openai_config(strict)) as adapter:
        rejected = await adapter.probe_parameter(
            request, parameter="logprobs", payload_patch=patch, detect_effect=saw_logprobs
        )

    assert accepted.support is ParamSupport.ACCEPTED
    assert ignored.support is ParamSupport.IGNORED
    assert rejected.support is ParamSupport.REJECTED
    assert rejected.status == 400


async def test_a_provider_error_carries_its_status_and_body(make_provider: Any) -> None:
    server = make_provider(Persona.HONEST, fail_status=503, fail_body="upstream exploded")
    async with get_adapter(openai_config(server)) as adapter:
        response, error = await adapter.try_chat(
            ChatRequest(messages=(Message(Role.USER, "hi"),))
        )
    assert response is None
    assert isinstance(error, ProviderError)
    assert error.status == 503
    assert "upstream exploded" in (error.body or "")


def test_raise_for_status_picks_the_right_exception_type() -> None:
    cases = (
        (401, AuthError),
        (403, AuthError),
        (429, RateLimited),
        (500, ProviderError),
    )
    for status, kind in cases:
        failure = HttpResult(
            status=status, headers={"retry-after": "3"}, json=None, text="no", total_s=0.0
        )
        with pytest.raises(kind):
            raise_for_status(failure, context="test")

    # A 2xx is not an error and must pass straight through.
    ok = HttpResult(status=204, headers={}, json=None, text="", total_s=0.0)
    raise_for_status(ok, context="test")


async def test_the_broken_persona_does_not_break_the_parser(make_provider: Any) -> None:
    """No usage, unparseable arguments and inverted logprob order, all survived."""
    server = make_provider(Persona.BROKEN)
    async with get_adapter(openai_config(server)) as adapter:
        response = await adapter.chat(
            ChatRequest(
                messages=(Message(Role.USER, "weather?"),),
                tools=(TOOL,),
                tool_choice="required",
                logprobs=True,
                top_logprobs=3,
            )
        )
    assert response.usage.input_tokens is None
    assert response.usage.raw == {}
    assert response.tool_calls[0].arguments_raw == "{"
    assert response.tool_calls[0].arguments is None
    if response.logprobs:
        alternatives = [logprob for _token, logprob in response.logprobs[0].top]
        # Recorded exactly as sent, ascending order and all: normalising it away
        # would erase the thing that makes it recognisable.
        assert alternatives == sorted(alternatives)


async def test_the_serving_stack_label_is_only_offered_when_something_says_so(
    make_provider: Any,
) -> None:
    litellm = make_provider(Persona.LITELLM)
    honest = make_provider(Persona.HONEST)

    async with get_adapter(openai_config(litellm)) as adapter:
        models = await adapter.list_models()
        sample = await adapter.chat(ChatRequest(messages=(Message(Role.USER, "hi"),)))
        assert adapter.detect_serving_stack(models, sample) == "litellm"

    async with get_adapter(openai_config(honest)) as adapter:
        models = await adapter.list_models()
        sample = await adapter.chat(ChatRequest(messages=(Message(Role.USER, "hi"),)))
        assert adapter.detect_serving_stack(models, sample) is None


async def test_openrouter_endpoint_lookup_is_skipped_off_openrouter(
    mock_provider: MockProvider,
) -> None:
    """It must never fire against an endpoint that is not OpenRouter."""
    async with get_adapter(openai_config(mock_provider)) as adapter:
        assert await adapter.fetch_openrouter_endpoints("author/slug") is None
    assert mock_provider.requests == []
