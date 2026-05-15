"""Anthropic-request -> OpenAI-request translator."""

from __future__ import annotations

import json

from api.translation.anthropic_to_openai import translate_request
from api.v1.schemas import MessagesRequest


def _req(**kwargs):
    base = {
        "model": "grok-4",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
    }
    base.update(kwargs)
    return MessagesRequest.model_validate(base)


def test_plain_text_user_message_round_trip():
    body = translate_request(_req(), "grok-4")
    assert body["model"] == "grok-4"
    assert body["max_tokens"] == 256
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["stream"] is False


def test_system_string_prepended_as_system_message():
    body = translate_request(_req(system="be terse"), "grok-4")
    assert body["messages"][0] == {"role": "system", "content": "be terse"}
    assert body["messages"][1] == {"role": "user", "content": "hello"}


def test_system_blocks_concatenated_with_newline():
    body = translate_request(
        _req(system=[{"type": "text", "text": "rule one"}, {"type": "text", "text": "rule two"}]),
        "grok-4",
    )
    assert body["messages"][0]["content"] == "rule one\nrule two"


def test_user_text_blocks_collapse_to_string_when_alone():
    body = translate_request(
        _req(messages=[{"role": "user", "content": [{"type": "text", "text": "hi there"}]}]),
        "grok-4",
    )
    assert body["messages"][-1] == {"role": "user", "content": "hi there"}


def test_user_image_block_becomes_image_url_part():
    img = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KG..."},
    }
    body = translate_request(
        _req(messages=[{"role": "user", "content": [{"type": "text", "text": "look"}, img]}]),
        "grok-4",
    )
    user_msg = body["messages"][-1]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0] == {"type": "text", "text": "look"}
    assert user_msg["content"][1]["type"] == "image_url"
    assert user_msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_assistant_tool_use_block_becomes_tool_call():
    body = translate_request(
        _req(
            messages=[
                {"role": "user", "content": "find"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"location": "SF"},
                        },
                    ],
                },
            ]
        ),
        "grok-4",
    )
    assistant = body["messages"][-1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Let me check."
    assert len(assistant["tool_calls"]) == 1
    call = assistant["tool_calls"][0]
    assert call["id"] == "toolu_1"
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"location": "SF"}


def test_tool_result_block_becomes_role_tool_message():
    body = translate_request(
        _req(
            messages=[
                {"role": "user", "content": "find"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "result text",
                        },
                        {"type": "text", "text": "what next?"},
                    ],
                },
            ]
        ),
        "grok-4",
    )
    # The user message above splits into: tool message + user message, in that order.
    last_three = body["messages"][-3:]
    assert [m["role"] for m in last_three] == ["assistant", "tool", "user"]
    assert last_three[1] == {"role": "tool", "tool_call_id": "toolu_1", "content": "result text"}
    assert last_three[2] == {"role": "user", "content": "what next?"}


def test_tool_result_with_list_content_flattens_text_blocks():
    body = translate_request(
        _req(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "line 1"},
                                {"type": "text", "text": "line 2"},
                            ],
                        }
                    ],
                }
            ]
        ),
        "grok-4",
    )
    tool_msg = body["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == "line 1\nline 2"


def test_tools_get_translated_to_openai_function_shape():
    body = translate_request(
        _req(
            tools=[
                {
                    "name": "get_weather",
                    "description": "lookup",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ]
        ),
        "grok-4",
    )
    assert "tools" in body
    assert body["tools"][0]["type"] == "function"
    fn = body["tools"][0]["function"]
    assert fn["name"] == "get_weather"
    assert fn["description"] == "lookup"
    assert fn["parameters"]["properties"]["location"]["type"] == "string"


def test_tool_choice_any_maps_to_required():
    body = translate_request(_req(tool_choice={"type": "any"}), "grok-4")
    assert body["tool_choice"] == "required"


def test_tool_choice_specific_tool_maps_to_function():
    body = translate_request(_req(tool_choice={"type": "tool", "name": "x"}), "grok-4")
    assert body["tool_choice"] == {"type": "function", "function": {"name": "x"}}


def test_tool_choice_auto_passes_through():
    body = translate_request(_req(tool_choice={"type": "auto"}), "grok-4")
    assert body["tool_choice"] == "auto"


def test_stop_sequences_become_openai_stop():
    body = translate_request(_req(stop_sequences=["</end>", "STOP"]), "grok-4")
    assert body["stop"] == ["</end>", "STOP"]


def test_streaming_request_includes_include_usage():
    body = translate_request(_req(stream=True), "grok-4")
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_cache_control_field_accepted_but_dropped():
    # cache_control on a text block must not raise; we just drop it.
    body = translate_request(
        _req(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral"}}
                    ],
                }
            ]
        ),
        "grok-4",
    )
    assert body["messages"][-1] == {"role": "user", "content": "ctx"}


def test_thinking_block_in_assistant_history_is_dropped():
    body = translate_request(
        _req(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "thinking aloud", "signature": "sig"},
                        {"type": "text", "text": "result"},
                    ],
                },
            ]
        ),
        "grok-4",
    )
    assert body["messages"][-1] == {"role": "assistant", "content": "result"}
