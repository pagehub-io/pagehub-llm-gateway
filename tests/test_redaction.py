"""Content redaction — confirm we strip the right fields and keep the right ones."""

from __future__ import annotations

from api.shared.redaction import redact


def test_redaction_disabled_is_pure_passthrough():
    payload = {"text": "secret", "model": "x"}
    assert redact(payload, enabled=False) is payload


def test_redacts_anthropic_text_block_text():
    body = {
        "model": "grok-4",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "secret prompt"}]}
        ],
    }
    out = redact(body, enabled=True)
    # Schema preserved, content gone.
    assert out["model"] == "grok-4"
    assert out["messages"][0]["role"] == "user"
    text_block = out["messages"][0]["content"][0]
    assert text_block["type"] == "text"
    assert text_block["text"] == {"redacted": True, "chars": len("secret prompt")}


def test_redacts_plain_string_content_in_user_message():
    body = {"messages": [{"role": "user", "content": "the actual user text"}]}
    out = redact(body, enabled=True)
    redacted = out["messages"][0]["content"]
    assert redacted == {"redacted": True, "chars": len("the actual user text")}


def test_redacts_tool_call_input_keeps_name_and_id():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"location": "secret place", "units": "f"},
                    }
                ],
            }
        ]
    }
    out = redact(body, enabled=True)
    block = out["messages"][0]["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "toolu_1"
    assert block["name"] == "get_weather"
    assert block["input"] == {"redacted": True, "keys": ["location", "units"]}


def test_redacts_canonical_tool_call_input_keeps_name_and_id():
    body = {
        "content_blocks": [
            {
                "type": "tool_call",
                "id": "c1",
                "name": "fn",
                "input": {"x": 1, "secret": "hush"},
            }
        ]
    }
    out = redact(body, enabled=True)
    block = out["content_blocks"][0]
    assert block["id"] == "c1"
    assert block["name"] == "fn"
    assert block["input"]["redacted"] is True
    assert sorted(block["input"]["keys"]) == ["secret", "x"]


def test_redacts_tool_result_keeps_block_shape():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "the lookup said 72F",
                    }
                ],
            }
        ]
    }
    out = redact(body, enabled=True)
    block = out["messages"][0]["content"][0]
    assert block["tool_use_id"] == "t1"
    assert block["content"]["redacted"] is True


def test_redacts_openai_role_tool_message():
    body = {"messages": [{"role": "tool", "tool_call_id": "c1", "content": "raw tool reply"}]}
    out = redact(body, enabled=True)
    msg = out["messages"][0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "c1"
    assert msg["content"]["redacted"] is True


def test_redacts_openai_tool_call_arguments_string():
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "fn", "arguments": '{"loc":"SF"}'},
                        }
                    ],
                }
            }
        ]
    }
    out = redact(body, enabled=True)
    call = out["choices"][0]["message"]["tool_calls"][0]
    assert call["id"] == "c1"
    assert call["function"]["name"] == "fn"
    assert call["function"]["arguments"]["redacted"] is True


def test_keeps_usage_and_stop_reason_intact():
    body = {
        "usage": {"input_tokens": 100, "output_tokens": 50},
        "stop_reason": "end_turn",
        "model": "grok-4",
    }
    out = redact(body, enabled=True)
    assert out == body  # nothing here is caller content
