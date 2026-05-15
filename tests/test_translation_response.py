"""OpenAI-response -> Anthropic-response translator."""

from __future__ import annotations

from api.translation.openai_to_anthropic import map_finish_reason, translate_response


def _oai(
    *,
    content="hello",
    tool_calls=None,
    finish_reason="stop",
    usage=None,
):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl_x",
        "model": "grok-4",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


def test_plain_text_response_round_trip():
    anth = translate_response(_oai(content="hi from grok"), anthropic_model="grok-4")
    assert anth["type"] == "message"
    assert anth["role"] == "assistant"
    assert anth["model"] == "grok-4"
    assert anth["content"] == [{"type": "text", "text": "hi from grok"}]
    assert anth["stop_reason"] == "end_turn"
    assert anth["stop_sequence"] is None
    assert anth["usage"] == {"input_tokens": 7, "output_tokens": 3}


def test_tool_call_response_becomes_tool_use_block():
    anth = translate_response(
        _oai(
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"SF","units":"f"}',
                    },
                }
            ],
            finish_reason="tool_calls",
        ),
        anthropic_model="grok-4",
    )
    # No text content => no text block.
    assert len(anth["content"]) == 1
    tu = anth["content"][0]
    assert tu["type"] == "tool_use"
    assert tu["id"] == "call_1"
    assert tu["name"] == "get_weather"
    assert tu["input"] == {"location": "SF", "units": "f"}
    assert anth["stop_reason"] == "tool_use"


def test_mixed_text_and_tool_call_response():
    anth = translate_response(
        _oai(
            content="thinking out loud...",
            tool_calls=[
                {"id": "c1", "function": {"name": "do", "arguments": "{}"}}
            ],
            finish_reason="tool_calls",
        ),
        anthropic_model="grok-4",
    )
    assert [b["type"] for b in anth["content"]] == ["text", "tool_use"]


def test_malformed_tool_arguments_dont_500():
    anth = translate_response(
        _oai(
            content=None,
            tool_calls=[
                {"id": "c", "function": {"name": "n", "arguments": "{ broken"}}
            ],
            finish_reason="tool_calls",
        ),
        anthropic_model="grok-4",
    )
    tu = anth["content"][0]
    assert tu["type"] == "tool_use"
    assert "_unparsed_arguments" in tu["input"]


def test_empty_response_gets_empty_text_block():
    oai = _oai(content=None)
    oai["choices"][0]["message"].pop("content", None)
    anth = translate_response(oai, anthropic_model="grok-4")
    assert anth["content"] == [{"type": "text", "text": ""}]


def test_finish_reason_map():
    assert map_finish_reason("stop") == "end_turn"
    assert map_finish_reason("length") == "max_tokens"
    assert map_finish_reason("tool_calls") == "tool_use"
    assert map_finish_reason("content_filter") == "stop_sequence"
    assert map_finish_reason(None) is None
    assert map_finish_reason("something_else") == "end_turn"


def test_response_id_gets_msg_prefix():
    oai = _oai()
    oai["id"] = "12345"
    anth = translate_response(oai, anthropic_model="grok-4")
    assert anth["id"].startswith("msg_")
