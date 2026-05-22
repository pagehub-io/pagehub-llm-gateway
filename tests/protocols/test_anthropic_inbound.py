"""Layer (b): AnthropicInbound.decode_request + encode_response against fixture bodies.

These tests exercise the Anthropic <-> Canonical translation in isolation — no
provider, no engine, no HTTP. If a future change to canonical types breaks the
Anthropic contract, these tests fail loudly.
"""

from __future__ import annotations

import json

from api.canonical.events import (
    ContentBlockDone,
    ContentBlockStart,
    ContentTextDelta,
    ContentToolCallDelta,
    MessageDelta,
    StreamDone,
    StreamStart,
)
from api.canonical.types import (
    CanonicalImage,
    CanonicalResponse,
    CanonicalText,
    CanonicalToolCall,
    CanonicalToolResult,
    CanonicalUsage,
)
from api.protocols.anthropic import AnthropicInbound
from api.v1.schemas import MessagesRequest

# ---------------------------------------------------------------------------
# decode_request
# ---------------------------------------------------------------------------


def _decode(body: dict):
    return AnthropicInbound.decode_request(MessagesRequest.model_validate(body))


def test_decode_plain_user_text():
    c = _decode(
        {
            "model": "claude-opus-4-7",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert c.model == "claude-opus-4-7"
    assert c.max_tokens == 32
    assert c.messages[0].role == "user"
    [block] = c.messages[0].content_blocks
    assert isinstance(block, CanonicalText)
    assert block.text == "hi"


def test_decode_system_string_to_canonical_text_list():
    c = _decode(
        {
            "model": "x",
            "system": "be terse",
            "messages": [{"role": "user", "content": "go"}],
        }
    )
    assert c.system is not None
    assert isinstance(c.system[0], CanonicalText)
    assert c.system[0].text == "be terse"


def test_decode_system_blocks_keep_individual_texts():
    c = _decode(
        {
            "model": "x",
            "system": [{"type": "text", "text": "rule one"}, {"type": "text", "text": "rule two"}],
            "messages": [{"role": "user", "content": "go"}],
        }
    )
    assert [b.text for b in c.system] == ["rule one", "rule two"]


def test_decode_assistant_tool_use():
    c = _decode(
        {
            "model": "x",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "checking..."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"loc": "SF"},
                        },
                    ],
                },
            ],
        }
    )
    assistant_blocks = c.messages[1].content_blocks
    assert isinstance(assistant_blocks[0], CanonicalText)
    assert isinstance(assistant_blocks[1], CanonicalToolCall)
    assert assistant_blocks[1].id == "toolu_1"
    assert assistant_blocks[1].name == "get_weather"
    assert assistant_blocks[1].input == {"loc": "SF"}


def test_decode_tool_result_with_list_content_flattens_text():
    c = _decode(
        {
            "model": "x",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [
                                {"type": "text", "text": "line 1"},
                                {"type": "text", "text": "line 2"},
                            ],
                        }
                    ],
                }
            ],
        }
    )
    block = c.messages[0].content_blocks[0]
    assert isinstance(block, CanonicalToolResult)
    assert block.tool_call_id == "toolu_1"
    assert block.content == "line 1\nline 2"


def test_decode_image_block_base64():
    c = _decode(
        {
            "model": "x",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVB...",
                            },
                        }
                    ],
                }
            ],
        }
    )
    img = c.messages[0].content_blocks[0]
    assert isinstance(img, CanonicalImage)
    assert img.media_type == "image/png"
    assert img.data == "iVB..."


def test_decode_tool_choice_any_to_required():
    c = _decode(
        {"model": "x", "messages": [{"role": "user", "content": "go"}], "tool_choice": {"type": "any"}}
    )
    assert c.tool_choice.mode == "required"


def test_decode_tool_choice_specific_tool():
    c = _decode(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "go"}],
            "tool_choice": {"type": "tool", "name": "myfn"},
        }
    )
    assert c.tool_choice.mode == "tool"
    assert c.tool_choice.name == "myfn"


def test_decode_tools_list():
    c = _decode(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "looks up",
                    "input_schema": {"type": "object", "properties": {"loc": {"type": "string"}}},
                }
            ],
        }
    )
    [t] = c.tools
    assert t.name == "get_weather"
    assert t.description == "looks up"
    assert t.json_schema["properties"]["loc"]["type"] == "string"


def test_decode_thinking_lands_in_metadata():
    c = _decode(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "go"}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }
    )
    assert c.metadata["anthropic_thinking"]["type"] == "enabled"


# ---------------------------------------------------------------------------
# encode_response
# ---------------------------------------------------------------------------


def test_encode_response_with_text_block():
    cr = CanonicalResponse(
        id="abc123",
        model="claude-opus-4-7",
        content_blocks=[CanonicalText(text="hi")],
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=7, output_tokens=3),
    )
    body = AnthropicInbound.encode_response(cr)
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-opus-4-7"
    assert body["id"].startswith("msg_")
    assert body["content"] == [{"type": "text", "text": "hi"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 3}


def test_encode_response_with_tool_call_block():
    cr = CanonicalResponse(
        id="msg_x",
        model="claude-opus-4-7",
        content_blocks=[
            CanonicalText(text="ok let me look"),
            CanonicalToolCall(id="c1", name="search", input={"q": "x"}),
        ],
        stop_reason="tool_use",
        usage=CanonicalUsage(input_tokens=1, output_tokens=1),
    )
    body = AnthropicInbound.encode_response(cr)
    assert [b["type"] for b in body["content"]] == ["text", "tool_use"]
    tu = body["content"][1]
    assert tu["id"] == "c1"
    assert tu["name"] == "search"
    assert tu["input"] == {"q": "x"}
    assert body["stop_reason"] == "tool_use"


def test_encode_response_empty_emits_safe_empty_text():
    cr = CanonicalResponse(id="msg_x", model="x", content_blocks=[], stop_reason="end_turn")
    body = AnthropicInbound.encode_response(cr)
    assert body["content"] == [{"type": "text", "text": ""}]


def test_encode_response_emits_cache_read_input_tokens_when_canonical_has_cache_tokens():
    """OpenAI's cached_tokens (split out of prompt_tokens by the canonical
    decoder) maps to Anthropic's cache_read_input_tokens — the two have the
    same semantic: tokens served at the cache-hit discount. We don't emit
    cache_creation_input_tokens because OpenAI has no separate cache-write
    rate (and we have no signal for it)."""
    cr = CanonicalResponse(
        id="msg_x",
        model="gpt-5.5",
        content_blocks=[CanonicalText(text="ok")],
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=100, output_tokens=10, cache_tokens=400),
    )
    body = AnthropicInbound.encode_response(cr)
    assert body["usage"]["input_tokens"] == 100
    assert body["usage"]["output_tokens"] == 10
    assert body["usage"]["cache_read_input_tokens"] == 400
    # cache_creation_input_tokens is NOT emitted — see the docstring above.
    assert "cache_creation_input_tokens" not in body["usage"]


def test_encode_response_omits_cache_read_when_no_cache_tokens():
    """xAI responses (no caching) and OpenAI cache-miss responses leave
    canonical.usage.cache_tokens as None — the encoder must NOT emit a
    zero-valued cache_read_input_tokens (clean wire output for the common case)."""
    cr = CanonicalResponse(
        id="msg_x",
        model="gpt-5",
        content_blocks=[CanonicalText(text="hi")],
        stop_reason="end_turn",
        usage=CanonicalUsage(input_tokens=5, output_tokens=2),
    )
    body = AnthropicInbound.encode_response(cr)
    assert body["usage"] == {"input_tokens": 5, "output_tokens": 2}


# ---------------------------------------------------------------------------
# encode_stream
# ---------------------------------------------------------------------------


async def _run_encode(events: list) -> list[tuple[str, dict]]:
    async def aiter():
        for e in events:
            yield e

    out: list[tuple[str, dict]] = []
    async for chunk in AnthropicInbound.encode_stream(aiter()):
        # Each chunk is one SSE event block ending in \n\n.
        for block in chunk.split(b"\n\n"):
            if not block.strip():
                continue
            evt, data = None, None
            for line in block.split(b"\n"):
                if line.startswith(b"event:"):
                    evt = line[6:].strip().decode()
                elif line.startswith(b"data:"):
                    data = json.loads(line[5:].strip())
            if evt is not None and data is not None:
                out.append((evt, data))
    return out


async def test_encode_stream_text_sequence():
    events = await _run_encode(
        [
            StreamStart(message_id="msg_1", model="grok-4", usage_estimate=CanonicalUsage(input_tokens=10)),
            ContentBlockStart(index=0, block=CanonicalText()),
            ContentTextDelta(index=0, text="Hel"),
            ContentTextDelta(index=0, text="lo"),
            ContentBlockDone(index=0),
            MessageDelta(stop_reason="end_turn", usage=CanonicalUsage(output_tokens=2)),
            StreamDone(),
        ]
    )
    types = [t for t, _ in events]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start_data = events[0][1]
    assert start_data["message"]["id"].startswith("msg_")
    assert start_data["message"]["usage"]["input_tokens"] == 10


async def test_encode_stream_tool_call_emits_input_json_delta():
    events = await _run_encode(
        [
            StreamStart(message_id="msg_2", model="grok-4"),
            ContentBlockStart(
                index=0, block=CanonicalToolCall(id="c1", name="fn", input={})
            ),
            ContentToolCallDelta(index=0, id="c1", name="fn", partial_input_json='{"a":'),
            ContentToolCallDelta(index=0, partial_input_json='1}'),
            ContentBlockDone(index=0),
            MessageDelta(stop_reason="tool_use", usage=CanonicalUsage(output_tokens=4)),
            StreamDone(),
        ]
    )
    json_deltas = [
        d for t, d in events
        if t == "content_block_delta" and d["delta"]["type"] == "input_json_delta"
    ]
    assembled = "".join(d["delta"]["partial_json"] for d in json_deltas)
    assert assembled == '{"a":1}'


async def test_encode_stream_empty_stream_still_terminates():
    events = await _run_encode([])
    types = [t for t, _ in events]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"


async def test_encode_stream_message_delta_carries_cache_read_input_tokens():
    """When canonical's MessageDelta has cache_tokens set, the streaming
    message_delta SSE event must include cache_read_input_tokens in its
    usage block — symmetric with the non-streaming encode_response path."""
    events = await _run_encode(
        [
            StreamStart(message_id="msg_c", model="gpt-5.5"),
            ContentBlockStart(index=0, block=CanonicalText()),
            ContentTextDelta(index=0, text="hi"),
            ContentBlockDone(index=0),
            MessageDelta(
                stop_reason="end_turn",
                usage=CanonicalUsage(input_tokens=50, output_tokens=2, cache_tokens=300),
            ),
            StreamDone(),
        ]
    )
    deltas = [d for t, d in events if t == "message_delta"]
    assert len(deltas) == 1
    usage = deltas[0]["usage"]
    assert usage["output_tokens"] == 2
    assert usage["cache_read_input_tokens"] == 300


async def test_encode_stream_message_delta_omits_cache_read_when_none():
    """The negative — no cache_tokens, no cache_read_input_tokens emitted
    in the streaming message_delta, matching the non-streaming behavior."""
    events = await _run_encode(
        [
            StreamStart(message_id="msg_d", model="grok-4"),
            ContentBlockStart(index=0, block=CanonicalText()),
            ContentTextDelta(index=0, text="hi"),
            ContentBlockDone(index=0),
            MessageDelta(stop_reason="end_turn", usage=CanonicalUsage(output_tokens=2)),
            StreamDone(),
        ]
    )
    [delta] = [d for t, d in events if t == "message_delta"]
    assert delta["usage"] == {"output_tokens": 2}
