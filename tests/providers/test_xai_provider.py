"""Layer (c): XAIProvider.encode_request + decode_response + decode_stream.

Exercises the Canonical <-> xAI wire-format translation in isolation. The HTTP
transport tests use ``httpx.MockTransport`` so nothing leaves the process.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx
import pytest

from api import twin
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
    CanonicalMessage,
    CanonicalRequest,
    CanonicalText,
    CanonicalToolCall,
    CanonicalToolChoice,
    CanonicalToolResult,
)
from api.providers.base import ProviderError
from api.providers.xai import (
    XAIProvider,
    decode_response,
    decode_stream,
    encode_request,
)

# ---------------------------------------------------------------------------
# encode_request (Canonical -> xAI body)
# ---------------------------------------------------------------------------


def _req(messages=None, **kwargs):
    return CanonicalRequest(
        model=kwargs.pop("model", "grok-4"),
        messages=messages or [CanonicalMessage(role="user", content_blocks=[CanonicalText(text="hi")])],
        max_tokens=kwargs.pop("max_tokens", 64),
        **kwargs,
    )


def test_encode_user_text():
    body = encode_request(_req(), target_model="grok-4")
    assert body["model"] == "grok-4"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_encode_system_blocks_concatenate():
    body = encode_request(
        _req(system=[CanonicalText(text="rule one"), CanonicalText(text="rule two")]),
        target_model="grok-4",
    )
    assert body["messages"][0] == {"role": "system", "content": "rule one\nrule two"}


def test_encode_assistant_tool_call():
    body = encode_request(
        _req(
            messages=[
                CanonicalMessage(role="user", content_blocks=[CanonicalText(text="x?")]),
                CanonicalMessage(
                    role="assistant",
                    content_blocks=[
                        CanonicalText(text="thinking"),
                        CanonicalToolCall(id="c1", name="fn", input={"a": 1}),
                    ],
                ),
            ]
        ),
        target_model="grok-4",
    )
    assistant = body["messages"][-1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "thinking"
    call = assistant["tool_calls"][0]
    assert call["id"] == "c1"
    assert call["function"]["name"] == "fn"
    assert json.loads(call["function"]["arguments"]) == {"a": 1}


def test_encode_tool_result_splits_into_role_tool():
    body = encode_request(
        _req(
            messages=[
                CanonicalMessage(role="user", content_blocks=[CanonicalText(text="weather?")]),
                CanonicalMessage(
                    role="assistant",
                    content_blocks=[CanonicalToolCall(id="c1", name="get_weather", input={})],
                ),
                CanonicalMessage(
                    role="user",
                    content_blocks=[
                        CanonicalToolResult(tool_call_id="c1", content="72F"),
                        CanonicalText(text="thanks"),
                    ],
                ),
            ]
        ),
        target_model="grok-4",
    )
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "tool", "user"]
    assert body["messages"][-2] == {"role": "tool", "tool_call_id": "c1", "content": "72F"}
    assert body["messages"][-1] == {"role": "user", "content": "thanks"}


def test_encode_tool_choice_modes():
    assert encode_request(
        _req(tool_choice=CanonicalToolChoice(mode="auto")), target_model="x"
    )["tool_choice"] == "auto"
    assert encode_request(
        _req(tool_choice=CanonicalToolChoice(mode="required")), target_model="x"
    )["tool_choice"] == "required"
    assert encode_request(
        _req(tool_choice=CanonicalToolChoice(mode="tool", name="myfn")), target_model="x"
    )["tool_choice"] == {"type": "function", "function": {"name": "myfn"}}


def test_encode_streaming_adds_include_usage():
    body = encode_request(_req(stream=True), target_model="x")
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# decode_response (xAI body -> Canonical)
# ---------------------------------------------------------------------------


def _oai(content="hi", tool_calls=None, finish_reason="stop", usage=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl_x",
        "model": "grok-4",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


def test_decode_text_response():
    cr = decode_response(_oai(content="hi from grok"), canonical_model="grok-4")
    assert cr.model == "grok-4"
    assert len(cr.content_blocks) == 1
    block = cr.content_blocks[0]
    assert isinstance(block, CanonicalText)
    assert block.text == "hi from grok"
    assert cr.stop_reason == "end_turn"
    assert cr.usage.input_tokens == 7
    assert cr.usage.output_tokens == 3


def test_decode_tool_call_response():
    cr = decode_response(
        _oai(
            content=None,
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "fn", "arguments": '{"a":1}'}}
            ],
            finish_reason="tool_calls",
        ),
        canonical_model="grok-4",
    )
    [block] = cr.content_blocks
    assert isinstance(block, CanonicalToolCall)
    assert block.id == "c1"
    assert block.name == "fn"
    assert block.input == {"a": 1}
    assert cr.stop_reason == "tool_use"


def test_decode_malformed_tool_arguments_survives():
    cr = decode_response(
        _oai(
            content=None,
            tool_calls=[{"id": "c", "function": {"name": "n", "arguments": "{ broken"}}],
            finish_reason="tool_calls",
        ),
        canonical_model="grok-4",
    )
    [block] = cr.content_blocks
    assert isinstance(block, CanonicalToolCall)
    assert "_unparsed_arguments" in block.input


# ---------------------------------------------------------------------------
# decode_stream (xAI SSE -> Canonical events)
# ---------------------------------------------------------------------------


def _chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _aiter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


async def _collect(chunks: list[bytes]) -> list:
    out = []
    async for ev in decode_stream(_aiter(chunks), canonical_model="grok-4", input_tokens_estimate=2):
        out.append(ev)
    return out


async def test_stream_decodes_text_sequence():
    events = await _collect(
        [
            _chunk({"id": "x", "choices": [{"index": 0, "delta": {"content": "Hel"}}]}),
            _chunk({"id": "x", "choices": [{"index": 0, "delta": {"content": "lo"}}]}),
            _chunk(
                {
                    "id": "x",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ),
            _chunk({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}),
            b"data: [DONE]\n\n",
        ]
    )
    types = [type(e).__name__ for e in events]
    assert types[0] == "StreamStart"
    assert "ContentBlockStart" in types
    assert types.count("ContentTextDelta") == 2
    assert types[-3] == "ContentBlockDone"
    assert types[-2] == "MessageDelta"
    assert types[-1] == "StreamDone"
    delta = next(e for e in events if isinstance(e, MessageDelta))
    assert delta.stop_reason == "end_turn"
    assert delta.usage.output_tokens == 2


async def test_stream_tool_call_assembles_partials():
    events = await _collect(
        [
            _chunk(
                {
                    "id": "x",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "fn", "arguments": ""},
                                    }
                                ]
                            },
                        }
                    ],
                }
            ),
            _chunk(
                {
                    "id": "x",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '{"a":'}}
                                ]
                            },
                        }
                    ],
                }
            ),
            _chunk(
                {
                    "id": "x",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '1}'}}
                                ]
                            },
                        }
                    ],
                }
            ),
            _chunk(
                {
                    "id": "x",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )
    starts = [e for e in events if isinstance(e, ContentBlockStart)]
    assert len(starts) == 1
    assert isinstance(starts[0].block, CanonicalToolCall)
    assert starts[0].block.name == "fn"
    deltas = [e for e in events if isinstance(e, ContentToolCallDelta)]
    assembled = "".join(d.partial_input_json for d in deltas)
    assert assembled == '{"a":1}'


# ---------------------------------------------------------------------------
# HTTP transport (httpx.MockTransport)
# ---------------------------------------------------------------------------


def _provider_with(handler) -> XAIProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=5.0)
    return XAIProvider(api_key="k", base_url="http://mock/v1", client=client)


async def test_send_hits_chat_completions_with_bearer_token():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = _provider_with(handler)
    out = await provider.send({"model": "grok-4", "messages": []})
    assert seen["url"] == "http://mock/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert out["choices"][0]["message"]["content"] == "ok"


async def test_send_4xx_becomes_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate"})

    provider = _provider_with(handler)
    with pytest.raises(ProviderError) as ei:
        await provider.send({"model": "grok-4", "messages": []})
    assert ei.value.status_code == 429


async def test_twin_override_changes_target_url():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = _provider_with(handler)
    twin.set_overrides({"GROK_BASE_URL": "http://twin/v1"})
    try:
        await provider.send({"model": "grok-4", "messages": []})
    finally:
        twin.reset()
    assert seen["url"].startswith("http://twin/v1/chat/completions")


def test_map_model_grok_passthrough_and_default_fallback():
    p = XAIProvider(api_key="k", base_url="x")
    assert p.map_model("grok-4") == "grok-4"
    assert p.map_model("grok-4-latest") == "grok-4-latest"
    assert p.map_model("claude-opus-4-7").startswith("grok")


def test_unused_import_for_helpers():
    # silence ruff F401 — these are exported for direct provider-layer tests above.
    _ = (ContentBlockDone, ContentTextDelta, StreamStart, StreamDone, ContentBlockStart)
