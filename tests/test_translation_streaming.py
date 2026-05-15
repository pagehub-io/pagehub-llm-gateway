"""OpenAI SSE -> Anthropic SSE event translator (unit tests, no HTTP)."""

from __future__ import annotations

import json
from typing import AsyncIterator

from api.translation.streaming import openai_stream_to_anthropic_events


async def _iter_bytes(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


def _make_chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _done() -> bytes:
    return b"data: [DONE]\n\n"


def _parse(stream_bytes: bytes) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for raw in stream_bytes.split(b"\n\n"):
        if not raw.strip():
            continue
        evt = None
        data = None
        for line in raw.split(b"\n"):
            if line.startswith(b"event:"):
                evt = line[6:].strip().decode()
            elif line.startswith(b"data:"):
                data = json.loads(line[5:].strip())
        if evt is not None and data is not None:
            events.append((evt, data))
    return events


async def _run(chunks: list[bytes]) -> bytes:
    out = b""
    async for ev in openai_stream_to_anthropic_events(
        _iter_bytes(chunks), anthropic_model="grok-4", input_tokens_estimate=5
    ):
        out += ev
    return out


async def test_simple_text_stream_produces_full_event_sequence():
    chunks = [
        _make_chunk({"id": "x1", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
        _make_chunk({"id": "x1", "choices": [{"index": 0, "delta": {"content": "Hel"}}]}),
        _make_chunk({"id": "x1", "choices": [{"index": 0, "delta": {"content": "lo"}}]}),
        _make_chunk({"id": "x1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _make_chunk(
            {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}}
        ),
        _done(),
    ]
    out = await _run(chunks)
    events = _parse(out)
    types = [t for t, _ in events]
    assert types[0] == "message_start"
    assert types[1] == "content_block_start"
    assert types[-1] == "message_stop"
    assert types[-2] == "message_delta"
    assert types[-3] == "content_block_stop"
    text_deltas = [
        d for t, d in events if t == "content_block_delta" and d["delta"]["type"] == "text_delta"
    ]
    assert "".join(d["delta"]["text"] for d in text_deltas) == "Hello"
    delta_event = next(d for t, d in events if t == "message_delta")
    assert delta_event["delta"]["stop_reason"] == "end_turn"
    assert delta_event["usage"]["output_tokens"] == 2


async def test_message_start_carries_input_tokens_estimate_then_real():
    chunks = [
        _make_chunk({"id": "x", "choices": [{"index": 0, "delta": {"content": "ok"}}]}),
        _make_chunk({"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _done(),
    ]
    out = await _run(chunks)
    events = _parse(out)
    start = events[0][1]
    # We seeded estimate=5 in _run.
    assert start["message"]["usage"]["input_tokens"] == 5


async def test_tool_call_stream_assembles_input_json_delta():
    chunks = [
        _make_chunk(
            {
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        _make_chunk(
            {
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"location":'}}
                            ]
                        },
                    }
                ],
            }
        ),
        _make_chunk(
            {
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"SF"}'}}
                            ]
                        },
                    }
                ],
            }
        ),
        _make_chunk(
            {"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        ),
        _done(),
    ]
    out = await _run(chunks)
    events = _parse(out)
    types = [t for t, _ in events]
    cb_start = next(d for t, d in events if t == "content_block_start")
    assert cb_start["content_block"]["type"] == "tool_use"
    assert cb_start["content_block"]["id"] == "call_abc"
    assert cb_start["content_block"]["name"] == "get_weather"
    json_deltas = [
        d for t, d in events if t == "content_block_delta" and d["delta"]["type"] == "input_json_delta"
    ]
    assembled = "".join(d["delta"]["partial_json"] for d in json_deltas)
    assert json.loads(assembled) == {"location": "SF"}
    delta_event = next(d for t, d in events if t == "message_delta")
    assert delta_event["delta"]["stop_reason"] == "tool_use"
    # content_block_stop must appear after the json_deltas.
    assert types.index("content_block_stop") > types.index("content_block_delta")


async def test_text_then_tool_call_closes_text_block_first():
    chunks = [
        _make_chunk({"id": "x", "choices": [{"index": 0, "delta": {"content": "thinking"}}]}),
        _make_chunk(
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
                                    "function": {"name": "run", "arguments": "{}"},
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        _make_chunk(
            {"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        ),
        _done(),
    ]
    out = await _run(chunks)
    events = _parse(out)
    types_indexed = list(enumerate([t for t, _ in events]))
    text_stop_idx = next(
        i for i, t in types_indexed if t == "content_block_stop" and events[i][1]["index"] == 0
    )
    tool_start_idx = next(
        i
        for i, t in types_indexed
        if t == "content_block_start" and events[i][1]["content_block"]["type"] == "tool_use"
    )
    assert text_stop_idx < tool_start_idx


async def test_stream_with_no_content_still_emits_final_events():
    chunks = [
        _make_chunk({"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _done(),
    ]
    out = await _run(chunks)
    events = _parse(out)
    types = [t for t, _ in events]
    assert "message_start" in types
    assert "message_delta" in types
    assert types[-1] == "message_stop"


async def test_sse_handles_crlf_separators():
    # xAI mostly uses \n\n, but the SSE spec also allows \r\n\r\n.
    crlf_chunk = b"data: " + json.dumps(
        {"id": "x", "choices": [{"index": 0, "delta": {"content": "ok"}}]}
    ).encode() + b"\r\n\r\n"
    chunks = [
        crlf_chunk,
        _make_chunk({"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _done(),
    ]
    out = await _run(chunks)
    events = _parse(out)
    text = next(
        d for t, d in events
        if t == "content_block_delta" and d["delta"]["type"] == "text_delta"
    )
    assert text["delta"]["text"] == "ok"
