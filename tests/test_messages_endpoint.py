"""End-to-end-ish: through the FastAPI app with FakeProvider as the backend."""

from __future__ import annotations

import json

from api.providers.base import ProviderError
from tests.fake_provider import FakeProvider


def _basic_body():
    return {
        "model": "grok-4",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_non_streaming_text_round_trip(client, auth_headers, fake_provider):
    fake_provider._non_streaming_response = {
        "id": "chatcmpl_xyz",
        "model": "grok-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi from grok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }
    resp = client.post("/v1/messages", json=_basic_body(), headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "grok-4"
    assert body["content"] == [{"type": "text", "text": "hi from grok"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] == 12
    assert body["usage"]["output_tokens"] == 4


def test_non_streaming_tool_use_round_trip(client, auth_headers, fake_provider):
    fake_provider._non_streaming_response = {
        "id": "chatcmpl_t",
        "model": "grok-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"SF"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }
    inbound = _basic_body()
    inbound["tools"] = [
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
    inbound["tool_choice"] = {"type": "auto"}
    resp = client.post("/v1/messages", json=inbound, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    [block] = body["content"]
    assert block["type"] == "tool_use"
    assert block["name"] == "get_weather"
    assert block["input"] == {"location": "SF"}

    # And confirm the outbound payload to the provider was correctly translated.
    sent = fake_provider.last_request
    assert sent["tools"][0]["function"]["name"] == "get_weather"
    assert sent["tool_choice"] == "auto"


def test_tool_result_follow_up_request_is_translated(client, auth_headers, fake_provider):
    """Simulate Claude Code's second turn: user provides a tool_result and the gateway
    must produce a `role:"tool"` OpenAI message keyed by tool_call_id."""
    fake_provider._non_streaming_response = {
        "id": "chatcmpl_done",
        "model": "grok-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "sunny, 72F"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
    }
    body = {
        "model": "grok-4",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "weather in SF?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"location": "SF"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "72F sunny",
                    }
                ],
            },
        ],
    }
    resp = client.post("/v1/messages", json=body, headers=auth_headers)
    assert resp.status_code == 200
    sent = fake_provider.last_request
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["user", "assistant", "tool"]
    assert sent["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "72F sunny",
    }


def test_provider_5xx_surfaces_as_provider_error(client, auth_headers, app):
    app.state.provider = FakeProvider(
        non_streaming_error=ProviderError(status_code=503, message="Grok unavailable")
    )
    resp = client.post("/v1/messages", json=_basic_body(), headers=auth_headers)
    assert resp.status_code == 503
    body = resp.json()
    assert body["detail"]["error"]["type"] == "provider_error"
    assert "Grok unavailable" in body["detail"]["error"]["message"]


def test_streaming_endpoint_returns_anthropic_sse(client, auth_headers, fake_provider):
    fake_provider._stream_chunks = [
        b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        b"data: [DONE]\n\n",
    ]
    body = dict(_basic_body(), stream=True)
    with client.stream("POST", "/v1/messages", json=body, headers=auth_headers) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = b"".join(resp.iter_bytes())

    # Parse the events we wrote.
    events = []
    for block in raw.split(b"\n\n"):
        if not block.strip():
            continue
        evt, data = None, None
        for line in block.split(b"\n"):
            if line.startswith(b"event:"):
                evt = line[6:].strip().decode()
            elif line.startswith(b"data:"):
                data = json.loads(line[5:].strip())
        events.append((evt, data))

    types = [t for t, _ in events]
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert "message_delta" in types
    assert types[-1] == "message_stop"
    text = "".join(
        d["delta"]["text"]
        for t, d in events
        if t == "content_block_delta" and d["delta"]["type"] == "text_delta"
    )
    assert text == "Hello"


def test_streaming_endpoint_emits_error_event_on_provider_error(client, auth_headers, app):
    app.state.provider = FakeProvider(
        stream_error=ProviderError(status_code=502, message="upstream gone")
    )
    body = dict(_basic_body(), stream=True)
    with client.stream("POST", "/v1/messages", json=body, headers=auth_headers) as resp:
        assert resp.status_code == 200
        raw = b"".join(resp.iter_bytes())
    assert b"event: error" in raw
    assert b"upstream gone" in raw


def test_metrics_increment_after_successful_request(client, auth_headers):
    pre = client.get("/metrics").json()
    pre_total = sum(pre["requests_total"].values())
    pre_tokens = pre["output_tokens_total"]

    client.post("/v1/messages", json=_basic_body(), headers=auth_headers)

    post = client.get("/metrics").json()
    post_total = sum(post["requests_total"].values())
    assert post_total > pre_total
    assert post["output_tokens_total"] >= pre_tokens
