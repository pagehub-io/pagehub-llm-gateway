"""End-to-end-through-the-app: Anthropic in -> Canonical -> xAI encode/decode (scripted) -> Anthropic out.

This test wires all four translation layers (AnthropicInbound.decode_request,
XAIProvider.encode_request, XAIProvider.decode_response, AnthropicInbound.encode_response)
through the FastAPI app with a scripted HTTP transport. It's the integration test
that confirms the layers compose correctly — every unit-level layer test above
runs in isolation.
"""

from __future__ import annotations

import json
import uuid

from api.providers.base import ProviderError
from tests.fakes import ScriptedXAIProvider


def _body():
    return {
        "model": "grok-4",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_non_streaming_text_round_trip(client, auth_headers, scripted_xai):
    scripted_xai._non_streaming_response = {
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
    resp = client.post("/v1/messages", json=_body(), headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["model"] == "grok-4"
    assert body["content"][0]["text"] == "hi from grok"
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 12, "output_tokens": 4}

    # The scripted provider saw a properly-translated xAI body.
    sent = scripted_xai.last_provider_body
    assert sent["model"] == "grok-4"
    assert sent["messages"] == [{"role": "user", "content": "hello"}]


def test_non_streaming_tool_use_round_trip(client, auth_headers, scripted_xai):
    scripted_xai._non_streaming_response = {
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
                            "function": {"name": "get_weather", "arguments": '{"location":"SF"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }
    inbound = _body()
    inbound["tools"] = [
        {
            "name": "get_weather",
            "description": "lookup",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }
    ]
    inbound["tool_choice"] = {"type": "auto"}
    resp = client.post("/v1/messages", json=inbound, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    [block] = body["content"]
    assert block["type"] == "tool_use"
    assert block["name"] == "get_weather"
    assert block["input"] == {"location": "SF"}
    assert body["stop_reason"] == "tool_use"

    sent = scripted_xai.last_provider_body
    assert sent["tools"][0]["function"]["name"] == "get_weather"
    assert sent["tool_choice"] == "auto"


def test_tool_result_follow_up(client, auth_headers, scripted_xai):
    scripted_xai._non_streaming_response = {
        "id": "chatcmpl_done",
        "model": "grok-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "72F sunny"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 5},
    }
    body = {
        "model": "grok-4",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "weather in SF?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"location": "SF"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F sunny"}
                ],
            },
        ],
    }
    resp = client.post("/v1/messages", json=body, headers=auth_headers)
    assert resp.status_code == 200
    sent = scripted_xai.last_provider_body
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["user", "assistant", "tool"]
    assert sent["messages"][-1] == {"role": "tool", "tool_call_id": "toolu_1", "content": "72F sunny"}


def test_provider_error_propagates_with_status(app_with_xai):
    from fastapi.testclient import TestClient

    from api.engine import Engine

    failing = ScriptedXAIProvider(error=ProviderError(status_code=503, message="Grok unavailable"))
    app_with_xai.state.provider = failing
    app_with_xai.state.engine = Engine(provider=failing, recorder=app_with_xai.state.recorder)
    client = TestClient(app_with_xai)
    resp = client.post("/v1/messages", json=_body(), headers={"x-api-key": "test-token"})
    assert resp.status_code == 503
    assert "Grok unavailable" in resp.json()["detail"]["error"]["message"]


def test_streaming_text_returns_valid_anthropic_sse(client, auth_headers, scripted_xai):
    scripted_xai._stream_chunks = [
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        b"data: [DONE]\n\n",
    ]
    with client.stream("POST", "/v1/messages", json=dict(_body(), stream=True), headers=auth_headers) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = b"".join(resp.iter_bytes())
    events = _parse_sse(raw)
    types = [t for t, _ in events]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    text = "".join(
        d["delta"]["text"]
        for t, d in events
        if t == "content_block_delta" and d["delta"]["type"] == "text_delta"
    )
    assert text == "Hello"
    delta = next(d for t, d in events if t == "message_delta")
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["usage"]["output_tokens"] == 2


async def test_recorder_captures_e2e_request(client, auth_headers, scripted_xai, app):
    scripted_xai._non_streaming_response = {
        "id": "chatcmpl",
        "model": "grok-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1},
    }
    resp = client.post("/v1/messages", json=_body(), headers=auth_headers)
    assert resp.status_code == 200

    recorder = app.state.recorder
    rows = await recorder.list_requests(limit=10, offset=0)
    detail = await recorder.get_request(uuid.UUID(rows[0]["id"]))
    assert rows[0]["provider"] == "xai"
    kinds = [e["kind"] for e in detail["events"]]
    assert "inbound_received" in kinds
    assert "decoded_canonical" in kinds
    assert "provider_request" in kinds
    assert "provider_response" in kinds
    assert "decoded_canonical_response" in kinds
    assert "encoded_outbound" in kinds


def _parse_sse(raw: bytes):
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
        if evt is not None and data is not None:
            events.append((evt, data))
    return events
