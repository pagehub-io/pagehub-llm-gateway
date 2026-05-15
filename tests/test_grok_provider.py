"""GrokProvider tests using httpx MockTransport — no real network."""

from __future__ import annotations

import httpx
import pytest

from api import twin
from api.providers.base import ProviderError
from api.providers.grok import GrokProvider


def _make_provider(handler) -> GrokProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=5.0)
    return GrokProvider(api_key="key", base_url="http://mock.xai/v1", client=client)


async def test_non_streaming_request_hits_chat_completions_url():
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

    provider = _make_provider(handler)
    out = await provider.chat_completion({"model": "grok-4", "messages": []})
    assert seen["url"] == "http://mock.xai/v1/chat/completions"
    assert seen["auth"] == "Bearer key"
    assert out["choices"][0]["message"]["content"] == "ok"


async def test_provider_error_on_4xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "denied"})

    provider = _make_provider(handler)
    with pytest.raises(ProviderError) as ei:
        await provider.chat_completion({"model": "grok-4", "messages": []})
    assert ei.value.status_code == 403


async def test_streaming_yields_provider_chunks():
    chunks_body = (
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        b'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=chunks_body, headers={"content-type": "text/event-stream"})

    provider = _make_provider(handler)
    out = b""
    async for c in provider.chat_completion_stream({"model": "grok-4", "messages": []}):
        out += c
    assert b'"content":"hi"' in out
    assert b"[DONE]" in out


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

    provider = _make_provider(handler)
    twin.set_overrides({"GROK_BASE_URL": "http://twin.local/v1"})
    try:
        await provider.chat_completion({"model": "grok-4", "messages": []})
    finally:
        twin.reset()
    assert seen["url"].startswith("http://twin.local/v1/chat/completions")


def test_map_model_passes_grok_ids_through():
    p = GrokProvider(api_key="k", base_url="x")
    assert p.map_model("grok-4") == "grok-4"
    assert p.map_model("grok-4-latest") == "grok-4-latest"


def test_map_model_falls_back_to_default_for_claude_ids():
    p = GrokProvider(api_key="k", base_url="x")
    # Anything that doesn't start with "grok" -> default.
    assert p.map_model("claude-opus-4-7").startswith("grok")
    assert p.map_model("") != ""
