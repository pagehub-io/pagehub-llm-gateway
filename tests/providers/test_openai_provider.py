"""Layer (c) for the OpenAI provider — Canonical <-> OpenAI translation +
HTTP transport, in isolation.

Most of the wire-format coverage lives in ``test_xai_provider.py`` already,
because xAI and OpenAI share the same chat-completions wire spec via
:class:`OpenAICompatibleProvider`. This file pins the OpenAI-specific
specialization: which model ids it claims, which env-var name it reads, which
twin-override key it honors, and that its HTTP transport hits OpenAI's
endpoint with the right Bearer header. We add one encode + one decode test to
prove the inheritance is wired through (catching e.g. a future regression
where a subclass forgets to call ``super().__init__``).
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx
import pytest

from api import twin
from api.canonical.events import ContentBlockStart, ContentTextDelta
from api.canonical.types import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalText,
    CanonicalToolCall,
)
from api.providers.base import ProviderError
from api.providers.openai import OpenAIProvider

# ---------------------------------------------------------------------------
# Specialization — what this subclass differs on
# ---------------------------------------------------------------------------


def test_claims_model_gpt_family():
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    assert p.claims_model("gpt-5") is True
    assert p.claims_model("gpt-4o") is True
    assert p.claims_model("gpt-5-mini") is True


def test_claims_model_o_series():
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    assert p.claims_model("o1") is True
    assert p.claims_model("o1-pro") is True
    assert p.claims_model("o3-mini") is True
    assert p.claims_model("o4-mini") is True


def test_does_not_claim_grok_or_claude_or_empty():
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    assert p.claims_model("grok-4") is False
    assert p.claims_model("grok-4.3") is False
    assert p.claims_model("claude-opus-4-7") is False
    assert p.claims_model("") is False


def test_map_model_passthrough_and_default_fallback():
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    assert p.map_model("gpt-5") == "gpt-5"
    assert p.map_model("o3-mini") == "o3-mini"
    assert p.map_model("claude-opus-4-7") == "gpt-5"  # unclaimed -> default
    assert p.map_model("") == "gpt-5"


# ---------------------------------------------------------------------------
# Inheritance smoke — encode/decode go through the shared base
# ---------------------------------------------------------------------------


def test_encode_request_via_subclass():
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    canonical = CanonicalRequest(
        model="gpt-5",
        messages=[CanonicalMessage(role="user", content_blocks=[CanonicalText(text="hi")])],
        max_tokens=32,
    )
    body = p.encode_request(canonical)
    assert body["model"] == "gpt-5"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    # OpenAI's GPT-5 / o-series 400 on ``max_tokens``; the subclass must emit
    # ``max_completion_tokens`` instead. Caught the hard way during the v2
    # diagnostic smoke — regressing this would break every OpenAI call.
    assert body["max_completion_tokens"] == 32
    assert "max_tokens" not in body


def test_decode_response_via_subclass():
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    body = {
        "id": "chatcmpl_o",
        "model": "gpt-5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ack"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    cr = p.decode_response(body, canonical_request_model="gpt-5")
    assert cr.model == "gpt-5"
    [block] = cr.content_blocks
    assert isinstance(block, CanonicalText)
    assert block.text == "ack"
    assert cr.stop_reason == "end_turn"
    assert cr.usage.input_tokens == 5
    assert cr.usage.output_tokens == 1


def test_reasoning_tokens_counted_in_output_tokens():
    """OpenAI's o-series / GPT-5 split completion_tokens into a 'reasoning'
    subcount inside completion_tokens_details. Those tokens ARE already
    included in completion_tokens; we propagate them to canonical
    output_tokens via the sum, not the breakdown."""
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    body = {
        "id": "chatcmpl_o",
        "model": "gpt-5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 50,  # already includes reasoning
            "completion_tokens_details": {"reasoning_tokens": 40, "accepted_prediction_tokens": 0},
            "total_tokens": 60,
        },
    }
    cr = p.decode_response(body, canonical_request_model="gpt-5")
    assert cr.usage.input_tokens == 10
    assert cr.usage.output_tokens == 50  # reasoning is part of this sum


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


def _provider_with(handler, *, base_url: str = "http://mock/v1") -> OpenAIProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=5.0)
    return OpenAIProvider(
        api_key="k", base_url=base_url, default_model="gpt-5", client=client
    )


async def test_send_hits_chat_completions_with_bearer_token():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = _provider_with(handler)
    out = await provider.send({"model": "gpt-5", "messages": []})
    assert seen["url"] == "http://mock/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert out["choices"][0]["message"]["content"] == "ok"


async def test_send_4xx_becomes_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate"})

    provider = _provider_with(handler)
    with pytest.raises(ProviderError) as ei:
        await provider.send({"model": "gpt-5", "messages": []})
    assert ei.value.status_code == 429


async def test_twin_override_changes_target_url():
    """Each provider has its own twin-override key. xAI's was ``GROK_BASE_URL``;
    OpenAI's is ``OPENAI_BASE_URL`` — verify the right key is honored."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = _provider_with(handler)
    twin.set_overrides({"OPENAI_BASE_URL": "http://twin-openai/v1"})
    try:
        await provider.send({"model": "gpt-5", "messages": []})
    finally:
        twin.reset()
    assert seen["url"].startswith("http://twin-openai/v1/chat/completions")


async def test_grok_override_does_not_affect_openai():
    """Cross-provider hygiene: a ``GROK_BASE_URL`` override must not redirect
    OpenAI traffic. Each subclass reads its own ``twin_override_env_var``."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = _provider_with(handler)
    twin.set_overrides({"GROK_BASE_URL": "http://twin-grok/v1"})
    try:
        await provider.send({"model": "gpt-5", "messages": []})
    finally:
        twin.reset()
    # Hits the configured OpenAI URL, NOT the grok-override.
    assert seen["url"].startswith("http://mock/v1/chat/completions")


def test_missing_api_key_raises_with_provider_name_in_message():
    """The base raises ``{NAME}_API_KEY not configured`` — confirm the OpenAI
    subclass surfaces ``OPENAI_API_KEY`` in the error, not the base's ``_API_KEY``."""
    provider = OpenAIProvider(api_key="", base_url="x", default_model="gpt-5")
    with pytest.raises(ProviderError) as ei:
        provider._headers()
    assert "OPENAI_API_KEY" in ei.value.message


# ---------------------------------------------------------------------------
# Stream specialization smoke — confirm streaming goes through the shared base
# ---------------------------------------------------------------------------


def _chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _aiter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


async def test_decode_stream_via_subclass():
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    events = []
    async for ev in p.decode_stream(
        _aiter(
            [
                _chunk({"id": "x", "choices": [{"index": 0, "delta": {"content": "He"}}]}),
                _chunk({"id": "x", "choices": [{"index": 0, "delta": {"content": "y"}}]}),
                _chunk(
                    {
                        "id": "x",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                ),
                _chunk({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}),
                b"data: [DONE]\n\n",
            ]
        ),
        canonical_request_model="gpt-5",
    ):
        events.append(ev)
    starts = [e for e in events if isinstance(e, ContentBlockStart)]
    assert len(starts) == 1
    text_deltas = [e for e in events if isinstance(e, ContentTextDelta)]
    assert "".join(d.text for d in text_deltas) == "Hey"


def test_unused_imports():
    # Silence ruff F401 — exported for direct asserts in tests above.
    _ = (CanonicalToolCall,)
