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


# ---------------------------------------------------------------------------
# Prompt caching: request-side cache_key and response-side cached_tokens.
#
# OpenAI's /v1/chat/completions does prompt caching automatically for prompts
# >= 1024 tokens, but routing-to-the-same-cache-shard across requests is
# best-effort unless you provide a stable ``prompt_cache_key``. The same
# response carries a ``usage.prompt_tokens_details.cached_tokens`` count that
# we need to split out of the total ``prompt_tokens`` (which INCLUDES it) so
# canonical usage maps cleanly to Anthropic's non-overlapping convention.
# ---------------------------------------------------------------------------


def _canonical_with(system: str = "", tools: list | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        model="gpt-5",
        system=[CanonicalText(text=system)] if system else None,
        messages=[CanonicalMessage(role="user", content_blocks=[CanonicalText(text="hi")])],
        tools=tools or [],
        max_tokens=32,
    )


def test_compute_prompt_cache_key_stable_for_same_system_and_tools():
    from api.providers.openai_compatible import compute_prompt_cache_key

    a = compute_prompt_cache_key(_canonical_with(system="You are X."))
    b = compute_prompt_cache_key(_canonical_with(system="You are X."))
    assert a == b
    # 32-char hex digest = 128 bits, well under OpenAI's 64-char cap
    assert len(a) == 32 and all(c in "0123456789abcdef" for c in a)


def test_compute_prompt_cache_key_differs_when_system_changes():
    from api.providers.openai_compatible import compute_prompt_cache_key

    a = compute_prompt_cache_key(_canonical_with(system="You are X."))
    b = compute_prompt_cache_key(_canonical_with(system="You are Y."))
    assert a != b


def test_compute_prompt_cache_key_ignores_message_history():
    """The message history is the PART OF THE PROMPT THAT CHANGES turn-to-turn;
    if it factored into the cache key, every continuation would route to a
    different cache shard and we'd defeat the whole point. Two requests with
    identical (system, tools) but different messages must produce the same key."""
    from api.providers.openai_compatible import compute_prompt_cache_key

    a = _canonical_with(system="You are X.")
    b = _canonical_with(system="You are X.")
    b.messages = [
        CanonicalMessage(role="user", content_blocks=[CanonicalText(text="completely different")])
    ]
    assert compute_prompt_cache_key(a) == compute_prompt_cache_key(b)


def test_openai_subclass_emits_prompt_cache_key_in_body():
    """The OpenAI provider must wire compute_prompt_cache_key into the
    outgoing chat-completions body. Without this, OpenAI's caching still
    works automatically but routing hit rate is best-effort."""
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    body = p.encode_request(_canonical_with(system="You are X."))
    assert "prompt_cache_key" in body
    assert isinstance(body["prompt_cache_key"], str)
    assert len(body["prompt_cache_key"]) == 32


def test_openai_subclass_cache_key_is_stable_across_calls_with_same_prefix():
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    b1 = p.encode_request(_canonical_with(system="same"))
    b2 = p.encode_request(_canonical_with(system="same"))
    assert b1["prompt_cache_key"] == b2["prompt_cache_key"]


def test_xai_subclass_does_NOT_emit_prompt_cache_key():
    """xAI's chat-completions endpoint accepts the parameter silently but
    doesn't do anything with it. We omit it entirely to keep the body clean
    and to avoid implying support that isn't there. The OpenAICompatible base
    ClassVar ``supports_prompt_cache_key`` defaults to False; only the OpenAI
    subclass flips it on."""
    from api.providers.xai import XAIProvider

    p = XAIProvider(api_key="k", base_url="x", default_model="grok-4.3")
    body = p.encode_request(_canonical_with(system="hello"))
    assert "prompt_cache_key" not in body


def test_decode_response_splits_cached_tokens_out_of_input():
    """OpenAI's ``prompt_tokens`` is the TOTAL input including any cached
    portion. Canonical's ``input_tokens`` is the UNCACHED portion only
    (matches Anthropic's convention and what downstream cost math assumes).
    So ``input_tokens = prompt_tokens - cached_tokens`` and ``cache_tokens``
    holds the cached subset separately."""
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    body = {
        "id": "x",
        "model": "gpt-5",
        "choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 800},
        },
    }
    cr = p.decode_response(body, canonical_request_model="gpt-5")
    assert cr.usage.input_tokens == 200  # uncached portion
    assert cr.usage.cache_tokens == 800  # cached subset
    assert cr.usage.output_tokens == 20


def test_decode_response_when_cached_tokens_zero_leaves_cache_tokens_none():
    """The common no-hit case: cached_tokens is 0 (or absent). Canonical
    cache_tokens stays None so downstream "cache_tokens is None" checks
    behave correctly (and the Anthropic encoder doesn't emit a noisy
    cache_read_input_tokens=0)."""
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    body_zero = {
        "id": "x",
        "model": "gpt-5",
        "choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }
    cr = p.decode_response(body_zero, canonical_request_model="gpt-5")
    assert cr.usage.input_tokens == 100
    assert cr.usage.cache_tokens is None


def test_decode_response_when_prompt_tokens_details_absent_unchanged():
    """A response with no prompt_tokens_details (older models, xAI, etc.)
    must behave identically to the pre-caching code path."""
    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    body = {
        "id": "x",
        "model": "gpt-5",
        "choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    cr = p.decode_response(body, canonical_request_model="gpt-5")
    assert cr.usage.input_tokens == 10
    assert cr.usage.cache_tokens is None


async def test_decode_stream_carries_cache_tokens_in_terminal_message_delta():
    """The streaming path is symmetric: OpenAI's terminal usage frame can carry
    prompt_tokens_details.cached_tokens, and our MessageDelta event must
    surface it via CanonicalUsage.cache_tokens so the Anthropic encoder can
    emit cache_read_input_tokens in the final message_delta SSE event."""
    from api.canonical.events import MessageDelta as MD

    p = OpenAIProvider(api_key="k", base_url="x", default_model="gpt-5")
    chunks = [
        _chunk({"id": "x", "choices": [{"index": 0, "delta": {"content": "ok"}}]}),
        _chunk({"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _chunk(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 400},
                },
            }
        ),
        b"data: [DONE]\n\n",
    ]
    events = []
    async for ev in p.decode_stream(_aiter(chunks), canonical_request_model="gpt-5"):
        events.append(ev)
    [delta] = [e for e in events if isinstance(e, MD)]
    assert delta.usage.input_tokens == 100  # 500 total - 400 cached
    assert delta.usage.cache_tokens == 400
    assert delta.usage.output_tokens == 10
