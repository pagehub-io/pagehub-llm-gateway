"""Fake adapters used by the three test layers.

- :class:`EchoProvider` — speaks Canonical natively. Used by the *engine* tests
  to exercise the pipeline independent of any wire format.
- :class:`ScriptedXAIProvider` — speaks the OpenAI/xAI wire format on the
  transport side (returns scripted dicts and SSE chunks) but uses the real
  ``XAIProvider`` encode/decode helpers. Used by the *e2e-through-app* test to
  cover all four layers (Anthropic decode -> canonical -> xAI encode/decode ->
  Anthropic encode) in one go without touching real HTTP.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, AsyncIterator

from api.canonical.events import (
    ContentBlockDone,
    ContentBlockStart,
    ContentTextDelta,
    MessageDelta,
    StreamDone,
    StreamStart,
)
from api.canonical.types import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalText,
    CanonicalUsage,
)
from api.providers.base import ProviderError
from api.providers.xai import XAIProvider


class EchoProvider:
    """ProviderAdapter that lives entirely in canonical-space.

    Records the canonical request it was given and returns a scripted canonical
    response (or streams scripted canonical events). No wire format involved.
    """

    name = "echo"

    def __init__(
        self,
        *,
        response: CanonicalResponse | None = None,
        stream_events: list | None = None,
        error: ProviderError | None = None,
    ) -> None:
        self._response = response
        self._stream_events = stream_events or []
        self._error = error
        self.last_canonical_request: CanonicalRequest | None = None
        self.last_provider_body: dict[str, Any] | None = None

    def encode_request(self, canonical: CanonicalRequest) -> dict[str, Any]:
        self.last_canonical_request = canonical
        return {"_canonical": canonical.model_dump(exclude_none=True)}

    def decode_response(
        self, provider_body: dict[str, Any], *, canonical_request_model: str
    ) -> CanonicalResponse:
        if self._response is not None:
            return self._response
        return CanonicalResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            model=canonical_request_model,
            content_blocks=[CanonicalText(text="echo")],
            stop_reason="end_turn",
            usage=CanonicalUsage(input_tokens=1, output_tokens=1),
        )

    async def decode_stream(
        self, raw_chunks: AsyncIterator[bytes], *, canonical_request_model: str
    ):
        # Ignore raw chunks; emit the scripted canonical events.
        events = list(self._stream_events) or [
            StreamStart(message_id=f"msg_{uuid.uuid4().hex[:24]}", model=canonical_request_model),
            ContentBlockStart(index=0, block=CanonicalText()),
            ContentTextDelta(index=0, text="echo"),
            ContentBlockDone(index=0),
            MessageDelta(stop_reason="end_turn", usage=CanonicalUsage(output_tokens=1)),
            StreamDone(),
        ]
        for ev in events:
            await asyncio.sleep(0)
            yield ev

    async def send(self, provider_body: dict[str, Any]) -> dict[str, Any]:
        self.last_provider_body = provider_body
        if self._error is not None:
            raise self._error
        # Engine.run_anthropic will then call decode_response on this — which ignores body.
        return {}

    async def send_stream(self, provider_body: dict[str, Any]) -> AsyncIterator[bytes]:
        self.last_provider_body = provider_body
        if self._error is not None:
            raise self._error
        # decode_stream is what actually runs; this just gives it an empty iterator to consume.
        if False:
            yield b""

    def map_model(self, canonical_model: str) -> str:
        return canonical_model or "echo-model"


class ScriptedXAIProvider(XAIProvider):
    """Real ``XAIProvider`` encode/decode logic; scripted HTTP transport.

    Used by the e2e-through-app test so we exercise the full canonical pipeline
    plus the actual xAI wire-format translation, without a real network call.
    """

    def __init__(
        self,
        *,
        non_streaming_response: dict[str, Any] | None = None,
        stream_chunks: list[bytes] | None = None,
        error: ProviderError | None = None,
    ) -> None:
        super().__init__(api_key="test", base_url="http://fake/v1")
        self._non_streaming_response = non_streaming_response
        self._stream_chunks = stream_chunks or []
        self._error = error
        self.last_provider_body: dict[str, Any] | None = None

    async def send(self, provider_body: dict[str, Any]) -> dict[str, Any]:
        self.last_provider_body = provider_body
        if self._error is not None:
            raise self._error
        return self._non_streaming_response or {
            "id": "chatcmpl_fake",
            "model": provider_body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def send_stream(self, provider_body: dict[str, Any]) -> AsyncIterator[bytes]:
        self.last_provider_body = provider_body
        if self._error is not None:
            raise self._error
        for chunk in self._stream_chunks:
            await asyncio.sleep(0)
            yield chunk
