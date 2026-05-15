"""In-process fake Provider used by every test in this suite.

Tests NEVER call the real xAI API: configure :class:`FakeProvider` with a scripted
non-streaming response or a list of OpenAI-shaped SSE chunks, then drop it in via
``app.state.provider = FakeProvider(...)``.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from api.providers.base import ProviderError


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        *,
        non_streaming_response: dict[str, Any] | None = None,
        stream_chunks: list[bytes] | None = None,
        non_streaming_error: ProviderError | None = None,
        stream_error: ProviderError | None = None,
        default_model: str = "grok-4",
    ) -> None:
        self._non_streaming_response = non_streaming_response
        self._stream_chunks = stream_chunks or []
        self._non_streaming_error = non_streaming_error
        self._stream_error = stream_error
        self._default_model = default_model
        # Captured: tests can inspect what the translator produced.
        self.last_request: dict[str, Any] | None = None
        self.last_stream_request: dict[str, Any] | None = None

    def map_model(self, anthropic_model_id: str) -> str:
        if anthropic_model_id and anthropic_model_id.lower().startswith("grok"):
            return anthropic_model_id
        return self._default_model

    async def chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        self.last_request = body
        if self._non_streaming_error is not None:
            raise self._non_streaming_error
        return self._non_streaming_response or {
            "id": "chatcmpl_fake",
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def chat_completion_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        self.last_stream_request = body
        if self._stream_error is not None:
            raise self._stream_error
        for chunk in self._stream_chunks:
            # Tiny await so the event loop turns between chunks — surfaces ordering bugs.
            await asyncio.sleep(0)
            yield chunk
