"""Provider protocol — what every backend (Grok, OpenAI, Bedrock, ...) must implement."""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol


class ProviderError(Exception):
    """Provider-level failure. ``status_code`` reflects what the gateway should return."""

    def __init__(self, status_code: int, message: str, *, provider_body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.provider_body = provider_body


class Provider(Protocol):
    name: str

    async def chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming call. Returns the parsed JSON response dict."""

    def chat_completion_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Streaming call. Returns an async iterator over raw upstream SSE bytes."""

    def map_model(self, anthropic_model_id: str) -> str:
        """Map the Anthropic-side model id from the inbound request to the provider id.

        Lets us accept Claude-style ids unchanged (Claude Code passes whatever ``--model``
        was set to) and route them at the gateway.
        """
