"""ProviderAdapter protocol — what every backend speaks at the Canonical boundary.

A ProviderAdapter is FOUR things rolled into one type:

  1. ``encode_request(CanonicalRequest) -> provider_body``      (canonical -> wire)
  2. ``decode_response(provider_body) -> CanonicalResponse``    (wire -> canonical)
  3. ``decode_stream(byte_stream) -> AsyncIterator[Canonical]`` (streaming wire -> canonical)
  4. ``send`` / ``send_stream`` — actually do the HTTP call

Inbound protocols never see (1)–(4); they hand a CanonicalRequest to the engine
and the engine drives whichever adapter is registered for the target provider.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol

from api.canonical.types import CanonicalRequest, CanonicalResponse


class ProviderError(Exception):
    def __init__(self, status_code: int, message: str, *, provider_body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.provider_body = provider_body


class ProviderAdapter(Protocol):
    name: str

    def encode_request(self, canonical: CanonicalRequest) -> dict[str, Any]:
        """Canonical -> provider wire body."""

    def decode_response(
        self, provider_body: dict[str, Any], *, canonical_request_model: str
    ) -> CanonicalResponse:
        """Provider wire body -> Canonical. ``canonical_request_model`` is the model id the
        canonical-side caller asked for — the adapter echoes it (or its own resolved id)
        on the response's ``model`` field."""

    def decode_stream(
        self, raw_chunks: AsyncIterator[bytes], *, canonical_request_model: str
    ) -> AsyncIterator:  # AsyncIterator[CanonicalStreamEvent]
        """Streaming wire -> Canonical events."""

    async def send(self, provider_body: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming HTTPS call. Returns the parsed JSON response."""

    def send_stream(self, provider_body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Streaming HTTPS call. Yields raw response chunks."""

    def map_model(self, canonical_model: str) -> str:
        """Map the canonical-side model id to the provider-side id."""
