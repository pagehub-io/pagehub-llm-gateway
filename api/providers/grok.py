"""xAI Grok provider — OpenAI-compatible chat-completions endpoint.

Base URL: ``https://api.x.ai/v1`` (configurable via ``GROK_BASE_URL`` env var and the
dev-only ``X-Twin-Grok-Base-Url`` request header).
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from api.config import settings
from api.providers.base import ProviderError
from api.twin import get_override


class GrokProvider:
    name = "grok"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.xai_api_key
        self._configured_base_url = base_url if base_url is not None else settings.grok_base_url
        self._timeout = timeout_seconds if timeout_seconds is not None else settings.provider_timeout_seconds
        self._client = client  # injectable for tests

    def _resolved_base_url(self) -> str:
        override = get_override("GROK_BASE_URL")
        return (override or self._configured_base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise ProviderError(
                status_code=500,
                message="XAI_API_KEY not configured on the gateway.",
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _stream_headers(self) -> dict[str, str]:
        h = self._headers()
        h["Accept"] = "text/event-stream"
        return h

    def map_model(self, anthropic_model_id: str) -> str:
        """If the inbound request already names a Grok model, use it verbatim.

        Anything that looks like a Claude id (or anything unrecognised) falls back to
        the configured default — this is what lets Claude Code's ``--model grok-4``
        work AND lets the SDK's default (a Claude id) round-trip without breaking.
        """
        if not anthropic_model_id:
            return settings.grok_default_model
        # Heuristic: starts with "grok" -> pass through. Otherwise -> default.
        lower = anthropic_model_id.lower()
        if lower.startswith("grok"):
            return anthropic_model_id
        return settings.grok_default_model

    def _new_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(timeout=self._timeout)

    async def chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._resolved_base_url()}/chat/completions"
        client = self._new_client()
        owns_client = self._client is None
        try:
            try:
                resp = await client.post(url, json=body, headers=self._headers())
            except httpx.RequestError as exc:
                raise ProviderError(
                    status_code=502, message=f"Upstream provider request failed: {exc}"
                ) from exc
            if resp.status_code >= 400:
                raise ProviderError(
                    status_code=resp.status_code,
                    message=f"Provider returned {resp.status_code}",
                    provider_body=resp.text,
                )
            return resp.json()
        finally:
            if owns_client:
                await client.aclose()

    async def chat_completion_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Open an SSE stream to the provider and yield raw chunks.

        Implemented as an async generator so the caller can ``async for`` directly; the
        underlying httpx response stays open until the generator is exhausted or the
        consumer breaks out.
        """
        url = f"{self._resolved_base_url()}/chat/completions"
        client = self._new_client()
        owns_client = self._client is None
        # ensure we send `stream: true` even if the caller forgot
        body = {**body, "stream": True}
        try:
            try:
                async with client.stream(
                    "POST", url, json=body, headers=self._stream_headers()
                ) as resp:
                    if resp.status_code >= 400:
                        text = (await resp.aread()).decode("utf-8", errors="replace")
                        raise ProviderError(
                            status_code=resp.status_code,
                            message=f"Provider returned {resp.status_code} on stream open",
                            provider_body=text,
                        )
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
            except httpx.RequestError as exc:
                raise ProviderError(
                    status_code=502, message=f"Upstream provider request failed: {exc}"
                ) from exc
        finally:
            if owns_client:
                await client.aclose()
