"""XAIProvider — xAI Grok backend.

Thin specialization of :class:`OpenAICompatibleProvider`. xAI's API is a
straight-up OpenAI chat-completions clone, so all of the actual translation
lives in ``api/providers/openai_compatible.py``. This file only encodes the
xAI-specific bindings: which canonical model ids we claim, which settings to
read, and which X-Twin-* env-var name overrides the base URL.

The ``encode_request`` / ``decode_response`` / ``decode_stream`` symbols are
re-exported below for direct provider-layer tests that import them by name.
"""

from __future__ import annotations

import httpx

from api.config import settings
from api.providers.openai_compatible import (
    OpenAICompatibleProvider,
    decode_response,
    decode_stream,
    encode_request,
)

__all__ = ["XAIProvider", "encode_request", "decode_response", "decode_stream"]


class XAIProvider(OpenAICompatibleProvider):
    name = "xai"
    # xAI publishes ids like ``grok-4``, ``grok-4-latest``, ``grok-4.3``,
    # ``grok-4.20-...`` — all share the ``grok`` stem.
    model_name_prefixes = ("grok",)
    twin_override_env_var = "GROK_BASE_URL"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_seconds: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key if api_key is not None else settings.xai_api_key,
            base_url=base_url if base_url is not None else settings.grok_base_url,
            default_model=(
                default_model if default_model is not None else settings.grok_default_model
            ),
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else settings.provider_timeout_seconds
            ),
            client=client,
        )
