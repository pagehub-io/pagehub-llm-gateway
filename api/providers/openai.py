"""OpenAIProvider — OpenAI (the company) backend.

Thin specialization of :class:`OpenAICompatibleProvider`. OpenAI's chat
completions endpoint is the de-facto spec other OpenAI-compatible providers
implement; the actual wire translation lives in
``api/providers/openai_compatible.py``. This file only declares the
OpenAI-specific bindings.

We exist primarily for **diagnostic value**: when the xAI Grok smoke produces
weak output, running the same end-to-end through OpenAI on a known-good
tool-use model (GPT-5, the o-series) tells us whether the gateway's canonical
translation is correct or whether we have a translation bug. If OpenAI works
and Grok doesn't, Grok is just weak on that prompt. If both fail in the same
shape, the bug is in our middle layer.
"""

from __future__ import annotations

import httpx

from api.config import settings
from api.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"
    # OpenAI's currently-marketed mainline ids: ``gpt-*`` (the GPT family,
    # including GPT-5) and ``o1-*`` / ``o3-*`` / ``o4-*`` (the reasoning
    # o-series). A future ``o5`` etc. is a one-line change to add.
    model_name_prefixes = ("gpt", "o1", "o3", "o4")
    twin_override_env_var = "OPENAI_BASE_URL"
    # OpenAI's GPT-5 and the o-series 400 on ``max_tokens`` — they require
    # ``max_completion_tokens``. Older OpenAI models (gpt-4o, gpt-4.1) accept
    # both, so the newer name is safe across the board. xAI Grok still wants
    # the historical ``max_tokens`` name.
    max_tokens_param_name = "max_completion_tokens"

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
            api_key=api_key if api_key is not None else settings.openai_api_key,
            base_url=base_url if base_url is not None else settings.openai_base_url,
            default_model=(
                default_model if default_model is not None else settings.openai_default_model
            ),
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else settings.provider_timeout_seconds
            ),
            client=client,
        )
