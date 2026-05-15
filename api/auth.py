"""Gateway authentication — accept either ``x-api-key`` or ``Authorization: Bearer``.

Claude Code's CLI sets ``ANTHROPIC_AUTH_TOKEN`` and sends it as a Bearer token; the
official Anthropic SDKs send ``x-api-key``. We accept both so any caller that thinks
it's talking to Anthropic works without modification.

Constant-time compare against ``GATEWAY_AUTH_TOKEN`` so a wrong-length token doesn't
leak length via timing.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status

from api.config import settings


def _extract_presented_token(request: Request) -> str | None:
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key
    auth = request.headers.get("authorization")
    if auth:
        parts = auth.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    return None


async def require_auth(request: Request) -> None:
    expected = settings.gateway_auth_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "type": "configuration_error",
                    "message": "GATEWAY_AUTH_TOKEN is not configured.",
                }
            },
        )
    presented = _extract_presented_token(request)
    if not presented or not hmac.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "type": "authentication_error",
                    "message": "Missing or invalid auth token. Provide x-api-key or Authorization: Bearer.",
                }
            },
        )
