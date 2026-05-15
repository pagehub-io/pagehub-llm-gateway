"""Gateway authentication.

- ``require_auth``: caller auth for /v1/messages and /v1/models. Accepts
  ``x-api-key`` (Anthropic SDK) OR ``Authorization: Bearer`` (Claude Code's
  ``ANTHROPIC_AUTH_TOKEN``). Compared constant-time against
  ``GATEWAY_AUTH_TOKEN``.

- ``require_admin_auth``: admin auth for /v1/admin/*. Accepts the SAME
  header schemes but compared against ``ADMIN_AUTH_TOKEN`` (a separate token,
  intentionally — a leaked caller token must not read request history).
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


def _enforce(expected: str, request: Request, *, missing_token_message: str) -> None:
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"type": "configuration_error", "message": missing_token_message}},
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


async def require_auth(request: Request) -> None:
    _enforce(
        settings.gateway_auth_token,
        request,
        missing_token_message="GATEWAY_AUTH_TOKEN is not configured.",
    )


async def require_admin_auth(request: Request) -> None:
    _enforce(
        settings.admin_auth_token,
        request,
        missing_token_message="ADMIN_AUTH_TOKEN is not configured.",
    )
