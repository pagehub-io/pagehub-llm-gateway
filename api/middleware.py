"""ASGI middleware: X-Twin-* dev-only gate + request body size limit."""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.config import settings
from api.twin import header_to_env_var, set_overrides

_DEV_ENV = "development"


class TwinHeaderMiddleware:
    """Honor ``X-Twin-*`` headers in development; silently strip them everywhere else.

    Dev: copy each ``X-Twin-{Dep}-Base-Url: <url>`` header into a request-scoped
    contextvar keyed by the matching env-var name (``{DEP}_BASE_URL``). Outbound
    helpers consult the contextvar before falling back to the env default.

    Non-dev (staging / prod / anything not exactly "development"): drop every
    ``X-Twin-*`` header before any handler sees it — never log the value, since
    these can carry internal hostnames or attacker-controlled URLs (SSRF risk).
    Deny-by-default for new envs.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._active = settings.env == _DEV_ENV

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if self._active:
            overrides: dict[str, str] = {}
            for name, value in scope.get("headers", []):
                name_str = name.decode("latin-1")
                if not name_str.lower().startswith("x-twin-"):
                    continue
                env_var = header_to_env_var(name_str)
                if env_var:
                    overrides[env_var] = value.decode("latin-1")
            if overrides:
                set_overrides(overrides)
            await self.app(scope, receive, send)
            return

        headers = [
            (name, value)
            for (name, value) in scope.get("headers", [])
            if not name.lower().startswith(b"x-twin-")
        ]
        scope = dict(scope)
        scope["headers"] = headers
        await self.app(scope, receive, send)


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes`` with a 413."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await self._reject(send)
                        return
                except ValueError:
                    pass
                break

        seen = 0

        async def limited_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await self._reject(send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps(
            {"error": {"type": "request_too_large", "message": f"Request body exceeds {self.max_bytes} bytes."}}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _BodyTooLarge(Exception):
    pass
