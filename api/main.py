"""FastAPI app factory + lifespan for pagehub-llm-gateway."""

from __future__ import annotations

import contextlib
import logging

from fastapi import FastAPI

from api.config import settings
from api.middleware import BodySizeLimitMiddleware, TwinHeaderMiddleware
from api.providers.base import Provider
from api.providers.grok import GrokProvider
from api.v1.messages_router import router as messages_router
from api.v1.service_router import router as service_router


def _build_provider() -> Provider:
    return GrokProvider()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = getattr(app.state, "provider", None) or _build_provider()
    try:
        yield
    finally:
        pass


def create_app() -> FastAPI:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = FastAPI(
        title="pagehub-llm-gateway",
        description=(
            "Anthropic-compatible gateway that routes Claude-style /v1/messages calls to "
            "non-Anthropic backends. v1 backend: xAI Grok. Point ANTHROPIC_BASE_URL + "
            "ANTHROPIC_AUTH_TOKEN at this gateway and Claude Code thinks it's talking to "
            "Anthropic."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.request_body_max_bytes)
    app.add_middleware(TwinHeaderMiddleware)

    app.include_router(service_router)
    app.include_router(messages_router)
    return app


app = create_app()
